"""Small fixtures for the Yb:LuAG hot-model corrections."""

from dataclasses import replace
import numpy as np
import pytest

from ybluag.model import YbLuAGMaterial, spectral_cross_sections_m2
from ybluag.gallery import (YbGallerySettings, _assembly_configuration,
                            _conservative_heat_polar)
from ybluag.assembly import solve_yb_assembly, solve_yb_cooler_temperature
from hoyag.thermal import DiskThermalMesh
from ybluag.regenerative import (RegenerativeCavity, amplify_regenerative,
                                 _local_fluence_transfer)
from hoyag.propagation import Grid2D, gaussian_beam
from ybluag.field_metrics import coherent_overlap
from ybluag.diagnostics import spectral_gain_screen
from ybluag.structured import propagate_structured_small_signal


def test_local_temperature_cross_sections_preserve_scalar_query():
    material = YbLuAGMaterial()
    temperatures = np.array([[293.15, 353.15], [413.15, 473.15]])
    absorption, emission = material.local_cross_sections_m2(
        material.signal_wavelength_nm, temperatures)
    assert absorption.shape == temperatures.shape
    assert emission.shape == temperatures.shape
    for index in np.ndindex(temperatures.shape):
        expected = spectral_cross_sections_m2(
            material.signal_wavelength_nm, float(temperatures[index]))
        np.testing.assert_allclose((absorption[index], emission[index]), expected,
                                   rtol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        material.local_cross_sections_m2(1030, np.array([293.15, 500.0]))


def test_local_coefficients_use_density_without_changing_scalar_api():
    material = YbLuAGMaterial()
    beta = np.array([[.1, .5], [.8, .9]])
    density = np.array([[0, 1], [2, .5]])*material.number_density_m3
    pump, gain = material.local_coefficients_m1(beta, density)
    pump_ref, gain_ref = material.coefficients_m1(beta)
    np.testing.assert_allclose(pump, pump_ref*density/material.number_density_m3)
    np.testing.assert_allclose(gain, gain_ref*density/material.number_density_m3)


def test_zero_heat_recovers_cold_assembly():
    settings = YbGallerySettings(grid_n=32, field_size_m=.012,
                                 thickness_m=100e-6, thermal_nr=4,
                                 thermal_nphi=4, thermal_nz=1)
    grid = Grid2D.square(32, .012)
    mesh = DiskThermalMesh.disk(nr=4, nphi=4, nz=1,
                                radius_m=settings.disk_radius_m,
                                thickness_m=settings.thickness_m)
    heat = _conservative_heat_polar(np.zeros((1, *grid.shape)), grid, mesh)
    assembly = solve_yb_assembly(
        mesh, heat, grid, _assembly_configuration(YbLuAGMaterial(), settings))
    assert np.all(heat == 0)
    np.testing.assert_allclose(assembly.temperature.disk_temperature_K,
                               293.15, atol=1e-10)
    assert np.max(np.abs(assembly.screens.mean_roundtrip_opd_m)) < 1e-18


def test_manufactured_uniform_thermal_index_opd():
    settings = YbGallerySettings(grid_n=32, field_size_m=.012,
                                 thickness_m=100e-6, thermal_nr=4,
                                 thermal_nphi=4, thermal_nz=1)
    grid = Grid2D.square(32, .012)
    mesh = DiskThermalMesh.disk(nr=4, nphi=4, nz=1,
                                radius_m=settings.disk_radius_m,
                                thickness_m=settings.thickness_m)
    config = _assembly_configuration(YbLuAGMaterial(), settings)
    cold = solve_yb_cooler_temperature(mesh, np.zeros(mesh.shape), config)
    manufactured = replace(cold, disk_temperature_K=np.full(mesh.shape, 298.15))
    result = solve_yb_assembly(mesh, np.zeros(mesh.shape), grid, config,
                               temperature=manufactured)
    x, y = grid.mesh
    inside = x*x+y*y < settings.disk_radius_m**2
    expected = config["optics"]["dn_dT_K1"]*5*settings.thickness_m
    np.testing.assert_allclose(
        result.screens.thermal_single_pass_opd_m[inside], expected,
        rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("doping", [11.7, 12.0, 12.3])
@pytest.mark.parametrize("thickness_um", [100, 150, 200])
def test_assembly_geometry_is_independent_of_doping(doping, thickness_um):
    material = YbLuAGMaterial(yb_at_percent=doping, lifetime_s=0.973e-3)
    settings = YbGallerySettings(thickness_m=thickness_um*1e-6,
                                 disk_radius_m=4.8e-3,
                                 assembly_property_model="proposal_12at")
    config = _assembly_configuration(material, settings)
    assert config["geometry"]["disk_radius_m"] == settings.disk_radius_m
    assert config["geometry"]["disk_thickness_m"] == settings.thickness_m
    assert config["thermal"]["disk"]["conductivity_W_mK"] == 7.2
    assert config["sources"]["selected_property_model"] == "proposal_12at"


def test_regenerative_screen_is_applied_at_encounters_not_lumped_at_output():
    grid = Grid2D.square(64, 12e-3)
    material = YbLuAGMaterial()
    seed = gaussian_beam(grid, .6e-3)
    x, y = grid.mesh
    opd = 2e-7 * (x/.8e-3)**2 * np.exp(-(x*x+y*y)/(1.5e-3)**2)
    args = (material, grid, seed, 10e-9, np.zeros(grid.shape),
            np.ones((2, *grid.shape)), 100e-6, 2, 1e4,
            RegenerativeCavity(round_trips=3), 1.9e-19, 0)
    cold = amplify_regenerative(*args)["output_field"]
    hot = amplify_regenerative(*args, encounter_opd_m=opd)["output_field"]
    lumped = cold * np.exp(2j*np.pi*6*opd/(1030e-9))
    assert coherent_overlap(hot, lumped) < .999
    assert not np.allclose(abs(hot)**2, abs(cold)**2, rtol=1e-4, atol=1e-18)


def test_local_temperature_regenerative_matches_cold_limit():
    grid = Grid2D.square(32, 12e-3)
    material = YbLuAGMaterial()
    seed = gaussian_beam(grid, .6e-3)
    args = (material, grid, seed, 10e-9, np.zeros(grid.shape),
            np.ones((2, *grid.shape)), 100e-6, 2, 1e4,
            RegenerativeCavity(round_trips=2), 1.9e-19, 0)
    cold = amplify_regenerative(*args)
    same = amplify_regenerative(
        *args, temperature_K_by_slice=np.full((2, *grid.shape), 293.15),
        encounter_opd_m=np.zeros(grid.shape))
    np.testing.assert_allclose(same["output_field"], cold["output_field"],
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(same["heat_W_m3_by_slice"],
                               cold["heat_W_m3_by_slice"], rtol=1e-8, atol=1e-6)
    hot = amplify_regenerative(
        *args, temperature_K_by_slice=np.full((2, *grid.shape), 353.15))
    assert hot["output_energy_J"] != pytest.approx(cold["output_energy_J"],
                                                     abs=1e-17)


def test_local_fluence_transfer_is_monotone_and_nonnegative():
    incoming = np.array([[0., 1., 10.], [100., 1e4, 1e6]])
    gain = np.full(incoming.shape, 2.)
    saturation = np.full(incoming.shape, 100.)
    output = _local_fluence_transfer(incoming, gain, saturation)
    assert np.all(output >= 0)
    assert np.all(np.diff(output.ravel()) >= 0)
    assert output[0, 0] == 0


def test_spatial_spectral_screen_is_padding_invariant_and_field_weighted():
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.973e-3)
    beta = np.zeros((2, 4, 4))
    beta[:, 1:3, 1:3] = np.array([[.3, .7], [.2, .6]])
    density = np.zeros_like(beta)
    density[:, 1:3, 1:3] = material.number_density_m3
    incident = np.zeros((4, 4))
    incident[1:3, 1:3] = np.array([[4., 1.], [1., 2.]])
    def screen(b, d, f):
        return spectral_gain_screen(
            material, source_fwhm_fs=300, stretched_fwhm_ps=10,
            material_traversals=2, thickness_m=100e-6,
            excited_fraction_by_slice=b, density_m3_by_slice=d,
            incident_fluence_J_m2=f)
    small = screen(beta, density, incident)
    padded = screen(np.pad(beta, ((0, 0), (3, 3), (3, 3))),
                    np.pad(density, ((0, 0), (3, 3), (3, 3))),
                    np.pad(incident, 3))
    np.testing.assert_allclose(small["unsaturated_gain"], padded["unsaturated_gain"],
                               rtol=1e-13)
    center = len(small["wavelength_nm"]) // 2
    sa, se = material.cross_sections_m2(material.signal_wavelength_nm)
    local = np.exp(np.sum(density*(se*beta-sa*(1-beta)), axis=0)*100e-6)
    expected = np.sum(incident*local)/np.sum(incident)
    assert small["unsaturated_gain"][center] == pytest.approx(expected, rel=1e-12)
    homogeneous = screen(np.full_like(beta, .45),
                         np.full_like(density, material.number_density_m3),
                         np.ones_like(incident))
    analytic = np.exp(material.number_density_m3*
                      (se*.45-sa*.55)*100e-6*2)
    assert homogeneous["unsaturated_gain"][center] == pytest.approx(analytic,
                                                                      rel=1e-12)


def test_spectral_center_matches_matching_weak_field_solver():
    material = YbLuAGMaterial()
    grid = Grid2D.square(32, .008)
    field = gaussian_beam(grid, .6e-3)
    x, y = grid.mesh
    pump = 1e8*np.exp(-2*(x*x+y*y)/(1e-3)**2)
    main = propagate_structured_small_signal(
        material, field, grid, pump, 100e-6, 2,
        include_diffraction=False)
    beta = main.excited_fraction_by_step
    screen = spectral_gain_screen(
        material, source_fwhm_fs=300, stretched_fwhm_ps=10,
        material_traversals=1, thickness_m=100e-6,
        excited_fraction_by_slice=beta,
        density_m3_by_slice=np.full(beta.shape, material.number_density_m3),
        incident_fluence_J_m2=abs(field)**2)
    center = len(screen["wavelength_nm"])//2
    assert screen["unsaturated_gain"][center] == pytest.approx(
        main.output_power_W/main.input_power_W, rel=1e-12)
