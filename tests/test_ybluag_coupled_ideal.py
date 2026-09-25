"""Small checks of the shared ideal-multipass disk and coupled closure."""

from dataclasses import replace
import numpy as np
import pytest

from hoyag.propagation import Grid2D, optical_power
from ybluag.gallery import (YbGallerySettings, _periodic_ideal_multipass,
                            simulate_pulsed_seed)
from ybluag.model import YbLuAGMaterial, trapezoid


def _fixture():
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.973e-3,
                              pump_wavelength_nm=938)
    settings = YbGallerySettings(pump_power_W=.01, pump_radius_m=1e-3,
                                 thickness_m=100e-6,
                                 assembly_property_model="proposal_12at",
                                 grid_n=32, field_size_m=.012, z_steps=1,
                                 thermal_nr=4, thermal_nphi=4, thermal_nz=1,
                                 cluster_contrast=0, waist_m=.6e-3,
                                 post_disk_distance_m=0)
    return material, settings


def test_coupled_ideal_runs_one_shared_thermal_optical_state():
    material, settings = _fixture()
    result = simulate_pulsed_seed(
        material, settings, "Gaussian TEM00", 1e-9, 10e-12, 1e4, 2,
        pump_passes=2, architecture="ideal_multipass",
        thermal_optical_mode="coupled_steady")
    assert result["coupled_steady_convergence"]["status"] == "converged"
    assert result["ideal_multipass"]["population_photon_balance_relative_L1"] < 1e-3
    assert abs(result["ideal_multipass"]["optical_energy_balance_residual_J"]) < 1e-20
    assert result["thermal_feedback_applied"]
    assert result["optical_temperature_K_by_slice"] is not None
    assert result["steady_coupled_temperature_max_C"] > 20
    assert result["coupled_steady_convergence"]["history"][-1][
        "coherent_field_overlap_error"] < 1e-6


def test_ideal_cold_limit_static_phase_and_shared_depletion():
    material, settings = _fixture()
    grid = Grid2D.square(32, .012)
    x, y = grid.mesh
    seed = np.exp(-(x*x+y*y)/(.6e-3)**2)
    seed /= np.sqrt(optical_power(seed, grid))
    pump = np.full(grid.shape, 1e4)
    scale = np.ones((1, *grid.shape))
    t = np.linspace(-30e-12, 30e-12, 41)
    shape = np.exp(-4*np.log(2)*(t/10e-12)**2)
    shape /= trapezoid(shape, t)
    args = (material, settings, grid, seed, 1e-9, pump, scale, 2, 1e4,
            2, t, shape, 1.9e-19)
    cold = _periodic_ideal_multipass(*args)
    at_reference = _periodic_ideal_multipass(
        *args, temperature_K_by_slice=np.full(scale.shape, 293.15),
        encounter_opd_m=np.zeros(grid.shape))
    np.testing.assert_allclose(at_reference["output_field"],
                               cold["output_field"], rtol=1e-9, atol=1e-12)
    assert np.mean(cold["excited_fraction_before_pulse_by_slice"]) < 1
    phase = np.full(grid.shape, .2)
    shifted = _periodic_ideal_multipass(
        *args, encounter_opd_m=phase*1030e-9/(2*np.pi))
    np.testing.assert_allclose(abs(shifted["output_field"]),
                               abs(cold["output_field"]), rtol=1e-10)
    np.testing.assert_allclose(shifted["output_field"],
                               cold["output_field"]*np.exp(.4j),
                               rtol=1e-9, atol=1e-12)


def test_static_disk_phase_does_not_change_ideal_material_gain():
    material, settings = _fixture()
    kwargs = dict(pump_passes=2, architecture="ideal_multipass",
                  compute_thermal=False)
    cold = simulate_pulsed_seed(material, settings, "Gaussian TEM00",
                                1e-9, 10e-12, 1e4, 2, **kwargs)
    phase = np.full((settings.grid_n, settings.grid_n), .12)
    changed = simulate_pulsed_seed(material, settings, "Gaussian TEM00",
                                   1e-9, 10e-12, 1e4, 2,
                                   static_cold_phase_rad=phase, **kwargs)
    assert changed["output_energy_J"] == pytest.approx(cold["output_energy_J"],
                                                        rel=1e-10)
    np.testing.assert_allclose(changed["pre_pulse_excited_fraction_by_slice"],
                               cold["pre_pulse_excited_fraction_by_slice"])
    in_beam = cold["output_fluence_J_m2"] > .01*cold["output_fluence_J_m2"].max()
    delta = np.angle(np.exp(1j*(changed["output_phase"]-cold["output_phase"])))
    np.testing.assert_allclose(delta[in_beam], .24, atol=1e-9)


def test_uniform_temperature_changes_gain_by_local_spectroscopy():
    material, settings = _fixture()
    beta = np.full((1, 2, 2), .5)
    density = material.number_density_m3
    for temperature in (293.15, 353.15):
        _, gain = material.coefficients_m1(beta, temperature)
        sa, se = material.local_cross_sections_m2(
            material.signal_wavelength_nm, np.full(beta.shape, temperature))
        expected = density*(.5*se-.5*sa)
        np.testing.assert_allclose(gain, expected, rtol=1e-12)
    _, cold = material.coefficients_m1(beta, 293.15)
    _, hot = material.coefficients_m1(beta, 353.15)
    assert not np.allclose(cold, hot, rtol=1e-3, atol=0)


def test_configured_ideal_relay_changes_next_encounter_energy():
    material, settings = _fixture()
    kwargs = dict(pump_passes=2, architecture="ideal_multipass",
                  compute_thermal=False)
    ideal = simulate_pulsed_seed(material, settings, "Gaussian TEM00",
                                 1e-9, 10e-12, 1e4, 2, **kwargs)
    lossy = simulate_pulsed_seed(
        material, replace(settings, ideal_relay_power_retention=.8),
        "Gaussian TEM00", 1e-9, 10e-12, 1e4, 2, **kwargs)
    assert lossy["output_energy_J"] < ideal["output_energy_J"]
    assert lossy["ideal_multipass"]["ideal_relay_loss_J"] > 0
    assert lossy["ideal_multipass"]["encounter_exit_energies_J"][1] < (
        ideal["ideal_multipass"]["encounter_exit_energies_J"][1])


def test_slm_candidate_improves_after_complete_coupled_forward_rerun():
    from ybluag.field_metrics import coherent_overlap
    material, initial = _fixture()
    settings = replace(initial, slm_to_disk_distance_m=.001)
    grid = Grid2D.square(settings.grid_n, settings.field_size_m)
    x, y = grid.mesh
    static = .12*(x*x+y*y)/(.6e-3)**2
    kwargs = dict(pump_passes=2, architecture="ideal_multipass",
                  thermal_optical_mode="coupled_steady")
    target = simulate_pulsed_seed(material, settings, "Gaussian TEM00",
                                  1e-9, 10e-12, 1e4, 2, **kwargs)
    uncorrected = simulate_pulsed_seed(
        material, settings, "Gaussian TEM00", 1e-9, 10e-12, 1e4, 2,
        static_cold_phase_rad=static, **kwargs)
    corrected = simulate_pulsed_seed(
        material, settings, "Gaussian TEM00", 1e-9, 10e-12, 1e4, 2,
        static_cold_phase_rad=static,
        slm_correction_phase_rad=-2*static, **kwargs)
    desired = target["output_complex_field_sqrt_J_m"]
    before = coherent_overlap(desired, uncorrected["output_complex_field_sqrt_J_m"])
    after = coherent_overlap(desired, corrected["output_complex_field_sqrt_J_m"])
    assert after > before + 1e-6


def test_browser_api_accepts_coupled_ideal_result():
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples"))
    from ybluag_app import calculate_pulsed
    result = calculate_pulsed({
        "pump_W": .01, "grid_n": 32, "field_size_mm": 12,
        "optical_z_steps": 1, "thermal_nr": 4,
        "thermal_nphi": 4, "thermal_nz": 1,
        "operation_duration_s": 0, "signal_traversals": 2,
        "pump_passes": 2, "thermal_optical_mode": "coupled_steady"})
    assert result["architecture"] == "ideal_multipass"
    assert result["steady_state_observation"]["mode"] == "steady_state"
    assert result["coupled_steady_convergence"]["status"] == "converged"
    json.dumps(result, allow_nan=False)
