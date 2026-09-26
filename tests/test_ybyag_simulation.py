"""YAG integration: data identity, material separation and amplifier ledgers."""
from pathlib import Path
import sys
import numpy as np
import pytest

from ybyag import YbYAGMaterial
from ybyag import material_data
from ybluag.model import YbLuAGMaterial, H, C
from ybluag.gallery import YbGallerySettings, _assembly_configuration, simulate_pulsed_seed
from ybluag.fluorescence import fluorescence_spectrum
from ybluag.regenerative import RegenerativeCavity
from hoyag.propagation import Grid2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
from ybluag_app import calculate_pulsed, calculate_structured, calculate
from desktop_simulation import YAG_FIELDS, validate_yb_payload


def request(**overrides):
    values = {key: default for key, _, default, _ in YAG_FIELDS}
    values.update(material='Yb:YAG', pump_W=.01, grid_n=32, optical_z_steps=1,
                  thermal_nr=4, thermal_nphi=4, thermal_nz=1,
                  signal_traversals=2, pump_passes=2, regen_round_trips=2,
                  operation_duration_s=0, cluster_contrast=0)
    values.update(overrides)
    return values


def test_runtime_spectra_match_github_data_and_y_site_density():
    m = YbYAGMaterial()
    table = np.genfromtxt(ROOT/'Yb-YAG/spectra/yb_yag_293K_legacy.csv', delimiter=',', names=True)
    a, e = m.cross_sections_m2(table['wavelength_nm'])
    np.testing.assert_array_equal(a, table['sigma_abs_cm2']*1e-4)
    np.testing.assert_array_equal(e, table['sigma_em_cm2']*1e-4)
    expected = 3*4552/.59362*6.02214076e23*.2
    assert m.number_density_m3 == pytest.approx(expected)
    assert m.lifetime_s == .00095
    assert m.number_density_m3 != YbLuAGMaterial(yb_at_percent=20, lifetime_s=.00095).number_density_m3


def test_vendored_data_are_byte_identical_to_imported_repository():
    for path in (ROOT/'src/ybyag/data').rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (ROOT/'Yb-YAG'/path.relative_to(ROOT/'src/ybyag/data')).read_bytes()


def test_yag_transparency_fluorescence_and_cavity_indices():
    m = YbYAGMaterial()
    _, gain = m.coefficients_m1(m.transparency_fraction())
    assert abs(gain) < 1e-10
    emission = fluorescence_spectrum(m)
    assert emission.wavelength_nm[0] == 905
    assert emission.wavelength_nm[-1] == 1095
    assert 905 < emission.energy_equivalent_wavelength_nm < 1095
    cavity = RegenerativeCavity().validate(m, 100e-6, Grid2D.square(32, .012), 1e4)
    assert cavity.host_index == pytest.approx(material_data.n_yag(1030))
    assert cavity.host_group_index > cavity.host_index
    assert cavity.host_index != pytest.approx(YbLuAGMaterial.cavity_phase_index, rel=1e-5)


def test_hot_gain_never_reuses_rt_or_luag_spectra():
    with pytest.raises(ValueError, match='pump spectra'):
        YbYAGMaterial().local_cross_sections_m2(969, 294)
    with pytest.raises(ValueError, match='905'):
        YbYAGMaterial().cross_sections_m2(900)
    with pytest.raises(ValueError, match='pump spectra'):
        calculate_pulsed(request(thermal_optical_mode='coupled_steady'))


def test_yag_assembly_replaces_crystal_properties_but_keeps_shared_copper():
    m = YbYAGMaterial()
    s = YbGallerySettings(assembly_property_model='yag_rt_proxy')
    with pytest.warns(material_data.ApproximationWarning):
        cfg = _assembly_configuration(m, s)
    assert cfg['thermal']['disk']['density_kg_m3'] < 5000
    assert cfg['thermal']['disk']['conductivity_W_mK'] == pytest.approx(
        material_data.thermal_conductivity_doped(300, 20, family='HT', interpolate_doping=True))
    assert cfg['optics']['index'] == pytest.approx(m.refractive_index())
    assert cfg['thermal']['plate']['conductivity_W_mK'] == 394
    assert 'YAG' in cfg['mechanical']['disk']['name']
    assert cfg['optics']['photoelastic_model'] is None


def test_15_percent_yag_assembly_uses_exact_concentration_table():
    material = YbYAGMaterial(yb_at_percent=15)
    cfg = _assembly_configuration(material, YbGallerySettings(
        assembly_property_model='yag_rt_proxy'))
    assert cfg['thermal']['disk']['conductivity_W_mK'] == pytest.approx(
        material_data.measured_thermal_conductivity_W_mK(293.15, 15))
    mass = material_data.doped_RT_density_heat_capacity(15)
    assert cfg['thermal']['disk']['density_kg_m3'] == mass['rho_kg_m3']
    assert cfg['thermal']['disk']['heat_capacity_J_kgK'] == mass['Cp_J_kgK']
    assert 'Aggarwal2005' in cfg['sources']['conductivity']


@pytest.mark.parametrize('architecture', ['ideal_multipass', 'regenerative'])
def test_both_yag_pulsed_architectures_share_conservative_population(architecture):
    r = calculate_pulsed(request(architecture=architecture), compute_thermal=False)
    assert r['material'] == 'Yb:YAG'
    assert r['yb_at_percent'] == 20
    assert r['lifetime_s_assumed'] == .00095
    assert r['output_energy_J'] > 0
    ledger = r['ideal_multipass'] or r['regenerative']
    assert ledger['population_photon_balance_relative_L1'] < 1e-3
    residual = ledger.get('optical_energy_balance_residual_J', ledger.get('cavity_energy_balance_residual_J'))
    assert abs(residual) < 1e-19
    assert r['thermal_optical_mode'] == 'cold'


def test_yag_copper_deformation_and_shaped_cw():
    r = calculate_pulsed(request(operation_duration_s=.01))
    assert r['thermal']['status'] == 'computed'
    assert r['thermal_timeline'] is not None
    assert r['thermal_feedback_applied']
    assert np.max(abs(np.asarray(r['thermal']['front_displacement_nm']))) > 0
    cw = calculate_structured(request(solver_mode='saturated_cw', signal_W=.001))
    assert cw['material'] == 'Yb:YAG'
    assert cw['thermal']['status'] == 'computed'
    assert len(cw['modes']) == 1


def test_yag_desktop_payload_and_plain_cw():
    payload = validate_yb_payload('pulsed', request())
    assert payload['yb_at_percent'] == 20
    cw = calculate(request(seed_W=.001))
    assert cw['material'] == 'Yb:YAG'
    assert cw['wavelength_nm'][0] == 905
    assert cw['signal_out_W'] > 0


def test_cold_thickness_and_rear_surface_phase_are_per_traversal():
    material = YbYAGMaterial()
    settings = YbGallerySettings(grid_n=32, z_steps=1, pump_power_W=.01)
    thickness_scale = np.ones((32, 32))
    thickness_scale[16, 16] = 1.01
    rear_height = np.zeros((32, 32))
    rear_height[16, 16] = 10e-9
    result = simulate_pulsed_seed(
        material, settings, 'Gaussian TEM00', 10e-9, 10e-12, 10_000, 2, 2,
        compute_thermal=False,
        dataset_physical={
            'thickness_scale': thickness_scale,
            'surface_figure_m': rear_height,
        },
    )
    wavelength_m = material.signal_wavelength_nm * 1e-9
    thickness_error_m = .01 * settings.thickness_m
    expected_opd_per_traversal_m = (
        (material.cavity_phase_index - 1) * thickness_error_m + 10e-9
    )
    assert result['static_cold_phase_rad'][16, 16] == pytest.approx(
        2 * np.pi * expected_opd_per_traversal_m / wavelength_m
    )
