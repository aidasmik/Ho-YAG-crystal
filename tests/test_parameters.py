import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MATERIAL = json.loads((ROOT / "config" / "hoyag_stage0_parameters.json").read_text())
PUMP = json.loads((ROOT / "config" / "picosecond_pump_generic.json").read_text())

C0 = 299_792_458.0
H = 6.62607015e-34


def _sellmeier_n(lambda_m):
    s = MATERIAL["optical"]["preferred_dispersion_model"]
    lam = np.asarray(lambda_m) * 1e6
    n2 = 1 + s["A"] * lam**2 / (lam**2 - s["B_um2"])
    n2 += s["C"] * lam**2 / (lam**2 - s["D_um2"])
    return np.sqrt(n2)


def test_stage0_quantum_defect():
    op = MATERIAL["operating_point"]
    expected = 1 - op["pump_wavelength_m"] / op["laser_wavelength_m"]
    assert np.isclose(expected, op["quantum_defect_fraction_simple"], rtol=1e-12)


def test_stage0_thermal_diffusivity_unit_conversion():
    th = MATERIAL["thermal"]
    expected = th["thermal_conductivity_W_mK"] / (
        th["specific_heat_J_kgK"] * th["density_kg_m3"]
    )
    assert np.isclose(expected, th["thermal_diffusivity_m2_s_derived"], rtol=1e-12)


def test_stage0_sellmeier_reference_indices():
    s = MATERIAL["optical"]["preferred_dispersion_model"]
    n_p = _sellmeier_n(MATERIAL["operating_point"]["pump_wavelength_m"])
    n_l = _sellmeier_n(MATERIAL["operating_point"]["laser_wavelength_m"])
    assert np.isclose(n_p, s["n_at_pump_1907p7_nm"], rtol=1e-12)
    assert np.isclose(n_l, s["n_at_laser_2090p3_nm"], rtol=1e-12)


def test_stage0p_average_power():
    p = PUMP["pulse_defaults"]
    assert np.isclose(
        p["pulse_energy_J"] * p["repetition_rate_Hz"],
        p["average_power_W_derived"],
        rtol=1e-12,
    )


def test_stage0p_group_index_and_transit_time():
    w = PUMP["wavelength_and_material"]
    lam = w["pump_center_wavelength_m"]
    h = 1e-12
    dn_dlambda = (_sellmeier_n(lam + h) - _sellmeier_n(lam - h)) / (2 * h)
    ng = float(_sellmeier_n(lam) - lam * dn_dlambda)
    transit = w["reference_crystal_length_m"] * ng / C0
    assert np.isclose(ng, w["yag_group_index_at_pump_derived"], rtol=2e-9)
    assert np.isclose(transit, w["group_transit_time_s_derived"], rtol=2e-9)


def test_stage0p_beta2_from_sellmeier():
    w = PUMP["wavelength_and_material"]
    lam0 = w["pump_center_wavelength_m"]
    omega0 = 2 * np.pi * C0 / lam0
    domega = 1e12

    def beta(omega):
        lam = 2 * np.pi * C0 / omega
        return _sellmeier_n(lam) * omega / C0

    beta2 = (beta(omega0 + domega) - 2 * beta(omega0) + beta(omega0 - domega))
    beta2 /= domega**2
    stored = PUMP["linear_dispersion_derived_from_sellmeier"][
        "beta2_s2_per_m_at_1907p7nm"
    ]
    assert np.isclose(beta2, stored, rtol=3e-4)


def test_stage0p_transform_limited_bandwidth():
    p = PUMP["pulse_defaults"]
    s = PUMP["transform_limited_gaussian_spectrum"]
    delta_nu = s["time_bandwidth_product_fwhm"] / p["pulse_duration_fwhm_s"]
    delta_lambda = (
        PUMP["wavelength_and_material"]["pump_center_wavelength_m"] ** 2
        * delta_nu
        / C0
    )
    assert np.isclose(delta_nu, s["nominal_10ps_bandwidth_Hz"], rtol=1e-12)
    assert np.isclose(delta_lambda * 1e9, s["nominal_10ps_bandwidth_nm"], rtol=1e-12)


def test_stage0p_saturation_fluence_and_nominal_peak_fluence():
    sat = PUMP["saturation_estimates"]
    lam = PUMP["wavelength_and_material"]["pump_center_wavelength_m"]
    photon_energy = H * C0 / lam
    fsat = photon_energy / (
        sat["sigma_abs_pump_m2"] + sat["sigma_em_pump_m2"]
    )
    assert np.isclose(photon_energy, sat["pump_photon_energy_J"], rtol=1e-12)
    assert np.isclose(fsat, sat["saturation_fluence_J_m2_derived"], rtol=1e-12)

    p = PUMP["pulse_defaults"]
    # For I(r)=I0 exp(-2r^2/w^2), D4sigma=2w.
    w = p["beam_D4sigma_waist_diameter_m"] / 2
    peak_fluence = 2 * p["pulse_energy_J"] / (np.pi * w**2)
    assert np.isclose(
        peak_fluence / 1e4,
        sat["nominal_gaussian_peak_fluence_J_cm2_derived"],
        rtol=1e-12,
    )
