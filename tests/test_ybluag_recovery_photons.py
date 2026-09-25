"""Boundary-photon checks independent of the heat-by-subtraction ledger."""
import numpy as np
import pytest

from ybluag.model import YbLuAGMaterial, H, C
from ybluag.multipass_pump import transport_multipass_pump, recover_pumped_population


def test_thick_pump_cell_rates_equal_boundary_photons():
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                              pump_wavelength_nm=969)
    scale = np.ones((1, 2))
    beta = np.array([[0.0, .2]])
    thickness = 1e-3
    mean, absorbed, _ = transport_multipass_pump(
        material, np.full(2, 1e8), scale, thickness, 3, beta)
    up, down = material.rates_s1(mean, 0)
    excitation_rate = material.number_density_m3*thickness*(up*(1-beta)-down*beta)
    np.testing.assert_allclose(excitation_rate, absorbed/(H*C/969e-9), rtol=2e-13)


def test_recovery_without_pump_is_exact_decay():
    material = YbLuAGMaterial()
    beta = np.full((2, 2), .7)
    duration = .4e-3
    final, absorbed, integral = recover_pumped_population(
        material, np.zeros(2), np.ones_like(beta), 100e-6, 2, beta, duration, 4)
    np.testing.assert_allclose(final, beta*np.exp(-duration/material.lifetime_s), rtol=2e-14)
    np.testing.assert_allclose(integral, beta*material.lifetime_s*(-np.expm1(-duration/material.lifetime_s)))
    assert np.all(absorbed == 0)


def test_depleted_pump_recovery_photon_error_converges():
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                              pump_wavelength_nm=969)
    beta = np.full((2, 1), .01)
    density = material.number_density_m3
    errors = []
    for steps in (4, 8, 16):
        final, absorbed, integral = recover_pumped_population(
            material, np.full(1, 1e8), np.ones_like(beta), 100e-6,
            10, beta, 1e-4, steps)
        photons = absorbed/(H*C/969e-9)
        stored_and_decayed = density*50e-6*(final-beta+integral/material.lifetime_s)
        errors.append(float(np.sum(abs(photons-stored_and_decayed))/np.sum(photons)))
        assert np.all((final >= 0) & (final <= 1))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-3


def test_recovery_rejects_negative_time():
    with pytest.raises(ValueError):
        recover_pumped_population(YbLuAGMaterial(), np.zeros(1), np.ones((1, 1)),
                                  100e-6, 2, np.zeros((1, 1)), -1)


def test_ideal_gain_bound_includes_each_relay_loss():
    from ybluag.diagnostics import gain_feasibility
    kwargs = dict(thickness_m=100e-6, pump_passes=10, signal_traversals=4)
    perfect = gain_feasibility(YbLuAGMaterial(), **kwargs)
    lossy = gain_feasibility(YbLuAGMaterial(), **kwargs, ideal_relay_power_retention=.8)
    assert lossy['configured_optical_retention'] == pytest.approx(.8**3)
    assert lossy['pump_ceiling_after_configured_losses'] == pytest.approx(
        perfect['pump_ceiling_after_configured_losses']*.8**3)


def test_desktop_reference_keeps_slm_correction_and_removes_static_disk():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
    from ybluag_app import calculate_pulsed
    phase = np.broadcast_to(np.linspace(-.3, .3, 32), (32, 32)).copy()
    request = dict(pump_W=.01, grid_n=32, optical_z_steps=1,
                   thermal_nr=4, thermal_nphi=4, thermal_nz=1,
                   cluster_contrast=0, signal_traversals=2, pump_passes=2,
                   operation_duration_s=0, disk_radius_mm=3,
                   slm_correction_phase_rad=phase)
    clean = calculate_pulsed(request, compute_thermal=False)
    changed = calculate_pulsed({**request, 'static_cold_phase_rad': np.full((32, 32), .12)},
                               compute_thermal=False)
    assert changed['pump_wavelength_nm'] == 969
    assert changed['disk_radius_mm'] == 3
    delta = np.angle(np.exp(1j*(np.array(changed['uniform_isothermal_output_phase'])-
                                np.array(clean['output_phase']))))
    mask = np.array(clean['output_fluence_J_m2']) > .01*np.max(clean['output_fluence_J_m2'])
    np.testing.assert_allclose(delta[mask], 0, atol=1e-10)
    actual = np.angle(np.exp(1j*(np.array(changed['output_phase'])-np.array(clean['output_phase']))))
    np.testing.assert_allclose(actual[mask], .24, atol=1e-10)
