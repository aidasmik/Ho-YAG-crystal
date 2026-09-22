import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate, gaussian_beam
from hoyag.temporal import (
    TimeGrid,
    apply_gdd,
    combine_spatial_temporal,
    gaussian_temporal_envelope,
    propagate_spatiotemporal,
    pulse_intensity_fwhm_s,
    spatiotemporal_energy,
    temporal_energy,
)


def test_gaussian_temporal_fwhm_matches_requested_duration():
    time = TimeGrid.centered(4096, 20e-12)
    duration = 1.0e-12
    pulse = gaussian_temporal_envelope(time, duration, normalize=False)
    measured = pulse_intensity_fwhm_s(pulse, time)
    assert abs(measured - duration) / duration < 2e-3


def test_gaussian_time_bandwidth_product_is_transform_limited():
    time = TimeGrid.centered(32768, 200e-12)
    duration = 1.0e-12
    pulse = gaussian_temporal_envelope(time, duration, normalize=False)
    spectrum = np.fft.fftshift(np.fft.fft(pulse))
    frequency = np.fft.fftshift(time.frequency_hz)
    spectral_intensity = np.abs(spectrum) ** 2
    idx = np.flatnonzero(spectral_intensity >= spectral_intensity.max() / 2)
    bandwidth = frequency[idx[-1]] - frequency[idx[0]]
    assert abs(duration * bandwidth - 0.441) < 0.015


def test_gdd_broadening_matches_analytic_gaussian_formula():
    time = TimeGrid.centered(8192, 30e-12)
    duration = 1.0e-12
    gdd = 0.5e-24
    pulse0 = gaussian_temporal_envelope(time, duration, normalize=False)
    pulse1 = apply_gdd(pulse0, time, gdd)
    measured = pulse_intensity_fwhm_s(pulse1, time)
    expected = duration * np.sqrt(1 + (4 * np.log(2) * gdd / duration**2) ** 2)
    assert abs(measured - expected) / expected < 5e-3


def test_gdd_conserves_temporal_energy():
    time = TimeGrid.centered(4096, 20e-12)
    pulse = gaussian_temporal_envelope(time, 1e-12)
    e0 = temporal_energy(pulse, time)
    e1 = temporal_energy(apply_gdd(pulse, time, 0.75e-24), time)
    assert abs(e1 - e0) / e0 < 1e-12


def test_spatiotemporal_energy_conserved_for_passive_propagation():
    grid = Grid2D.square(96, 4e-3)
    time = TimeGrid.centered(256, 30e-12)
    spatial = gaussian_beam(grid, 0.45e-3)
    temporal = gaussian_temporal_envelope(time, 5e-12)
    field = combine_spatial_temporal(spatial, temporal, grid, time)
    e0 = spatiotemporal_energy(field, grid, time)
    out = propagate_spatiotemporal(
        field, grid, time,
        wavelength_m=1.9077e-6,
        distance_m=18e-3,
        refractive_index=1.8018687,
        beta2_s2_per_m=-4.460681783044079e-26,
    )
    e1 = spatiotemporal_energy(out, grid, time)
    assert abs(e1 - e0) / e0 < 1e-11


def test_zero_beta2_reduces_to_stage1_spatial_propagation():
    grid = Grid2D.square(64, 4e-3)
    time = TimeGrid.centered(64, 20e-12)
    spatial = gaussian_beam(grid, 0.45e-3)
    temporal = gaussian_temporal_envelope(time, 5e-12, normalize=False)
    field = combine_spatial_temporal(spatial, temporal, grid, time)
    z = 18e-3

    out3d = propagate_spatiotemporal(
        field, grid, time,
        wavelength_m=1.9077e-6,
        distance_m=z,
        refractive_index=1.8018687,
        beta2_s2_per_m=0.0,
    )
    out2d = angular_spectrum_propagate(
        spatial, grid,
        wavelength_m=1.9077e-6,
        distance_m=z,
        refractive_index=1.8018687,
    )

    peak_i = np.argmax(np.abs(temporal))
    assert np.allclose(out3d[peak_i], temporal[peak_i] * out2d, rtol=1e-11, atol=1e-11)


def test_10ps_pulse_has_negligible_gvd_broadening_in_18mm_yag():
    time = TimeGrid.centered(2048, 80e-12)
    pulse = gaussian_temporal_envelope(time, 10e-12, normalize=False)
    f0 = pulse_intensity_fwhm_s(pulse, time)
    gdd = -4.460681783044079e-26 * 18e-3
    f1 = pulse_intensity_fwhm_s(apply_gdd(pulse, time, gdd), time)
    assert abs(f1 - f0) / f0 < 1e-4
