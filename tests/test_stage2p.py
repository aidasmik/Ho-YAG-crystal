import numpy as np

from hoyag.populations import (
    H,
    C0,
    I7,
    HoYAGFourLevelParams,
    four_level_rhs,
    ground_state_populations,
    integrate_populations,
    laser_gain_coefficient_m1,
    propagate_single_pulse_hoyag,
    pump_material_step,
    scale_pulse_to_energy,
)
from hoyag.propagation import Grid2D, normalize_power
from hoyag.spectroscopy import effective_pump_absorption_cross_section_295K
from hoyag.temporal import (
    TimeGrid,
    combine_spatial_temporal,
    gaussian_temporal_envelope,
    spatiotemporal_energy,
)


def _uniform_pulse(grid, time, duration_s):
    spatial = normalize_power(np.ones(grid.shape, dtype=complex), grid)
    temporal = gaussian_temporal_envelope(time, duration_s)
    return combine_spatial_temporal(spatial, temporal, grid, time)


def test_rate_equations_conserve_total_ho_population():
    p = HoYAGFourLevelParams()
    state = np.array([1e24, 2e24, 3e24, p.N_total_m3 - 6e24])
    derivative = four_level_rhs(
        state, p,
        pump_abs_rate_s1=1e6,
        pump_em_rate_s1=2e5,
        laser_abs_rate_s1=3e4,
        laser_em_rate_s1=1e4,
    )
    assert abs(np.sum(derivative)) < 1e-12 * np.max(np.abs(derivative))


def test_ground_state_without_light_is_stationary():
    p = HoYAGFourLevelParams()
    state = ground_state_populations(p)
    assert np.allclose(four_level_rhs(state, p), 0.0)


def test_physical_pulse_scaling_integrates_to_requested_energy():
    grid = Grid2D.square(8, 1e-3)
    time = TimeGrid.centered(256, 50e-12)
    field = _uniform_pulse(grid, time, 10e-12)
    physical = scale_pulse_to_energy(field, grid, time, 100e-6)
    assert np.isclose(spatiotemporal_energy(physical, grid, time), 100e-6, rtol=1e-12)


def test_weak_pulse_excitation_matches_sigma_times_fluence():
    p = HoYAGFourLevelParams()
    time = TimeGrid.centered(4096, 20e-12)
    temporal = gaussian_temporal_envelope(time, 1e-12, normalize=False)

    fluence = 10.0
    intensity = np.abs(temporal) ** 2
    intensity *= fluence / (np.sum(intensity) * time.dt)

    final = integrate_populations(intensity, time, p)
    fraction = final[I7] / p.N_total_m3
    expected = p.sigma_abs_pump_m2 * fluence / (H * C0 / p.pump_wavelength_m)
    assert abs(fraction - expected) / expected < 2e-3


def test_picosecond_bandwidth_changes_effective_absorption():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    spatial = normalize_power(np.ones(grid.shape, dtype=complex), grid)

    t10 = TimeGrid.centered(4096, 100e-12)
    e10 = combine_spatial_temporal(
        spatial, gaussian_temporal_envelope(t10, 10e-12), grid, t10
    )
    s10 = effective_pump_absorption_cross_section_295K(
        e10, t10, p.pump_wavelength_m
    )

    t1 = TimeGrid.centered(8192, 50e-12)
    e1 = combine_spatial_temporal(
        spatial, gaussian_temporal_envelope(t1, 1e-12), grid, t1
    )
    s1 = effective_pump_absorption_cross_section_295K(
        e1, t1, p.pump_wavelength_m
    )

    assert 1.15e-24 < s10 < 1.27e-24
    assert 7.0e-25 < s1 < 9.0e-25
    assert s1 < s10


def test_low_fluence_propagation_recovers_beer_lambert():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    time = TimeGrid.centered(256, 50e-12)
    field = _uniform_pulse(grid, time, 10e-12)
    length = 1e-3

    result = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=1e-10,
        length_m=length,
        nz=20,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
    )

    expected = np.exp(-p.N_total_m3 * p.sigma_abs_pump_m2 * length)
    assert abs(result.transmission - expected) / expected < 1e-6


def test_saturation_increases_transmission_and_creates_gain():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    time = TimeGrid.centered(256, 50e-12)
    field = _uniform_pulse(grid, time, 10e-12)
    length = 18e-3

    result = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=0.05,
        length_m=length,
        nz=100,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        store_full_populations=True,
    )
    linear = np.exp(-p.N_total_m3 * p.sigma_abs_pump_m2 * length)

    assert result.transmission > 1.3 * linear
    assert np.max(result.peak_I7_fraction_by_slice) > 0.30

    gains = [
        np.max(laser_gain_coefficient_m1(s, p))
        for s in result.final_populations_by_slice
    ]
    assert max(gains) > 0


def test_low_fluence_absorbed_energy_matches_stored_I7_energy():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    time = TimeGrid.centered(256, 50e-12)
    field = _uniform_pulse(grid, time, 10e-12)
    pulse_energy = 1e-6
    length = 18e-3
    nz = 100

    result = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=pulse_energy,
        length_m=length,
        nz=nz,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        store_full_populations=True,
    )

    dz = length / nz
    number_I7 = 0.0
    for populations in result.final_populations_by_slice:
        number_I7 += np.sum(populations[I7]) * grid.dx * grid.dy * dz

    stored = number_I7 * (H * C0 / p.pump_wavelength_m)
    absorbed = result.input_energy_J - result.output_energy_J
    assert abs(stored - absorbed) / absorbed < 1e-3



def test_peak_i7_metric_tracks_temporal_peak_not_only_final_state():
    # Artificially short I7 lifetime makes the temporal maximum occur before
    # the end of the time window, exercising the diagnostic definition.
    p = HoYAGFourLevelParams(
        tau7_s=5e-12,
        M78_s1=0.0,
        k75_m3_s=0.0,
        k76_m3_s=0.0,
        C57_m3_s=0.0,
        C67_m3_s=0.0,
    )
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(512, 80e-12)
    field = _uniform_pulse(grid, time, 10e-12)
    physical = scale_pulse_to_energy(field, grid, time, 5e-3)
    step = pump_material_step(
        physical, grid, time, 0.1e-3, p,
        sigma_abs_m2=p.sigma_abs_pump_m2,
        sigma_em_m2=p.sigma_em_pump_m2,
    )
    final_fraction = np.max(step.final_populations[I7]) / p.N_total_m3
    assert step.peak_I7_fraction > final_fraction
    assert step.peak_I7_fraction > 0


def test_stage2p_z_discretization_converges_for_saturating_pulse():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(192, 60e-12)
    field = _uniform_pulse(grid, time, 10e-12)

    coarse = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=5e-3,
        length_m=18e-3,
        nz=30,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
    )
    fine = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=5e-3,
        length_m=18e-3,
        nz=60,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
    )
    assert abs(coarse.transmission - fine.transmission) / fine.transmission < 0.015


def test_population_time_discretization_converges_at_high_fluence():
    p = HoYAGFourLevelParams()
    fluence = 3.0e4  # 3 J/cm^2

    def solve(nt):
        time = TimeGrid.centered(nt, 80e-12)
        temporal = gaussian_temporal_envelope(time, 10e-12, normalize=False)
        intensity = np.abs(temporal) ** 2
        intensity *= fluence / (np.sum(intensity) * time.dt)
        return integrate_populations(intensity, time, p)[I7] / p.N_total_m3

    coarse = solve(256)
    fine = solve(512)
    assert abs(coarse - fine) / fine < 2e-3
