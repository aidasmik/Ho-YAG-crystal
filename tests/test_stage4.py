import numpy as np

from hoyag.inhomogeneity import HoDensityField, uniform_ho_density_field
from hoyag.populations import (
    H,
    C0,
    I7,
    I8,
    HoYAGFourLevelParams,
)
from hoyag.propagation import Grid2D, laguerre_gaussian, normalize_power
from hoyag.signal import (
    laser_transparency_i7_fraction,
    propagate_structured_signal_saturated,
    propagate_structured_signal_small_signal,
    signal_material_step,
    small_signal_gain_coefficient_m1,
)
from hoyag.temporal import (
    TimeGrid,
    combine_spatial_temporal,
    gaussian_temporal_envelope,
    spatiotemporal_energy,
)


def _two_level_population_field(density, i7_fraction):
    state = np.zeros((4, *density.values_m3.shape), dtype=float)
    state[I7] = i7_fraction * density.values_m3
    state[I8] = (1.0 - i7_fraction) * density.values_m3
    return state


def _uniform_signal_pulse(grid, time, duration_s=10e-12):
    spatial = normalize_power(np.ones(grid.shape, dtype=complex), grid)
    temporal = gaussian_temporal_envelope(time, duration_s)
    return combine_spatial_temporal(spatial, temporal, grid, time)


def test_laser_transparency_fraction_gives_zero_gain():
    p = HoYAGFourLevelParams()
    f = laser_transparency_i7_fraction(p)
    state = np.zeros(4)
    state[I7] = f * p.N_total_m3
    state[I8] = (1-f) * p.N_total_m3
    assert abs(small_signal_gain_coefficient_m1(state, p)) < 1e-12


def test_uniform_small_signal_matches_exponential_gain():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    density = uniform_ho_density_field(grid, 6, 6e-3, p.N_total_m3)
    populations = _two_level_population_field(density, 0.35)
    field = normalize_power(np.ones(grid.shape, dtype=complex), grid)

    result = propagate_structured_signal_small_signal(
        field,
        grid,
        populations,
        density,
        p,
        include_passive_propagation=False,
    )

    g = p.N_total_m3 * (
        p.sigma_em_laser_m2 * 0.35
        - p.sigma_abs_laser_m2 * 0.65
    )
    expected = np.exp(g * density.length_m)
    assert abs(result.power_gain - expected) / expected < 1e-12


def test_ground_state_signal_is_absorbed_analytically():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(6, 1e-3)
    density = uniform_ho_density_field(grid, 4, 4e-3, p.N_total_m3)
    populations = _two_level_population_field(density, 0.0)
    field = normalize_power(np.ones(grid.shape, dtype=complex), grid)

    result = propagate_structured_signal_small_signal(
        field, grid, populations, density, p,
        include_passive_propagation=False,
    )
    expected = np.exp(
        -p.sigma_abs_laser_m2 * p.N_total_m3 * density.length_m
    )
    assert abs(result.power_gain - expected) / expected < 1e-12


def test_spatial_gain_map_imprints_structured_amplitude_without_phase_change():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 1e-3)
    density_values = np.full((1, grid.ny, grid.nx), p.N_total_m3)
    density = HoDensityField(density_values, 2e-3)

    populations = _two_level_population_field(density, 0.10)
    populations[I7, :, :, grid.nx//2:] = 0.40 * p.N_total_m3
    populations[I8, :, :, grid.nx//2:] = 0.60 * p.N_total_m3

    field = laguerre_gaussian(
        grid, p=0, l=1, waist_radius_m=0.25e-3, normalize=False
    )
    result = propagate_structured_signal_small_signal(
        field, grid, populations, density, p,
        include_passive_propagation=False,
    )

    nonzero = np.abs(field) > np.max(np.abs(field)) * 1e-6
    phase_delta = np.angle(result.field_out[nonzero] / field[nonzero])
    assert np.max(np.abs(phase_delta)) < 1e-12

    left = np.mean(np.abs(result.field_out[:, :grid.nx//2])**2)
    right = np.mean(np.abs(result.field_out[:, grid.nx//2:])**2)
    assert right > left


def test_signal_material_step_conserves_local_total_ho_population():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(256, 50e-12)
    density = np.full(grid.shape, p.N_total_m3)
    state = np.zeros((4, grid.ny, grid.nx))
    state[I7] = 0.4 * density
    state[I8] = 0.6 * density

    pulse = _uniform_signal_pulse(grid, time)
    pulse *= np.sqrt(1e-3 / spatiotemporal_energy(pulse, grid, time))

    step = signal_material_step(
        pulse, grid, time, 0.5e-3, density, state, p
    )
    assert np.allclose(
        np.sum(step.final_populations, axis=0),
        density,
        rtol=1e-12,
        atol=1e8,
    )


def test_finite_energy_signal_saturates_gain_and_depletes_i7():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(256, 60e-12)
    density = uniform_ho_density_field(grid, 4, 4e-3, p.N_total_m3)
    populations = _two_level_population_field(density, 0.55)
    pulse = _uniform_signal_pulse(grid, time)

    result = propagate_structured_signal_saturated(
        pulse,
        grid,
        time,
        pulse_energy_J=5e-3,
        populations_by_slice=populations,
        density=density,
        params=p,
        include_passive_propagation=False,
    )

    assert result.energy_gain > 1.0
    assert result.energy_gain < result.small_signal_power_gain_reference
    assert np.mean(result.final_populations_by_slice[I7]) < np.mean(populations[I7])


def test_signal_energy_gain_matches_i7_depletion_in_isolated_two_level_limit():
    p = HoYAGFourLevelParams(
        tau5_s=1e99,
        tau6_s=1e99,
        tau7_s=1e99,
        M56_s1=0.0,
        M67_s1=0.0,
        M78_s1=0.0,
        k75_m3_s=0.0,
        k76_m3_s=0.0,
        C57_m3_s=0.0,
        C67_m3_s=0.0,
    )
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(512, 80e-12)
    density = np.full(grid.shape, p.N_total_m3)

    state = np.zeros((4, grid.ny, grid.nx))
    state[I7] = 0.6 * density
    state[I8] = 0.4 * density

    pulse = _uniform_signal_pulse(grid, time)
    pulse *= np.sqrt(5e-4 / spatiotemporal_energy(pulse, grid, time))

    step = signal_material_step(
        pulse, grid, time, 0.2e-3, density, state, p
    )

    delta_n7 = np.sum(state[I7] - step.final_populations[I7])
    volume = grid.dx * grid.dy * 0.2e-3
    released = delta_n7 * volume * (H*C0/p.laser_wavelength_m)

    assert step.signal_energy_change_J > 0
    assert abs(step.signal_energy_change_J - released) / released < 0.03
