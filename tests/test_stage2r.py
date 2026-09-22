import numpy as np

from hoyag.populations import (
    I7,
    I8,
    HoYAGFourLevelParams,
    ground_state_populations,
    relax_populations_dark,
)
from hoyag.propagation import Grid2D, normalize_power
from hoyag.pulse_train import simulate_pulse_train_hoyag
from hoyag.temporal import (
    TimeGrid,
    combine_spatial_temporal,
    gaussian_temporal_envelope,
)


def _uniform_pulse(grid, time, duration_s=10e-12):
    spatial = normalize_power(np.ones(grid.shape, dtype=complex), grid)
    temporal = gaussian_temporal_envelope(time, duration_s)
    return combine_spatial_temporal(spatial, temporal, grid, time)


def test_dark_relaxation_matches_simple_i7_exponential_limit():
    p = HoYAGFourLevelParams(
        M56_s1=0.0,
        M67_s1=0.0,
        M78_s1=0.0,
        k75_m3_s=0.0,
        k76_m3_s=0.0,
        C57_m3_s=0.0,
        C67_m3_s=0.0,
    )
    state = ground_state_populations(p)
    state[I7] = 0.2 * p.N_total_m3
    state[I8] = 0.8 * p.N_total_m3

    duration = 1.0e-3
    out = relax_populations_dark(
        state, duration, p, max_step_s=10e-6
    )
    expected_i7 = 0.2 * np.exp(-duration / p.tau7_s)
    assert abs(out[I7] / p.N_total_m3 - expected_i7) < 2e-9
    assert np.isclose(np.sum(out), p.N_total_m3, rtol=1e-12)


def test_long_dark_interval_returns_close_to_ground_state():
    p = HoYAGFourLevelParams()
    state = ground_state_populations(p)
    state[I7] = 0.1 * p.N_total_m3
    state[I8] = 0.9 * p.N_total_m3

    out = relax_populations_dark(
        state, 0.1, p, max_step_s=2e-6
    )
    assert out[I7] / p.N_total_m3 < 1e-5
    assert out[I8] / p.N_total_m3 > 0.99998


def test_10khz_pulse_train_accumulates_i7_population():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(2, 1e-3)
    time = TimeGrid.centered(64, 60e-12)
    field = _uniform_pulse(grid, time)

    result = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10_000.0,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=400,
        convergence_tolerance=1e-5,
        max_dark_step_s=2e-6,
    )

    assert result.converged
    assert result.pulses_simulated < 400
    assert result.pre_pulse_peak_I7_fraction_history[0] == 0.0
    assert result.pre_pulse_peak_I7_fraction_history[-1] > 0.02
    assert result.transmission_history[-1] > result.transmission_history[0]
    assert result.convergence_history[-1] <= 1e-5


def test_converged_state_is_periodic_fixed_point():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(2, 1e-3)
    time = TimeGrid.centered(64, 60e-12)
    field = _uniform_pulse(grid, time)

    first = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10_000.0,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=400,
        convergence_tolerance=1e-5,
        max_dark_step_s=2e-6,
    )
    assert first.converged

    one_more = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10_000.0,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        initial_populations_by_slice=first.pre_pulse_populations_by_slice,
        max_pulses=1,
        min_pulses=1,
        convergence_tolerance=1e-3,
        max_dark_step_s=2e-6,
    )

    residual = np.max(
        np.abs(
            one_more.pre_pulse_populations_by_slice
            - first.pre_pulse_populations_by_slice
        )
    ) / p.N_total_m3
    assert residual < 1.2e-5


def test_low_repetition_rate_has_far_less_population_memory():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(2, 1e-3)
    time = TimeGrid.centered(64, 60e-12)
    field = _uniform_pulse(grid, time)

    high = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10_000.0,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=250,
        convergence_tolerance=3e-5,
        max_dark_step_s=2e-6,
    )
    low = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10.0,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=3,
        min_pulses=2,
        convergence_tolerance=1e-6,
        max_dark_step_s=20e-6,
    )

    assert high.pre_pulse_peak_I7_fraction_history[-1] > 100 * low.pre_pulse_peak_I7_fraction_history[-1]
