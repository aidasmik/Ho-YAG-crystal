import numpy as np

from hoyag.inhomogeneity import (
    HoDensityField,
    axial_linear_ho_density_field,
    gaussian_ho_density_perturbation,
    pump_material_step_inhomogeneous,
    propagate_single_pulse_inhomogeneous,
    relax_inhomogeneous_populations_dark,
    simulate_inhomogeneous_pulse_train,
    smooth_random_ho_density_field,
    uniform_ho_density_field,
)
from hoyag.populations import (
    I7,
    I8,
    HoYAGFourLevelParams,
    propagate_single_pulse_hoyag,
)
from hoyag.propagation import Grid2D, normalize_power
from hoyag.pulse_train import simulate_pulse_train_hoyag
from hoyag.temporal import (
    TimeGrid,
    combine_spatial_temporal,
    gaussian_temporal_envelope,
    spatiotemporal_energy,
)


def _uniform_pulse(grid, time, duration_s=10e-12):
    spatial = normalize_power(np.ones(grid.shape, dtype=complex), grid)
    temporal = gaussian_temporal_envelope(time, duration_s)
    return combine_spatial_temporal(spatial, temporal, grid, time)


def test_uniform_density_field_matches_requested_value():
    grid = Grid2D.square(5, 1e-3)
    density = uniform_ho_density_field(grid, 4, 2e-3, 1.52e26)
    assert density.values_m3.shape == (4, 5, 5)
    assert np.all(density.values_m3 == 1.52e26)
    assert density.dz_m == 0.5e-3


def test_random_density_field_is_reproducible_nonnegative_and_mean_preserving():
    grid = Grid2D.square(8, 2e-3)
    a = smooth_random_ho_density_field(
        grid, 6, 6e-3, 1.52e26, 0.08, seed=123,
        minimum_fraction=0.5,
    )
    b = smooth_random_ho_density_field(
        grid, 6, 6e-3, 1.52e26, 0.08, seed=123,
        minimum_fraction=0.5,
    )
    assert np.allclose(a.values_m3, b.values_m3)
    assert np.min(a.values_m3) >= 0.5 * 1.52e26
    assert abs(np.mean(a.values_m3) / 1.52e26 - 1.0) < 2e-3


def test_zero_doped_voxels_are_transparent_and_keep_zero_population():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(96, 60e-12)
    field = _uniform_pulse(grid, time)
    physical = field * np.sqrt(100e-6 / spatiotemporal_energy(field, grid, time))

    density = np.full(grid.shape, p.N_total_m3)
    density[:, :2] = 0.0

    step = pump_material_step_inhomogeneous(
        physical,
        grid,
        time,
        0.5e-3,
        density,
        p,
        sigma_abs_m2=p.sigma_abs_pump_m2,
        sigma_em_m2=p.sigma_em_pump_m2,
    )

    assert np.allclose(step.field_out[:, :, :2], physical[:, :, :2])
    assert np.all(step.final_populations[:, :, :2] == 0.0)
    assert np.mean(np.abs(step.field_out[:, :, 2:]) ** 2) < np.mean(
        np.abs(physical[:, :, 2:]) ** 2
    )


def test_uniform_stage3_single_pulse_reproduces_stage2p():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(4, 1e-3)
    time = TimeGrid.centered(96, 60e-12)
    field = _uniform_pulse(grid, time)
    length = 2e-3
    nz = 4

    baseline = propagate_single_pulse_hoyag(
        field, grid, time,
        pulse_energy_J=50e-6,
        length_m=length,
        nz=nz,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        store_full_populations=True,
    )
    density = uniform_ho_density_field(
        grid, nz, length, p.N_total_m3
    )
    stage3 = propagate_single_pulse_inhomogeneous(
        field, grid, time,
        pulse_energy_J=50e-6,
        density=density,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        store_full_populations=True,
    )

    assert np.isclose(stage3.transmission, baseline.transmission, rtol=1e-12)
    assert np.allclose(
        stage3.final_populations_by_slice,
        baseline.final_populations_by_slice,
        rtol=1e-12,
        atol=1e8,
    )


def test_axial_gradient_low_fluence_matches_integrated_beer_lambert():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(3, 1e-3)
    time = TimeGrid.centered(96, 60e-12)
    field = _uniform_pulse(grid, time)
    density = axial_linear_ho_density_field(
        grid, 8, 8e-3, p.N_total_m3, relative_end_to_end=0.8
    )

    result = propagate_single_pulse_inhomogeneous(
        field, grid, time,
        pulse_energy_J=1e-10,
        density=density,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
    )

    column = np.sum(density.values_m3[:, 0, 0]) * density.dz_m
    expected = np.exp(-p.sigma_abs_pump_m2 * column)
    assert abs(result.transmission - expected) / expected < 2e-6


def test_dark_relaxation_preserves_local_density_map():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(3, 1e-3)
    density = gaussian_ho_density_perturbation(
        grid, 3, 3e-3, p.N_total_m3, -0.4, 0.25e-3, 0.5e-3
    )
    state = np.zeros((4, *density.values_m3.shape))
    state[I7] = 0.2 * density.values_m3
    state[I8] = 0.8 * density.values_m3

    out = relax_inhomogeneous_populations_dark(
        state, density.values_m3, 100e-6, p, max_step_s=1e-6
    )
    assert np.allclose(
        np.sum(out, axis=0), density.values_m3, rtol=1e-12, atol=1e8
    )


def test_uniform_stage3_pulse_train_reproduces_stage2r_history():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(2, 1e-3)
    time = TimeGrid.centered(64, 60e-12)
    field = _uniform_pulse(grid, time)

    baseline = simulate_pulse_train_hoyag(
        field, grid, time,
        pulse_energy_J=50e-6,
        length_m=1e-3,
        nz=1,
        repetition_rate_Hz=10_000,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=3,
        min_pulses=3,
        convergence_tolerance=1e-12,
        max_dark_step_s=1e-6,
    )
    density = uniform_ho_density_field(
        grid, 1, 1e-3, p.N_total_m3
    )
    stage3 = simulate_inhomogeneous_pulse_train(
        field, grid, time,
        pulse_energy_J=50e-6,
        density=density,
        repetition_rate_Hz=10_000,
        params=p,
        spectral_absorption=False,
        include_passive_propagation=False,
        max_pulses=3,
        min_pulses=3,
        convergence_tolerance=1e-12,
        max_dark_step_s=1e-6,
    )

    assert np.allclose(
        stage3.transmission_history,
        baseline.transmission_history,
        rtol=2e-12,
        atol=1e-14,
    )
    assert np.allclose(
        stage3.pre_pulse_peak_I7_fraction_history,
        baseline.pre_pulse_peak_I7_fraction_history,
        rtol=2e-12,
        atol=1e-14,
    )
