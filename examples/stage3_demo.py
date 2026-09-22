"""Stage 3 example: 10 kHz pump through a smooth Ho-density map."""

import numpy as np

from hoyag import Grid2D, TimeGrid, gaussian_beam, gaussian_temporal_envelope
from hoyag.inhomogeneity import (
    smooth_random_ho_density_field,
    simulate_inhomogeneous_pulse_train,
)
from hoyag.populations import HoYAGFourLevelParams
from hoyag.temporal import combine_spatial_temporal


def main() -> None:
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(8, 2e-3)
    time = TimeGrid.centered(96, 80e-12)

    density = smooth_random_ho_density_field(
        grid,
        nz=8,
        length_m=18e-3,
        mean_density_m3=p.N_total_m3,
        relative_rms=0.05,
        seed=7,
        minimum_fraction=0.7,
    )

    field = combine_spatial_temporal(
        gaussian_beam(grid, 0.5e-3),
        gaussian_temporal_envelope(time, 10e-12),
        grid,
        time,
    )

    result = simulate_inhomogeneous_pulse_train(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        density=density,
        repetition_rate_Hz=10_000.0,
        params=p,
        max_pulses=500,
        convergence_tolerance=1e-5,
    )

    print(f"density min/mean/max : {density.min_density_m3:.3e} / "
          f"{density.mean_density_m3:.3e} / {density.max_density_m3:.3e} m^-3")
    print(f"converged            : {result.converged}")
    print(f"pulses simulated     : {result.pulses_simulated}")
    print(f"final transmission   : {result.transmission_history[-1]:.6f}")
    print(f"final pre-pulse I7   : "
          f"{result.pre_pulse_peak_I7_fraction_history[-1]:.6f}")


if __name__ == "__main__":
    main()
