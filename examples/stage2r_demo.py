"""Stage 2R example: generic 10 kHz, 10 ps pump train."""

from hoyag import Grid2D, TimeGrid, gaussian_beam, gaussian_temporal_envelope
from hoyag.temporal import combine_spatial_temporal
from hoyag.pulse_train import simulate_pulse_train_hoyag


def main() -> None:
    grid = Grid2D.square(8, 2e-3)
    time = TimeGrid.centered(96, 80e-12)

    spatial = gaussian_beam(grid, 0.5e-3)
    temporal = gaussian_temporal_envelope(time, 10e-12)
    field = combine_spatial_temporal(spatial, temporal, grid, time)

    result = simulate_pulse_train_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=18e-3,
        nz=12,
        repetition_rate_Hz=10_000.0,
        max_pulses=1000,
        convergence_tolerance=1e-5,
    )

    print(f"converged             : {result.converged}")
    print(f"pulses simulated      : {result.pulses_simulated}")
    print(f"final transmission    : {result.transmission_history[-1]:.6f}")
    print(
        "final pre-pulse I7   : "
        f"{result.pre_pulse_peak_I7_fraction_history[-1]:.6f}"
    )
    print(
        "final post-pulse I7  : "
        f"{result.post_pulse_peak_I7_fraction_history[-1]:.6f}"
    )
    print(f"final residual         : {result.convergence_history[-1]:.3e}")


if __name__ == "__main__":
    main()
