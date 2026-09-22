"""Stage 3 example: 10 kHz pump through the 10 mm x 1 mm Ho:YAG thin disk."""

from hoyag import Grid2D, TimeGrid, gaussian_beam, gaussian_temporal_envelope
from hoyag.geometry import DEFAULT_THIN_DISK_GEOMETRY
from hoyag.inhomogeneity import (
    thin_disk_ho_density_field,
    simulate_inhomogeneous_pulse_train,
)
from hoyag.populations import HoYAGFourLevelParams
from hoyag.temporal import combine_spatial_temporal


def main() -> None:
    p = HoYAGFourLevelParams()
    geometry = DEFAULT_THIN_DISK_GEOMETRY

    # 10 mm x 10 mm numerical window containing the circular 10 mm disk.
    grid = Grid2D.square(64, geometry.diameter_m)
    time = TimeGrid.centered(96, 80e-12)

    density = thin_disk_ho_density_field(
        grid,
        nz=8,
        density_m3=p.N_total_m3,
        geometry=geometry,
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

    print(f"disk diameter         : {geometry.diameter_m * 1e3:.3f} mm")
    print(f"disk thickness        : {geometry.thickness_m * 1e3:.3f} mm")
    print(f"converged             : {result.converged}")
    print(f"pulses simulated      : {result.pulses_simulated}")
    print(f"final transmission    : {result.transmission_history[-1]:.6f}")
    print(
        "final pre-pulse I7    : "
        f"{result.pre_pulse_peak_I7_fraction_history[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
