"""Stage 2P example: one generic 10 ps pump pulse through homogeneous Ho:YAG."""

import numpy as np

from hoyag import Grid2D, TimeGrid, gaussian_beam, gaussian_temporal_envelope
from hoyag.temporal import combine_spatial_temporal
from hoyag.populations import HoYAGFourLevelParams, propagate_single_pulse_hoyag


def main() -> None:
    grid = Grid2D.square(32, 4e-3)
    time = TimeGrid.centered(256, 80e-12)

    spatial = gaussian_beam(grid, 0.5e-3)
    temporal = gaussian_temporal_envelope(time, 10e-12)
    field = combine_spatial_temporal(spatial, temporal, grid, time)

    result = propagate_single_pulse_hoyag(
        field,
        grid,
        time,
        pulse_energy_J=100e-6,
        length_m=18e-3,
        nz=40,
        params=HoYAGFourLevelParams(),
        spectral_absorption=True,
        include_passive_propagation=True,
    )

    print(f"effective sigma_a: {result.effective_sigma_abs_m2:.6e} m^2")
    print(f"input energy     : {result.input_energy_J * 1e6:.6f} uJ")
    print(f"output energy    : {result.output_energy_J * 1e6:.6f} uJ")
    print(f"transmission     : {result.transmission:.6f}")
    print(f"peak I7 fraction : {np.max(result.peak_I7_fraction_by_slice):.6f}")


if __name__ == "__main__":
    main()
