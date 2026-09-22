"""Stage 4 example: amplify an LG structured signal through a gain map."""

import numpy as np

from hoyag import Grid2D, laguerre_gaussian
from hoyag.inhomogeneity import uniform_ho_density_field
from hoyag.populations import HoYAGFourLevelParams, I7, I8
from hoyag.signal import propagate_structured_signal_small_signal


def main() -> None:
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(128, 4e-3)
    density = uniform_ho_density_field(
        grid, nz=20, length_m=18e-3, density_m3=p.N_total_m3
    )

    populations = np.zeros((4, density.nz, grid.ny, grid.nx))
    populations[I7] = 0.30 * density.values_m3
    populations[I8] = 0.70 * density.values_m3

    # Make the right half more strongly inverted.
    populations[I7, :, :, grid.nx//2:] = 0.40 * p.N_total_m3
    populations[I8, :, :, grid.nx//2:] = 0.60 * p.N_total_m3

    signal = laguerre_gaussian(
        grid, p=0, l=1, waist_radius_m=0.6e-3
    )

    result = propagate_structured_signal_small_signal(
        signal, grid, populations, density, p
    )

    print(f"input relative power : {result.input_power:.6f}")
    print(f"output relative power: {result.output_power:.6f}")
    print(f"power gain           : {result.power_gain:.6f}")
    print(
        "g min/max            : "
        f"{result.gain_coefficient_by_slice_m1.min():.3f} / "
        f"{result.gain_coefficient_by_slice_m1.max():.3f} m^-1"
    )


if __name__ == "__main__":
    main()
