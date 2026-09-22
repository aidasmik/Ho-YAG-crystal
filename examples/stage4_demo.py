"""Stage 4 example: amplify an LG signal through the Ho:YAG thin disk."""

import numpy as np

from hoyag import Grid2D, laguerre_gaussian
from hoyag.geometry import DEFAULT_THIN_DISK_GEOMETRY
from hoyag.inhomogeneity import thin_disk_ho_density_field
from hoyag.populations import HoYAGFourLevelParams, I7, I8
from hoyag.signal import propagate_structured_signal_small_signal


def main() -> None:
    p = HoYAGFourLevelParams()
    geometry = DEFAULT_THIN_DISK_GEOMETRY
    grid = Grid2D.square(128, geometry.diameter_m)

    density = thin_disk_ho_density_field(
        grid,
        nz=8,
        density_m3=p.N_total_m3,
        geometry=geometry,
    )

    populations = np.zeros((4, density.nz, grid.ny, grid.nx))
    populations[I7] = 0.30 * density.values_m3
    populations[I8] = 0.70 * density.values_m3

    signal = laguerre_gaussian(
        grid, p=0, l=1, waist_radius_m=0.6e-3
    )

    result = propagate_structured_signal_small_signal(
        signal, grid, populations, density, p
    )

    print(f"disk diameter         : {geometry.diameter_m * 1e3:.3f} mm")
    print(f"disk thickness        : {geometry.thickness_m * 1e3:.3f} mm")
    print(f"input relative power  : {result.input_power:.6f}")
    print(f"output relative power : {result.output_power:.6f}")
    print(f"power gain            : {result.power_gain:.6f}")


if __name__ == "__main__":
    main()
