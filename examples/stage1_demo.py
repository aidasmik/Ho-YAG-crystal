"""Minimal Stage 1 demonstration."""

import numpy as np
from hoyag.propagation import Grid2D, laguerre_gaussian, angular_spectrum_propagate, optical_power


def main() -> None:
    grid = Grid2D.square(512, 8e-3)
    wavelength = 2.0903e-6
    n_yag = 1.7991

    field0 = laguerre_gaussian(grid, p=0, l=1, waist_radius_m=0.6e-3)
    field1 = angular_spectrum_propagate(
        field0,
        grid,
        wavelength_m=wavelength,
        distance_m=18e-3,
        refractive_index=n_yag,
    )

    print(f"Input relative power : {optical_power(field0, grid):.12f}")
    print(f"Output relative power: {optical_power(field1, grid):.12f}")
    print(f"Peak input intensity : {np.max(np.abs(field0)**2):.6e}")
    print(f"Peak output intensity: {np.max(np.abs(field1)**2):.6e}")


if __name__ == "__main__":
    main()
