"""Stage 1P demo using the generic 10 ps Ho:YAG pump placeholder."""

from hoyag import (
    Grid2D,
    TimeGrid,
    combine_spatial_temporal,
    gaussian_beam,
    gaussian_temporal_envelope,
    propagate_spatiotemporal,
    pulse_intensity_fwhm_s,
    spatiotemporal_energy,
)


def main() -> None:
    grid = Grid2D.square(128, 4e-3)
    time = TimeGrid.centered(512, 80e-12)

    spatial = gaussian_beam(grid, 0.5e-3)
    temporal = gaussian_temporal_envelope(time, 10e-12)
    pulse = combine_spatial_temporal(spatial, temporal, grid, time)

    output = propagate_spatiotemporal(
        pulse, grid, time,
        wavelength_m=1.9077e-6,
        distance_m=18e-3,
        refractive_index=1.8018686989409411,
        beta2_s2_per_m=-4.460681783044079e-26,
    )

    iy, ix = grid.ny // 2, grid.nx // 2
    f0 = pulse_intensity_fwhm_s(pulse[:, iy, ix], time)
    f1 = pulse_intensity_fwhm_s(output[:, iy, ix], time)

    print(f"Input relative energy : {spatiotemporal_energy(pulse, grid, time):.12f}")
    print(f"Output relative energy: {spatiotemporal_energy(output, grid, time):.12f}")
    print(f"Input temporal FWHM   : {f0 * 1e12:.6f} ps")
    print(f"Output temporal FWHM  : {f1 * 1e12:.6f} ps")


if __name__ == "__main__":
    main()
