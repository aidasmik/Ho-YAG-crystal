import numpy as np

from hoyag.propagation import (
    Grid2D,
    angular_spectrum_propagate,
    apply_phase_mask,
    gaussian_beam,
    hermite_gaussian,
    laguerre_gaussian,
    optical_power,
)


def test_zero_distance_is_identity():
    grid = Grid2D.square(128, 4e-3)
    field = laguerre_gaussian(grid, p=0, l=1, waist_radius_m=0.5e-3)
    out = angular_spectrum_propagate(field, grid, 2.0903e-6, 0.0, refractive_index=1.7991)
    assert np.allclose(out, field)


def test_power_conserved_in_passive_uniform_medium():
    grid = Grid2D.square(256, 8e-3)
    field = gaussian_beam(grid, 0.5e-3)
    p0 = optical_power(field, grid)
    out = angular_spectrum_propagate(field, grid, 2.0903e-6, 0.05, refractive_index=1.7991)
    p1 = optical_power(out, grid)
    assert abs(p1 - p0) / p0 < 1e-10


def test_gaussian_radius_matches_paraxial_solution():
    grid = Grid2D.square(512, 8e-3)
    wavelength = 2.0903e-6
    n = 1.7991
    w0 = 0.35e-3
    z = 0.20
    field = gaussian_beam(grid, w0)
    out = angular_spectrum_propagate(field, grid, wavelength, z, refractive_index=n)

    x, y = grid.mesh
    intensity = np.abs(out) ** 2
    p = np.sum(intensity)
    r2_mean = np.sum((x**2 + y**2) * intensity) / p
    w_numeric = np.sqrt(2 * r2_mean)

    z_r = np.pi * n * w0**2 / wavelength
    w_expected = w0 * np.sqrt(1 + (z / z_r) ** 2)
    assert abs(w_numeric - w_expected) / w_expected < 0.01


def test_hg10_is_odd_in_x():
    grid = Grid2D.square(129, 4e-3)
    field = hermite_gaussian(grid, 1, 0, 0.5e-3, normalize=False)
    assert np.allclose(field[:, ::-1], -field, atol=1e-12)


def test_lg_vortex_has_phase_winding():
    grid = Grid2D.square(512, 4e-3)
    l = 2
    field = laguerre_gaussian(grid, p=0, l=l, waist_radius_m=0.6e-3, normalize=False)
    radius = 0.45e-3
    theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ix = np.rint(x / grid.dx + grid.nx / 2).astype(int)
    iy = np.rint(y / grid.dy + grid.ny / 2).astype(int)
    phase = np.unwrap(np.angle(field[iy, ix]))
    winding = phase[-1] - phase[0]
    assert abs(winding - 2 * np.pi * l) < 0.15


def test_phase_mask_preserves_intensity():
    grid = Grid2D.square(128, 4e-3)
    field = gaussian_beam(grid, 0.5e-3)
    x, y = grid.mesh
    phase = 4e5 * (x**2 - y**2)
    masked = apply_phase_mask(field, phase, grid)
    assert np.allclose(np.abs(masked) ** 2, np.abs(field) ** 2)
