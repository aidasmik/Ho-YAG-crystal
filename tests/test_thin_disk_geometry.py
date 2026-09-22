import math
import numpy as np

from hoyag.geometry import ThinDiskGeometry, DEFAULT_THIN_DISK_GEOMETRY
from hoyag.inhomogeneity import thin_disk_ho_density_field
from hoyag.populations import HoYAGFourLevelParams
from hoyag.propagation import Grid2D


def test_default_thin_disk_dimensions():
    g = DEFAULT_THIN_DISK_GEOMETRY
    assert g.diameter_m == 10e-3
    assert g.radius_m == 5e-3
    assert g.thickness_m == 1e-3
    assert np.isclose(g.face_area_m2, math.pi * (5e-3)**2)
    assert np.isclose(g.volume_m3, math.pi * (5e-3)**2 * 1e-3)


def test_thin_disk_density_has_circular_10mm_aperture_and_1mm_length():
    p = HoYAGFourLevelParams()
    grid = Grid2D.square(201, 10e-3)
    density = thin_disk_ho_density_field(
        grid, nz=5, density_m3=p.N_total_m3
    )

    assert density.length_m == 1e-3
    assert density.nz == 5
    assert np.isclose(density.dz_m, 0.2e-3)

    mid = density.values_m3[0]
    assert mid[grid.ny//2, grid.nx//2] == p.N_total_m3
    assert mid[0, 0] == 0.0
    assert mid[0, -1] == 0.0
    assert mid[-1, 0] == 0.0
    assert mid[-1, -1] == 0.0

    # Pixelized circular area should agree with pi R^2 to better than ~1%.
    pixel_area = np.count_nonzero(mid) * grid.dx * grid.dy
    exact_area = math.pi * (5e-3)**2
    assert abs(pixel_area - exact_area) / exact_area < 0.01

    # All z slices represent the same physical cylindrical disk.
    assert np.all(density.values_m3 == density.values_m3[0][None, :, :])


def test_thin_disk_requires_window_at_least_disk_diameter():
    grid = Grid2D.square(64, 8e-3)
    g = ThinDiskGeometry(diameter_m=10e-3, thickness_m=1e-3)
    try:
        g.aperture_mask(grid)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for undersized numerical window")
