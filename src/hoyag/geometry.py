"""Working crystal geometry definitions.

The original Rupp 4 mm diameter x 18 mm rod remains in the Stage 0 validation
configuration. The default working geometry below is the user-selected Ho:YAG
thin disk: 10 mm diameter x 1 mm thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .propagation import Grid2D


@dataclass(frozen=True)
class ThinDiskGeometry:
    """Circular thin-disk crystal geometry."""

    diameter_m: float = 10e-3
    thickness_m: float = 1e-3

    def __post_init__(self) -> None:
        if self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive")
        if self.thickness_m <= 0:
            raise ValueError("thickness_m must be positive")

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def face_area_m2(self) -> float:
        return math.pi * self.radius_m**2

    @property
    def volume_m3(self) -> float:
        return self.face_area_m2 * self.thickness_m

    def validate_grid(self, grid: Grid2D) -> None:
        width_x = grid.nx * grid.dx
        width_y = grid.ny * grid.dy
        if width_x + 1e-15 < self.diameter_m or width_y + 1e-15 < self.diameter_m:
            raise ValueError(
                "transverse numerical window must be at least the disk diameter "
                f"({self.diameter_m} m)"
            )

    def aperture_mask(self, grid: Grid2D) -> np.ndarray:
        """Boolean mask for the circular crystal face."""
        self.validate_grid(grid)
        x, y = grid.mesh
        return x**2 + y**2 <= self.radius_m**2


DEFAULT_THIN_DISK_GEOMETRY = ThinDiskGeometry()
