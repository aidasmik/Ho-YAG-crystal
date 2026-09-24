"""Phase-only shaping of a Gaussian seed before the Yb:LuAG disk.

The target names describe requested fields at the disk, not fields already
present at the source. The phase plate changes phase but leaves source-plane
intensity Gaussian. Scalar free-space propagation forms the structured field.
"""

from __future__ import annotations

import numpy as np

from hoyag.propagation import (Grid2D, angular_spectrum_propagate,
                               gaussian_beam, normalize_power, optical_power)


def _flat_top_hologram(grid: Grid2D, source: np.ndarray, wavelength_m: float,
                       distance_m: float, waist_m: float) -> np.ndarray:
    """Phase-only alternating-projection design for a disk-plane flat top."""
    x, y = grid.mesh
    radius = np.hypot(x, y)
    target_amplitude = np.exp(-(radius / waist_m)**8)
    target_amplitude = normalize_power(target_amplitude, grid,
                                       optical_power(source, grid))
    source_amplitude = abs(source)
    field = source_amplitude * np.exp(-1j * 0.7 * (radius / waist_m)**2)
    for _ in range(48):
        disk = angular_spectrum_propagate(field, grid, wavelength_m, distance_m)
        constrained_disk = target_amplitude * np.exp(1j * np.angle(disk))
        back = angular_spectrum_propagate(constrained_disk, grid,
                                          wavelength_m, -distance_m)
        field = source_amplitude * np.exp(1j * np.angle(back))
    return np.angle(field)


def gaussian_seed_and_target_mask(grid: Grid2D, waist_m: float, power: float,
                                  target: str, wavelength_m: float,
                                  slm_to_disk_m: float):
    """Return the Gaussian source and phase-only mask for one target shape."""
    if not np.isfinite(slm_to_disk_m) or slm_to_disk_m <= 0:
        raise ValueError("SLM-to-disk propagation distance must be positive")
    source = normalize_power(gaussian_beam(grid, waist_m), grid, power)
    x, y = grid.mesh
    r = np.hypot(x, y)
    if target == "Gaussian TEM00":
        phase = np.zeros(grid.shape)
    elif target == "Helical LG(0,+1)":
        phase = np.arctan2(y, x)
    elif target == "Double helix LG(0,+2)":
        phase = 2 * np.arctan2(y, x)
    elif target == "Hermite-Gaussian HG(1,1)":
        phase = np.where(x * y >= 0, 0.0, np.pi)
    elif target == "Needle Bessel-Gaussian":
        # A converging axicon: first J0 zero near 0.32 waist in the
        # paraxial overlap region. Finite Gaussian aperture limits length.
        transverse_wavenumber = 2.4048255577 / (0.32 * waist_m)
        phase = -transverse_wavenumber * r
    elif target == "Flattop super-Gaussian":
        phase = _flat_top_hologram(grid, source, wavelength_m,
                                   slm_to_disk_m, waist_m)
    else:
        raise ValueError("unknown beam-shaping target")
    return source, np.angle(np.exp(1j * phase))
