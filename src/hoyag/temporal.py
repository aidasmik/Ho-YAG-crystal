"""Picosecond complex-envelope propagation for Stage 1P.

The pulse is represented as E(tau, y, x), where tau is retarded time. The
carrier is not resolved. Stage 1P is passive and linear: transverse diffraction
comes from Stage 1 and temporal evolution includes second-order GVD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .propagation import Grid2D


@dataclass(frozen=True)
class TimeGrid:
    """Uniform retarded-time grid tau = t - z/v_g."""

    nt: int
    dt: float

    def __post_init__(self) -> None:
        if self.nt < 4:
            raise ValueError("nt must be >= 4")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @classmethod
    def centered(cls, nt: int, window_s: float) -> "TimeGrid":
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        return cls(nt=nt, dt=window_s / nt)

    @property
    def tau(self) -> np.ndarray:
        return (np.arange(self.nt) - (self.nt - 1) / 2) * self.dt

    @property
    def frequency_hz(self) -> np.ndarray:
        return np.fft.fftfreq(self.nt, d=self.dt)

    @property
    def omega_rad_s(self) -> np.ndarray:
        return 2 * np.pi * self.frequency_hz


def _temporal(envelope: np.ndarray, time: TimeGrid) -> np.ndarray:
    arr = np.asarray(envelope, dtype=np.complex128)
    if arr.shape != (time.nt,):
        raise ValueError(f"temporal envelope shape {arr.shape} != {(time.nt,)}")
    return arr


def _spatiotemporal(field: np.ndarray, grid: Grid2D, time: TimeGrid) -> np.ndarray:
    arr = np.asarray(field, dtype=np.complex128)
    expected = (time.nt, grid.ny, grid.nx)
    if arr.shape != expected:
        raise ValueError(f"field shape {arr.shape} != {expected}")
    return arr


def temporal_energy(envelope: np.ndarray, time: TimeGrid) -> float:
    arr = _temporal(envelope, time)
    return float(np.sum(np.abs(arr) ** 2) * time.dt)


def normalize_temporal_energy(
    envelope: np.ndarray,
    time: TimeGrid,
    target_energy: float = 1.0,
) -> np.ndarray:
    if target_energy <= 0:
        raise ValueError("target_energy must be positive")
    arr = _temporal(envelope, time)
    energy = temporal_energy(arr, time)
    if energy == 0:
        raise ValueError("cannot normalize a zero envelope")
    return arr * math.sqrt(target_energy / energy)


def gaussian_temporal_envelope(
    time: TimeGrid,
    duration_fwhm_s: float,
    *,
    delay_s: float = 0.0,
    gdd_s2: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Gaussian pulse; duration_fwhm_s is the transform-limited intensity FWHM."""
    if duration_fwhm_s <= 0:
        raise ValueError("duration_fwhm_s must be positive")
    tau = time.tau - delay_s
    envelope = np.exp(-2 * np.log(2) * (tau / duration_fwhm_s) ** 2).astype(np.complex128)
    if gdd_s2 != 0:
        envelope = apply_gdd(envelope, time, gdd_s2)
    return normalize_temporal_energy(envelope, time) if normalize else envelope


def apply_gdd(envelope: np.ndarray, time: TimeGrid, gdd_s2: float) -> np.ndarray:
    """Apply quadratic spectral phase exp(i*GDD*Omega^2/2)."""
    arr = _temporal(envelope, time)
    if gdd_s2 == 0:
        return arr.copy()
    spectrum = np.fft.fft(arr)
    phase = np.exp(0.5j * gdd_s2 * time.omega_rad_s**2)
    return np.fft.ifft(spectrum * phase)


def apply_gvd(
    envelope: np.ndarray,
    time: TimeGrid,
    beta2_s2_per_m: float,
    distance_m: float,
) -> np.ndarray:
    return apply_gdd(envelope, time, beta2_s2_per_m * distance_m)


def pulse_intensity_fwhm_s(envelope: np.ndarray, time: TimeGrid) -> float:
    """Measure intensity FWHM using linear interpolation at both crossings."""
    arr = _temporal(envelope, time)
    intensity = np.abs(arr) ** 2
    half = float(np.max(intensity)) / 2
    above = np.flatnonzero(intensity >= half)
    if len(above) < 2:
        raise ValueError("time grid does not resolve pulse FWHM")
    il = int(above[0])
    ir = int(above[-1])
    if il == 0 or ir == time.nt - 1:
        raise ValueError("pulse FWHM touches time-window boundary")

    def crossing(i0: int, i1: int) -> float:
        y0, y1 = intensity[i0], intensity[i1]
        x0, x1 = time.tau[i0], time.tau[i1]
        return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))

    return crossing(ir, ir + 1) - crossing(il - 1, il)


def combine_spatial_temporal(
    spatial_field: np.ndarray,
    temporal_envelope: np.ndarray,
    grid: Grid2D,
    time: TimeGrid,
) -> np.ndarray:
    spatial = np.asarray(spatial_field, dtype=np.complex128)
    if spatial.shape != grid.shape:
        raise ValueError(f"spatial field shape {spatial.shape} != {grid.shape}")
    temporal = _temporal(temporal_envelope, time)
    return temporal[:, None, None] * spatial[None, :, :]


def spatiotemporal_energy(field: np.ndarray, grid: Grid2D, time: TimeGrid) -> float:
    arr = _spatiotemporal(field, grid, time)
    return float(np.sum(np.abs(arr) ** 2) * time.dt * grid.dx * grid.dy)


def propagate_spatiotemporal(
    field: np.ndarray,
    grid: Grid2D,
    time: TimeGrid,
    wavelength_m: float,
    distance_m: float,
    *,
    refractive_index: float = 1.0,
    beta2_s2_per_m: float = 0.0,
    bandlimit: bool = True,
) -> np.ndarray:
    """Propagate E(tau,y,x) with Stage 1 diffraction plus second-order GVD."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if refractive_index <= 0:
        raise ValueError("refractive_index must be positive")

    arr = _spatiotemporal(field, grid, time)
    if distance_m == 0:
        return arr.copy()

    fx, fy = np.meshgrid(grid.fx, grid.fy, indexing="xy")
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    k = 2 * np.pi * refractive_index / wavelength_m
    kt2 = kx**2 + ky**2
    kz = np.sqrt((k**2 - kt2).astype(np.complex128))
    hxy = np.exp(1j * kz * distance_m)
    if bandlimit:
        hxy = np.where(kt2 <= k**2, hxy, 0.0)

    spectrum_xy = np.fft.fft2(arr, axes=(-2, -1))
    out = np.fft.ifft2(spectrum_xy * hxy[None, :, :], axes=(-2, -1))

    if beta2_s2_per_m != 0:
        spectrum_t = np.fft.fft(out, axis=0)
        ht = np.exp(0.5j * beta2_s2_per_m * distance_m * time.omega_rad_s**2)
        out = np.fft.ifft(spectrum_t * ht[:, None, None], axis=0)

    return out
