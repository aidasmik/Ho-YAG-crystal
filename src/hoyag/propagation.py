"""Passive scalar complex-field propagation for Stage 1.

The field is a slowly varying complex envelope E(x, y).  Stage 1 assumes a
uniform, linear, isotropic medium with no gain, absorption, thermal lensing,
or nonlinear response.  Those effects are added in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """Uniform Cartesian transverse grid."""

    nx: int
    ny: int
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be >= 2")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("dx and dy must be positive")

    @classmethod
    def square(cls, n: int, size_m: float) -> "Grid2D":
        if size_m <= 0:
            raise ValueError("size_m must be positive")
        return cls(nx=n, ny=n, dx=size_m / n, dy=size_m / n)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.nx) - (self.nx - 1) / 2) * self.dx

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.ny) - (self.ny - 1) / 2) * self.dy

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="xy")

    @property
    def fx(self) -> np.ndarray:
        return np.fft.fftfreq(self.nx, d=self.dx)

    @property
    def fy(self) -> np.ndarray:
        return np.fft.fftfreq(self.ny, d=self.dy)


def _check_field(field: np.ndarray, grid: Grid2D) -> np.ndarray:
    arr = np.asarray(field, dtype=np.complex128)
    if arr.shape != grid.shape:
        raise ValueError(f"field shape {arr.shape} does not match grid {grid.shape}")
    return arr


def optical_power(field: np.ndarray, grid: Grid2D) -> float:
    """Return relative optical power integral int |E|^2 dx dy."""
    arr = _check_field(field, grid)
    return float(np.sum(np.abs(arr) ** 2) * grid.dx * grid.dy)


def normalize_power(field: np.ndarray, grid: Grid2D, target_power: float = 1.0) -> np.ndarray:
    """Scale a field to the requested relative power."""
    if target_power <= 0:
        raise ValueError("target_power must be positive")
    arr = _check_field(field, grid)
    p = optical_power(arr, grid)
    if p == 0:
        raise ValueError("cannot normalize a zero field")
    return arr * math.sqrt(target_power / p)

def gaussian_beam(
    grid: Grid2D,
    waist_radius_m: float,
    *,
    x0_m: float = 0.0,
    y0_m: float = 0.0,
    phase_rad: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Fundamental Gaussian field at a waist plane.

    ``waist_radius_m`` is the usual 1/e field radius (1/e^2 intensity radius).
    """
    if waist_radius_m <= 0:
        raise ValueError("waist_radius_m must be positive")
    x, y = grid.mesh
    r2 = (x - x0_m) ** 2 + (y - y0_m) ** 2
    field = np.exp(-r2 / waist_radius_m**2 + 1j * phase_rad)
    return normalize_power(field, grid) if normalize else field

def _hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    if n < 0:
        raise ValueError("Hermite order must be nonnegative")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(1, n):
        h0, h1 = h1, 2 * x * h1 - 2 * k * h0
    return h1


def hermite_gaussian(
    grid: Grid2D,
    m: int,
    n: int,
    waist_radius_m: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """HG_mn field at its waist plane."""
    if waist_radius_m <= 0:
        raise ValueError("waist_radius_m must be positive")
    x, y = grid.mesh
    xs = math.sqrt(2) * x / waist_radius_m
    ys = math.sqrt(2) * y / waist_radius_m
    field = (
        _hermite_polynomial(m, xs)
        * _hermite_polynomial(n, ys)
        * np.exp(-(x**2 + y**2) / waist_radius_m**2)
    ).astype(np.complex128)
    return normalize_power(field, grid) if normalize else field


def _generalized_laguerre(p: int, alpha: int, x: np.ndarray) -> np.ndarray:
    if p < 0 or alpha < 0:
        raise ValueError("p and alpha must be nonnegative")
    result = np.zeros_like(x, dtype=float)
    for k in range(p + 1):
        coeff = ((-1) ** k) * math.comb(p + alpha, p - k) / math.factorial(k)
        result += coeff * x**k
    return result


def laguerre_gaussian(
    grid: Grid2D,
    p: int,
    l: int,
    waist_radius_m: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """LG_p^l field at its waist plane, including vortex phase exp(i*l*phi)."""
    if p < 0:
        raise ValueError("p must be nonnegative")
    if waist_radius_m <= 0:
        raise ValueError("waist_radius_m must be positive")
    x, y = grid.mesh
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    a = abs(l)
    rho2 = 2 * r**2 / waist_radius_m**2
    radial = (math.sqrt(2) * r / waist_radius_m) ** a
    radial *= _generalized_laguerre(p, a, rho2)
    field = radial * np.exp(-r**2 / waist_radius_m**2) * np.exp(1j * l * phi)
    return normalize_power(field, grid) if normalize else field


def apply_phase_mask(field: np.ndarray, phase_rad: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Apply an arbitrary phase-only mask to a field."""
    arr = _check_field(field, grid)
    phase = np.asarray(phase_rad, dtype=float)
    if phase.shape != grid.shape:
        raise ValueError(f"phase shape {phase.shape} does not match grid {grid.shape}")
    return arr * np.exp(1j * phase)


def angular_spectrum_propagate(
    field: np.ndarray,
    grid: Grid2D,
    wavelength_m: float,
    distance_m: float,
    *,
    refractive_index: float = 1.0,
    bandlimit: bool = True,
) -> np.ndarray:
    """Propagate a scalar complex envelope through a uniform medium.

    Uses the exact angular-spectrum transfer function

        H(kx, ky) = exp(i * kz * z),
        kz = sqrt((n*k0)^2 - kx^2 - ky^2).

    Evanescent components are retained as decaying exponentials when
    ``bandlimit=False``. With ``bandlimit=True``, components outside the
    propagating circle are suppressed. Despite the historical argument name,
    this is *not* a full anti-alias angular-spectrum band-limit. FFT propagation
    is periodic, so the transverse window must still be large enough to prevent
    diffracted light from wrapping around the array boundaries.
    """
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if refractive_index <= 0:
        raise ValueError("refractive_index must be positive")

    arr = _check_field(field, grid)
    if distance_m == 0:
        return arr.copy()

    fx, fy = np.meshgrid(grid.fx, grid.fy, indexing="xy")
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    k = 2 * np.pi * refractive_index / wavelength_m
    kt2 = kx**2 + ky**2

    kz = np.sqrt((k**2 - kt2).astype(np.complex128))
    transfer = np.exp(1j * kz * distance_m)
    if bandlimit:
        transfer = np.where(kt2 <= k**2, transfer, 0.0)

    spectrum = np.fft.fft2(arr)
    return np.fft.ifft2(spectrum * transfer)
