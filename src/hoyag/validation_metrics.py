"""Stage 7V: independent-grid metrics and mesh-independent physical inputs.

No solver or fitted Gaussian mode is substituted here. Fields are compared on
one declared physical domain. OPD comparison removes piston only; polarization
and angular-mode content are not inferred from intensity alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j1


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DiagnosticGrid:
    """Uniform cell-centred grid implementing the repository Grid2D contract."""
    nx: int
    ny: int
    dx: float
    dy: float

    def __post_init__(self):
        for n in (self.nx, self.ny):
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
                raise ValueError("grid sizes must be integers >=2")
        if not np.all(np.isfinite([self.dx, self.dy])) or min(self.dx, self.dy) <= 0:
            raise ValueError("grid spacings must be positive")

    @classmethod
    def square(cls, n: int, width_m: float):
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("n must be an integer >=2")
        return cls(n, n, width_m / n, width_m / n)

    @property
    def x(self): return (np.arange(self.nx) - (self.nx - 1) / 2) * self.dx
    @property
    def y(self): return (np.arange(self.ny) - (self.ny - 1) / 2) * self.dy
    @property
    def shape(self): return self.ny, self.nx
    @property
    def mesh(self): return np.meshgrid(self.x, self.y, indexing="xy")
    @property
    def fx(self): return np.fft.fftfreq(self.nx, self.dx)
    @property
    def fy(self): return np.fft.fftfreq(self.ny, self.dy)


def grid_from_axes(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    for a in (x, y):
        if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
            raise ValueError("finite coordinate vectors required")
        if not np.allclose(np.diff(a), a[1] - a[0], rtol=1e-10, atol=1e-15) or a[1] <= a[0]:
            raise ValueError("coordinates must be uniformly increasing")
        if not np.allclose(a, -a[::-1], rtol=0, atol=1e-12):
            raise ValueError("coordinates must be cell-centred around zero")
    return DiagnosticGrid(len(x), len(y), float(x[1] - x[0]), float(y[1] - y[0]))


@dataclass(frozen=True)
class ContinuousDensity:
    """One continuous positive Ho map, sampled rather than regenerated per mesh.

    Fourier wavevectors are in rad/m. The analytical cylindrical volume mean
    of each component is subtracted. contrast_bound bounds the sum of absolute
    cosine amplitudes, NOT an RMS or a pixel-scale correlation length. Positive
    density is guaranteed for contrast_bound < 0.5. No per-grid renormalization.
    """
    mean_m3: float = 1.52e26
    radius_m: float = 0.005
    thickness_m: float = 0.001
    contrast_bound: float = 0.0
    correlation_m: float = 0.0006
    seed: int = 17
    terms: int = 16

    def __post_init__(self):
        vals = [self.mean_m3, self.radius_m, self.thickness_m, self.correlation_m]
        if not np.all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("positive finite density, lengths and correlation required")
        if not np.isfinite(self.contrast_bound) or not 0 <= self.contrast_bound < 0.5:
            raise ValueError("contrast_bound must be in [0,0.5)")
        for v in (self.seed, self.terms):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError("seed and terms must be integers")
        if self.terms < 1 or self.seed < 0:
            raise ValueError("invalid terms or seed")

    def coefficients(self):
        rng = np.random.default_rng(self.seed)
        wavevectors = rng.normal(size=(self.terms, 3)) / self.correlation_m
        phase = rng.uniform(-np.pi, np.pi, self.terms)
        amplitudes = rng.uniform(0.2, 1.0, self.terms)
        amplitudes *= self.contrast_bound / amplitudes.sum()
        kr = np.hypot(wavevectors[:, 0], wavevectors[:, 1]) * self.radius_m
        disk = np.ones_like(kr)
        nonzero = kr != 0
        disk[nonzero] = 2 * j1(kr[nonzero]) / kr[nonzero]
        kd = wavevectors[:, 2] * self.thickness_m / 2
        means = disk * np.sinc(kd / np.pi) * np.cos(phase + kd)
        return wavevectors, phase, amplitudes, means

    def __call__(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        if not all(np.all(np.isfinite(a)) for a in (x, y, z)):
            raise ValueError("coordinates must be finite")
        value = np.ones(x.shape)
        for k, p, a, m in zip(*self.coefficients()):
            value += a * (np.cos(k[0] * x + k[1] * y + k[2] * z + p) - m)
        return self.mean_m3 * value

    @property
    def fingerprint(self): return stable_hash(asdict(self))

    def cell_average(self, mesh, order=4):
        """Gauss quadrature in z,r²,phi with the same continuous coefficients."""
        if order < 2:
            raise ValueError("quadrature order must be >=2")
        if not np.isclose(mesh.r_edges_m[-1], self.radius_m, rtol=0, atol=1e-12) or not np.isclose(mesh.z_edges_m[-1], self.thickness_m, rtol=0, atol=1e-12):
            raise ValueError("density definition and physical mesh dimensions differ")
        nodes, weights = leggauss(order)
        z = (mesh.z_edges_m[:-1, None] + mesh.z_edges_m[1:, None]) / 2 + np.diff(mesh.z_edges_m)[:, None] * nodes / 2
        r2 = mesh.r_edges_m ** 2
        r = np.sqrt((r2[:-1, None] + r2[1:, None]) / 2 + np.diff(r2)[:, None] * nodes / 2)
        # Axisymmetric cells still integrate the asymmetric physical source.
        phi_order = max(32, order) if mesh.nphi == 1 else order
        pn, pw = leggauss(phi_order)
        phi = (np.arange(mesh.nphi)[:, None] + .5) * 2*np.pi/mesh.nphi + np.pi/mesh.nphi * pn
        answer = np.zeros(mesh.shape)
        for iz, wz in enumerate(weights / 2):
            for ir, wr in enumerate(weights / 2):
                for ip, wp in enumerate(pw / 2):
                    xx = r[:, ir, None] * np.cos(phi[:, ip])[None, :]
                    yy = r[:, ir, None] * np.sin(phi[:, ip])[None, :]
                    answer += wz * wr * wp * self(xx[None], yy[None], z[:, iz, None, None])
        return answer


def interval_average_matrix(old_edges, new_edges):
    """Exact positive averaging of a fixed piecewise-constant 1-D function."""
    old, new = np.asarray(old_edges, float), np.asarray(new_edges, float)
    for a in (old, new):
        if a.ndim != 1 or len(a) < 2 or np.any(np.diff(a) <= 0) or not np.all(np.isfinite(a)):
            raise ValueError("strictly increasing finite edges required")
    if not np.allclose(old[[0, -1]], new[[0, -1]], rtol=0, atol=1e-14):
        raise ValueError("remapping must preserve the physical domain")
    overlap = np.maximum(0, np.minimum(new[1:, None], old[None, 1:]) -
                           np.maximum(new[:-1, None], old[None, :-1]))
    return overlap / np.diff(new)[:, None]


def conservative_remap(values, old_mesh, new_mesh):
    """Conservative cell averages in (z,r,phi), optionally with leading axes."""
    a = np.asarray(values, float)
    if a.shape[-3:] != old_mesh.shape or not np.all(np.isfinite(a)):
        raise ValueError("finite array ending in old (z,r,phi) shape required")
    z = interval_average_matrix(old_mesh.z_edges_m, new_mesh.z_edges_m)
    r = interval_average_matrix(old_mesh.r_edges_m ** 2, new_mesh.r_edges_m ** 2)
    p = interval_average_matrix(np.linspace(0, 2*np.pi, old_mesh.nphi+1),
                                np.linspace(0, 2*np.pi, new_mesh.nphi+1))
    return np.einsum("ia,jb,kc,...abc->...ijk", z, r, p, a, optimize=True)


def sample_cartesian(values, grid, x_query, y_query, *, fill_value=0):
    """Interpolate trailing (y,x) axes, preserving complex phase and polarization."""
    a = np.asarray(values)
    if a.shape[-2:] != grid.shape or not np.all(np.isfinite(a)):
        raise ValueError("field or map must end in (ny,nx) and be finite")
    xq, yq = np.broadcast_arrays(x_query, y_query)
    source = np.moveaxis(a, (-2, -1), (0, 1))
    result = RegularGridInterpolator((grid.y, grid.x), source, bounds_error=False,
                                     fill_value=fill_value)(np.stack((yq, xq), axis=-1))
    return np.moveaxis(result, tuple(range(xq.ndim, result.ndim)), tuple(range(a.ndim-2)))


@dataclass(frozen=True)
class ComparisonDomain:
    radius_m: float = 0.001
    weight_radius_m: float = 0.000408
    nr: int = 64
    nphi: int = 128

    def quadrature(self):
        if not 0 < self.weight_radius_m or not 0 < self.radius_m or self.nr < 4 or self.nphi < 8:
            raise ValueError("invalid comparison domain")
        t, w = leggauss(self.nr)
        r = self.radius_m * (t + 1) / 2
        theta = 2*np.pi*np.arange(self.nphi) / self.nphi
        x, y = r[:, None] * np.cos(theta), r[:, None] * np.sin(theta)
        area = np.broadcast_to((r * w * self.radius_m/2 * 2*np.pi/self.nphi)[:, None], x.shape)
        weights = area * np.exp(-2*r[:, None]**2/self.weight_radius_m**2)
        weights /= weights.sum()
        return x, y, area, weights


def vector_overlap(a, b, weights=None):
    """Normalized squared coherent overlap; global phase alone is irrelevant."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("matching finite complex fields required")
    w = 1 if weights is None else np.asarray(weights, float)
    na, nb = np.sum(abs(a)**2*w), np.sum(abs(b)**2*w)
    if min(na, nb) <= 0:
        raise ValueError("cannot compare zero fields")
    return float(np.clip(abs(np.sum(a.conj()*b*w))**2/(na*nb), 0, 1))


def modal_mixture_fidelity(a, b, powers_a, powers_b, weights):
    """Low-rank density-operator fidelity of INCOHERENT modal mixtures.

    No coherent sum of unrelated modal phases is constructed. Valid with unequal
    mode counts, permutations and arbitrary rotations of equally populated
    orthonormal degenerate subspaces. Insignificant modes are not discarded.
    """
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("sample fields must have the same non-mode axes")
    pa, pb = np.asarray(powers_a, float), np.asarray(powers_b, float)
    if pa.shape != (len(a),) or pb.shape != (len(b),):
        raise ValueError("one nonnegative power per mode required")
    if np.any(~np.isfinite(np.r_[pa, pb])) or min(pa.min(), pb.min()) < 0 or min(pa.sum(), pb.sum()) <= 0:
        raise ValueError("finite nonnegative modal powers with positive sums required")
    w = np.broadcast_to(weights, a.shape[1:]).ravel()
    aa, bb = a.reshape(len(a), -1), b.reshape(len(b), -1)
    na = np.sqrt((abs(aa)**2) @ w)
    nb = np.sqrt((abs(bb)**2) @ w)
    if min(na.min(), nb.min()) <= 0:
        raise ValueError("zero mode field")
    aa = aa / na[:, None]; bb = bb / nb[:, None]
    overlap = (aa.conj()*w) @ bb.T
    cross = np.sqrt(pa/pa.sum())[:, None] * overlap * np.sqrt(pb/pb.sum())[None]
    return float(np.clip(np.linalg.svd(cross, compute_uv=False).sum()**2, 0, 1))


def compare_fields(a, ga, b, gb, domain, powers_a=None, powers_b=None):
    """Compare all vector components on identical physical quadrature nodes."""
    for g in (ga, gb):
        if domain.radius_m > min(abs(g.x[0]), abs(g.y[0])):
            raise ValueError("comparison domain extends outside a sampled optical window")
    x, y, area, _ = domain.quadrature()
    aa = sample_cartesian(a, ga, x, y)
    bb = sample_cartesian(b, gb, x, y)
    if aa.ndim == 3: aa = aa[None]
    if bb.ndim == 3: bb = bb[None]
    if aa.shape[1] != 2 or bb.shape[1] != 2:
        raise ValueError("fields must be (modes,2,ny,nx) or (2,ny,nx)")
    pa = np.ones(len(aa)) if powers_a is None else powers_a
    pb = np.ones(len(bb)) if powers_b is None else powers_b
    fid = modal_mixture_fidelity(aa, bb, pa, pb, area)
    # The ROI may hide power loss; record how much each original field it covers.
    original_a = np.asarray(a); original_b = np.asarray(b)
    if original_a.ndim == 3: original_a = original_a[None]
    if original_b.ndim == 3: original_b = original_b[None]
    xa, ya = ga.mesh; xb, yb = gb.mesh
    roi_a = xa*xa+ya*ya <= domain.radius_m**2
    roi_b = xb*xb+yb*yb <= domain.radius_m**2
    ca = np.sum(abs(original_a)**2*roi_a, axis=(1,2,3))/np.sum(abs(original_a)**2, axis=(1,2,3))
    cb = np.sum(abs(original_b)**2*roi_b, axis=(1,2,3))/np.sum(abs(original_b)**2, axis=(1,2,3))
    ia = np.sum(abs(aa)**2*area, axis=(1,2,3)) / (np.sum(abs(original_a)**2*roi_a, axis=(1,2,3))*ga.dx*ga.dy)
    ib = np.sum(abs(bb)**2*area, axis=(1,2,3)) / (np.sum(abs(original_b)**2*roi_b, axis=(1,2,3))*gb.dx*gb.dy)
    return {"mixture_fidelity": fid, "roi_power_fraction_a": ca.tolist(), "roi_power_fraction_b": cb.tolist(),
            "interpolation_norm_ratio_a":ia.tolist(), "interpolation_norm_ratio_b":ib.tolist(),
            "single_mode_overlap": fid if len(aa) == len(bb) == 1 else None}


def compare_opd(a, ga, b, gb, domain):
    for g in (ga, gb):
        if domain.radius_m > min(abs(g.x[0]), abs(g.y[0])):
            raise ValueError("comparison domain extends outside a sampled optical window")
    x, y, _, w = domain.quadrature()
    da, db = sample_cartesian(a, ga, x, y), sample_cartesian(b, gb, x, y)
    delta = db - da
    piston = float(np.sum(w*delta))
    residual = delta - piston
    return {"piston_difference_m": piston,
            "piston_removed_difference_rms_m": float(np.sqrt(np.sum(w*residual**2))),
            "piston_removed_difference_peak_m": float(abs(residual).max()),
            "reference_radius_m": domain.radius_m, "weight_radius_m": domain.weight_radius_m,
            "removed_terms": ["piston"], "tilt_or_defocus_removed": False}


def relative_change(a, b, absolute_floor=1e-12):
    if not np.all(np.isfinite([a, b, absolute_floor])) or absolute_floor <= 0:
        raise ValueError("finite values and positive absolute floor required")
    return float(abs(a-b)/max(abs(a), abs(b), absolute_floor))


def oam_spectrum(field, grid, *, radius_m=0.0015, radial_nodes=96, angular_nodes=256, max_charge=12):
    """Azimuthal spectrum of both vector components about the fixed lab axis.

    Coefficients c_l(r) use FFT(exp(+il phi)) -> positive l. Power outside the
    reported charge range is retained as tail, not silently renormalized away.
    A finite-radius/cell-interpolation quadrature error is separately reported.
    """
    if max_charge >= angular_nodes // 2 or max_charge < 0:
        raise ValueError("charge range must lie within angular Nyquist limit")
    if radius_m > min(abs(grid.x[0]), abs(grid.y[0])):
        raise ValueError("OAM integration circle must fit inside sampled window")
    field = np.asarray(field, complex)
    if field.shape != (2, *grid.shape):
        raise ValueError("field must be (2,ny,nx)")
    domain = ComparisonDomain(radius_m, radius_m, radial_nodes, angular_nodes)
    x, y, area, _ = domain.quadrature()
    sampled = sample_cartesian(field, grid, x, y)
    coefficients = np.fft.fft(sampled, axis=-1)/angular_nodes
    angular_power = np.sum(abs(coefficients)**2 * (area[:, 0]*angular_nodes)[None, :, None], axis=(0,1))
    total = float(angular_power.sum())
    if total <= 0:
        raise ValueError("zero field in OAM aperture")
    charges = np.arange(-max_charge, max_charge+1)
    powers = angular_power[charges % angular_nodes] / total
    cartesian = float(np.sum(abs(field)**2)*grid.dx*grid.dy)
    return {"charges": charges.tolist(), "fractions": powers.tolist(),
            "unreported_charge_tail": float(max(0., 1-powers.sum())),
            "polar_to_full_cartesian_power": total/cartesian,
            "dominant_charge": int(charges[np.argmax(powers)]),
            "radius_m": radius_m, "axis": "fixed lab origin, not recentered for each field"}


def closed_phase_winding(field_component, grid, radius_m, samples=512, amplitude_floor=1e-5):
    theta = np.linspace(0, 2*np.pi, samples, endpoint=False)
    f = sample_cartesian(field_component, grid, radius_m*np.cos(theta), radius_m*np.sin(theta))
    if np.min(abs(f)) <= amplitude_floor*np.max(abs(field_component)):
        return {"resolved": False, "charge": None, "reason": "contour crosses negligible amplitude"}
    # Includes the last-to-first edge, unlike endpoint differences of an open trace.
    increments = np.angle(np.roll(f, -1)*f.conj())
    if np.max(abs(increments)) >= .95*np.pi:
        return {"resolved": False, "charge": None, "reason": "angular phase increment undersampled"}
    return {"resolved": True, "charge": float(increments.sum()/(2*np.pi))}
