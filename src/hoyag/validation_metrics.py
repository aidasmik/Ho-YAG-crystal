"""Stage 7V: physical-coordinate comparisons, OAM and conservative mesh exchange.

A small eigensolver residual is not a discretization validation. These helpers
compare physical quantities without aligning away beam shifts, tilt or defocus.
Only one scalar piston/global phase is irrelevant. OAM is Fourier power versus
azimuth, summed over polarization, not a phase winding measured on one circle.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment


def fingerprint(value):
    """Stable JSON fingerprint; rejects NaN, which cannot identify a physical case."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _axis(axis):
    x = np.asarray(axis, float)
    if x.ndim != 1 or len(x) < 4 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('at least four finite increasing axis coordinates required')
    if not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-10, atol=1e-15):
        raise ValueError('Cartesian comparison grid must be uniform')
    return x


def sample_cartesian(values, x_m, y_m, x_query, y_query, *, order=3):
    """Interpolate real and imaginary parts, never wrapped phase; zero outside.

    Input shape (...,ny,nx); query x/y arrays have the same shape. Cubic spline
    interpolation is itself tested by refinement; it does not create resolution.
    """
    x, y = _axis(x_m), _axis(y_m)
    a = np.asarray(values)
    if a.shape[-2:] != (len(y), len(x)) or np.any(~np.isfinite(a)):
        raise ValueError('finite data and Cartesian grid shapes must match')
    xx, yy = np.broadcast_arrays(np.asarray(x_query, float), np.asarray(y_query, float))
    coords = np.array([(yy-y[0])/(y[1]-y[0]), (xx-x[0])/(x[1]-x[0])])
    output = []
    for plane in a.reshape(-1, len(y), len(x)):
        b = map_coordinates(plane.real, coords, order=order, mode='constant', cval=0.)
        if np.iscomplexobj(plane):
            b = b + 1j*map_coordinates(plane.imag, coords, order=order, mode='constant', cval=0.)
        output.append(b)
    return np.asarray(output).reshape(a.shape[:-2] + xx.shape)


def common_coordinates(xa, ya, xb, yb):
    """Union of physical pixel windows at the finer spacing, no beam recentering."""
    def one(a, b):
        a, b = _axis(a), _axis(b)
        da, db = a[1]-a[0], b[1]-b[0]
        step = min(da, db)
        lo = min(a[0]-da/2, b[0]-db/2)
        hi = max(a[-1]+da/2, b[-1]+db/2)
        n = int(np.ceil((hi-lo)/step-1e-10))
        if n > 4096:
            raise ValueError('comparison grid exceeds safety limit')
        return lo + (np.arange(n)+.5)*step
    return one(xa, xb), one(ya, yb)


def vector_overlap(a, b):
    """Squared normalized complex overlap, including both polarization components."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape != b.shape or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError('finite fields of the same shape required')
    aa, bb = np.vdot(a, a).real, np.vdot(b, b).real
    if aa <= 0 or bb <= 0:
        raise ValueError('overlap is undefined for a zero field')
    return float(np.clip(abs(np.vdot(a, b))**2/(aa*bb), 0, 1))


def compare_fields(a, xa, ya, b, xb, yb):
    """Compare vector eigenbranches on a common physical grid.

    Matching permutations are allowed; arbitrary unitary rotation is not used to
    manufacture branch fidelity. Subspace capture is reported separately to
    identify polarization/eigenvalue degeneracy and differing mode counts.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 4 or b.ndim != 4 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError('expected (modes,2,ny,nx) fields')
    x, y = common_coordinates(xa, ya, xb, yb)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    ar = sample_cartesian(a, xa, ya, xx, yy).reshape(len(a), -1)
    br = sample_cartesian(b, xb, yb, xx, yy).reshape(len(b), -1)
    ar /= np.linalg.norm(ar, axis=1)[:, None]
    br /= np.linalg.norm(br, axis=1)[:, None]
    overlaps = np.clip(abs(ar.conj() @ br.T)**2, 0, 1)
    row, col = linear_sum_assignment(-overlaps)
    qa, ra = np.linalg.qr(ar.T, mode='reduced')
    qb, rb = np.linalg.qr(br.T, mode='reduced')
    if np.min(abs(np.diag(ra))) < 1e-8 or np.min(abs(np.diag(rb))) < 1e-8:
        raise ValueError('linearly dependent modes cannot define a comparison subspace')
    singular = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return {'matched_branch_overlaps': overlaps[row, col].tolist(),
            'minimum_matched_overlap': float(overlaps[row, col].min()),
            'minimum_subspace_capture': float(np.clip(singular.min()**2, 0, 1)),
            'same_mode_count': bool(len(a) == len(b)),
            'comparison_shape': [len(y), len(x)],
            'alignment': 'permutation and one global phase only; no shift, tilt or defocus removal'}


def piston_removed_opd_error(opda, xa, ya, opdb, xb, yb, *, radius_m=.001, waist_m=.000408):
    """Gaussian-weighted RMS of UNWRAPPED OPD difference, subtracting piston only."""
    if not np.isfinite(radius_m+waist_m) or min(radius_m, waist_m) <= 0:
        raise ValueError('positive optical comparison radius and waist required')
    x, y = common_coordinates(xa, ya, xb, yb)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    for axis in (xa, ya, xb, yb):
        if np.min(axis) > -radius_m or np.max(axis) < radius_m:
            raise ValueError('both grids must cover the full OPD comparison pupil')
    a = sample_cartesian(opda, xa, ya, xx, yy)
    b = sample_cartesian(opdb, xb, yb, xx, yy)
    r2 = xx*xx+yy*yy
    weight = np.exp(-2*r2/waist_m**2)*(r2 <= radius_m**2)
    weight /= weight.sum()
    delta = a-b
    piston = float(np.sum(weight*delta))
    return {'rms_difference_m': float(np.sqrt(np.sum(weight*(delta-piston)**2))),
            'removed_piston_m': piston, 'radius_m': radius_m, 'weight_waist_m': waist_m,
            'tilt_removed': False, 'defocus_removed': False}


def oam_spectrum(field, x_m, y_m, *, radius_m, radial_points=160, angular_points=256,
                 charges=tuple(range(-8, 9))):
    """Azimuthal Fourier power, integrated over radius and both lab polarizations.

    Fractions use ALL sampled azimuthal frequencies as denominator, not only the
    requested charge list. Captured pupil energy is checked against the Cartesian
    integral. High-order aliasing and interpolation require separate refinement.
    """
    if radial_points < 8 or angular_points < 32 or angular_points % 2:
        raise ValueError('use >=8 radial nodes and even >=32 angular samples')
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('pupil radius must be finite and positive')
    x, y = _axis(x_m), _axis(y_m)
    if radius_m > min(-x[0], x[-1], -y[0], y[-1]):
        raise ValueError('OAM pupil must lie inside Cartesian grid')
    f = np.asarray(field, complex)
    if f.shape != (2, len(y), len(x)):
        raise ValueError('expected vector field (2,ny,nx)')
    nodes, weights = leggauss(radial_points)
    r = (nodes+1)*radius_m/2
    wr = weights*radius_m/2*r
    phi = np.arange(angular_points)*2*np.pi/angular_points
    polar = sample_cartesian(f, x, y, r[:, None]*np.cos(phi), r[:, None]*np.sin(phi))
    coef = np.fft.fft(polar, axis=-1)/angular_points
    power = 2*np.pi*np.sum(abs(coef)**2*wr[None, :, None], axis=(0, 1))
    ell = np.rint(np.fft.fftfreq(angular_points)*angular_points).astype(int)
    total = float(power.sum())
    plane_total = float(np.sum(abs(f)**2)*(x[1]-x[0])*(y[1]-y[0]))
    if min(total, plane_total) <= 0:
        raise ValueError('nonzero field required for OAM')
    result = {str(l): float(power[ell == l].sum()/total) for l in charges}
    return {'charge_fractions': result, 'reported_fraction': float(sum(result.values())),
            'unreported_fraction': float(max(0., 1-sum(result.values()))),
            'mean_azimuthal_order': float(np.sum(ell*power)/total),
            'pupil_power_over_cartesian_power': total/plane_total,
            'radial_points': radial_points, 'angular_points': angular_points,
            'definition': 'azimuthal-mode power; not a single-contour winding number'}


def edge_power_fraction(field, *, outer_fraction=.1):
    a = np.asarray(field)
    power = np.sum(abs(a)**2, axis=tuple(range(a.ndim-2)))
    ny, nx = power.shape
    if not 0 < outer_fraction < .5 or power.sum() <= 0:
        raise ValueError('nonzero field and a valid border fraction required')
    ix = max(1, int(nx*outer_fraction)); iy = max(1, int(ny*outer_fraction))
    mask = np.ones(power.shape, bool); mask[iy:-iy, ix:-ix] = False
    return float(power[mask].sum()/power.sum())


def _overlaps(new_edges, old_edges):
    return np.maximum(0., np.minimum(new_edges[1:, None], old_edges[None, 1:])
                        - np.maximum(new_edges[:-1, None], old_edges[None, :-1]))


def conservative_heat_remap(q, old_mesh, new_mesh):
    """Exact volume-overlap remap of a piecewise-constant (z,r,phi) heat field.

    Preserves total heat for the same cylindrical domain. This freezes the OLD
    optical heat distribution, so a thermal-only refinement is not full coupling.
    """
    q = np.asarray(q, float)
    if q.shape != old_mesh.shape or np.any(~np.isfinite(q)):
        raise ValueError('finite source with the old thermal shape required')
    for old, new in ((old_mesh.r_edges_m, new_mesh.r_edges_m),
                     (old_mesh.z_edges_m, new_mesh.z_edges_m)):
        if not np.allclose([old[0], old[-1]], [new[0], new[-1]], rtol=0, atol=1e-14):
            raise ValueError('conservative remapping requires identical physical domains')
    oz = _overlaps(new_mesh.z_edges_m, old_mesh.z_edges_m)
    or2 = .5*_overlaps(new_mesh.r_edges_m**2, old_mesh.r_edges_m**2)
    op = _overlaps(np.linspace(0, 2*np.pi, new_mesh.nphi+1),
                   np.linspace(0, 2*np.pi, old_mesh.nphi+1))
    energy = np.einsum('az,br,cp,zrp->abc', oz, or2, op, q, optimize=True)
    return energy/new_mesh.volumes_m3


@dataclass(frozen=True)
class ContinuousDopant:
    """Bounded physical-coordinate random Fourier field, independent of any mesh.

    amplitude_bound is NOT a promised RMS. Frequencies/phases use a fixed seed
    in physical metres; no per-grid recentering/rescaling changes the specimen.
    """
    mean_m3: float = 1.52e26
    amplitude_bound: float = 0.
    correlation_length_m: float = .0007
    seed: int = 11
    terms: int = 8

    def __post_init__(self):
        if not np.isfinite(self.mean_m3) or self.mean_m3 <= 0:
            raise ValueError('positive density required by the modal model')
        if not np.isfinite(self.amplitude_bound) or not 0 <= self.amplitude_bound < 1:
            raise ValueError('amplitude bound must be in [0,1)')
        if not np.isfinite(self.correlation_length_m) or self.correlation_length_m <= 0:
            raise ValueError('correlation length parameter must be positive')
        if not isinstance(self.seed, int) or not isinstance(self.terms, int) or self.terms < 1:
            raise ValueError('integer seed and positive integer term count required')

    def value(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        rng = np.random.default_rng(self.seed)
        vectors = rng.normal(size=(self.terms, 3))/self.correlation_length_m
        phases = rng.uniform(0, 2*np.pi, self.terms)
        value = np.zeros(x.shape)
        for k, phase in zip(vectors, phases):
            value += np.cos(k[0]*x+k[1]*y+k[2]*z+phase)/self.terms
        return self.mean_m3*(1+self.amplitude_bound*value)

    def cell_averages(self, mesh, *, order=3, tolerance=1e-10, max_order=96):
        """Exact axial and adaptively checked radial/azimuthal Fourier averages.

        Integrating in r with the explicit r weight avoids a square-root endpoint
        in the r-squared variable. No per-grid density renormalization is used.
        """
        if not isinstance(order,int) or order < 2 or tolerance <= 0 or max_order < 2*order:
            raise ValueError('invalid cell quadrature settings')
        if self.amplitude_bound == 0:
            return np.full(mesh.shape,self.mean_m3)
        rng=np.random.default_rng(self.seed)
        vectors=rng.normal(size=(self.terms,3))/self.correlation_length_m
        phases=rng.uniform(0,2*np.pi,self.terms)
        zlo,zhi=mesh.z_edges_m[:-1],mesh.z_edges_m[1:]
        rlo,rhi=mesh.r_edges_m[:-1],mesh.r_edges_m[1:]
        plo=np.arange(mesh.nphi)*2*np.pi/mesh.nphi
        dphi=2*np.pi/mesh.nphi
        def integrate(n):
            nodes,weights=leggauss(n)
            r=(rlo[:,None]+rhi[:,None])/2+(rhi-rlo)[:,None]/2*nodes
            wr=weights[None,:]*r*(rhi-rlo)[:,None]/(rhi*rhi-rlo*rlo)[:,None]
            phi=plo[:,None]+dphi/2*(nodes+1)
            xx=r[:,None,:,None]*np.cos(phi)[None,:,None,:]
            yy=r[:,None,:,None]*np.sin(phi)[None,:,None,:]
            total=np.zeros(mesh.shape)
            for k,phase in zip(vectors,phases):
                plane=np.sum(np.exp(1j*(k[0]*xx+k[1]*yy))*wr[:,None,:,None]*weights[None,None,None,:]/2,axis=(2,3))
                axial=np.exp(1j*(k[2]*(zlo+zhi)/2+phase))*np.sinc(k[2]*(zhi-zlo)/(2*np.pi))
                total+=(axial[:,None,None]*plane[None]).real/self.terms
            return total
        n=order;previous=integrate(n)
        while 2*n <= max_order:
            n*=2;current=integrate(n)
            if np.max(abs(current-previous))*self.amplitude_bound <= tolerance:
                return self.mean_m3*(1+self.amplitude_bound*current)
            previous=current
        raise ValueError('continuous dopant cell quadrature did not converge; refine quadrature/mesh')
