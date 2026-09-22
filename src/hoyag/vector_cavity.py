"""Stage 7A: grid-resolved vector cavity eigenfields and material-grid exchange.

The eigenproblem uses the Stage 4R/6 collapsed-disk, scalar-diffraction/Jones
operator. No Gaussian fit, implicit mode filter or phase conjugation is applied.
Frozen cycle-averaged gain is used to select the spatial modes. Degenerate
polarization eigenspaces are aligned to the preceding field, without selecting
an artificial vortex handedness.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
from scipy.optimize import linear_sum_assignment
from numpy.polynomial.legendre import leggauss


def normalize_vector(field):
    """Unit discrete L2 norm. Physical area normalization occurs at projection."""
    a = np.asarray(field, complex)
    norm = np.linalg.norm(a)
    if not np.all(np.isfinite(a)) or norm <= 0:
        raise ValueError('finite nonzero vector field required')
    return a / norm


def aligned_distance(a, b):
    """L2 field change with one irrelevant global phase removed."""
    aa, bb = normalize_vector(a), normalize_vector(b)
    overlap = np.vdot(aa, bb)
    return float(np.sqrt(max(0., 2. - 2.*min(abs(overlap), 1.))))


def subspace_distance(a, b):
    """Largest principal angle sine, independent of basis in a degenerate pair."""
    aa = np.asarray(a).reshape(len(a), -1).T
    bb = np.asarray(b).reshape(len(b), -1).T
    if aa.shape != bb.shape:
        raise ValueError('subspaces must have the same shape')
    qa = np.linalg.qr(aa)[0]; qb = np.linalg.qr(bb)[0]
    singular = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return float(np.sqrt(max(0., 1.-min(1., singular[-1])**2)))


class VectorRoundTrip:
    """Cached Stage 6 operator for fields (2,ny,nx), fixed laboratory x/y axes.

    Extra physical loss is allowed only through explicit nonnegative apertures.
    The disk material has two gain and two thermal/photoelastic traversals.
    Geometry OPD is a round-trip quantity and is applied once in total.
    """
    def __init__(self, grid, cavity, screens=None, single_pass_log_gain=0.):
        self.grid, self.cavity = grid, cavity
        self.shape = (2, *grid.shape)
        if min(grid.nx*grid.dx, grid.ny*grid.dy) < cavity.disk_diameter_m:
            raise ValueError('padded optical window must contain the physical disk')
        x, y = grid.mesh
        self.mask = (x*x+y*y <= (cavity.disk_diameter_m/2)**2)
        self.gain = np.broadcast_to(np.asarray(single_pass_log_gain, float), grid.shape)
        if not np.all(np.isfinite(self.gain)) or np.max(abs(self.gain)) > 50:
            raise ValueError('invalid gain screen')
        if screens is None:
            self.jout = self.jin = None
            self.geometry_half = np.ones(grid.shape, complex)
        else:
            if not np.isclose(screens.wavelength_m, cavity.wavelength_m, rtol=1e-12, atol=0):
                raise ValueError('screen/cavity wavelength mismatch')
            if screens.inward_jones.shape != (*grid.shape, 2, 2):
                raise ValueError('screen/grid mismatch')
            for value in (screens.inward_jones, screens.outward_jones,
                          screens.geometry_roundtrip_opd_m):
                if not np.all(np.isfinite(value)):
                    raise ValueError('nonfinite optical screen')
            self.jout, self.jin = screens.outward_jones, screens.inward_jones
            self.geometry_half = np.exp(1j*np.pi*screens.geometry_roundtrip_opd_m/cavity.wavelength_m)
        fx, fy = np.meshgrid(grid.fx, grid.fy, indexing='xy')
        k0 = 2*np.pi/cavity.wavelength_m
        self.propagator = np.exp(-.5j*cavity.reduced_length_m/k0*(2*np.pi)**2*(fx*fx+fy*fy))
        self.curvature = np.exp(-1j*k0*(x*x+y*y)/cavity.output_mirror_radius_m)
        self.back_amplitude = math.sqrt(cavity.disk_hr_reflectivity*(1-cavity.other_roundtrip_loss))
        self.pass_amplitude = self.mask*np.exp(self.gain/2)

    def travel(self, field):
        return np.fft.ifft2(np.fft.fft2(field, axes=(-2, -1))*self.propagator, axes=(-2, -1))

    @staticmethod
    def jones(field, matrix):
        return field if matrix is None else np.einsum('yxij,jyx->iyx', matrix, field, optimize=True)

    def propagate(self, field, *, return_visits=False):
        f = np.asarray(field, complex)
        if f.shape != self.shape or not np.all(np.isfinite(f)):
            raise ValueError(f'field must be finite with shape {self.shape}')
        outward = self.jones(f, self.jout)*self.geometry_half
        incident_oc = self.travel(outward*self.pass_amplitude)
        output = np.sqrt(self.cavity.output_transmission)*incident_oc
        inward = self.travel(incident_oc*np.sqrt(1-self.cavity.output_transmission)*self.curvature)
        back = inward*self.pass_amplitude*self.back_amplitude
        back = self.jones(back, self.jin)*self.geometry_half
        if return_visits:
            return back, output, outward, inward
        return back, output

    def linear_operator(self):
        size = int(np.prod(self.shape))
        def mv(a):
            return self.propagate(a.reshape(self.shape))[0].ravel()
        return LinearOperator((size, size), matvec=mv, dtype=np.complex128)


@dataclass
class EigenfieldResult:
    fields: np.ndarray
    eigenvalues: np.ndarray
    residuals: np.ndarray
    candidate_eigenvalues: np.ndarray
    converged: bool
    operator_calls: int
    status: str


def solve_vector_eigenfields(operator: VectorRoundTrip, previous_fields, *,
                             candidates=4, tolerance=2e-7, maxiter=600,
                             degeneracy_tolerance=2e-7, seed=7):
    """Largest-modulus non-Hermitian Arnoldi modes, checked in the full grid.

    A converged eigenpair is not proof of nonlinear/temporal multimode stability.
    Failure or missing candidates is returned explicitly, never as convergence.
    """
    prev = np.asarray(previous_fields, complex)
    if prev.ndim != 4 or prev.shape[1:] != operator.shape:
        raise ValueError('previous_fields must be (nmodes,2,ny,nx)')
    nm = len(prev)
    if not 1 <= nm <= candidates or candidates >= np.prod(operator.shape)-1:
        raise ValueError('require 1 <= nmodes <= candidates < vector size - 1')
    if tolerance <= 0 or maxiter < 1:
        raise ValueError('invalid eigenproblem tolerances')
    a = operator.linear_operator(); calls = [0]
    def mv(x):
        calls[0] += 1
        return a @ x
    counted = LinearOperator(a.shape, matvec=mv, dtype=complex)
    rng = np.random.default_rng(seed)
    v0 = np.sum(prev.reshape(nm, -1), axis=0)
    # Tiny full-space seed prevents silently constraining parity/polarization.
    v0 = v0 + 1e-5*(rng.normal(size=v0.size)+1j*rng.normal(size=v0.size))/np.sqrt(v0.size)
    success = True; status = 'Arnoldi converged'
    try:
        val, vec = eigs(counted, k=candidates, which='LM', v0=v0,
                        ncv=min(a.shape[0]-1, max(4*candidates+1, 64)), maxiter=maxiter, tol=tolerance*.1)
    except ArpackNoConvergence as err:
        val, vec = err.eigenvalues, err.eigenvectors
        success = False; status = 'Arnoldi iteration limit; incomplete candidate spectrum'
    if vec is None or len(val) < nm:
        return EigenfieldResult(prev.copy(), np.full(nm, np.nan+0j),
                                np.full(nm, np.inf), np.asarray(val), False, calls[0], status)
    order = np.argsort(-abs(val), kind='stable'); val, vec = val[order], vec[:, order]
    # Keep the nm strongest candidates; permit rotation only in truly degenerate
    # complex-eigenvalue clusters. Same |lambda| alone is not degeneracy.
    selected = []; chosen_values = []; consumed = set()
    for i in range(len(val)):
        if i in consumed or len(selected) >= nm:
            continue
        cluster = [j for j in range(i, len(val)) if j not in consumed and
                   abs(val[j]-val[i]) <= degeneracy_tolerance*max(abs(val[i]), 1.)]
        basis = np.linalg.qr(vec[:, cluster])[0]
        slots = min(len(cluster), nm-len(selected))
        refs = prev[len(selected):len(selected)+slots].reshape(slots, -1).T
        # Procrustes alignment is allowed here because eigenvalues are degenerate.
        left, _, right = np.linalg.svd(basis.conj().T @ refs, full_matrices=False)
        oriented = basis @ left @ right
        for j in range(slots):
            selected.append(normalize_vector(oriented[:, j]))
            chosen_values.append(val[i])
        consumed.update(cluster)
    fields = np.asarray(selected).reshape(nm, *operator.shape)
    # Match labels by overlap; never rotate different eigenvalues into each other.
    overlap = abs(prev.reshape(nm, -1).conj() @ fields.reshape(nm, -1).T)
    rows, cols = linear_sum_assignment(-overlap)
    fields = fields[cols[np.argsort(rows)]]
    vals = []; residuals = []
    for i in range(nm):
        phase = np.vdot(prev[i], fields[i])
        if abs(phase) > 0:
            fields[i] *= np.exp(-1j*np.angle(phase))
        out = operator.propagate(fields[i])[0]
        eigenvalue = np.vdot(fields[i], out)
        residual = np.linalg.norm(out-eigenvalue*fields[i])/max(np.linalg.norm(out), 1e-30)
        vals.append(eigenvalue); residuals.append(residual)
    if max(residuals) > tolerance:
        success = False; status = 'full-grid eigenfield residual exceeds tolerance'
    return EigenfieldResult(fields, np.asarray(vals), np.asarray(residuals), val,
                            success, calls[0], status)


class PlaneExchange:
    """Positive quadrature exchange between the Cartesian field and disk cells.

    Intensity is bilinearly interpolated and integrated over each exact annular
    wedge. Mode normalization is explicit and the pre-normalization quadrature
    error is returned. Large missing-power errors are rejected, not hidden.
    """
    def __init__(self, grid, mesh, order=4):
        if order < 2:
            raise ValueError('quadrature order must be >=2')
        self.grid, self.mesh = grid, mesh
        r_nodes, weights = leggauss(order)
        self.area = mesh.face_areas_m2.ravel()
        rows=[]; cols=[]; values=[]
        nphi_quad = max(order, 32) if mesh.nphi == 1 else order
        phi_nodes, phi_weights = leggauss(nphi_quad)
        for ir in range(mesh.nr):
            lo, hi = mesh.r_edges_m[ir:ir+2]**2
            r = np.sqrt((lo+hi)/2+(hi-lo)/2*r_nodes)
            for ip in range(mesh.nphi):
                dphi = 2*np.pi/mesh.nphi
                phi = (ip+.5)*dphi+.5*dphi*phi_nodes
                xx = (r[:, None]*np.cos(phi)).ravel()
                yy = (r[:, None]*np.sin(phi)).ravel()
                weight = ((hi-lo)/4*weights[:, None]*(dphi/2)*phi_weights[None, :]).ravel()
                ix = xx/grid.dx+(grid.nx-1)/2
                iy = yy/grid.dy+(grid.ny-1)/2
                jx = np.floor(ix).astype(int); jy = np.floor(iy).astype(int)
                if np.any(jx<0) or np.any(jx+1>=grid.nx) or np.any(jy<0) or np.any(jy+1>=grid.ny):
                    raise ValueError('optical window needs at least one pixel padding outside disk')
                fx, fy = ix-jx, iy-jy
                for ax, ay, bilinear in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                                         (0,1,(1-fx)*fy),(1,1,fx*fy)]:
                    rows.extend(np.full(len(xx),ir*mesh.nphi+ip))
                    cols.extend((jy+ay)*grid.nx+jx+ax)
                    values.extend(weight*bilinear)
        self.matrix = coo_matrix((values, (rows, cols)),
                                 shape=(mesh.nr*mesh.nphi, grid.nx*grid.ny)).tocsr()

    def mode_intensity(self, vector_field, *, max_quadrature_error=.05):
        f = np.asarray(vector_field)
        if f.shape != (2,*self.grid.shape):
            raise ValueError('vector field shape mismatch')
        intensity = np.sum(abs(f)**2, axis=0)
        full = np.sum(intensity)*self.grid.dx*self.grid.dy
        power = self.matrix @ intensity.ravel()
        captured = float(power.sum())
        error = abs(captured/full-1.)
        if captured <= 0 or error > max_quadrature_error:
            raise ValueError(f'material/optical quadrature mismatch {error:.3%}; refine grids/window')
        return power/(self.area*captured), error

    def surface_on_grid(self, value):
        v = np.asarray(value, float).reshape(self.mesh.nr,self.mesh.nphi)
        x,y = self.grid.mesh; r=np.hypot(x,y)
        if not np.all(np.isfinite(v)):
            raise ValueError('nonfinite material field')
        rr=np.clip(r, self.mesh.r_m[0], self.mesh.r_m[-1])
        if self.mesh.nphi == 1:
            result=np.interp(rr,self.mesh.r_m,v[:,0])
        else:
            ph=np.mod(np.arctan2(y,x),2*np.pi)
            angles=np.r_[self.mesh.phi_rad[-1]-2*np.pi,self.mesh.phi_rad,
                         self.mesh.phi_rad[0]+2*np.pi]
            vv=np.concatenate((v[:,-1:],v,v[:,:1]),axis=1)
            result=RegularGridInterpolator((self.mesh.r_m,angles),vv)(np.stack((rr,ph),axis=-1))
        return np.where(r<=self.mesh.r_edges_m[-1],result,0.)

    def visit_averaged_mode(self, field, passive_operator):
        _,_,forward,backward = passive_operator.propagate(field,return_visits=True)
        mf,ef=self.mode_intensity(forward)
        mb,eb=self.mode_intensity(backward)
        return .5*(mf+mb),max(ef,eb)


def passive_mode_losses(operator, fields):
    """Total logarithmic round-trip losses and mirror output fractions."""
    losses=[];outputs=[]
    for field in fields:
        f=normalize_vector(field);back,out=operator.propagate(f)
        retention=float(np.sum(abs(back)**2))
        if not 0<retention<=1+1e-9:
            raise ValueError('passive operator must not generate power')
        losses.append(-math.log(min(retention,1.)))
        outputs.append(float(np.sum(abs(out)**2)))
    return np.asarray(losses),np.asarray(outputs)
