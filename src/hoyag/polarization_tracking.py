"""Stage 7W: candidate-bank tracking without rotating split eigenbranches.

The polarization-family test guards a mode-count boundary. It does not declare
nearby complex eigenvalues identical and does not rotate a split pair. Candidate
assignment precedes selection; a newly dominant branch is never hidden merely
to preserve a label. This is finite-bank tracking, not global laser stability.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
from .vector_cavity import EigenfieldResult, normalize_vector, aligned_distance


@dataclass(frozen=True)
class TrackingSettings:
    exact_eigenvalue_tolerance: float = 1e-12
    pair_eigenvalue_tolerance: float = 1e-3
    pair_spatial_overlap: float = .995
    pair_vector_overlap_max: float = .1
    branch_reset_overlap: float = .25

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isscalar(value) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0,1]')
        if self.exact_eigenvalue_tolerance > 1e-10:
            raise ValueError('exact degeneracy must not be used to merge physically split modes')
        if self.exact_eigenvalue_tolerance >= self.pair_eigenvalue_tolerance:
            raise ValueError('exact eigenvalue tolerance must be smaller than pair tolerance')


@dataclass
class TrackedEigenfields(EigenfieldResult):
    diagnostics: dict = field(default_factory=dict)


def normalized_modes(fields):
    a = np.asarray(fields, complex)
    if a.ndim != 4 or a.shape[1] != 2 or len(a) < 1:
        raise ValueError('fields must have shape (nmodes,2,ny,nx)')
    a = np.array([normalize_vector(f) for f in a])
    gram = a.reshape(len(a), -1).conj() @ a.reshape(len(a), -1).T
    if np.linalg.eigvalsh(gram).min() < 1e-10:
        raise ValueError('modal fields are linearly dependent or numerically rank deficient')
    return a


def match_branches(previous, candidates):
    """One-to-one rectangular maximum-overlap match to the FULL candidate bank."""
    a, b = normalized_modes(previous), normalized_modes(candidates)
    if a.shape[1:] != b.shape[1:] or len(a) > len(b):
        raise ValueError('candidate bank must contain at least as many compatible fields')
    overlap = abs(a.reshape(len(a), -1).conj() @ b.reshape(len(b), -1).T)**2
    rows, cols = linear_sum_assignment(-overlap)
    cols = cols[np.argsort(rows)]
    return cols, overlap[np.arange(len(a)), cols]


def invariant_subspace_distance(a, b):
    """Largest principal-angle sine; QR is diagnostic only, never a field update."""
    a, b = normalized_modes(a), normalized_modes(b)
    if a.shape != b.shape:
        raise ValueError('subspaces must have the same dimensions')
    qa = np.linalg.qr(a.reshape(len(a), -1).T)[0]
    qb = np.linalg.qr(b.reshape(len(b), -1).T)[0]
    # Residual form avoids loss of precision in sqrt(1 - sigma_min**2).
    residual = qb - qa @ (qa.conj().T @ qb)
    return float(np.linalg.svd(residual, compute_uv=False)[0])


def mixture_distance(fields_a, powers_a, fields_b, powers_b):
    """Relative Hilbert-Schmidt distance of the incoherent spatial coherency.

    J = sum(P_i |E_i><E_i|). Uses only small Gram matrices, not a pixels-squared
    matrix. Both polarization and spatial coherence are retained. Unequal modal
    powers cannot be hidden by rotating a two-dimensional subspace.
    """
    a, b = normalized_modes(fields_a), normalized_modes(fields_b)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError('mixtures must use the same optical grid')
    p, q = np.asarray(powers_a, float), np.asarray(powers_b, float)
    if p.shape != (len(a),) or q.shape != (len(b),):
        raise ValueError('one power per mode required')
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)) or np.any(p < 0) or np.any(q < 0):
        raise ValueError('modal powers must be finite and nonnegative')
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    aa = float(np.sum(p[:,None]*p[None,:]*abs(a.conj()@a.T)**2))
    bb = float(np.sum(q[:,None]*q[None,:]*abs(b.conj()@b.T)**2))
    ab = float(np.sum(p[:,None]*q[None,:]*abs(a.conj()@b.T)**2))
    return float(np.sqrt(max(0., aa+bb-2*ab)/max(aa, bb, 1e-60)))


def polarization_family(a, b, eigenvalue_a, eigenvalue_b, settings):
    """Identify nearly split polarization partners for a TRUNCATION guard only."""
    a, b = normalize_vector(a), normalize_vector(b)
    gap = abs(eigenvalue_a-eigenvalue_b)/max(abs(eigenvalue_a), abs(eigenvalue_b), 1e-30)
    vector_overlap = float(abs(np.vdot(a,b))**2)
    cross = b.reshape(2,-1) @ a.reshape(2,-1).conj().T
    spatial_overlap = float(min(1., np.linalg.svd(cross,compute_uv=False).sum()**2))
    paired = (gap <= settings.pair_eigenvalue_tolerance and
              vector_overlap <= settings.pair_vector_overlap_max and
              spatial_overlap >= settings.pair_spatial_overlap)
    return bool(paired), {'complex_relative_gap':float(gap),
                         'vector_overlap_squared':vector_overlap,
                         'best_constant_polarization_overlap_squared':spatial_overlap}


def _failure(previous, values, calls, status, diagnostics=None):
    n=len(previous)
    return TrackedEigenfields(np.asarray(previous).copy(), np.full(n,np.nan+0j),
          np.full(n,np.inf), np.asarray(values), False, calls, status, diagnostics or {})


def select_tracked_fields(operator, previous, values, bank, *, tolerance=2e-7,
                          tracking=None, operator_calls=0):
    """Pure selection step; independently testable using exact matrix spectra.

    Match identities before ranking. Retain the strongest nm actual eigenfields;
    if that cuts a polarization pair, return insufficient_mode_capacity rather
    than blend them or quietly promote an extra mode. Rank crossings inside the
    retained bank preserve labels and their separate photon-state slots.
    """
    cfg=tracking or TrackingSettings()
    previous,bank=normalized_modes(previous),normalized_modes(bank)
    values=np.asarray(values,complex)
    nm=len(previous)
    if bank.shape[1:]!=previous.shape[1:] or values.shape!=(len(bank),) or len(bank)<nm+1:
        raise ValueError('candidate bank must include at least one omitted mode')
    if np.any(~np.isfinite(values)) or np.any(abs(values)==0) or tolerance<=0:
        raise ValueError('finite nonzero eigenvalues and positive tolerance required')
    # Every candidate must satisfy the full operator, including omitted partners.
    candidate_residual=[]
    for v,lam in zip(bank,values):
        out=operator.propagate(v)[0]
        candidate_residual.append(float(np.linalg.norm(out-lam*v)/max(np.linalg.norm(out),1e-30)))
    if max(candidate_residual)>tolerance:
        return _failure(previous,values,operator_calls,'candidate eigenpair residual exceeds tolerance',
                        {'candidate_residuals':candidate_residual})
    full_match,full_overlap=match_branches(previous,bank)
    rank=np.argsort(-abs(values),kind='stable')
    retained=rank[:nm]
    omitted=rank[nm:]
    split_pairs=[]
    for i in retained:
        for j in omitted:
            paired,diag=polarization_family(bank[i],bank[j],values[i],values[j],cfg)
            if paired:
                split_pairs.append({'retained_index':int(i),'omitted_index':int(j),**diag})
    diag={'full_bank_assignment':full_match.tolist(),'full_bank_overlap_squared':full_overlap.tolist(),
          'retained_candidate_indices':retained.tolist(), 'split_boundary_pairs':split_pairs,
          'candidate_residuals':candidate_residual,
          'maximum_omitted_log_power_growth':float(np.max(2*np.log(abs(values[omitted])))),
          'minimum_retained_log_power_growth':float(np.min(2*np.log(abs(values[retained])))),
          'minimum_complex_boundary_gap':float(min(abs(values[i]-values[j]) for i in retained for j in omitted)),
          'candidate_completeness_verified':False}
    if split_pairs:
        return _failure(previous,values,operator_calls,'insufficient_mode_capacity: retained boundary splits a polarization family',diag)
    indices,overlap=match_branches(previous,bank[retained])
    selected=retained[indices]
    fields=bank[selected].copy()
    chosen=values[selected].copy()
    diag['retained_candidate_indices']=selected.tolist()
    # Rotate only machine-near-identical eigenvalues, never the broader family.
    done=set();exact_groups=[]
    for i in range(nm):
        if i in done:continue
        group=[i]
        for j in range(i+1,nm):
            if j not in done and all(abs(chosen[j]-chosen[k]) <= cfg.exact_eigenvalue_tolerance*max(abs(chosen[j]),abs(chosen[k]),1e-30) for k in group):
                group.append(j)
        if len(group)>1:
            basis=np.linalg.qr(fields[group].reshape(len(group),-1).T)[0]
            refs=previous[group].reshape(len(group),-1).T
            u,_,vh=np.linalg.svd(basis.conj().T@refs,full_matrices=False)
            fields[group]=(basis@u@vh).T.reshape(len(group),*previous.shape[1:])
            exact_groups.append(group)
        done.update(group)
    residuals=[];rayleigh=[];matched=[]
    for i,v in enumerate(fields):
        phase=np.vdot(previous[i],v)
        if abs(phase)>0:fields[i]*=np.exp(-1j*np.angle(phase))
        out=operator.propagate(fields[i])[0]
        lam=np.vdot(fields[i],out)
        rayleigh.append(lam)
        residuals.append(float(np.linalg.norm(out-lam*fields[i])/max(np.linalg.norm(out),1e-30)))
        matched.append(float(abs(np.vdot(previous[i],fields[i]))**2))
    reset=[i for i in range(nm) if matched[i]<cfg.branch_reset_overlap or
           (int(full_match[i])!=int(selected[i]) and full_overlap[i]>=cfg.branch_reset_overlap)]
    diag.update(matched_overlap_squared=matched, photon_seed_reset_branches=reset,
                exact_degenerate_groups=exact_groups,
                maximum_complex_eigenvalue_separation_rotated=cfg.exact_eigenvalue_tolerance)
    success=max(residuals)<=tolerance
    return TrackedEigenfields(fields,np.asarray(rayleigh),np.asarray(residuals),values,
            success,operator_calls,'tracked eigenbranches converged' if success else 'tracked eigenpair residual failure',diag)


def solve_tracked_eigenfields(operator, previous, *, candidates=6,tolerance=2e-7,
                             maxiter=600,tracking=None,seed=7):
    previous=normalized_modes(previous)
    nm=len(previous);size=int(np.prod(operator.shape))
    if previous.shape[1:]!=operator.shape or not isinstance(candidates,int) or not nm+2<=candidates<size-1:
        raise ValueError('require nmodes+2 <= candidates < vector_size-1')
    if not np.isfinite(tolerance) or tolerance<=0 or not isinstance(maxiter,int) or maxiter<1:
        raise ValueError('invalid eigensolver controls')
    calls=[0];a=operator.linear_operator()
    def mv(x):
        calls[0]+=1
        return a@x
    counted=LinearOperator(a.shape,matvec=mv,dtype=np.complex128)
    rng=np.random.default_rng(seed)
    v0=previous.reshape(nm,-1).sum(axis=0)
    v0+=1e-5*(rng.normal(size=size)+1j*rng.normal(size=size))/np.sqrt(size)
    try:
        values,vectors=eigs(counted,k=candidates,which='LM',v0=v0,
            ncv=min(size-1,max(4*candidates+1,64)),maxiter=maxiter,tol=tolerance*.1)
    except ArpackNoConvergence as exc:
        return _failure(previous,exc.eigenvalues,calls[0],
                        'Arnoldi did not complete the requested candidate bank')
    if len(values)!=candidates:
        return _failure(previous,values,calls[0],'incomplete candidate bank')
    bank=vectors.T.reshape(candidates,*operator.shape)
    return select_tracked_fields(operator,previous,values,bank,tolerance=tolerance,
                                tracking=tracking,operator_calls=calls[0])
