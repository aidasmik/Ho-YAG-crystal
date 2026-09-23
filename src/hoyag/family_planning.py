"""One-spectrum retained-boundary diagnosis, without changing Stage 7W tracking."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .polarization_tracking import TrackingSettings, normalized_modes, polarization_family


def diagnose_candidate_bank(operator, values, fields, *, counts=(2, 4, 6, 8),
                            previous=None, tolerance=2e-7, settings=None) -> dict:
    """Check full-operator residuals and all retained/omitted family boundaries.

    A finite candidate bank can reject a boundary; it cannot prove global mode
    completeness. The active coupled solver must independently repeat its guard.
    """
    cfg = settings or TrackingSettings()
    values = np.asarray(values, complex)
    fields = normalized_modes(fields)
    if len(values) != len(fields) or fields.shape[1:] != operator.shape:
        raise ValueError('candidate spectrum and operator dimensions disagree')
    if not np.all(np.isfinite(values)) or np.any(abs(values)==0) or min(counts) < 1:
        raise ValueError('invalid spectrum or retained counts')
    rank = np.argsort(-abs(values), kind='stable')
    residuals = []
    for field, value in zip(fields, values):
        applied = operator.propagate(field)[0]
        residuals.append(float(np.linalg.norm(applied-value*field)/max(np.linalg.norm(applied), 1e-30)))
    overlap = None
    if previous is not None:
        old = normalized_modes(previous)
        if old.shape[1:] != operator.shape:
            raise ValueError('previous fields use another grid')
        overlap = abs(old.reshape(len(old), -1).conj() @ fields.reshape(len(fields), -1).T)**2
    boundaries = {}
    admissible = []
    for count in counts:
        if count + 2 > len(values):
            boundaries[str(count)] = {'status': 'candidate_bank_insufficient'}
            continue
        retained, omitted = rank[:count], rank[count:]
        split = []
        for i in retained:
            for j in omitted:
                paired, relation = polarization_family(fields[i], fields[j], values[i], values[j], cfg)
                if paired:
                    split.append({'retained_index': int(i), 'omitted_index': int(j), **relation})
        status = ('candidate_eigenpair_failure' if max(residuals) > tolerance else
                  'split_polarization_family' if split else 'admissible_within_candidate_bank')
        boundaries[str(count)] = {'status': status, 'retained_indices': retained.tolist(),
                                  'omitted_indices': omitted.tolist(), 'split_pairs': split,
                                  'minimum_retained_log_power_growth': float(np.min(2*np.log(abs(values[retained])))),
                                  'maximum_omitted_log_power_growth': float(np.max(2*np.log(abs(values[omitted]))))}
        if status == 'admissible_within_candidate_bank':
            admissible.append(count)
    return {'status': 'candidate_eigenpair_failure' if max(residuals)>tolerance else
            'candidate_bank_insufficient' if any(x['status']=='candidate_bank_insufficient' for x in boundaries.values()) else
            'finite_bank_diagnostic',
            'candidate_count': len(values), 'candidate_eigenvalues': [[float(v.real),float(v.imag)] for v in values],
            'full_operator_residuals': residuals, 'gain_ordering': rank.tolist(),
            'branch_overlaps_squared': None if overlap is None else overlap.tolist(),
            'boundaries': boundaries, 'admissible_retained_counts': admissible,
            'modes_6_complete_family_candidate': 6 in admissible,
            'global_completeness_verified': False}


def solve_candidate_diagnostic(operator, *, candidate_count=10, counts=(2,4,6,8),
                               tolerance=2e-7, maxiter=600, seed=7, previous=None):
    """Use one non-Hermitian Arnoldi spectrum; failure remains an explicit status."""
    size = int(np.prod(operator.shape))
    if not 10 <= candidate_count <= 12 or candidate_count >= size-1:
        raise ValueError('candidate bank must initially contain 10 to 12 modes')
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=size) + 1j*rng.normal(size=size)
    a = operator.linear_operator()
    try:
        values, vectors = eigs(a, k=candidate_count, which='LM', v0=v0,
                               ncv=min(size-1, max(4*candidate_count+1,64)),
                               maxiter=maxiter, tol=tolerance*.1)
    except ArpackNoConvergence as exc:
        return {'status':'candidate_bank_insufficient', 'candidate_count':len(exc.eigenvalues),
                'reason':'Arnoldi did not complete the requested bank',
                'global_completeness_verified':False}
    return diagnose_candidate_bank(operator, values,
        vectors.T.reshape(candidate_count,*operator.shape), counts=counts,
        previous=previous, tolerance=tolerance)
