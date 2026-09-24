"""Same-plane, registered complex-field comparison utilities."""

from __future__ import annotations

import numpy as np


def coherent_overlap(reference, field) -> float:
    """Global-phase-invariant normalized overlap, with matching sample grids."""
    a = np.asarray(reference, dtype=complex)
    b = np.asarray(field, dtype=complex)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("fields must share a transverse plane and grid")
    denom = float(np.vdot(a, a).real * np.vdot(b, b).real)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("nonzero finite field energies required")
    return float(abs(np.vdot(a, b))**2 / denom)


def intensity_overlap(reference, field) -> float:
    """Bhattacharyya intensity overlap, independent of global power scale."""
    a = np.abs(np.asarray(reference, complex))**2
    b = np.abs(np.asarray(field, complex))**2
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("fields must share a transverse plane and grid")
    denom = float(a.sum()*b.sum())
    if denom <= 0:
        raise ValueError("nonzero field energies required")
    return float(np.sum(np.sqrt(a*b))**2/denom)
