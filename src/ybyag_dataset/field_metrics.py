"""One field comparison for dataset labels and correction teachers.

Complex fidelity is invariant to an arbitrary global optical phase. Phase RMS
is a diagnostic evaluated on the *same* baseline/reference support for every
candidate in a search.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldMetrics:
    coherent_fidelity: float
    shape_overlap: float
    phase_rms_rad: float
    output_norm: float


def phase_support(baseline: np.ndarray, desired: np.ndarray,
                  threshold: float = 0.01) -> np.ndarray:
    baseline_power = np.abs(baseline) ** 2
    desired_power = np.abs(desired) ** 2
    if baseline_power.shape != desired_power.shape or baseline_power.ndim != 2:
        raise ValueError("baseline and desired fields must share a 2-D grid")
    if baseline_power.max() <= 0 or desired_power.max() <= 0:
        raise ValueError("fields must carry nonzero power")
    return ((baseline_power >= threshold * baseline_power.max()) &
            (desired_power >= threshold * desired_power.max()))


def phase_residual(field: np.ndarray, desired: np.ndarray,
                   weights: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, float]:
    field = np.asarray(field, complex)
    desired = np.asarray(desired, complex)
    weights = np.asarray(weights, float)
    support = np.asarray(support, bool)
    if (field.shape != desired.shape or field.shape != weights.shape or
            field.shape != support.shape or not np.any(support)):
        raise ValueError("phase arrays must be aligned with nonempty support")
    if (not np.all(np.isfinite(field)) or not np.all(np.isfinite(desired)) or
            not np.all(np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("phase arrays must be finite with nonnegative weights")
    selected = weights[support]
    if selected.sum() <= 0:
        raise ValueError("phase support has zero weight")
    relative = np.angle(field * np.conj(desired))
    piston = np.angle(np.sum(selected * np.exp(1j * relative[support])))
    residual = np.angle(np.exp(1j * (relative - piston)))
    rms = float(np.sqrt(np.average(residual[support] ** 2, weights=selected)))
    return np.where(support, residual, 0.0), rms


def field_metrics(field: np.ndarray, desired: np.ndarray,
                  weights: np.ndarray, support: np.ndarray) -> FieldMetrics:
    field = np.asarray(field, complex)
    desired = np.asarray(desired, complex)
    if field.shape != desired.shape or field.shape != np.shape(weights):
        raise ValueError("field comparison arrays must be aligned")
    norm = float(np.sum(np.abs(field) ** 2))
    target_norm = float(np.sum(np.abs(desired) ** 2))
    if norm <= 0 or target_norm <= 0:
        raise ValueError("field comparison requires nonzero fields")
    coherent = float(np.abs(np.vdot(desired, field)) ** 2 /
                     (norm * target_norm))
    amplitude = float(np.sum(np.abs(field) * np.abs(desired)) ** 2 /
                      (norm * target_norm))
    _, rms = phase_residual(field, desired, weights, support)
    return FieldMetrics(coherent, amplitude, rms, norm)
