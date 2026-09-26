"""Phase comparisons on an illuminated optical pupil, in radians."""

from __future__ import annotations

import numpy as np


def piston_removed_residual(actual_field, reference_field, *, threshold=0.01):
    """Return wrapped phase error against a complex reference field.

    The reference carries the intended structured phase. A global piston is
    removed because intensity diagnostics cannot constrain it. Pixels with
    little light in either field are excluded from the displayed map and RMS.
    """
    actual = np.asarray(actual_field, complex)
    reference = np.asarray(reference_field, complex)
    if actual.shape != reference.shape or actual.ndim != 2:
        raise ValueError("phase fields must be matching two-dimensional arrays")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError("phase fields must be finite")
    if not 0 < threshold < 1:
        raise ValueError("phase pupil threshold must lie between zero and one")

    actual_intensity = abs(actual) ** 2
    reference_intensity = abs(reference) ** 2
    if actual_intensity.max() <= 0 or reference_intensity.max() <= 0:
        raise ValueError("phase fields must contain illuminated pixels")
    illuminated = (
        (actual_intensity >= threshold * actual_intensity.max())
        & (reference_intensity >= threshold * reference_intensity.max())
    )
    if not np.any(illuminated):
        raise ValueError("phase fields have no shared illuminated pupil")

    cross = actual * np.conj(reference)
    piston = float(np.angle(np.sum(cross[illuminated])))
    residual = np.angle(cross * np.exp(-1j * piston))
    weights = np.sqrt(actual_intensity[illuminated] * reference_intensity[illuminated])
    rms = float(np.sqrt(np.average(residual[illuminated] ** 2, weights=weights)))
    return np.where(illuminated, residual, np.nan), rms


def compensation_residual(actual_phase, ideal_phase, illumination, *, threshold=0.001):
    """Wrapped correction error on the illuminated SLM, weighted by seed fluence.

    Global piston has no effect on a phase-only beam-shaping command. The
    intensity weight prevents nearly dark edge pixels from dominating RMS.
    ``ideal_phase`` may contain NaN outside its physically defined pupil.
    """
    actual = np.asarray(actual_phase, float)
    ideal = np.asarray(ideal_phase, float)
    weight = np.asarray(illumination, float)
    if actual.ndim != 2 or ideal.shape != actual.shape or weight.shape != actual.shape:
        raise ValueError("correction maps and illumination must share a 2D grid")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0) or weight.max() <= 0:
        raise ValueError("SLM illumination must be finite and nonnegative")
    if not 0 < threshold < 1:
        raise ValueError("illumination threshold must lie between zero and one")
    valid = (np.isfinite(actual) & np.isfinite(ideal)
             & (weight >= threshold * weight.max()))
    if not np.any(valid):
        raise ValueError("no shared illuminated correction pupil")
    phasor = np.exp(1j * (actual[valid] - ideal[valid]))
    piston = float(np.angle(np.sum(weight[valid] * phasor)))
    residual = np.full(actual.shape, np.nan)
    residual[valid] = np.angle(phasor * np.exp(-1j * piston))
    rms = float(np.sqrt(np.average(residual[valid] ** 2, weights=weight[valid])))
    return residual, rms
