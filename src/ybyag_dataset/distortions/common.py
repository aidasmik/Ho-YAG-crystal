"""Correlated random fields with explicit RMS and deterministic seeds."""
import numpy as np
from scipy.ndimage import gaussian_filter


def correlated_unit_map(shape, correlation_pixels, seed):
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=shape), correlation_pixels, mode="reflect")
    field -= field.mean()
    rms = np.sqrt(np.mean(field**2))
    return field/rms if rms else np.zeros(shape)


def child_seeds(seed, count):
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(count)]
