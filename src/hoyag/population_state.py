"""Canonical population contract: (manifold, z, y, x), I5/I6/I7/I8.

Legacy arrays must be converted explicitly; dimensions of length four are not
used to guess an axis. Only roundoff-size population errors may be normalized.
"""
from dataclasses import dataclass
import numpy as np

POPULATION_AXES = ('manifold', 'z', 'y', 'x')


def validate_populations(values, density_m3, *, rtol=1e-8, error_type=ValueError):
    """Copy a manifold-first state and check local positivity and conservation.

    The density may be a scalar (one voxel) or any spatial array. There must be
    exactly four manifolds followed by the density's shape. Undoped voxels must
    contain exactly zero ions; there is no division by a global reference density.
    """
    n = np.asarray(values, dtype=float)
    density = np.asarray(density_m3, dtype=float)
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive')
    if n.shape != (4, *density.shape):
        raise error_type('population shape must be (4, *density.shape); '
                         'convert legacy (z,manifold,y,x) arrays explicitly')
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(density)) or np.any(density < 0):
        raise error_type('populations and density must be finite; density nonnegative')
    tol = rtol * density
    if np.any(n < -tol[None]):
        raise error_type('negative population: invalid state or insufficient resolution')
    if np.any(np.abs(n.sum(axis=0) - density) > tol):
        raise error_type('local population sum does not equal Ho density; '
                         'check axis order rather than renormalizing the state')
    n = np.maximum(n, 0.0)
    total = n.sum(axis=0)
    scale = np.divide(density, total, out=np.zeros_like(density), where=total > 0)
    return n * scale[None]


def convert_population_layout(values, *, source_layout):
    """Explicit conversion of archived arrays; never infer a layout from shape."""
    n = np.asarray(values, dtype=float)
    if n.ndim != 4:
        raise ValueError('archived spatial population array must be four-dimensional')
    if source_layout == 'manifold,z,y,x':
        if n.shape[0] != 4:
            raise ValueError('manifold axis must have length four')
        return n.copy()
    if source_layout == 'z,manifold,y,x':
        if n.shape[1] != 4:
            raise ValueError('manifold axis must have length four')
        return np.moveaxis(n, 1, 0).copy()
    raise ValueError('source_layout must be manifold,z,y,x or z,manifold,y,x')


@dataclass(frozen=True)
class PopulationField:
    """Validated tagged field for serialization and inter-stage handoffs."""
    values_m3: np.ndarray
    density_m3: np.ndarray
    axes = POPULATION_AXES

    def __post_init__(self):
        density = np.array(self.density_m3, dtype=float, copy=True)
        if density.ndim != 3:
            raise ValueError('PopulationField density order must be (z,y,x)')
        values = validate_populations(self.values_m3, density)
        values.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, 'values_m3', values)
        object.__setattr__(self, 'density_m3', density)

    @classmethod
    def from_legacy_z_major(cls, values, density_m3):
        return cls(convert_population_layout(values, source_layout='z,manifold,y,x'), density_m3)
