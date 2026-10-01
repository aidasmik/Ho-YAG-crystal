"""Measured relative Yb distributions (PL 969/1030 nm maps) on the optical grid.

The maps in ``src/ybyag_dataset/data/doping_maps`` are within-sample relative ratios
(value 1 = the sample's median) measured on a 0.1 mm grid for 5, 10 and 15 at.%
Yb:YAG. They are a spectroscopic proxy, not calibrated concentration; see
``provenance.json``. The relative Yb deviation used here is
``deviation_scale * (relative_ratio - 1)``.

Sign and scale: 969 nm PL overlaps the strong zero-phonon absorption, so more
Yb reabsorbs more of it and *lowers* the 969/1030 ratio. The measured sample
medians give d ln R / d ln C = -0.59 (5 -> 10 at.%) and -0.26 (10 -> 15 at.%,
whose absolute 15% ratio is not reconciled). The default scale is therefore
1/(-0.59) = -1.7; -3.9 is the less certain upper bound.

The measured region (4-8 mm across) is centred on the optical axis, optionally
offset and rotated per setup. Interior holes are filled from the nearest
measured point, after trimming the unstable boundary ring
(``edge_trim_mm``). Outside the measured region the deviation fades smoothly to
zero over ``taper_mm`` instead of stopping at an artificial edge.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

DATA = Path(__file__).resolve().parents[1] / 'data' / 'doping_maps'
DEFAULT_DEVIATION_SCALE = -1.7
SAMPLES = (5., 10., 15.)


def provenance():
    return json.loads((DATA / 'provenance.json').read_text(encoding='utf-8'))


def sample_for(yb_at_percent, *, nearest=False):
    """Measured sample (at.%) for a concentration; exact match unless ``nearest``."""
    value = float(yb_at_percent)
    best = min(SAMPLES, key=lambda s: abs(s-value))
    if not nearest and abs(best-value) > 1e-9:
        raise ValueError(f'no measured Yb map for {value:g} at.%; measured: 5, 10, 15 at.%')
    return best


@lru_cache(maxsize=16)
def _regular(sample, edge_trim_mm=.3):
    """Measured map on its native regular grid: values, validity, axes (mm).

    The outer ``edge_trim_mm`` of the selected interior is dropped: the
    PL-ratio is unstable at the mask boundary and shows an artificial ring.
    """
    path = DATA / f'YbYag{int(sample)}_relative_distribution.csv'
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    x, y, ratio = data[:, 0], data[:, 1], data[:, 2]
    step = .1
    xs = np.round((x-x.min())/step).astype(int)
    ys = np.round((y-y.min())/step).astype(int)
    grid = np.full((ys.max()+1, xs.max()+1), np.nan)
    grid[ys, xs] = ratio
    valid = np.isfinite(grid)
    if edge_trim_mm > 0:
        core = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]*step > edge_trim_mm
        valid = valid & core
    # Fill holes and the outside from the nearest measured point.
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    filled = grid[iy, ix]
    return filled, valid, x.min(), y.min(), step


def relative_map(grid, yb_at_percent, *, offset_m=(0., 0.), rotation_rad=0.,
                 deviation_scale=DEFAULT_DEVIATION_SCALE, smooth_mm=.1, taper_mm=.5,
                 edge_trim_mm=.3,
                 nearest=False):
    """Relative Yb scale (mean ~1) on a ``Grid2D`` from the measured sample map."""
    sample = sample_for(yb_at_percent, nearest=nearest)
    values, valid, x0, y0, step = _regular(sample, float(edge_trim_mm))
    ny, nx = values.shape
    rows, cols = np.nonzero(valid)
    centre = np.array([cols.mean(), rows.mean()])
    deviation = values-1.
    if smooth_mm > 0:
        deviation = gaussian_filter(deviation, smooth_mm/step, mode='nearest')
    # Distance outside the measured support (mm) for a smooth fade to zero.
    outside = distance_transform_edt(~valid)*step
    fade = np.exp(-(outside/taper_mm)**2) if taper_mm > 0 else valid.astype(float)
    x, y = grid.mesh
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    u = (x-offset_m[0])*1e3
    v = (y-offset_m[1])*1e3
    col = centre[0]+(c*u+s*v)/step
    row = centre[1]+(-s*u+c*v)/step
    inside = (col >= -.5) & (col <= nx-.5) & (row >= -.5) & (row <= ny-.5)
    sampled = map_coordinates(deviation*fade, [row, col], order=1, mode='nearest')
    # Beyond the stored array the fade continues with the distance to it.
    beyond = np.hypot(np.maximum(0, np.maximum(-.5-col, col-(nx-.5))),
                      np.maximum(0, np.maximum(-.5-row, row-(ny-.5))))*step
    edge_fade = np.exp(-(beyond/taper_mm)**2) if taper_mm > 0 else inside.astype(float)
    return 1.+deviation_scale*sampled*np.where(inside, 1., edge_fade)


def sampled_placement(ranges, rng, *, stress=1.):
    """Per-setup placement of the measured map (offset in m, rotation in rad)."""
    offset = stress*float(ranges.get('yb_map_offset_mm', 0.))*1e-3
    shift = tuple(rng.uniform(-offset, offset, 2)) if offset else (0., 0.)
    rotation = float(rng.uniform(0, 2*math.pi)) if ranges.get('yb_map_rotation', False) else 0.
    return shift, rotation
