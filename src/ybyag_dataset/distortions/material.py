"""One fixed, smooth map set per virtual Yb:YAG crystal."""
import numpy as np
from .common import correlated_unit_map, child_seeds
from .measured_doping import DEFAULT_DEVIATION_SCALE, relative_map, sampled_placement


def sample_material(shape, ranges, seed, *, enabled=True, stress=1.0, grid=None,
                    yb_at_percent=None):
    """``ranges['yb_distribution']``: "measured" uses the PL-mapped relative Yb
    distribution of the matching sample (5/10/15 at.%), placed with a seeded
    offset and rotation; "random" (default) uses a seeded correlated map."""
    if not enabled:
        return dict(yb_concentration_scale=np.ones(shape),
                    thickness_scale=np.ones(shape),
                    background_absorption_m1=np.zeros(shape),
                    surface_figure_m=np.zeros(shape))
    s = child_seeds(seed, 4)
    width = max(2, min(shape)/7)
    unit = correlated_unit_map(shape, width, s[0])
    if ranges.get("yb_distribution", "random") == "measured":
        if grid is None or yb_at_percent is None:
            raise ValueError("measured Yb distribution needs the optical grid and concentration")
        if tuple(grid.shape) != tuple(shape):
            raise ValueError("optical grid and material map shapes disagree")
        offset, rotation = sampled_placement(ranges, np.random.default_rng(s[0]), stress=stress)
        yb = relative_map(grid, yb_at_percent, offset_m=offset, rotation_rad=rotation,
            deviation_scale=float(ranges.get("yb_map_deviation_scale", DEFAULT_DEVIATION_SCALE)),
            smooth_mm=float(ranges.get("yb_map_smooth_mm", .1)),
            taper_mm=float(ranges.get("yb_map_taper_mm", .5)),
            edge_trim_mm=float(ranges.get("yb_map_edge_trim_mm", .3)),
            nearest=bool(ranges.get("yb_map_nearest_sample", False)))
    elif ranges.get("yb_distribution", "random") != "random":
        raise ValueError("yb_distribution must be measured or random")
    elif "yb_concentration_max_fraction" in ranges:
        # Hard physical bound: the largest local doping deviation equals the
        # configured fraction. It is a measured-crystal limit, so the stress
        # split does not exceed it.
        peak = float(np.max(np.abs(unit)))
        yb = 1 + ranges["yb_concentration_max_fraction"]*(unit/peak if peak else unit)
    else:
        yb = 1 + stress*ranges["yb_concentration_rms_fraction"]*unit
    thickness = 1 + stress*ranges["thickness_rms_fraction"]*correlated_unit_map(shape, width, s[1])
    background = ranges["background_absorption_mean_m1"]*(
        1+stress*ranges["background_absorption_rms_fraction"]*
        correlated_unit_map(shape, width, s[2]))
    surface = stress*ranges["surface_figure_rms_nm"]*1e-9*correlated_unit_map(shape, width, s[3])
    return dict(yb_concentration_scale=np.maximum(yb, .1),
                thickness_scale=np.maximum(thickness, .1),
                background_absorption_m1=np.maximum(background, 0),
                surface_figure_m=surface)
