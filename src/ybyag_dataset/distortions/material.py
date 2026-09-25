"""One fixed, smooth map set per virtual Yb:YAG crystal."""
import numpy as np
from .common import correlated_unit_map, child_seeds


def sample_material(shape, ranges, seed, *, enabled=True, stress=1.0):
    if not enabled:
        return dict(yb_concentration_scale=np.ones(shape),
                    thickness_scale=np.ones(shape),
                    background_absorption_m1=np.zeros(shape),
                    surface_figure_m=np.zeros(shape))
    s = child_seeds(seed, 4)
    width = max(2, min(shape)/7)
    yb = 1 + stress*ranges["yb_concentration_rms_fraction"]*correlated_unit_map(shape, width, s[0])
    thickness = 1 + stress*ranges["thickness_rms_fraction"]*correlated_unit_map(shape, width, s[1])
    background = ranges["background_absorption_mean_m1"]*(
        1+stress*ranges["background_absorption_rms_fraction"]*
        correlated_unit_map(shape, width, s[2]))
    surface = stress*ranges["surface_figure_rms_nm"]*1e-9*correlated_unit_map(shape, width, s[3])
    return dict(yb_concentration_scale=np.maximum(yb, .1),
                thickness_scale=np.maximum(thickness, .1),
                background_absorption_m1=np.maximum(background, 0),
                surface_figure_m=surface)
