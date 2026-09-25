"""Seed and alignment errors sampled before SLM-to-disk propagation."""
import numpy as np


def sample_beam(nominal, ranges, seed, *, enabled=True, stress=1.0, elapsed_s=0.0):
    rng = np.random.default_rng(seed)
    if not enabled:
        return dict(seed_energy_nj=nominal["seed_energy_nj"], waist_mm=nominal["waist_mm"],
                    seed_center_m=(0., 0.), seed_angle_rad=(0., 0.), seed_ellipticity=1.)
    radius_m = nominal["radius_mm"]*1e-3
    shift = stress*rng.uniform(0, ranges["seed_offset_radius_fraction"])*radius_m
    angle = rng.uniform(0, 2*np.pi)
    drift_m = rng.normal(0, ranges["alignment_drift_um_per_s"]*1e-6*elapsed_s, 2)
    center = shift*np.array([np.cos(angle), np.sin(angle)])+drift_m
    return dict(
        seed_energy_nj=nominal["seed_energy_nj"]*(1+stress*rng.uniform(-1, 1)*ranges["seed_energy_fraction"]),
        waist_mm=nominal["waist_mm"]*(1+stress*rng.uniform(-1, 1)*ranges["seed_waist_fraction"]),
        seed_center_m=tuple(center),
        seed_angle_rad=tuple(rng.normal(0, stress*ranges["seed_angle_urad"]*1e-6, 2)),
        seed_ellipticity=1+stress*rng.uniform(-1, 1)*ranges["seed_ellipticity_fraction"])
