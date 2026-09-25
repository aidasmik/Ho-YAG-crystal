"""Nonthermal upstream optics: low Zernikes plus a weak residual screen."""
import numpy as np
from .common import correlated_unit_map


def sample_optical_phase(x_m, y_m, aperture_radius_m, wavelength_m, ranges,
                         seed, *, enabled=True, stress=1.0):
    shape = np.shape(x_m)
    if not enabled:
        return np.zeros(shape), {}
    rng = np.random.default_rng(seed)
    radius = np.hypot(x_m, y_m)/aperture_radius_m
    angle = np.arctan2(y_m, x_m)
    r = radius
    basis = {
        "tip": r*np.cos(angle), "tilt": r*np.sin(angle),
        "defocus": 2*r*r-1,
        "astigmatism_0": r*r*np.cos(2*angle),
        "astigmatism_45": r*r*np.sin(2*angle),
        "coma_x": (3*r**3-2*r)*np.cos(angle),
        "coma_y": (3*r**3-2*r)*np.sin(angle),
        "spherical": 6*r**4-6*r*r+1,
        "trefoil_x": r**3*np.cos(3*angle),
        "trefoil_y": r**3*np.sin(3*angle),
        "secondary_astigmatism": (4*r**4-3*r*r)*np.cos(2*angle),
    }
    coefficients = {name: float(rng.normal()) for name in basis}
    inside = r <= 1
    zernike = sum(coefficients[key]*basis[key] for key in basis)
    zernike -= zernike[inside].mean()
    rms = np.sqrt(np.mean(zernike[inside]**2))
    target = stress*rng.uniform(0, ranges["external_aberration_rms_waves"])
    residual = correlated_unit_map(shape, max(2, min(shape)/12),
                                   int(rng.integers(0, 2**32-1)))
    phase_waves = ((zernike*target/rms if rms else zernike) +
                   stress*ranges["external_residual_rms_waves"]*residual)
    phase = 2*np.pi*phase_waves
    return phase, dict(zernike_random_coefficients=coefficients,
                       achieved_rms_waves=float(np.sqrt(np.mean(
                           (phase_waves[inside]-phase_waves[inside].mean())**2))),
                       plane="SLM/upstream_optics", wavelength_m=wavelength_m)
