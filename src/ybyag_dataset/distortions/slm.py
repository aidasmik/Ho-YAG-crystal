"""Requested structured phase, finite-resolution SLM response and delay."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .common import correlated_unit_map, child_seeds


def sample_slm_setup(shape, ranges, seed, *, enabled=True, stress=1.0):
    s = child_seeds(seed, 3)
    rng = np.random.default_rng(s[0])
    if not enabled:
        return dict(global_gain=1., spatial_gain=np.ones(shape), pixel_gain=np.ones(shape),
                    phase_offset=np.zeros(shape), bits=0, crosstalk_sigma_pixels=0.)
    return dict(global_gain=1+stress*rng.normal(0, ranges["slm_global_gain_fraction"]),
                spatial_gain=np.maximum(.1, 1+stress*ranges["slm_spatial_gain_rms_fraction"]*
                    correlated_unit_map(shape, max(2,min(shape)/8), s[1])),
                pixel_gain=np.maximum(.1, 1+stress*ranges["slm_pixel_gain_rms_fraction"]*
                    np.random.default_rng(s[2]).normal(size=shape)),
                phase_offset=np.zeros(shape),
                bits=int(ranges["slm_bits"]),
                crosstalk_sigma_pixels=float(ranges["slm_crosstalk_sigma_pixels"]))


def apply_slm(requested_phase_rad, setup, *, drift_fraction=0., previous_command=None,
              delayed=False):
    requested = np.mod(np.asarray(requested_phase_rad, float), 2*np.pi)
    command = (np.mod(previous_command, 2*np.pi) if delayed and previous_command is not None
               else requested)
    bits = setup["bits"]
    if bits:
        levels = 2**bits
        address = np.rint(command/(2*np.pi)*(levels-1)).astype(np.int32)
        command = address/(levels-1)*(2*np.pi)
        if "phase_lut_rad" in setup:
            lut = np.asarray(setup["phase_lut_rad"],float)
            if lut.shape != (levels,) or not np.all(np.isfinite(lut)):
                raise ValueError("phase_lut_rad must contain one finite phase per SLM code")
            drive_phase = lut[address]
        else:
            drive_phase = command
    else:
        if "phase_lut_rad" in setup:
            raise ValueError("phase_lut_rad requires finite-bit SLM addressing")
        drive_phase = command
    actual = ((setup["global_gain"]+drift_fraction)*setup["spatial_gain"]*
              setup["pixel_gain"]*drive_phase+setup["phase_offset"])
    # Smooth the physical complex response after per-pixel calibration. This
    # respects the 0/2pi wrap and weak inter-pixel optical crosstalk.
    sigma = setup["crosstalk_sigma_pixels"]
    if sigma:
        phasor = gaussian_filter(np.exp(1j*actual).real, sigma)+1j*gaussian_filter(
            np.exp(1j*actual).imag, sigma)
        actual = np.angle(phasor)
    return np.mod(actual, 2*np.pi), requested
