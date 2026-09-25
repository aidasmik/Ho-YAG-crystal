"""Interactive camera observation of an already solved Yb output field.

Only downstream optics and detector settings change here. Crystal, pump,
gain and thermal states remain those of the saved physical result.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybyag_dataset.distortions.camera import capture
from ybluag.camera_dataset import CameraSettings


@dataclass(frozen=True)
class PreviewSettings:
    tip_x_waves: float = 0.0
    tip_y_waves: float = 0.0
    astigmatism_0_waves: float = 0.0
    astigmatism_45_waves: float = 0.0
    coma_x_waves: float = 0.0
    spherical_waves: float = 0.0
    defocus_waves: float = 0.0
    residual_rms_waves: float = 0.0
    focus_offset_mm: float = 0.0
    fov_width_mm: float = 6.0
    optical_throughput: float = 1e-8
    pulses_per_exposure: int = 100
    qe: float = 0.05
    read_noise_e: float = 3.0
    prnu_rms: float = 0.01
    background_e: float = 2.0
    dark_current_e_s: float = 0.1
    dsnu_rms_e: float = 0.5
    psf_sigma_pixels: float = 1.0
    hot_pixel_fraction: float = 1e-5
    dead_pixel_fraction: float = 1e-5
    camera_shift_pixels: float = 0.0
    camera_rotation_deg: float = 0.0
    seed: int = 0


def preview_frame(result: dict, settings: PreviewSettings):
    """Return clean fluence and 1080p ADU without rerunning the amplifier.

    Aberration coefficients are phase waves at the signal wavelength. The
    residual screen and fixed pixel pattern share a seed; changing the seed
    gives another virtual camera session and noise realization.
    """
    fluence = np.asarray(result["output_fluence_J_m2"], float)
    phase = np.asarray(result["output_phase"], float)
    x = np.asarray(result["x_mm"], float)*1e-3
    y = np.asarray(result["y_mm"], float)*1e-3
    if (fluence.ndim != 2 or phase.shape != fluence.shape or
            fluence.shape != (len(y), len(x)) or min(fluence.shape) < 16 or
            not np.all(np.isfinite(fluence)) or np.any(fluence < 0) or
            not np.all(np.isfinite(phase))):
        raise ValueError("saved output fluence and phase must be finite matching maps")
    if not isinstance(settings.seed, int) or settings.seed < 0:
        raise ValueError("preview seed must be a nonnegative integer")
    if not isinstance(settings.pulses_per_exposure, int):
        raise ValueError("exposure pulse count must be an integer")
    for key in ("tip_x_waves", "tip_y_waves", "astigmatism_0_waves", "astigmatism_45_waves", "coma_x_waves",
                "spherical_waves", "defocus_waves", "residual_rms_waves",
                "focus_offset_mm", "camera_shift_pixels", "camera_rotation_deg"):
        value = getattr(settings, key)
        if not math.isfinite(value) or abs(value) > (100 if key == "focus_offset_mm" else 5):
            raise ValueError(f"{key} is outside the camera-preview range")
    if settings.residual_rms_waves < 0:
        raise ValueError("residual RMS must be nonnegative")
    if not np.allclose(np.diff(x), x[1]-x[0]) or not np.allclose(np.diff(y), y[1]-y[0]):
        raise ValueError("camera preview requires a uniform optical grid")
    grid = Grid2D(len(x), len(y), float(x[1]-x[0]), float(y[1]-y[0]))
    xx, yy = np.meshgrid(x, y)
    # Radius containing 99% of calculated light gives an effective downstream
    # optical pupil. It is bounded so narrow beams do not create pixel artifacts.
    rr = np.hypot(xx, yy)
    order = np.argsort(rr.ravel())
    integrated = np.cumsum(fluence.ravel()[order])
    if integrated[-1] <= 0:
        raise ValueError("saved output has no illuminated pixels")
    pupil_radius_m = max(3*grid.dx, float(rr.ravel()[order]
                         [np.searchsorted(integrated, .99*integrated[-1])]))
    u, v = xx/pupil_radius_m, yy/pupil_radius_m
    r2 = u*u+v*v
    phase_waves = (settings.tip_x_waves*u + settings.tip_y_waves*v +
                   settings.astigmatism_0_waves*(u*u-v*v) +
                   settings.astigmatism_45_waves*(2*u*v) +
                   settings.coma_x_waves*(3*r2-2)*u +
                   settings.spherical_waves*(6*r2*r2-6*r2+1) +
                   settings.defocus_waves*(2*r2-1))
    if settings.residual_rms_waves:
        rng = np.random.default_rng(settings.seed)
        screen = gaussian_filter(rng.normal(size=fluence.shape), max(2, min(fluence.shape)/16))
        pupil = (rr <= pupil_radius_m) & (fluence > .01*fluence.max())
        screen -= screen[pupil].mean()
        screen_rms = float(np.sqrt(np.mean(screen[pupil]**2)))
        phase_waves += settings.residual_rms_waves*screen/max(screen_rms, 1e-12)
    field = np.sqrt(fluence)*np.exp(1j*(phase+2*np.pi*phase_waves))
    wavelength_m = float(result["signal_wavelength_nm"])*1e-9
    if settings.focus_offset_mm:
        field = angular_spectrum_propagate(field, grid, wavelength_m,
                                           settings.focus_offset_mm*1e-3)
    repetition_rate_hz = (float(result["average_output_W"])/
                          float(result["output_energy_J"]))
    camera = CameraSettings(
        width=1920, height=1080, object_fov_width_mm=settings.fov_width_mm,
        qe_at_signal=settings.qe, optical_throughput=settings.optical_throughput,
        pulses_per_exposure=settings.pulses_per_exposure,
        exposure_s=settings.pulses_per_exposure/repetition_rate_hz,
        read_noise_e=settings.read_noise_e, prnu_rms=settings.prnu_rms,
        background_e=settings.background_e, dark_current_e_s=settings.dark_current_e_s,
        dsnu_rms_e=settings.dsnu_rms_e,
        psf_sigma_pixels=settings.psf_sigma_pixels,
        hot_pixel_fraction=settings.hot_pixel_fraction,
        dead_pixel_fraction=settings.dead_pixel_fraction)
    rng = np.random.default_rng(settings.seed)
    shape = (camera.height, camera.width)
    defect = rng.random(shape)
    setup = {
        "prnu": np.maximum(0, 1+rng.normal(0, camera.prnu_rms, shape)).astype(np.float32),
        "dsnu_e": rng.normal(0, camera.dsnu_rms_e, shape).astype(np.float32),
        "hot": defect < camera.hot_pixel_fraction,
        "dead": (defect >= camera.hot_pixel_fraction) &
                (defect < camera.hot_pixel_fraction+camera.dead_pixel_fraction),
        "shift_pixels": (0., settings.camera_shift_pixels),
        "rotation_rad": math.radians(settings.camera_rotation_deg),
        "scale": 1.0,
    }
    adu, clean, diagnostics = capture(
        abs(field)**2, x, y, wavelength_m, camera, setup, settings.seed+1)
    diagnostics.update(pupil_radius_mm=pupil_radius_m*1e3,
                       phase_screen_rms_waves=settings.residual_rms_waves,
                       output_energy_J=float(np.sum(abs(field)**2)*grid.dx*grid.dy))
    return adu, clean, diagnostics
