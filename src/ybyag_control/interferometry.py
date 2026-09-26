"""Four-step phase-shifting interferometry for simulated camera measurements.

Each frame is formed from the same propagated optical field and a calibrated
real local oscillator. Shot/read noise and ADC clipping are applied by the
ordinary detector model before reconstructing the complex field. The four
exposures require four pulse batches; the optical field is held fixed during
this short acquisition, while the thermal state advances to its end time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ybyag_dataset.distortions.camera import capture, sample_camera_setup
from ybluag.camera_dataset import C, H, CameraSettings


PHASE_STEPS_RAD = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)


def reference_amplitude(reference_field, x_m, y_m, waist_m):
    """A broad, calibrated Gaussian reference arm in sqrt(J)/m units."""
    field = np.asarray(reference_field, complex)
    x, y = np.meshgrid(x_m, y_m)
    if field.shape != x.shape or waist_m <= 0 or not np.all(np.isfinite(field)):
        raise ValueError("invalid field or interferometer reference geometry")
    peak = float(np.max(abs(field)))
    if peak <= 0:
        raise ValueError("interferometer reference needs an illuminated field")
    reference_radius = 2 * waist_m
    return peak * np.exp(-(x * x + y * y) / (2 * reference_radius**2))


def calibrated_interferometer(
    reference_field,
    local_oscillator,
    x_m,
    y_m,
    wavelength_m,
    base_camera: CameraSettings,
    ranges,
    seed,
    *,
    noisy=True,
):
    """Choose a fixed exposure with 15% peak full-well occupancy."""
    field = np.asarray(reference_field, complex)
    lo = np.asarray(local_oscillator, float)
    if field.shape != lo.shape or field.shape != (len(y_m), len(x_m)):
        raise ValueError("interferometer fields and axes must match")
    if len(x_m) != len(y_m) or len(x_m) < 16:
        raise ValueError("interferometer optical grid must be square")
    camera = replace(
        base_camera,
        width=len(x_m),
        height=len(y_m),
        # Keep the outer detector sample just inside the interpolation grid.
        object_fov_width_mm=len(x_m) * float(x_m[1] - x_m[0]) * 1e3 * (1 - 1e-9),
        optical_throughput=1.0,
        psf_sigma_pixels=0.35,
    )
    setup = sample_camera_setup(camera, ranges, seed, enabled=noisy)
    # Registration is assumed calibrated to the model grid. Pixel response
    # and defects remain fixed and sampled independently of the main cameras.
    setup.update(shift_pixels=(0.0, 0.0), rotation_rad=0.0, scale=1.0)
    peak_e = 0.0
    for index, shift in enumerate(PHASE_STEPS_RAD):
        _, _, info = capture(
            abs(field + lo * np.exp(1j * shift)) ** 2,
            x_m, y_m, wavelength_m, camera, setup, seed + index,
            enabled=False,
        )
        peak_e = max(peak_e, info["expected_electron_peak"])
    if not np.isfinite(peak_e) or peak_e <= 0:
        raise ValueError("interferometer calibration has no signal")
    camera = replace(
        camera,
        optical_throughput=min(1.0, .15 * camera.full_well_e / peak_e),
    )
    return camera, setup


def capture_interferograms(
    field,
    local_oscillator,
    x_m,
    y_m,
    wavelength_m,
    camera,
    setup,
    seed,
    *,
    noisy=True,
):
    """Recover a PSF-limited field in sqrt(J)/m from four CCD exposures.

    The reference-arm phase and detector throughput/flat field are calibrated.
    The phase-step subtraction removes fixed additive backgrounds. Finite PSF,
    read noise, shot noise and ADC quantization remain in the measurement.
    """
    field = np.asarray(field, complex)
    lo = np.asarray(local_oscillator, float)
    if field.shape != lo.shape or field.shape != (len(y_m), len(x_m)):
        raise ValueError("interferometer fields and axes must match")
    frames = []
    saturated = []
    for index, shift in enumerate(PHASE_STEPS_RAD):
        frame, _, info = capture(
            abs(field + lo * np.exp(1j * shift)) ** 2,
            x_m, y_m, wavelength_m, camera, setup, seed + index,
            enabled=noisy,
        )
        frames.append(frame)
        saturated.append(info["saturated_fraction"])
    frames = np.stack(frames)
    # I(0)-I(pi) = 4 G LO Re(E); I(pi/2)-I(3pi/2) = 4 G LO Im(E).
    # Opposite phase steps cancel the signal/reference intensities, black
    # level, dark background and fixed additive pixel offsets.
    real = frames[0].astype(float) - frames[2].astype(float)
    imag = frames[1].astype(float) - frames[3].astype(float)
    visibility = real + 1j * imag

    pixel_m = camera.object_fov_width_mm * 1e-3 / camera.width
    electrons_per_fluence = (
        pixel_m**2 * camera.pulses_per_exposure * camera.optical_throughput
        * camera.qe_at_signal * wavelength_m / (H * C)
    )
    max_adu = 2**camera.adc_bits - 1
    adu_per_fluence = (
        electrons_per_fluence * setup["prnu"]
        * (max_adu - camera.black_level_adu) / camera.full_well_e
    )
    lo_floor = .01 * float(lo.max())
    usable = (
        (lo >= lo_floor)
        & (adu_per_fluence > 0)
        & ~setup["dead"]
        & ~setup["hot"]
        & np.all(frames < max_adu, axis=0)
    )
    denominator = 4 * lo * adu_per_fluence
    recovered = np.zeros(field.shape, complex)
    recovered[usable] = visibility[usable] / denominator[usable]
    return frames, recovered, float(max(saturated))
