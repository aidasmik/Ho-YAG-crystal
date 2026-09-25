"""Optional image-formation layer for exploratory synthetic Yb camera data.

Physical disk changes must arrive as separately solved result states.  Sensor
temperature drift below changes only the camera's dark current.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


H = 6.62607015e-34
C = 299792458.0


def suggest_optical_throughput(result, settings, *, target_well_fraction=.5):
    """Choose one attenuator setting from the first state's expected peak charge."""
    mapped, _, _ = _optical_map(result, settings)
    wavelength_m = float(result["signal_wavelength_nm"]) * 1e-9
    if not 0 < target_well_fraction < 1 or wavelength_m <= 0:
        raise ValueError("invalid camera exposure target or signal wavelength")
    pixel_area_m2 = (settings.object_fov_width_mm*1e-3/settings.width)**2
    unattenuated_e = (float(mapped.max()) * pixel_area_m2 *
                      settings.pulses_per_exposure * wavelength_m/(H*C) *
                      settings.qe_at_signal)
    if unattenuated_e <= 0:
        raise ValueError("source result has no illuminated camera pixels")
    return min(1.0, target_well_fraction*settings.full_well_e/unattenuated_e)


@dataclass(frozen=True)
class CameraSettings:
    width: int = 1920
    height: int = 1080
    object_fov_width_mm: float = 10.0
    qe_at_signal: float = .05
    optical_throughput: float = 1e-8
    pulses_per_exposure: int = 100
    exposure_s: float = .01
    full_well_e: float = 30000.0
    read_noise_e: float = 3.0
    dark_current_e_s: float = .1
    background_e: float = 0.0
    prnu_rms: float = .01
    dsnu_rms_e: float = .5
    hot_pixel_fraction: float = 1e-5
    dead_pixel_fraction: float = 1e-5
    psf_sigma_pixels: float = 1.0
    adc_bits: int = 12
    black_level_adu: int = 32
    sensor_temperature_C: float = 20.0
    sensor_temperature_jitter_C: float = .1
    sensor_temperature_correlation: float = .9
    dark_current_doubling_C: float = 6.0

    def __post_init__(self):
        if (not isinstance(self.width, int) or not isinstance(self.height, int) or
                not 16 <= self.width <= 1920 or not 16 <= self.height <= 1080 or
                not isinstance(self.pulses_per_exposure, int) or
                not 1 <= self.pulses_per_exposure <= 100000 or
                not isinstance(self.adc_bits, int) or not 8 <= self.adc_bits <= 16):
            raise ValueError("invalid camera dimensions, ADC, or exposure pulse count")
        nonnegative = ("read_noise_e", "dark_current_e_s", "background_e", "prnu_rms",
                       "dsnu_rms_e", "hot_pixel_fraction", "dead_pixel_fraction",
                       "psf_sigma_pixels", "sensor_temperature_jitter_C")
        positive = ("object_fov_width_mm", "optical_throughput", "exposure_s",
                    "full_well_e", "dark_current_doubling_C")
        for key in nonnegative + positive + ("qe_at_signal", "sensor_temperature_C",
                                              "sensor_temperature_correlation"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"nonfinite camera setting: {key}")
        if any(getattr(self, key) < 0 for key in nonnegative) or any(
                getattr(self, key) <= 0 for key in positive):
            raise ValueError("camera noise and exposure settings must be nonnegative")
        if (not 0 < self.qe_at_signal <= 1 or
                self.optical_throughput > 1 or
                self.hot_pixel_fraction + self.dead_pixel_fraction >= 1 or
                not 0 <= self.sensor_temperature_correlation < 1 or
                not 0 <= self.black_level_adu < 2**self.adc_bits):
            raise ValueError("camera fraction, correlation, or black level is invalid")


def _optical_map(result, settings):
    """Map pulse fluence to object-plane camera pixels, preserving SI units."""
    fluence = np.asarray(result.get("output_fluence_J_m2"), dtype=float)
    x = np.asarray(result.get("x_mm"), dtype=float)
    y = np.asarray(result.get("y_mm"), dtype=float)
    if (fluence.ndim != 2 or fluence.shape != (len(y), len(x)) or
            len(x) < 2 or len(y) < 2 or
            not np.all(np.isfinite(fluence)) or np.any(fluence < 0) or
            not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)):
        raise ValueError("finite nonnegative optical fluence and increasing x/y axes required")
    fov_y = settings.object_fov_width_mm * settings.height/settings.width
    xp = (np.arange(settings.width) + .5 - settings.width/2) * (
        settings.object_fov_width_mm/settings.width)
    yp = (np.arange(settings.height) + .5 - settings.height/2) * (
        fov_y/settings.height)
    if xp[0] < x[0] or xp[-1] > x[-1] or yp[0] < y[0] or yp[-1] > y[-1]:
        raise ValueError("camera field of view exceeds simulated optical field")
    xi = np.interp(xp, x, np.arange(len(x)))
    yi = np.interp(yp, y, np.arange(len(y)))
    yy, xx = np.meshgrid(yi, xi, indexing="ij")
    mapped = map_coordinates(fluence, [yy, xx], order=1, mode="nearest")
    if settings.psf_sigma_pixels:
        mapped = gaussian_filter(mapped, settings.psf_sigma_pixels, mode="constant")
    return np.maximum(mapped, 0).astype(np.float32), xp, yp


def capture_frames(states, settings=CameraSettings(), *, seed=0, frames_per_state=1):
    """Return camera observables and distinct low-resolution physical truth.

    Each state contains a saved solver result and its exact request. Separate
    states may encode physical pump/cooling changes only if solved independently.
    Repeated camera frames of one state have independent temporal noise and
    shared fixed pixel defects; they are not simulated optical transients.
    """
    if (not isinstance(seed, int) or seed < 0 or not isinstance(frames_per_state, int)
            or not 1 <= frames_per_state <= 8 or not 1 <= len(states) <= 8 or
            len(states)*frames_per_state > 8):
        raise ValueError("use 1–8 physical states, 1–8 frames each, at most 8 frames")
    rng = np.random.default_rng(seed)
    shape = (settings.height, settings.width)
    prnu = rng.normal(1, settings.prnu_rms, shape).clip(0).astype(np.float32)
    dsnu = rng.normal(0, settings.dsnu_rms_e, shape).astype(np.float32)
    defect_choice = rng.random(shape)
    dead = defect_choice < settings.dead_pixel_fraction
    hot = (defect_choice >= settings.dead_pixel_fraction) & (
        defect_choice < settings.dead_pixel_fraction + settings.hot_pixel_fraction)
    frames = []
    clean = []
    state_index = []
    sensor_temps = []
    saturated = []
    truth = []
    state_metadata = []
    temperature_delta = 0.0
    native_shape = None
    material = None
    for index, state in enumerate(states):
        result, request = state["result"], state["request"]
        if result.get("material") not in ("Yb:YAG", "Yb:LuAG") or request.get(
                "material") != result.get("material"):
            raise ValueError("matching Yb material in request and result required")
        if material is not None and result["material"] != material:
            raise ValueError("all physical states must use the same material")
        material = result["material"]
        current_shape = np.shape(result.get("output_fluence_J_m2"))
        if native_shape is not None and current_shape != native_shape:
            raise ValueError("all physical states must use the same optical grid")
        native_shape = current_shape
        if result.get("thermal_optical_mode") == "coupled_steady" and (
                result.get("coupled_steady_convergence") or {}).get("status") != "converged":
            raise ValueError("unsettled coupled optical result cannot be an observation")
        repetition_kHz = request.get("repetition_rate_kHz")
        if repetition_kHz is not None and (
                settings.pulses_per_exposure >
                settings.exposure_s * float(repetition_kHz) * 1000 + 1e-9):
            raise ValueError("exposure is too short for the selected pulse count and repetition rate")
        wavelength_m = float(result.get("signal_wavelength_nm", request.get("signal_nm", 1030))) * 1e-9
        if not math.isfinite(wavelength_m) or wavelength_m <= 0:
            raise ValueError("invalid signal wavelength")
        mapped, xp, yp = _optical_map(result, settings)
        pixel_area_m2 = (settings.object_fov_width_mm*1e-3/settings.width)**2
        photon_factor = (pixel_area_m2*settings.optical_throughput *
                         settings.pulses_per_exposure*wavelength_m/(H*C))
        expected_e = mapped.astype(np.float64) * photon_factor * settings.qe_at_signal * prnu
        expected_e[dead] = 0
        truth.append(np.asarray(result["output_fluence_J_m2"], np.float32))
        state_metadata.append({
            "source_run": state.get("source_run"),
            "material": result["material"], "request": request,
            "physical_state_kind": result.get("thermal_optical_mode", "static"),
            "thermal_feedback_applied": bool(result.get("thermal_feedback_applied", False)),
            "hot_phase_validity": result.get("hot_phase_validity"),
            "source_provenance": state.get("source_provenance"),
            "source_result_sha256": state.get("source_result_sha256"),
            "source_request_sha256": state.get("source_request_sha256"),
        })
        for _ in range(frames_per_state):
            temperature_delta = (settings.sensor_temperature_correlation*temperature_delta +
                                 settings.sensor_temperature_jitter_C * math.sqrt(
                                     1-settings.sensor_temperature_correlation**2)*rng.normal())
            sensor_T = settings.sensor_temperature_C + temperature_delta
            dark_mean = settings.dark_current_e_s*settings.exposure_s * (
                2 ** (temperature_delta/settings.dark_current_doubling_C))
            dark_mean = min(dark_mean, 1e7)
            electrons = rng.poisson(np.clip(expected_e + dark_mean + settings.background_e, 0,
                                            settings.full_well_e * 50)).astype(np.float32)
            electrons += dsnu + rng.normal(0, settings.read_noise_e, shape)
            electrons[hot] += settings.full_well_e*.5
            clipped = np.clip(electrons, 0, settings.full_well_e)
            max_adu = 2**settings.adc_bits-1
            adu = np.rint(settings.black_level_adu + clipped/settings.full_well_e *
                          (max_adu-settings.black_level_adu)).clip(0, max_adu).astype(np.uint16)
            frames.append(adu)
            clean.append(mapped)
            state_index.append(index)
            sensor_temps.append(sensor_T)
            saturated.append(float(np.mean(electrons >= settings.full_well_e)))
    return {
        "observable__camera_adu": np.stack(frames),
        "truth__camera_plane_fluence_J_m2": np.stack(clean),
        "truth__native_output_fluence_J_m2": np.stack(truth),
        "observable__state_index": np.asarray(state_index, np.uint8),
        "observable__sensor_temperature_C": np.asarray(sensor_temps, np.float32),
        "camera_x_mm": xp.astype(np.float32), "camera_y_mm": yp.astype(np.float32),
    }, state_metadata, saturated


def export_camera_dataset(path, states, settings=CameraSettings(), *, seed=0,
                          frames_per_state=1):
    """Save bounded camera frames and provenance beside exact optical targets."""
    arrays, state_metadata, saturated = capture_frames(
        states, settings, seed=seed, frames_per_state=frames_per_state)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    npz = destination.with_suffix(".npz")
    meta = destination.with_suffix(".json")
    np.savez_compressed(npz, **arrays)
    source_path = Path(__file__)
    metadata = {
        "schema": "yb_camera_preview_v1", "dataset_ready": False,
        "experimental_calibration": "missing", "purpose": "exploratory_synthetic_data",
        "sensor_type": "generic_monochrome_CCD_parameterization_not_a_camera_model",
        "camera_settings": asdict(settings), "seed": seed,
        "camera_model_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "frames_per_state": frames_per_state,
        "frame_state_indices": arrays["observable__state_index"].tolist(),
        "saturated_pixel_fraction_by_frame": saturated,
        "states": state_metadata,
        "labels": {
            "camera_adu": "observable digital camera output",
            "camera_plane_fluence_J_m2": "exact resampled per-pulse optical fluence before sensor noise",
            "native_output_fluence_J_m2": "native solver output at its original resolution",
            "sensor_temperature_C": "synthetic detector temperature; not crystal temperature",
        },
        "limits": [
            "Camera parameters are illustrative until calibrated from a real camera and optics.",
            "1080p sampling does not increase the solver optical resolution.",
            "Repeated frames of one state share one optical/thermal state; only sensor noise varies.",
            "Physical thermal instability requires separately solved and labeled states.",
            "Yb:YAG hot gain and full training-ground-truth qualification are unavailable.",
        ],
    }
    meta.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return npz, meta


def load_saved_state(directory):
    directory = Path(directory)
    request_path, result_path = directory/"request.json", directory/"result.json"
    request_raw = request_path.read_bytes()
    raw = result_path.read_bytes()
    provenance_path = directory/"provenance.json"
    return {
        "source_run": str(directory.resolve()),
        "request": json.loads(request_raw),
        "result": json.loads(raw),
        "source_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "source_result_sha256": hashlib.sha256(raw).hexdigest(),
        "source_provenance": json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file() else None,
    }
