"""Sensor layer checks; these do not qualify the optical model for NN ground truth."""
import json

import numpy as np
import pytest

from ybluag.camera_dataset import (CameraSettings, capture_frames,
                                   export_camera_dataset, suggest_optical_throughput)


def state(amplitude=2.0):
    axis = np.linspace(-6, 6, 32)
    xx, yy = np.meshgrid(axis, axis)
    fluence = amplitude*np.exp(-2*(xx*xx + yy*yy))
    return {"request": {"material": "Yb:YAG", "pump_W": 40},
            "result": {"material": "Yb:YAG", "x_mm": axis.tolist(),
                       "y_mm": axis.tolist(),
                       "output_fluence_J_m2": fluence.tolist(),
                       "signal_wavelength_nm": 1030,
                       "thermal_optical_mode": "lumped_phase",
                       "thermal_feedback_applied": True}}


def settings(**kwargs):
    values = dict(width=64, height=36, object_fov_width_mm=10,
                  optical_throughput=1e-7)
    values.update(kwargs)
    return CameraSettings(**values)


def test_reproducible_and_truth_kept_separate():
    a, states, _ = capture_frames([state()], settings(), seed=9, frames_per_state=2)
    b, _, _ = capture_frames([state()], settings(), seed=9, frames_per_state=2)
    assert np.array_equal(a["observable__camera_adu"], b["observable__camera_adu"])
    assert a["observable__camera_adu"].shape == (2, 36, 64)
    assert a["truth__native_output_fluence_J_m2"].shape == (1, 32, 32)
    assert np.array_equal(a["truth__camera_plane_fluence_J_m2"][0],
                          a["truth__camera_plane_fluence_J_m2"][1])
    assert not np.array_equal(a["observable__camera_adu"][0],
                              a["observable__camera_adu"][1])
    assert states[0]["physical_state_kind"] == "lumped_phase"


def test_distinct_solver_states_and_metadata(tmp_path):
    a, _, _ = capture_frames([state(1), state(2)], settings(), seed=1)
    assert a["observable__state_index"].tolist() == [0, 1]
    assert a["truth__camera_plane_fluence_J_m2"][1].max() > (
        a["truth__camera_plane_fluence_J_m2"][0].max()*1.9)
    npz, meta = export_camera_dataset(tmp_path/"camera", [state()], settings(), seed=1)
    assert npz.is_file()
    detail = json.loads(meta.read_text())
    assert detail["dataset_ready"] is False
    assert detail["npz_sha256"]
    assert detail["camera_model_source_sha256"]


def test_rejects_invalid_fov_and_material():
    with pytest.raises(ValueError, match="field of view"):
        capture_frames([state()], settings(object_fov_width_mm=20))
    other = state()
    other["request"]["material"] = "Yb:LuAG"
    with pytest.raises(ValueError, match="matching Yb material"):
        capture_frames([other], settings())


def test_sensor_saturation_is_reported():
    _, _, clipped = capture_frames([state(1e5)], settings(), seed=3)
    assert clipped[0] > 0


def test_pulse_count_fits_camera_exposure():
    slow = state()
    slow["request"]["repetition_rate_kHz"] = 1
    with pytest.raises(ValueError, match="exposure is too short"):
        capture_frames([slow], settings())


def test_suggested_attenuation_scales_with_signal():
    low = suggest_optical_throughput(state(1)["result"], settings())
    high = suggest_optical_throughput(state(2)["result"], settings())
    assert 0 < high < low <= 1
    assert np.isclose(high, low/2)
