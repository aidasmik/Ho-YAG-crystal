from dataclasses import replace

import numpy as np
import pytest

from ybluag.camera_preview import PreviewSettings, preview_frame


def _saved_optical_state():
    axis = (np.arange(64)-31.5)*12/64
    x, y = np.meshgrid(axis, axis)
    fluence = .01*np.exp(-2*(x*x+y*y)/.6**2)
    return {
        "x_mm": axis, "y_mm": axis,
        "output_fluence_J_m2": fluence,
        "output_phase": np.zeros_like(fluence),
        "signal_wavelength_nm": 1030,
        "output_energy_J": float(fluence.sum()*(.012/64)**2),
        "average_output_W": float(fluence.sum()*(.012/64)**2)*10000,
    }


def test_preview_uses_saved_energy_and_deterministic_camera_noise():
    result = _saved_optical_state()
    settings = PreviewSettings(optical_throughput=.005, seed=42)
    first, clean, info = preview_frame(result, settings)
    repeat, repeat_clean, _ = preview_frame(result, settings)
    assert first.shape == (1080, 1920)
    assert first.dtype == np.uint16
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(clean, repeat_clean)
    assert info["output_energy_J"] == pytest.approx(result["output_energy_J"])
    noisy, _, _ = preview_frame(result, replace(settings, read_noise_e=15))
    assert np.any(noisy != first)


def test_aberration_changes_intensity_after_propagation_only():
    result = _saved_optical_state()
    settings = PreviewSettings(optical_throughput=.005, seed=8)
    _, baseline, _ = preview_frame(result, settings)
    _, same_plane, _ = preview_frame(result, replace(settings, astigmatism_0_waves=.25))
    np.testing.assert_allclose(same_plane, baseline, rtol=1e-5, atol=1e-7)
    _, propagated, info = preview_frame(
        result, replace(settings, astigmatism_0_waves=.25, focus_offset_mm=100))
    assert np.mean(np.abs(propagated-baseline)) > 1e-7
    assert info["output_energy_J"] == pytest.approx(result["output_energy_J"])
