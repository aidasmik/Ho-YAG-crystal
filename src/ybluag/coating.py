"""Screen Yb:LuAG output-coupler transmission with the Yb CW rate model."""

from dataclasses import dataclass

import numpy as np

from .model import YbLuAGMaterial, propagate_cw


@dataclass(frozen=True)
class OutputCouplerScan:
    transmission: np.ndarray
    predicted_output_intensity_W_m2: np.ndarray
    selected_transmission: float
    selected_output_intensity_W_m2: float
    scope: str = ("CW equal counterpropagating signal intensity approximation; "
                  "fixed effective pump intensity; no pump recycling or cavity mode")


def scan_output_coupler(material: YbLuAGMaterial, pump_intensity_W_m2: float,
                        thickness_m: float, steps: int, transmissions, *,
                        disk_hr_reflectivity: float = 0.9995,
                        other_roundtrip_survival: float = 0.995) -> OutputCouplerScan:
    """Find the best transmission for one declared pump and loss scenario.

    Two equal counterpropagating signal intensities enter the shared CW rate
    equation as their sum. The resulting single-pass gain is squared for an
    active-mirror round trip. This is a coating-selection screen, not a
    self-consistent multi-pass resonator prediction.
    """
    for name, value in (("pump_intensity_W_m2", pump_intensity_W_m2),
                        ("thickness_m", thickness_m)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("disk_hr_reflectivity", disk_hr_reflectivity),
                        ("other_roundtrip_survival", other_roundtrip_survival)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    choices = np.asarray(transmissions, dtype=float)
    if (choices.ndim != 1 or len(choices) == 0 or
            np.any(~np.isfinite(choices)) or np.any((choices <= 0) | (choices >= 1))):
        raise ValueError("transmissions must be a nonempty 1D array in (0, 1)")

    def gain_ratio(one_direction_intensity):
        # Both directions deplete the same upper-state population.
        combined = max(2 * one_direction_intensity, 1e-15)
        result = propagate_cw(material, thickness_m, steps,
                              pump_intensity_W_m2, combined)
        return float(result.signal_out_W_m2 / combined)

    small_gain = gain_ratio(0.0)
    output = np.zeros_like(choices)
    passive = disk_hr_reflectivity * other_roundtrip_survival
    for i, transmission in enumerate(choices):
        survival = passive * (1.0 - transmission)
        if survival * small_gain**2 <= 1.0:
            continue
        high = 1e6
        while survival * gain_ratio(high)**2 > 1:
            high *= 2
            if high > 1e15:
                raise RuntimeError("could not bracket saturated Yb cavity intensity")
        low = 0.0
        for _ in range(45):
            middle = (low + high) / 2
            if survival * gain_ratio(middle)**2 > 1:
                low = middle
            else:
                high = middle
        intracavity = (low + high) / 2
        output[i] = intracavity * transmission / (1.0 - transmission)
    best = int(np.argmax(output))
    return OutputCouplerScan(choices.copy(), output,
                             float(choices[best]), float(output[best]))
