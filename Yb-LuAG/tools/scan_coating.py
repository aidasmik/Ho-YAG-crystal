"""Repeat the documented Yb:LuAG output-coupler design screen.

Run with PYTHONPATH=src. This is a bounded CW screen, not a cavity prediction.
"""

import json
from pathlib import Path

import numpy as np

from ybluag import YbLuAGMaterial, scan_output_coupler


def main():
    path = Path(__file__).resolve().parents[2] / "config" / "ybluag_10at_coatings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    oc = config["output_coupler"]
    pump_intensity = oc["screening_assumed_pump_W"] / (
        np.pi * oc["screening_assumed_pump_radius_m"]**2)
    result = scan_output_coupler(
        YbLuAGMaterial(pump_wavelength_nm=config["wavelengths_nm"]["pump"],
                       signal_wavelength_nm=config["wavelengths_nm"]["laser"]),
        pump_intensity, oc["screening_disk_thickness_m"], 8,
        oc["screening_candidates"],
        disk_hr_reflectivity=config["disk_rear"]["target_min_reflectance_laser"],
        other_roundtrip_survival=oc["screening_other_roundtrip_survival"])
    for transmission, intensity in zip(result.transmission,
                                       result.predicted_output_intensity_W_m2):
        print(f"T={transmission:.3%}, screened output intensity={intensity:.6g} W/m2")
    print(f"Selected candidate: {result.selected_transmission:.3%}")


if __name__ == "__main__":
    main()
