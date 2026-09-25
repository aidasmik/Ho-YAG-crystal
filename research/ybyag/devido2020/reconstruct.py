"""Reconstruct the published ZPL cross section without silently clipping data.

The original eight-column files remain untouched. This produces a derived table
for inspection, not a 5-20 at.% or >300 K amplifier calibration.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parent
TEMPERATURES_K = (80, 100, 125, 150, 175, 200, 225, 250, 275, 300)
N_ION_CM3 = 1.52e20
L_CM = .108
INCIDENT_MONITOR_FACTOR = 22.8
WAVELENGTH_OFFSET_NM = -.176


def reconstruct(output: Path = ROOT / "devido_zpl_reconstructed.csv") -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"source_doi": "10.5286/edata/737",
               "paper_doi": "10.1364/OME.386436",
               "sample_yb_at_percent": 1.1,
               "sample_type": "ceramic",
               "temperature_range_K": [80, 300],
               "wavelength_offset_nm": WAVELENGTH_OFFSET_NM,
               "incident_monitor_factor": INCIDENT_MONITOR_FACTOR,
               "ion_density_cm3": N_ION_CM3,
               "path_length_cm": L_CM,
               "power_handling": "Nonpositive powers and transmission >1 are retained as invalid; no clipping or inferred baseline correction.",
               "80K_censoring": "Transmission <=0.001 is marked as a lower bound; source spectral impurity can dominate near the peak.",
               "files": {}}
    names = ("source_file", "row", "nominal_temperature_K", "measured_temperature_K",
             "wavelength_nm", "incident_power_calibrated_W", "transmitted_power_W",
             "transmission", "sigma_abs_raw_cm2", "sigma_abs_usable_cm2",
             "quality_flag")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for temp in TEMPERATURES_K:
            source = ROOT / f"{temp}K_raw_data.txt"
            raw = np.loadtxt(source)
            if raw.ndim != 2 or raw.shape[1] != 8:
                raise ValueError(f"Unexpected format: {source}")
            counts = {"valid": 0, "invalid_power": 0, "invalid_transmission": 0,
                      "censored_lower_bound": 0, "isolated_high_spike": 0}
            rows = []
            for index, row in enumerate(raw, start=1):
                wavelength = row[0] + WAVELENGTH_OFFSET_NM
                incident = row[2] * INCIDENT_MONITOR_FACTOR
                transmitted = row[3]
                ratio = transmitted / incident if incident > 0 else float("nan")
                if incident <= 0 or transmitted <= 0:
                    flag = "invalid_power"
                elif ratio > 1:
                    flag = "invalid_transmission"
                elif temp == 80 and ratio <= .001:
                    flag = "censored_lower_bound"
                else:
                    flag = "valid"
                sigma = (np.log(1 / ratio) / (N_ION_CM3 * L_CM)
                         if flag in ("valid", "censored_lower_bound") else float("nan"))
                rows.append([source.name, index, temp, row[4], wavelength,
                             incident, transmitted, ratio, sigma, flag])
            # A few raw power glitches imply impossible isolated peaks >50x the
            # published room-temperature value. Flag them, preserving the raw
            # value in its own column. The median spans 11 acquisition points.
            order = np.argsort([row[4] for row in rows])
            sorted_sigma = np.array([rows[i][8] for i in order])
            finite = np.isfinite(sorted_sigma)
            if finite.sum() < 11:
                raise ValueError(f"Too few valid power ratios in {source}")
            filled = np.interp(np.arange(len(rows)), np.flatnonzero(finite),
                               sorted_sigma[finite])
            local = median_filter(filled, size=11, mode="nearest")
            for place, original_index in enumerate(order):
                row = rows[original_index]
                if (row[9] == "valid" and
                        row[8] > max(1.75 * local[place], local[place] + 2e-21)):
                    row[9] = "isolated_high_spike"
            valid_rows = []
            for row in rows:
                counts[row[9]] += 1
                usable = row[8] if row[9] == "valid" else float("nan")
                writer.writerow((*row[:9], usable, row[9]))
                if row[9] == "valid" and 967.8 <= row[4] <= 970.0:
                    valid_rows.append((row[4], row[8]))
            peak_wavelength, peak_sigma = max(valid_rows, key=lambda item: item[1])
            summary["files"][source.name] = {
                "raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "rows": int(raw.shape[0]), "quality_counts": counts,
                "valid_peak_wavelength_nm": peak_wavelength,
                "valid_peak_sigma_abs_cm2": peak_sigma,
                "largest_censored_lower_bound_cm2": (
                    max((row[8] for row in rows
                         if row[9] == "censored_lower_bound"), default=None))}
    summary["derived_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / "reconstruction.json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = reconstruct()
    for name, record in result["files"].items():
        print(name, record["valid_peak_wavelength_nm"],
              record["valid_peak_sigma_abs_cm2"], record["quality_counts"])
