"""Trace the two endpoint curves of Esmaeilzadeh et al. (2012), Figs. 4, 6.

Run with a locally downloaded copy of https://zenodo.org/records/1334966/files/13958.pdf
as ``python digitize.py PATH_TO_PDF``. Output is figure-derived, not raw data.
The figure temperature legend conflicts with a 380 K reference in its text;
only the unambiguous 300 K and 450 K endpoint curves are traced.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent


def trace(pixels, *, xmin, xmax, ymin, ymax, wavelength_min, wavelength_max,
          ordinate_max, color):
    rgb = np.asarray(pixels.convert("RGB"), dtype=np.int16)
    r, g, b = [rgb[..., i] for i in range(3)]
    if color == "blue":
        mask = (b > r * 1.4) & (b > g * 1.25) & (r < 110)
    elif color == "red":
        mask = (r > g * 1.6) & (r > b * 1.45) & (r > 100)
    elif color == "cyan":
        mask = (g > r * 1.5) & (b > r * 1.4) & (g > 75) & (r < 125)
    else:
        raise ValueError(color)
    axis_wavelengths = np.arange(wavelength_min, wavelength_max + 1, dtype=float)
    samples = []
    detections = []
    for wavelength in axis_wavelengths:
        x = xmin + (wavelength - wavelength_min) * (xmax - xmin) / (
            wavelength_max - wavelength_min)
        lo, hi = max(xmin + 2, round(x) - 2), min(xmax - 2, round(x) + 2)
        ys, _ = np.nonzero(mask[ymin:ymax + 1, lo:hi + 1])
        ys = ys + ymin
        # Legends overlay the graphs; exclude those pixels before readout.
        if wavelength >= 1015 and wavelength_min == 860:
            ys = ys[ys >= 285]
        if wavelength <= 940 and wavelength_min == 920:
            ys = ys[ys >= 275]
        if len(ys):
            y = float(np.median(ys))
            samples.append((ymax - y) / (ymax - ymin) * ordinate_max)
            detections.append(True)
        else:
            samples.append(float("nan"))
            detections.append(False)
    samples = np.asarray(samples)
    good = np.isfinite(samples)
    if good.sum() < len(samples) // 2:
        raise RuntimeError(f"Too few detected {color} pixels: {good.sum()}")
    filled = np.interp(np.arange(len(samples)), np.flatnonzero(good), samples[good])
    return axis_wavelengths, filled, detections


def digitize(pdf: Path, output: Path = HERE / "endpoint_spectra.csv") -> dict:
    with tempfile.TemporaryDirectory() as temp:
        prefix3 = Path(temp) / "page3"
        prefix4 = Path(temp) / "page4"
        for page, prefix in ((3, prefix3), (4, prefix4)):
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page),
                            "-r", "200", "-png", "-singlefile", str(pdf),
                            str(prefix)], check=True, stdout=subprocess.DEVNULL)
        absorption = Image.open(prefix3.with_suffix(".png")).crop(
            (125, 1510, 820, 1990)).resize((1390, 960))
        emission = Image.open(prefix4.with_suffix(".png")).crop(
            (175, 300, 820, 680)).resize((1290, 760))
    # Axis locations are pixel measurements in the two rendered crop images.
    # Units of the ordinates are 1e-21 cm2 (abs) and 1e-20 cm2 (em).
    curves = {}
    for key, im, kw in (
        ("abs_300", absorption, dict(xmin=139, xmax=1301, ymin=7, ymax=667,
             wavelength_min=860, wavelength_max=1040, ordinate_max=8, color="blue")),
        ("abs_450", absorption, dict(xmin=139, xmax=1301, ymin=7, ymax=667,
             wavelength_min=860, wavelength_max=1040, ordinate_max=8, color="red")),
        ("em_300", emission, dict(xmin=64, xmax=1222, ymin=35, ymax=655,
             wavelength_min=920, wavelength_max=1040, ordinate_max=2.2, color="blue")),
        ("em_450", emission, dict(xmin=64, xmax=1222, ymin=35, ymax=655,
             wavelength_min=920, wavelength_max=1040, ordinate_max=2.2, color="cyan")),
    ):
        curves[key] = trace(im, **kw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("wavelength_nm", "sigma_abs_300K_cm2", "sigma_abs_450K_cm2",
                         "sigma_em_300K_cm2", "sigma_em_450K_cm2",
                         "abs_300_detected", "abs_450_detected", "em_300_detected",
                         "em_450_detected"))
        for wavelength in range(920, 1041):
            index_a = wavelength - 860
            index_e = wavelength - 920
            writer.writerow((wavelength,
                             curves["abs_300"][1][index_a] * 1e-21,
                             curves["abs_450"][1][index_a] * 1e-21,
                             curves["em_300"][1][index_e] * 1e-20,
                             curves["em_450"][1][index_e] * 1e-20,
                             *(curves[key][2][index_a if key.startswith("abs") else index_e]
                               for key in ("abs_300", "abs_450", "em_300", "em_450"))))
    summary = {"source_doi": "10.5281/zenodo.1334966",
               "paper": "Esmaeilzadeh, Roohbakhsh, Ghaedzadeh (2012), Figs. 4 and 6",
               "sample_yb_at_percent": 25,
               "sample_thickness_mm": 1,
               "temperature_endpoints_K": [300, 450],
               "method": "Color-pixel tracing at 200 dpi, 1 nm intervals; missing pixels interpolated; Fig. 6 emission is reciprocity-derived by the authors.",
               "calibration": "Fig. 4 axes 860-1040 nm, 0-8e-21 cm2; Fig. 6 axes 920-1040 nm, 0-2.2e-20 cm2.",
               "uncertainty": "Approximate curve readout; at least +/-0.2e-21 cm2 absorption and +/-0.05e-20 cm2 emission, larger where curves overlap or pixels are interpolated.",
               "caveat": "25 at.% measured sample only. This table does not calibrate 5, 10, 15 or 20 at.% samples or 523 K operation.",
               "detected_fraction": {key: float(np.mean(value[2]))
                                     for key, value in curves.items()}}
    (HERE / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(digitize(Path(sys.argv[1])), indent=2))
