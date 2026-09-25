"""Trace room-temperature 5/10/15 at.% absorption coefficients, Tang Fig. 5.

Usage: python digitize.py PATH_TO_TANG_PDF
Source: https://files.secure.website/wscfus/7885803/32406066/tang-yb-yag.pdf
These are approximate image readouts from individual ceramics, not emission
cross sections or a high-temperature gain calibration.
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


def digitize(pdf: Path, output: Path = HERE / "rt_absorption_coefficients.csv"):
    with tempfile.TemporaryDirectory() as temp:
        prefix = Path(temp) / "page3"
        subprocess.run(["pdftoppm", "-f", "3", "-l", "3", "-r", "170",
                        "-png", "-singlefile", str(pdf), str(prefix)],
                       check=True, stdout=subprocess.DEVNULL)
        im = Image.open(prefix.with_suffix(".png")).crop(
            (190, 600, 650, 990)).resize((1380, 1170))
    rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    r, g, b = [rgb[..., i] for i in range(3)]
    masks = {
        5: (g > 1.3 * r) & (b > 1.2 * r) & (g > 55) & (r < 155),
        10: (b > 1.35 * r) & (b > 1.2 * g) & (r < 155),
        15: (r > 1.5 * g) & (r > 1.35 * b) & (r > 95),
    }
    # The published axis is 800-1100 nm and 0-25 cm^-1.
    xmin, xmax, ymin, ymax = 90, 1267, 163, 1034
    data = {}
    for concentration, mask in masks.items():
        values = []
        detected = []
        for wavelength in range(935, 1041):
            x = round(xmin + (wavelength - 800) * (xmax - xmin) / 300)
            lo, hi = max(xmin + 2, x - 2), min(xmax - 2, x + 2)
            ys, _ = np.nonzero(mask[ymin:ymax + 1, lo:hi + 1])
            ys += ymin
            # Colored legend is in the upper-right; measured curves here are
            # all below this region beyond 980 nm.
            if wavelength >= 980:
                ys = ys[ys >= 800]
            if len(ys):
                value = (ymax - float(np.median(ys))) * 25 / (ymax - ymin)
                values.append(max(0.0, value))
                detected.append(True)
            else:
                values.append(float("nan"))
                detected.append(False)
        values = np.asarray(values)
        good = np.isfinite(values)
        if good.sum() < 70:
            raise RuntimeError(f"Only {good.sum()} detections for {concentration} at.%")
        filled = np.interp(np.arange(len(values)), np.flatnonzero(good), values[good])
        data[concentration] = (filled, detected)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("wavelength_nm", "alpha_5at_cm-1", "alpha_10at_cm-1",
                         "alpha_15at_cm-1", "detected_5at", "detected_10at",
                         "detected_15at"))
        for wavelength in range(935, 1041):
            i = wavelength - 935
            writer.writerow((wavelength, *(data[c][0][i] for c in (5, 10, 15)),
                             *(data[c][1][i] for c in (5, 10, 15))))
    provenance = {
        "source": "F. Tang et al., Journal of Alloys and Compounds 593 (2014) 123-127, Fig. 5",
        "source_pdf": "https://files.secure.website/wscfus/7885803/32406066/tang-yb-yag.pdf",
        "sample": "Three separate transparent ceramics, nominal 5, 10, 15 at.% Yb:YAG",
        "temperature": "room temperature; exact specimen temperature not stated",
        "quantity": "unsaturated absorption coefficient, cm^-1; not a per-ion cross section",
        "method": "Color-pixel figure trace over 935-1040 nm at 170 dpi and 1 nm increments; the 915 nm region is obscured by annotation arrows; gaps linearly interpolated",
        "minimum_ordinate_uncertainty_cm-1": 0.5,
        "limits": "Cannot supply hot spectra, fluorescence absolute calibration, concentration-dependent lifetime or gain by itself.",
        "detected_fraction": {str(c): float(np.mean(data[c][1])) for c in data},
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                           encoding="utf-8")
    return provenance


if __name__ == "__main__":
    print(json.dumps(digitize(Path(sys.argv[1])), indent=2))
