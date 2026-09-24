"""Separate local Yb:LuAG CW material and coating screen.

From the repository root on Windows:
    python examples/ybluag_app.py
Open http://127.0.0.1:8781/ . This is not the Ho:YAG calculator.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ybluag import (YbLuAGMaterial, YbGallerySettings, fluorescence_spectrum,
                    propagate_cw, scan_output_coupler, simulate_structured_gallery,
                    simulate_pulsed_seed)
from ybluag.model import _spectra

PAGE = ROOT / "Yb-LuAG" / "app.html"
COATINGS = ROOT / "config" / "ybluag_10at_coatings.json"


def number(data, key, low, high):
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return result


def integer(data, key, low, high):
    value = number(data, key, low, high)
    if not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return int(value)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def calculate(data):
    """One 20 C CW material pass and an explicitly approximate OC screen."""
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    pump_nm = number(data, "pump_nm", 880, 1150)
    signal_nm = number(data, "signal_nm", 880, 1150)
    pump_W = number(data, "pump_W", 0.001, 1000)
    radius_mm = number(data, "radius_mm", 0.01, 10)
    thickness_um = number(data, "thickness_um", 1, 2000)
    seed_W = number(data, "seed_W", 0, 1000)
    config = json.loads(COATINGS.read_text(encoding="utf-8"))
    material = YbLuAGMaterial(pump_wavelength_nm=pump_nm,
                              signal_wavelength_nm=signal_nm)
    area = math.pi * (radius_mm * 1e-3) ** 2
    result = propagate_cw(material, thickness_um * 1e-6, 16,
                          pump_W / area, seed_W / area)
    coating = config["output_coupler"]
    scan = scan_output_coupler(
        material, pump_W / area, thickness_um * 1e-6, 8,
        coating["screening_candidates"],
        disk_hr_reflectivity=config["disk_rear"]["target_min_reflectance_laser"],
        other_roundtrip_survival=coating["screening_other_roundtrip_survival"])
    wavelength, _, _, _ = _spectra()
    sample = wavelength[::2]
    cross = [material.cross_sections_m2(float(w)) for w in sample]
    fluorescence = fluorescence_spectrum(material)
    return {
        "material": "Yb:LuAG", "temperature_C": 20, "yb_at_percent": 10,
        "pump_out_W": float(result.pump_out_W_m2) * area,
        "absorbed_pump_W": float(result.absorbed_pump_W_m2) * area,
        "signal_out_W": float(result.signal_out_W_m2) * area,
        "signal_change_W": float(result.signal_change_W_m2) * area,
        "mean_excited_fraction": float(np.mean(result.excited_fraction_by_step)),
        "transparency_fraction": material.transparency_fraction(),
        "absorption_coefficient_m1": material.coefficients_m1(0)[0],
        "fluorescence_energy_equivalent_nm": fluorescence.energy_equivalent_wavelength_nm,
        "coating_transmission": scan.transmission.tolist(),
        "coating_output_intensity_W_m2": scan.predicted_output_intensity_W_m2.tolist(),
        "selected_coating_transmission": scan.selected_transmission,
        "wavelength_nm": sample.tolist(),
        "absorption_cross_section_cm2": [float(v[0] * 1e4) for v in cross],
        "emission_cross_section_cm2": [float(v[1] * 1e4) for v in cross],
        "scope": "Single collinear CW pass at fixed 20 C; the OC scan uses an equal-counterpropagating-intensity approximation. Spectra are figure-guided reconstructions, and coating results are a design screen, not calibrated laser output.",
    }


def calculate_structured(data):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    from hoyag.structured_beam_gallery import PHASE_MASKS
    mask = data.get("phase_mask", "none")
    if mask not in PHASE_MASKS:
        raise ValueError("unknown phase mask")
    mode = data.get("solver_mode", "weak_probe")
    if mode not in ("weak_probe", "saturated_cw", "modal_cw"):
        raise ValueError("unknown solver mode")
    settings = YbGallerySettings(
        pump_power_W=number(data, "pump_W", 0.001, 1000),
        pump_radius_m=number(data, "radius_mm", 0.1, 5) * 1e-3,
        thickness_m=number(data, "thickness_um", 1, 2000) * 1e-6,
        input_power_W=number(data, "signal_W", 0.001, 100),
        waist_m=number(data, "waist_mm", 0.1, 2) * 1e-3,
        post_disk_distance_m=number(data, "distance_m", 0, 2),
        phase_mask_name=mask,
        phase_strength_rad=number(data, "phase_strength_rad", -50, 50),
        density_seed=integer(data, "density_seed", -2e9, 2e9),
        cluster_count=integer(data, "cluster_count", 2, 64),
        cluster_contrast=number(data, "cluster_contrast", 0, 1),
        fluorescence_escape_yield=number(data, "escape_yield", 0, 1),
        solver_mode=mode)
    result = simulate_structured_gallery(YbLuAGMaterial(), settings)
    grid = result["grid"]
    return {
        "material": "Yb:LuAG", "solver_mode": mode,
        "grid_n": grid.nx, "x_mm": (grid.x * 1e3).tolist(),
        "phase_mask": result["phase_mask"].tolist(),
        "yb_density_entrance_1e26_m3":
            (result["yb_density_m3"][0] / 1e26).tolist(),
        "yb_density_middle_1e26_m3":
            (result["yb_density_m3"][len(result["yb_density_m3"]) // 2] / 1e26).tolist(),
        "yb_density_xz_1e26_m3":
            (result["yb_density_m3"][:, grid.ny // 2, :] / 1e26).tolist(),
        "pump_intensity_W_m2": result["pump_intensity_W_m2"].tolist(),
        "fluorescence_escape_yield_assumed": result["fluorescence_escape_yield_assumed"],
        "scope": result["scope"],
        "thermal": {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in result["thermal"].items()},
        "resonator": result["resonator"],
        "modes": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                         for key, value in case.items()}
                  for name, case in result["outcomes"].items()},
    }


def calculate_pulsed(data):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    from hoyag.structured_beam_gallery import PHASE_MASKS, BEAM_NAMES
    mask = data.get("phase_mask", "none")
    beam = data.get("selected_beam", "Gaussian TEM00")
    if mask not in PHASE_MASKS or beam not in BEAM_NAMES:
        raise ValueError("unknown phase mask or seed beam")
    settings = YbGallerySettings(
        pump_power_W=number(data, "pump_W", 0.001, 1000),
        pump_radius_m=number(data, "radius_mm", 0.1, 5) * 1e-3,
        thickness_m=number(data, "thickness_um", 1, 2000) * 1e-6,
        waist_m=number(data, "waist_mm", 0.1, 2) * 1e-3,
        post_disk_distance_m=number(data, "distance_m", 0, 2),
        phase_mask_name=mask,
        phase_strength_rad=number(data, "phase_strength_rad", -50, 50),
        density_seed=integer(data, "density_seed", -2e9, 2e9),
        cluster_count=integer(data, "cluster_count", 2, 64),
        cluster_contrast=number(data, "cluster_contrast", 0, 1),
        fluorescence_escape_yield=number(data, "escape_yield", 0, 1))
    result = simulate_pulsed_seed(
        YbLuAGMaterial(), settings, beam,
        number(data, "seed_energy_nj", 0.001, 100000) * 1e-9,
        number(data, "seed_fwhm_ps", 0.1, 1000) * 1e-12,
        number(data, "repetition_rate_kHz", 0.01, 100) * 1e3,
        integer(data, "signal_traversals", 1, 10))
    return {key: jsonable(value) for key, value in result.items() if key != "grid"}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path not in ("/", "/app.html"):
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        self.respond(200, PAGE.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in ("/api/calculate", "/api/structured", "/api/pulsed"):
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON content type required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            response = (calculate_structured(payload) if endpoint == "/api/structured"
                        else calculate_pulsed(payload) if endpoint == "/api/pulsed"
                        else calculate(payload))
            status = 200
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            response, status = {"error": str(exc)}, 400
        body = json.dumps(response, allow_nan=False).encode("utf-8")
        self.respond(status, body, "application/json; charset=utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Yb:LuAG local app: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
