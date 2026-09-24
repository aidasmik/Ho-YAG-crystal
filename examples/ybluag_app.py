"""Separate local Yb:LuAG CW material and coating screen.

From the repository root on Windows:
    python examples/ybluag_app.py
Open http://127.0.0.1:8781/ . This is not the Ho:YAG calculator.
"""

from __future__ import annotations

from dataclasses import replace
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
from ybluag.regenerative import RegenerativeCavity
from ybluag.diagnostics import gain_feasibility, hardware_validity, spectral_gain_screen

PAGE = ROOT / "Yb-LuAG" / "app.html"
COATINGS = ROOT / "config" / "ybluag_10at_coatings.json"
PROPOSAL = ROOT / "config" / "ybslam_proposal_luag.json"


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


def sampling_diagnostics(grid, fluence):
    """Map-level FFT checks; convergence still requires independent refinement."""
    values = np.asarray(fluence, dtype=float)
    x, y = grid.mesh
    total = float(values.sum())
    edge = ((np.abs(x) > 0.4*grid.nx*grid.dx) |
            (np.abs(y) > 0.4*grid.ny*grid.dy))
    return {
        "grid_points_per_axis": grid.nx,
        "window_mm": grid.nx*grid.dx*1e3,
        "pixel_um": grid.dx*1e6,
        "second_moment_radius_um": (math.sqrt(2*float(np.sum(values*(x*x+y*y)))/total)*1e6
                                    if total > 0 else None),
        "edge_energy_fraction": float(values[edge].sum()/total) if total > 0 else None,
        "status": "preview_not_convergence_certified",
        "note": "Check radius, complex-field overlap, vortex core and mask structure across independent grid and window refinements.",
    }


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
    data = {"slm_to_disk_m": 0.25, "grid_n": 96,
            "field_size_mm": 12.0, "optical_z_steps": 4,
            "thermal_nr": 8, "thermal_nphi": 12, "thermal_nz": 4, **data}
    from hoyag.structured_beam_gallery import PHASE_MASKS, BEAM_NAMES
    mask = data.get("phase_mask", "none")
    if mask not in PHASE_MASKS:
        raise ValueError("unknown phase mask")
    mode = data.get("solver_mode", "weak_probe")
    if mode not in ("weak_probe", "saturated_cw", "modal_cw"):
        raise ValueError("unknown solver mode")
    beam = data.get("selected_beam", "Gaussian TEM00")
    if beam not in BEAM_NAMES:
        raise ValueError("unknown selected beam")
    settings = YbGallerySettings(
        pump_power_W=number(data, "pump_W", 0.001, 1000),
        pump_radius_m=number(data, "radius_mm", 0.1, 5) * 1e-3,
        thickness_m=number(data, "thickness_um", 1, 2000) * 1e-6,
        input_power_W=number(data, "signal_W", 0.001, 100),
        waist_m=number(data, "waist_mm", 0.1, 2) * 1e-3,
        post_disk_distance_m=number(data, "distance_m", 0, 2),
        slm_to_disk_distance_m=number(data, "slm_to_disk_m", 0.001, 2),
        phase_mask_name=mask,
        phase_strength_rad=number(data, "phase_strength_rad", -50, 50),
        density_seed=integer(data, "density_seed", -2e9, 2e9),
        cluster_count=integer(data, "cluster_count", 2, 64),
        cluster_contrast=number(data, "cluster_contrast", 0, 1),
        fluorescence_escape_yield=number(data, "escape_yield", 0, 1),
        solver_mode=mode,
        grid_n=integer(data, "grid_n", 32, 768),
        field_size_m=number(data, "field_size_mm", 8, 24) * 1e-3,
        z_steps=integer(data, "optical_z_steps", 1, 16),
        thermal_nr=integer(data, "thermal_nr", 4, 48),
        thermal_nphi=integer(data, "thermal_nphi", 4, 96),
        thermal_nz=integer(data, "thermal_nz", 1, 24))
    result = simulate_structured_gallery(YbLuAGMaterial(), settings, selected_beam=beam)
    ideal = (result if settings.cluster_contrast == 0 else
             simulate_structured_gallery(YbLuAGMaterial(),
                                         replace(settings, cluster_contrast=0.0),
                                         selected_beam=beam,
                                         compute_thermal=False))
    grid = result["grid"]
    return {
        "material": "Yb:LuAG", "solver_mode": mode,
        "concentration_dependent_index_status": "not_calculated: no measured bulk dn/dYb for this crystal",
        "grid_n": grid.nx, "field_size_mm": settings.field_size_m * 1e3,
        "optical_z_steps": settings.z_steps,
        "thermal_mesh": {"nr": settings.thermal_nr, "nphi": settings.thermal_nphi,
                         "nz": settings.thermal_nz},
        "selected_beam": beam,
        "sampling": sampling_diagnostics(grid, result["outcomes"][beam]["output_intensity"]),
        "x_mm": (grid.x * 1e3).tolist(), "y_mm": (grid.y * 1e3).tolist(),
        "phase_mask": result["phase_mask"].tolist(),
        "target_phase_mask": result["target_phase_mask"].tolist(),
        "aberration_phase_mask": result["aberration_phase_mask"].tolist(),
        "yb_density_entrance_1e26_m3":
            (result["yb_density_m3"][0] / 1e26).tolist(),
        "yb_density_middle_1e26_m3":
            (result["yb_density_m3"][len(result["yb_density_m3"]) // 2] / 1e26).tolist(),
        "yb_density_xz_1e26_m3":
            (result["yb_density_m3"][:, grid.ny // 2, :] / 1e26).tolist(),
        "pump_intensity_W_m2": result["pump_intensity_W_m2"].tolist(),
        "fluorescence_escape_yield_assumed": result["fluorescence_escape_yield_assumed"],
        "scope": result["scope"],
        "thermal": jsonable(result["thermal"]),
        "resonator": result["resonator"],
        "modes": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                         for key, value in case.items()}
                  for name, case in result["outcomes"].items()},
        "uniform_reference_profiles": {
            name: case["output_profile"].tolist()
            for name, case in ideal["outcomes"].items()},
        "uniform_reference_intensity": ideal["outcomes"][beam]["output_intensity"].tolist(),
    }


def calculate_pulsed(data, *, compute_thermal=True, summary_only=False):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    from hoyag.structured_beam_gallery import PHASE_MASKS, BEAM_NAMES
    mask = data.get("phase_mask", "none")
    beam = data.get("selected_beam", "Gaussian TEM00")
    if mask not in PHASE_MASKS or beam not in BEAM_NAMES:
        raise ValueError("unknown phase mask or seed beam")
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    data = {
        "pump_W": proposal["pump"]["incident_average_power_W"],
        "radius_mm": proposal["geometry"]["pump_beam_diameter_mm"] / 2,
        "thickness_um": proposal["geometry"]["disk_thickness_um"],
        "seed_energy_nj": proposal["seed"]["energy_nJ"],
        "seed_fwhm_ps": proposal["seed"]["amplifier_intensity_fwhm_ps"],
        "source_fwhm_fs": proposal["seed"]["source_intensity_fwhm_fs"],
        "repetition_rate_kHz": proposal["seed"]["repetition_rate_kHz"],
        "signal_traversals": proposal["seed"]["signal_traversals"],
        "pump_passes": proposal["pump"]["passes"],
        "waist_mm": 0.6,
        "distance_m": 0.0,
        "phase_strength_rad": math.pi,
        "density_seed": 17,
        "cluster_count": 24,
        "cluster_contrast": 0.0,
        "grid_n": 96,
        "field_size_mm": 12.0,
        "optical_z_steps": 4,
        "thermal_nr": 8,
        "thermal_nphi": 12,
        "thermal_nz": 4,
        "escape_yield": 0.0,
        "operation_duration_s": 30.0,
        "slm_to_disk_m": 0.25,
        "cooling_mode": "feedback",
        "cooling_target_C": 40.0,
        "architecture": "ideal_multipass",
        "regen_round_trips": 10,
        "cavity_length_m": 0.25,
        "mirror_radius_m": 0.5,
        "disk_hr_reflectivity": 0.9995,
        "held_retention": 0.98,
        "injection_efficiency": 0.9,
        "extraction_efficiency": 0.9,
        "cooling_h_max_W_m2K": 100000.0,
        **data,
    }
    source_fwhm_fs = number(data, "source_fwhm_fs", 50, 10000)
    amplifier_fwhm_ps = number(data, "seed_fwhm_ps", 0.1, 1000)
    if amplifier_fwhm_ps * 1000 < source_fwhm_fs:
        raise ValueError("amplifier pulse must be at least as long as the femtosecond source pulse")
    settings = YbGallerySettings(
        pump_power_W=number(data, "pump_W", 0.001, 1000),
        pump_radius_m=number(data, "radius_mm", 0.1, 5) * 1e-3,
        thickness_m=number(data, "thickness_um", 1, 2000) * 1e-6,
        waist_m=number(data, "waist_mm", 0.1, 2) * 1e-3,
        post_disk_distance_m=number(data, "distance_m", 0, 2),
        slm_to_disk_distance_m=number(data, "slm_to_disk_m", 0.001, 2),
        phase_mask_name=mask,
        phase_strength_rad=number(data, "phase_strength_rad", -50, 50),
        density_seed=integer(data, "density_seed", -2e9, 2e9),
        cluster_count=integer(data, "cluster_count", 2, 64),
        cluster_contrast=number(data, "cluster_contrast", 0, 1),
        fluorescence_escape_yield=number(data, "escape_yield", 0, 1),
        grid_n=integer(data, "grid_n", 32, 768),
        field_size_m=number(data, "field_size_mm", 8, 24) * 1e-3,
        z_steps=integer(data, "optical_z_steps", 1, 16),
        thermal_nr=integer(data, "thermal_nr", 4, 48),
        thermal_nphi=integer(data, "thermal_nphi", 4, 96),
        thermal_nz=integer(data, "thermal_nz", 1, 24))
    material = YbLuAGMaterial(
            yb_at_percent=proposal["material"]["yb_at_percent"],
            lifetime_s=proposal["material"]["lifetime_s"],
            pump_wavelength_nm=proposal["optics"]["pump_wavelength_nm"],
            signal_wavelength_nm=proposal["optics"]["signal_wavelength_nm"])
    pulse_args = (
        number(data, "seed_energy_nj", 0.001, 100000) * 1e-9,
        amplifier_fwhm_ps * 1e-12,
        number(data, "repetition_rate_kHz", 0.01, 100) * 1e3,
        integer(data, "signal_traversals", 1, 10))
    pump_passes = integer(data, "pump_passes", 1, 48)
    architecture = data.get("architecture", "ideal_multipass")
    if architecture not in ("ideal_multipass", "regenerative"):
        raise ValueError("unknown amplifier architecture")
    regenerative_cavity = RegenerativeCavity(
        round_trips=integer(data, "regen_round_trips", 1, 60),
        air_gap_m=number(data, "cavity_length_m", 0.01, 2),
        mirror_radius_m=number(data, "mirror_radius_m", 0.02, 10),
        disk_hr_reflectivity=number(data, "disk_hr_reflectivity", 0.5, 1),
        held_roundtrip_retention=number(data, "held_retention", 0.01, 1),
        injection_efficiency=number(data, "injection_efficiency", 0.01, 1),
        extraction_efficiency=number(data, "extraction_efficiency", 0.01, 1)) if architecture == "regenerative" else None
    cooling_mode = data.get("cooling_mode", "feedback")
    if cooling_mode not in ("fixed", "feedback"):
        raise ValueError("unknown cooling mode")
    result = simulate_pulsed_seed(
        material,
        settings, beam,
        *pulse_args, pump_passes=pump_passes, compute_thermal=compute_thermal,
        operation_duration_s=number(data, "operation_duration_s", 0, 120),
        cooling_mode=cooling_mode,
        cooling_target_C=number(data, "cooling_target_C", 20, 250),
        cooling_h_max_W_m2K=number(data, "cooling_h_max_W_m2K", 10000, 200000),
        architecture=architecture, regenerative_cavity=regenerative_cavity)
    if summary_only:
        return {"incident_pump_W": settings.pump_power_W,
                "average_output_W": result["output_energy_J"]*pulse_args[2],
                "net_energy_gain": result["output_energy_J"]/result["input_energy_J"]}
    reference = (result if settings.cluster_contrast == 0 and not result["thermal_feedback_applied"] else
                 simulate_pulsed_seed(material, replace(settings, cluster_contrast=0.0),
                                      beam, *pulse_args, pump_passes=pump_passes,
                                      compute_thermal=False, architecture=architecture,
                                      regenerative_cavity=regenerative_cavity))
    actual_phase = result["output_phase"]
    reference_phase = reference["output_phase"]
    weights = result["output_fluence_J_m2"]
    mask = weights >= 0.01 * float(weights.max())
    raw_residual = np.angle(np.exp(1j * (actual_phase - reference_phase)))
    piston = float(np.angle(np.sum(weights[mask] * np.exp(1j * raw_residual[mask]))))
    cold_density_residual = np.angle(np.exp(1j * (raw_residual - piston)))
    cold_density_rms = float(np.sqrt(np.average(
        cold_density_residual[mask]**2, weights=weights[mask])))
    hot_phase_validity = ("extrapolated_unvalidated"
                          if result["thermal_timeline"] is not None else
                          "not_calculated")
    hot_phase_reason = (
        "Material parameters are within their stated range, but the generic assembly "
        "is uncalibrated and thermal phase is applied after amplification only."
        if result["thermal_feedback_applied"] else
        "Requested temperature exceeds the supported thermo-mechanical property range; "
        "the displayed beam is cold-only, and hot wavefront error is unknown."
        if result["thermal_timeline"] is not None else
        "No requested-time thermal optical calculation was performed; the displayed beam is cold-only.")
    wavelength_nm = proposal["optics"]["signal_wavelength_nm"]
    spectral_fwhm_nm = (wavelength_nm * 1e-9)**2 / 299792458.0 * (
        0.441 / (source_fwhm_fs * 1e-15)) * 1e9
    peak_pump_kW_cm2 = 2 * settings.pump_power_W / (
        math.pi * settings.pump_radius_m**2) / 1e7
    feasibility = gain_feasibility(
        material, thickness_m=settings.thickness_m,
        pump_passes=pump_passes, signal_traversals=pulse_args[3],
        regenerative_round_trips=(regenerative_cavity.round_trips
                                    if regenerative_cavity is not None else None),
        input_energy_J=pulse_args[0],
        requested_output_energy_J=proposal["targets"]["output_energy_uJ_min"]*1e-6,
        injection_efficiency=(regenerative_cavity.injection_efficiency
                              if regenerative_cavity is not None else 1),
        extraction_efficiency=(regenerative_cavity.extraction_efficiency
                               if regenerative_cavity is not None else 1),
        held_roundtrip_retention=(regenerative_cavity.held_roundtrip_retention
                                  if regenerative_cavity is not None else 1),
        disk_hr_reflectivity=(regenerative_cavity.disk_hr_reflectivity
                              if regenerative_cavity is not None else 1))
    hardware = hardware_validity(
        pump_nm=material.pump_wavelength_nm,
        coating_band_nm=tuple(proposal["coatings"]["HR_band_nm"]),
        cavity_roundtrip_time_s=(result["regenerative"]["roundtrip_time_s"]
                                 if result["regenerative"] is not None else None))
    spectral_screen = spectral_gain_screen(
        material, source_fwhm_fs=source_fwhm_fs,
        stretched_fwhm_ps=amplifier_fwhm_ps,
        shared_inversion=result["mean_excited_fraction_before_pulse"],
        material_traversals=feasibility["material_traversals"],
        thickness_m=settings.thickness_m)
    return {
        **{key: jsonable(value) for key, value in result.items() if key != "grid"},
        "yb_at_percent": proposal["material"]["yb_at_percent"],
        "concentration_dependent_index_status": "not_calculated: no measured bulk dn/dYb for this crystal",
        "lifetime_s_assumed": proposal["material"]["lifetime_s"],
        "lifetime_status": proposal["material"]["lifetime_status"],
        "source_fwhm_fs_assumed": source_fwhm_fs,
        "amplifier_fwhm_ps_assumed": amplifier_fwhm_ps,
        "stretch_factor": amplifier_fwhm_ps * 1000 / source_fwhm_fs,
        "transform_limited_seed_spectral_fwhm_nm": spectral_fwhm_nm,
        "peak_pump_intensity_kW_cm2": peak_pump_kW_cm2,
        "incident_pump_W": settings.pump_power_W,
        "average_output_W": result["output_energy_J"]*pulse_args[2],
        "net_energy_gain": result["output_energy_J"]/result["input_energy_J"],
        "proposal_targets": proposal["targets"],
        "gain_feasibility": feasibility,
        "hardware_validity": hardware,
        "spectral_gain_screen": spectral_screen,
        "sampling": sampling_diagnostics(result["grid"], result["output_fluence_J_m2"]),
        "optical_z_steps": settings.z_steps,
        "thermal_mesh": {"nr": settings.thermal_nr, "nphi": settings.thermal_nphi,
                         "nz": settings.thermal_nz},
        "x_mm": jsonable(result["grid"].x * 1e3),
        "y_mm": jsonable(result["grid"].y * 1e3),
        "uniform_isothermal_output_fluence_J_m2": jsonable(reference["output_fluence_J_m2"]),
        "uniform_isothermal_output_phase": jsonable(reference_phase),
        "phase_residual_rad": jsonable(cold_density_residual) if result["thermal_feedback_applied"] else None,
        "phase_residual_rms_rad": cold_density_rms if result["thermal_feedback_applied"] else None,
        "cold_density_phase_residual_rad": jsonable(cold_density_residual)
            if not result["thermal_feedback_applied"] else None,
        "cold_density_phase_residual_rms_rad": cold_density_rms
            if not result["thermal_feedback_applied"] else None,
        "hot_phase_validity": hot_phase_validity,
        "hot_phase_reason": hot_phase_reason,
        "reference_scope": "Dashed output profiles use the same Gaussian source, target-shaping mask, added phase, SLM-to-disk propagation, pump and selected amplifier architecture with uniform Yb concentration and no thermal phase. The selected output receives transient thermal OPD only when the requested-time temperature is within the stated material range; temperature-dependent gain and thermal cavity feedback remain omitted.",
        "spectral_scope": "Pulse gain uses the 1030 nm center cross sections. The femtosecond source bandwidth, chirp, gain narrowing, dispersion and nonlinear phase are not propagated spectrally; pulse energy is a monochromatic engineering estimate.",
    }


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
        if endpoint not in ("/api/calculate", "/api/structured", "/api/pulsed", "/api/pump-sweep"):
            self.respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON content type required")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if endpoint == "/api/pump-sweep":
                top = number(payload, "pump_W", 0.001, 1000)
                response = {"points": [calculate_pulsed(
                    {**payload, "pump_W": max(0.001, top*fraction),
                     "operation_duration_s": 0},
                    compute_thermal=False, summary_only=True)
                    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)]}
            else:
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
