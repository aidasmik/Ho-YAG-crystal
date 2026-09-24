"""Convergence-gated synthetic Yb:LuAG research samples.

The caller must supply independently measured refinement evidence. This module
never promotes preview output or an uncalibrated model to experimental truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


REQUIRED_REFINEMENTS = ("optical_window", "optical_pitch", "optical_depth",
                        "thermal_mesh", "mechanical_mesh", "time_step")


def check_convergence_evidence(evidence, thresholds):
    """Require separate, finite errors for every independent refinement axis."""
    if not isinstance(evidence, dict) or not isinstance(thresholds, dict):
        raise ValueError("convergence evidence and thresholds must be mappings")
    for axis in REQUIRED_REFINEMENTS:
        row = evidence.get(axis)
        limit = thresholds.get(axis)
        if not isinstance(row, dict) or row.get("status") != "passed":
            raise ValueError(f"unresolved convergence axis: {axis}")
        error = row.get("relative_error")
        if (not isinstance(error, (float, int)) or
                not np.isfinite(error) or error < 0 or
                not isinstance(limit, (float, int)) or
                not np.isfinite(limit) or limit <= 0 or error > limit):
            raise ValueError(f"failed convergence threshold: {axis}")
        if not row.get("configuration_pair"):
            raise ValueError(f"missing refinement provenance: {axis}")


def export_research_sample(path, *, result, configuration, evidence, thresholds,
                           slm_command, intended_target_field, random_seeds,
                           material_version, solver_commit=None,
                           required_outputs=("hot_optical", "probes")):
    """Write one labeled sample only after numerical and physics-validity checks.

    Arrays go to a compressed NPZ; provenance and qualification go to JSON.
    Unknown hot-wavefront or spectral results are rejected rather than zeroed.
    """
    check_convergence_evidence(evidence, thresholds)
    if result.get("thermal_optical_mode") != "coupled_steady" or not result.get("thermal_feedback_applied"):
        raise ValueError("sample requires a calculated coupled thermal-optical field")
    coupled = result.get("coupled_steady_convergence") or {}
    if coupled.get("status") != "converged":
        raise ValueError("coupled optical/thermal iteration did not converge")
    thermal = result.get("thermal", {})
    if thermal.get("validity") == "outside_supported_conditions":
        raise ValueError("material properties are outside supported conditions")
    if any(thermal.get(field) is None for field in
           ("scalar_roundtrip_opd_nm", "front_displacement_nm",
            "rear_displacement_nm")):
        raise ValueError("hot phase or deformation is unavailable")
    if ("spectral" in required_outputs and
            result.get("spectral_gain_screen", {}).get("status") in
            ("not_calculated", "outside_supported_conditions")):
        raise ValueError("requested spectral output is unavailable")
    timeline = result.get("thermal_timeline")
    if not isinstance(timeline, dict) or not timeline.get("temperature_probes"):
        raise ValueError("five probe observations are required")
    probes = timeline["temperature_probes"]
    if len(probes.get("specifications", [])) != 5 or not probes.get("samples"):
        raise ValueError("five probe observations are required")
    config_json = json.dumps(configuration, sort_keys=True, allow_nan=False)
    commit = solver_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[2]).strip()
    arrays = {}
    for key in ("input_fluence_J_m2", "disk_input_fluence_J_m2",
                "output_fluence_J_m2", "output_phase", "yb_density_m3",
                "pre_pulse_excited_fraction_by_slice", "optical_temperature_K_by_slice"):
        if result.get(key) is None:
            raise ValueError(f"missing physical field: {key}")
        values = np.asarray(result[key], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite physical field: {key}")
        arrays[key] = values
    arrays["slm_command_rad"] = np.asarray(slm_command, dtype=float)
    arrays["intended_target_field"] = np.asarray(intended_target_field)
    for label, values in (("steady_disk_temperature_K", thermal.get("disk_temperature_K")),
                          ("steady_plate_temperature_K", thermal.get("plate_temperature_K")),
                          ("requested_disk_temperature_K", timeline.get("requested_disk_temperature_K")),
                          ("requested_plate_temperature_K", timeline.get("requested_plate_temperature_K"))):
        if values is None:
            raise ValueError(f"missing exact thermal field: {label}")
        arrays[label] = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arrays[label])):
            raise ValueError(f"nonfinite exact thermal field: {label}")
    if arrays["slm_command_rad"].shape != arrays["output_phase"].shape or not np.all(
            np.isfinite(arrays["slm_command_rad"])):
        raise ValueError("SLM command must match the output map and be finite")
    if arrays["intended_target_field"].shape != arrays["output_phase"].shape or not np.all(
            np.isfinite(arrays["intended_target_field"])):
        raise ValueError("target field must match the output map and be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    metadata = {
        "kind": "synthetic_research_unvalidated_material_assumptions",
        "experimental_calibration": "missing",
        "numerical_status": "verified_by_supplied_refinement_evidence",
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "material_version": material_version,
        "solver_commit": commit,
        "random_seeds": random_seeds,
        "approximation": result.get("scope"),
        "convergence_evidence": evidence,
        "convergence_thresholds": thresholds,
        "probe_specifications": probes["specifications"],
        "probe_samples": probes["samples"],
        "thermal_timeline_exact_disk_max_C": timeline.get("disk_max_C"),
        "spectral_status": result["spectral_gain_screen"]["status"],
        "required_outputs": list(required_outputs),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2, default=lambda v: v.tolist()
                   if isinstance(v, np.ndarray) else v), encoding="utf-8")
    return destination.with_suffix(".npz"), destination.with_suffix(".json")
