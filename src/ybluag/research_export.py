"""Time-consistent, convergence-gated synthetic Yb:LuAG research samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .field_metrics import coherent_overlap


REQUIRED_REFINEMENTS = ("optical_window", "optical_pitch", "optical_depth",
                        "thermal_nr", "thermal_nphi", "thermal_nz",
                        "mechanical_mesh", "coupled_tolerance", "time_step")
REQUIRED_SHAPES = ("Gaussian TEM00", "Helical LG(0,+1)",
                   "Flattop super-Gaussian", "Needle Bessel-Gaussian")
REQUIRED_METRICS = ("coherent_field", "phase_rms", "output_energy",
                    "beam_radius", "peak_temperature", "opd", "deformation")


def source_fingerprint():
    """Hash the implemented Yb physics/export source, including local edits."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "pulsed.py", "multipass_pump.py", "regenerative.py",
                 "gallery.py", "assembly.py", "sensors.py", "research_export.py"):
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def check_convergence_evidence(evidence, thresholds):
    """Require finite errors and provenance for each independent axis."""
    if not isinstance(evidence, dict) or not isinstance(thresholds, dict):
        raise ValueError("convergence evidence and thresholds must be mappings")
    for axis in REQUIRED_REFINEMENTS:
        row, limit = evidence.get(axis), thresholds.get(axis)
        if not isinstance(row, dict) or row.get("status") != "passed":
            raise ValueError(f"unresolved convergence axis: {axis}")
        error = row.get("relative_error")
        if (not isinstance(error, (float, int)) or not np.isfinite(error) or
                error < 0 or not isinstance(limit, (float, int)) or
                not np.isfinite(limit) or limit <= 0 or error > limit):
            raise ValueError(f"failed convergence threshold: {axis}")
        if not row.get("configuration_pair"):
            raise ValueError(f"missing refinement provenance: {axis}")
        for shape in REQUIRED_SHAPES:
            metrics = (row.get("shape_metric_errors") or {}).get(shape)
            if not isinstance(metrics, dict):
                raise ValueError(f"missing {shape} convergence on {axis}")
            for name in REQUIRED_METRICS:
                value = metrics.get(name)
                if (not isinstance(value, (float, int)) or
                        not np.isfinite(value) or value < 0 or value > limit):
                    raise ValueError(f"unresolved {name} for {shape} on {axis}")


def _array(result, key):
    value = result.get(key)
    if value is None:
        raise ValueError(f"missing physical field: {key}")
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite physical field: {key}")
    return value


def export_research_sample(path, *, result, configuration, evidence, thresholds,
                           slm_command, intended_target_field, random_seeds,
                           material_version, solver_commit=None,
                           required_outputs=("hot_optical", "probes"),
                           dataset_mode="steady_state", correction_phase_rad=None,
                           forward_model=None, camera_noise_std_fraction=.01):
    """Export settled observables and hidden truth after a corrected forward rerun.

    ``forward_model(correction_phase_rad)`` must rerun the same coupled physical
    setup with that additional SLM-plane command. This function never derives
    a correction by negating an output-plane phase.
    """
    if dataset_mode != "steady_state":
        raise ValueError("transient dataset export unavailable without synchronized coupled dynamics")
    check_convergence_evidence(evidence, thresholds)
    config_json = json.dumps(configuration, sort_keys=True, allow_nan=False)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    code_hash = source_fingerprint()
    if (evidence.get("configuration_sha256") != config_hash or
            evidence.get("source_sha256") != code_hash):
        raise ValueError("convergence evidence does not match configuration and source")
    if result.get("thermal_optical_mode") != "coupled_steady" or not result.get("thermal_feedback_applied"):
        raise ValueError("sample requires a calculated coupled thermal-optical field")
    if (result.get("coupled_steady_convergence") or {}).get("status") != "converged":
        raise ValueError("coupled optical/thermal iteration did not converge")
    thermal = result.get("thermal") or {}
    if thermal.get("validity") == "outside_supported_conditions":
        raise ValueError("material properties are outside supported conditions")
    for key in ("scalar_roundtrip_opd_nm", "front_displacement_nm",
                "rear_displacement_nm", "disk_temperature_K", "plate_temperature_K"):
        if thermal.get(key) is None:
            raise ValueError(f"hot thermal field unavailable: {key}")
    if ("spectral" in required_outputs and
            (result.get("spectral_gain_screen") or {}).get("status") != "calculated"):
        raise ValueError("requested spectral output is unavailable")
    observation = result.get("steady_state_observation") or {}
    if observation.get("mode") != "steady_state":
        raise ValueError("settled same-state probe observation is required")
    timestamp = observation.get("state_timestamp_s")
    if (not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or
            observation.get("beam_timestamp_s") != timestamp or
            observation.get("slm_target_timestamp_s") != timestamp):
        raise ValueError("probe, beam, and SLM timestamps disagree")
    samples = observation.get("probe_samples") or []
    specs = observation.get("probe_specifications") or []
    if (len(samples) != 5 or len(specs) != 5 or
            any(sample.get("sample_time_s") != timestamp for sample in samples)):
        raise ValueError("five settled probes from the same physical state are required")
    if correction_phase_rad is None or forward_model is None:
        raise ValueError("forward-validated SLM correction is required")
    command = np.asarray(slm_command, float)
    correction = np.asarray(correction_phase_rad, float)
    target = np.asarray(intended_target_field, complex)
    baseline = _array(result, "output_complex_field_sqrt_J_m")
    if (command.shape != baseline.shape or correction.shape != baseline.shape or
            target.shape != baseline.shape or not np.all(np.isfinite(command)) or
            not np.all(np.isfinite(correction)) or not np.all(np.isfinite(target))):
        raise ValueError("SLM command, correction, and target must match output grid")
    corrected = forward_model(correction)
    if (corrected.get("thermal_optical_mode") != "coupled_steady" or
            (corrected.get("coupled_steady_convergence") or {}).get("status") != "converged"):
        raise ValueError("corrected full forward model did not converge")
    for key in ("architecture", "selected_beam", "input_energy_J", "pump_passes"):
        if corrected.get(key) != result.get(key):
            raise ValueError(f"corrected forward model changed {key}")
    before = coherent_overlap(target, baseline)
    after = coherent_overlap(target, _array(corrected, "output_complex_field_sqrt_J_m"))
    if not np.isfinite(after) or after <= before + 1e-6:
        raise ValueError("correction did not improve full-forward coherent fidelity")
    if (not isinstance(random_seeds, dict) or "camera" not in random_seeds or
            random_seeds.get("probe") != observation.get("probe_seed")):
        raise ValueError("camera and matching probe random seeds must be recorded")
    if not np.isfinite(camera_noise_std_fraction) or camera_noise_std_fraction < 0:
        raise ValueError("invalid camera noise level")
    exact_fluence = _array(result, "output_fluence_J_m2")
    rng = np.random.default_rng(random_seeds["camera"])
    measured = np.maximum(0, exact_fluence + rng.normal(
        0, camera_noise_std_fraction*float(np.max(exact_fluence)), exact_fluence.shape))
    observables = {
        "measured_output_fluence_J_m2": measured,
        "slm_command_rad": command,
        "measured_probe_K": np.array([np.nan if s.get("measured_K") is None
                                      else s["measured_K"] for s in samples]),
        "probe_valid": np.array([bool(s.get("valid")) for s in samples]),
        "probe_sample_time_s": np.array([s["sample_time_s"] for s in samples]),
        "pump_power_W": np.asarray(configuration["pump_power_W"]),
        "seed_energy_J": np.asarray(result["input_energy_J"]),
    }
    truth = {key: _array(result, key) for key in (
        "input_fluence_J_m2", "disk_input_fluence_J_m2",
        "output_complex_field_sqrt_J_m", "yb_density_m3",
        "pre_pulse_excited_fraction_by_slice", "optical_temperature_K_by_slice",
        "heat_W_m3_by_slice", "static_cold_phase_rad")}
    truth.update({
        "disk_temperature_K": np.asarray(thermal["disk_temperature_K"]),
        "plate_temperature_K": np.asarray(thermal["plate_temperature_K"]),
        "roundtrip_opd_nm": np.asarray(thermal["scalar_roundtrip_opd_nm"]),
        "correction_phase_rad": correction, "intended_target_field": target})
    if not all(np.all(np.isfinite(v)) for v in truth.values()):
        raise ValueError("nonfinite hidden truth field")
    commit = solver_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[2]).strip()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix(".npz"),
                        **{"observable__"+k: v for k, v in observables.items()},
                        **{"truth__"+k: v for k, v in truth.items()})
    metadata = {
        "kind": "synthetic_research_unvalidated_material_assumptions",
        "experimental_calibration": "missing", "dataset_mode": dataset_mode,
        "physical_state_timestamp_s": timestamp,
        "numerical_status": "verified_by_supplied_refinement_evidence",
        "configuration": configuration,
        "configuration_sha256": config_hash, "source_sha256": code_hash,
        "material_version": material_version, "solver_commit": commit,
        "random_seeds": random_seeds, "camera_noise_std_fraction": camera_noise_std_fraction,
        "convergence_evidence": evidence, "convergence_thresholds": thresholds,
        "probe_specifications": specs, "probe_samples": samples,
        "correction_validation": {"baseline_coherent_overlap": before,
                                  "corrected_coherent_overlap": after,
                                  "forward_model": "full_coupled_rerun"},
        "observable_keys": list(observables), "hidden_truth_keys": list(truth),
        "limitations": result.get("scope")}
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2, default=lambda v: v.tolist()
                   if isinstance(v, np.ndarray) else v), encoding="utf-8")
    return destination.with_suffix(".npz"), destination.with_suffix(".json")
