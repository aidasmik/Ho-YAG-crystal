"""Stage 7V: resumable cases, explicit evidence and conservative qualification.

Missing, failed, stale and nonconverged cases cannot become implicit passes.
Sensitivity is reported separately from numerical convergence. A component-only
frozen-source study never qualifies a coupled operating point for NN data.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .validation_metrics import (stable_hash, grid_from_axes, ComparisonDomain, relative_change,
                                 compare_fields, compare_opd)
from .validation_plan import validate_plan, Acceptance, SCHEMA
from .validation_backend import frozen_case, coupled_case, sha256_file, source_manifest


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value): raise ValueError("nonfinite numbers cannot enter a validation record")
        return float(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def execute_case(case, directory, manifest, *, reference_directory=None, resume=True, progress=None):
    """Run exactly one case. Write status before computation and retain failures."""
    out = Path(directory) / case["id"]; out.mkdir(parents=True, exist_ok=True)
    key = stable_hash({"schema": SCHEMA, "spec_hash": case["spec_hash"], "source_hash": manifest["source_hash"]})
    previous_path = out / "summary.json"
    if previous_path.exists() and resume:
        previous = json.loads(previous_path.read_text())
        if previous.get("execution_key") != key:
            raise ValueError(f"{case['id']}: stale cache (source or case changed); choose a new output directory")
        if previous.get("status") == "completed":
            state = out / "state.npz"
            if not state.is_file() or sha256_file(state) != previous.get("state_sha256"):
                raise ValueError(f"{case['id']}: stored numerical state is missing or corrupt")
            return previous
    begin = time.perf_counter()
    record = {"schema": SCHEMA, "id": case["id"], "kind": case["kind"], "purpose": case["purpose"],
              "spec_hash": case["spec_hash"], "physics_hash": case["physics_hash"],
              "execution_key": key, "source_hash": manifest["source_hash"], "status": "running",
              "case": case, "dataset_ready": False}
    write_json(previous_path, record)
    def checkpoint(row):
        # Persist partial fixed-point residuals even when an expensive later solve
        # times out or is killed. This is not a completed numerical state.
        with (out / "iterations.jsonl").open("a") as stream:
            stream.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
        if progress: progress(row)
    try:
        if case["kind"] == "frozen":
            if reference_directory is None:
                raise ValueError("reference_directory is required for frozen cases")
            data, arrays = frozen_case(case, reference_directory)
        elif case["kind"] == "coupled":
            data, arrays = coupled_case(case, progress=checkpoint)
        else:
            raise ValueError("unknown case kind")
        for name, arr in arrays.items():
            if arr is None: continue
            a = np.asarray(arr)
            if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
                raise FloatingPointError(f"nonfinite output array: {name}")
        np.savez_compressed(out / "state.npz", **{k:v for k,v in arrays.items() if v is not None})
        record.update(data)
        record["state_sha256"] = sha256_file(out / "state.npz")
    except Exception as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    record["wall_seconds"] = time.perf_counter()-begin
    write_json(previous_path, record)
    return record


def load_record(case, directory, manifest):
    path = Path(directory) / case["id"] / "summary.json"
    if not path.is_file(): return {"status": "not_run", "id": case["id"]}
    record = json.loads(path.read_text())
    if record.get("spec_hash") != case["spec_hash"] or record.get("source_hash") != manifest["source_hash"]:
        return {"status": "stale", "id": case["id"], "reason": "source or physical/numerical inputs changed"}
    if record.get("status") == "completed":
        state = path.parent / "state.npz"
        if not state.is_file() or sha256_file(state) != record.get("state_sha256"):
            return {"status": "corrupt", "id": case["id"]}
    return record


def _finite_metric(record, key):
    value = record.get("metrics", {}).get(key)
    return value is not None and np.isscalar(value) and np.isfinite(value)


def quality_gates(record, tolerances):
    """One-case consistency; not the cross-grid qualification itself."""
    gates = {}
    gates["execution_completed"] = record.get("status") == "completed"
    if not gates["execution_completed"]: return gates
    metrics = record.get("metrics", {})
    for key, threshold in (("thermal_energy_error_relative", tolerances.energy_balance_relative),
                           ("mechanical_residual", tolerances.mechanical_residual)):
        gates[key] = _finite_metric(record,key) and abs(metrics[key]) <= threshold
    if record.get("kind") == "coupled":
        gates["periodic_hot_state"] = metrics.get("coupled_converged") is True
        gates["optical_energy_balance"] = _finite_metric(record,"optical_population_error_relative") and abs(metrics["optical_population_error_relative"]) <= tolerances.energy_balance_relative
        gates["full_grid_eigenpair"] = _finite_metric(record,"eigen_residual") and metrics["eigen_residual"] <= tolerances.eigen_residual
        gates["mirror_sampling_over_disk"] = record.get("sampling", {}).get("mirror_nyquist_over_disk_radius") is True
    else:
        gates["fixed_source_conservation"] = _finite_metric(record,"heat_remap_relative_error") and metrics["heat_remap_relative_error"] <= 1e-10
    return gates


def _states(record, directory):
    with np.load(Path(directory)/record["id"] / "state.npz", allow_pickle=False) as archive:
        a = {k: archive[k].copy() for k in ("fields_used", "x_m", "y_m", "mean_roundtrip_opd_m")}
    return a, grid_from_axes(a["x_m"], a["y_m"])


def compare_records(a, b, directory, tolerances, domain):
    pair = {"from": a.get("id"), "to": b.get("id"), "status": "not_evaluated"}
    if a.get("status") != "completed" or b.get("status") != "completed":
        pair["reason"] = "both numerical runs must be completed"
        return pair
    if a.get("physics_hash") != b.get("physics_hash"):
        pair.update(status="invalid_comparison", reason="different physical problems; use a sensitivity comparison")
        return pair
    if a.get("kind") != b.get("kind") or a.get("source_hash") != b.get("source_hash"):
        pair.update(status="invalid_comparison", reason="solver scope or source revision differs")
        return pair
    if a.get("field_semantics") != b.get("field_semantics"):
        pair.update(status="invalid_comparison", reason="field meanings differ")
        return pair
    try:
        sa, ga = _states(a,directory); sb, gb = _states(b,directory)
        metrics, passed = {}, {}
        qa, qb = quality_gates(a,tolerances), quality_gates(b,tolerances)
        passed["run_quality"] = all(qa.values()) and all(qb.values())
        pair["run_quality"] = {"from": qa, "to": qb}
        for key in ("output_W", "pump_absorbed_W", "heat_W"):
            if key != "heat_W" and a["kind"] == "frozen": continue
            if not _finite_metric(a,key) or not _finite_metric(b,key):
                passed[key] = False; metrics[key] = None
            else:
                metrics[key] = relative_change(a["metrics"][key], b["metrics"][key], tolerances.power_absolute_floor_W)
                passed[key] = metrics[key] <= tolerances.power_relative
        for key in ("peak_disk_K", "peak_plate_K"):
            if not _finite_metric(a,key) or not _finite_metric(b,key):
                passed[key] = False; metrics[key] = None
            else:
                metrics[key] = abs(a["metrics"][key]-b["metrics"][key])
                passed[key] = metrics[key] <= tolerances.temperature_K
        opd = compare_opd(sa["mean_roundtrip_opd_m"],ga,sb["mean_roundtrip_opd_m"],gb,domain)
        metrics["opd"] = opd
        passed["piston_removed_opd"] = opd["piston_removed_difference_rms_m"] <= tolerances.opd_difference_rms_m
        if a["field_semantics"] == "independent_weak_probes":
            if a["probe_labels"] != b["probe_labels"]: raise ValueError("probe modes differ")
            fm = [compare_fields(fa,ga,fb,gb,domain) for fa,fb in zip(sa["fields_used"],sb["fields_used"])]
            fidelity = min(x["mixture_fidelity"] for x in fm)
            for key in ("power_gain", "target_oam_fraction", "crossed_polarization_fraction"):
                changes = [abs(ra[key]-rb[key]) for ra,rb in zip(a["probes"],b["probes"])]
                metrics["probe_"+key+"_max_absolute_change"] = max(changes)
        else:
            pa, pb = a["metrics"].get("mode_power_W"), b["metrics"].get("mode_power_W")
            if pa is None or pb is None: raise ValueError("physical modal output powers missing")
            if min(sum(pa),sum(pb)) <= tolerances.power_absolute_floor_W:
                raise ValueError("below-threshold modal field selection is not qualified by this comparison")
            fm = [compare_fields(sa["fields_used"],ga,sb["fields_used"],gb,domain,pa,pb)]
            fidelity = fm[0]["mixture_fidelity"]
            # Report participation separately; ordering/phase changes must not
            # masquerade as a different power distribution. This is a conservative
            # count-completeness gate, not proof that omitted branches are stable.
            na, nb = np.sort(np.asarray(pa)/sum(pa))[::-1], np.sort(np.asarray(pb)/sum(pb))[::-1]
            count = max(len(na),len(nb))
            change = float(np.max(abs(np.pad(na,(0,count-len(na)))-np.pad(nb,(0,count-len(nb))))))
            metrics["sorted_mode_power_fraction_change"] = change
            passed["modal_power_sharing"] = change <= tolerances.max_mode_power_fraction_change
        metrics["field_comparison"] = fm
        metrics["minimum_field_fidelity"] = fidelity
        passed["vector_field"] = fidelity >= tolerances.field_overlap
        fractions = [f for r in fm for name in ("roi_power_fraction_a", "roi_power_fraction_b") for f in r[name]]
        passed["comparison_aperture"] = min(fractions) >= tolerances.minimum_comparison_power_fraction and max(fractions) <= tolerances.maximum_comparison_power_fraction
        pair.update(status="pass" if all(passed.values()) else "fail", metrics=metrics, gates=passed)
    except Exception as error:
        pair.update(status="invalid_comparison", reason=str(error))
    return pair


def build_report(plan, directory, manifest):
    validate_plan(plan); directory = Path(directory)
    tol = Acceptance(**plan["acceptance"]); domain = ComparisonDomain(**plan["comparison_domain"])
    records = {c["id"]: load_record(c,directory,manifest) for c in plan["cases"]}
    groups = []
    for group in plan["groups"]:
        entry = dict(group)
        if group["comparison"] == "sensitivity":
            reference = records[group["case_ids"][0]]
            rows = []
            for cid in group["case_ids"][1:]:
                r = records[cid]
                row = {"id":cid, "status":r["status"], "numerically_qualified":False}
                if reference.get("status") == r.get("status") == "completed":
                    row["absolute_changes"] = {k:r["metrics"][k]-reference["metrics"][k] for k in
                        ("output_W","heat_W","peak_disk_K","peak_plate_K") if _finite_metric(reference,k) and _finite_metric(r,k)}
                    row["physics_hash"] = r["physics_hash"]
                rows.append(row)
            entry.update(status="reported" if all(r["status"]=="completed" for r in rows) else "incomplete", cases=rows,
                         note="Physical changes are not numerical errors. Each new operating point still needs its own refinement.")
        else:
            pairs = [compare_records(records[x], records[y], directory, tol, domain)
                     for x,y in zip(group["case_ids"][:-1],group["case_ids"][1:])]
            need = group["required_successive_pairs"]
            tail = pairs[-need:]
            if len(tail) < need or any(p["status"]=="not_evaluated" for p in tail): status = "incomplete"
            elif all(p["status"]=="pass" for p in tail): status = "pass"
            else: status = "fail"
            entry.update(status=status, pairs=pairs)
        groups.append(entry)
    required = [g for g in groups if g["required"]]
    qualified = bool(required) and all(g["status"]=="pass" for g in required)
    if any(g["status"]=="fail" for g in required): overall = "failed_acceptance"
    elif not qualified: overall = "incomplete"
    elif plan["kind"] == "frozen": overall = "component_checks_passed_not_coupled_validation"
    else: overall = "numerically_qualified_within_declared_model"
    report = {"schema":SCHEMA, "kind":plan["kind"], "status":overall,
              "source_hash":manifest["source_hash"], "plan_hash":stable_hash(plan),
              "case_status":{k:v["status"] for k,v in records.items()}, "groups":groups,
              "component_numerical_checks_passed":qualified,
              "coupled_numerically_qualified":qualified and plan["kind"]=="coupled",
              "dataset_ready":False, "experimentally_calibrated":False,
              "limits":plan["limits"], "acceptance":plan["acceptance"],
              "note":"Missing/failed/stale cases never count as passes. Passing here is not physical-model or contact/spectroscopy calibration."}
    write_json(directory/"qualification.json",report)
    rows = []
    for cid,r in records.items():
        rows.append({"case":cid,"kind":plan["kind"],"status":r["status"],
                     **{k:r.get("metrics",{}).get(k) for k in
                        ("output_W","pump_absorbed_W","heat_W","peak_disk_K","peak_plate_K")}})
    with (directory/"case_summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=["# Stage 7V qualification report", "", f"Status: **{overall}**.", "",
          f"Scope: `{plan['kind']}`. Completed cases: {sum(x['status']=='completed' for x in rows)}/{len(rows)}.", "",
          "| Comparison family | Status |", "|---|---|"]
    text += [f"| {g['name']} | {g['status']} |" for g in groups]
    text += ["", "No case is approved for NN training merely because the runner executed.", ""] + plan["limits"]
    (directory/"REPORT.md").write_text("\n".join(text)+"\n")
    return report
