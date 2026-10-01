"""Check generated group separation, file integrity and observable diversity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check_trial_sequence(rows):
    ordered=sorted(rows,key=lambda row:row["trial_index"])
    if ([row["trial_index"] for row in ordered] != list(range(len(ordered))) or
            len(ordered)<3 or ordered[0]["trial_outcome"]!="baseline_accepted"):
        raise ValueError("incomplete closed-loop trial sequence")
    retained=float(ordered[0]["measured_camera_shape_loss"])
    fixed=lambda row:(row["setup_npz_sha256"],
        json.dumps(row["camera_settings"],sort_keys=True),
        row["physical_parameters"]["yb_at_percent"],row["selected_target"],
        json.dumps(row["sensor_parameters"]["bias_K"],sort_keys=True),
        json.dumps(row["physical_parameters"]["external_optics"],sort_keys=True))
    if any(fixed(row)!=fixed(ordered[0]) for row in ordered[1:]):
        raise ValueError("crystal or instrument parameters changed inside a setup")
    for row in ordered[1:]:
        loss=float(row["measured_camera_shape_loss"])
        tolerance=float(row["trial_improvement_tolerance"])
        outcome=row["trial_outcome"]
        if outcome=="accepted":
            if not loss < retained-tolerance:
                raise ValueError("accepted trial did not improve measured camera loss")
            retained=loss
        elif outcome=="rejected":
            if loss < retained-tolerance:
                raise ValueError("rejected trial improved measured camera loss")
        elif outcome!="delayed_measured":
            raise ValueError("unknown measured trial outcome")


def check(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    new_labels=manifest["schema"]=="ybyag_nn_closed_loop_v3"
    smoke=bool(manifest.get("smoke_unqualified",False))
    if not new_labels and manifest["schema"]!="ybyag_nn_closed_loop_v2":
        raise ValueError("unknown NN dataset schema")
    groups={}
    details=[]
    sequence_rows={}
    coverage={}
    outcomes={}
    correction_labels={"verified":0,"rejected":0}
    for split,files in manifest["splits"].items():
        groups[split]=set()
        coverage[split]=set()
        outcomes[split]={}
        for name in files:
            path=root/name
            metadata=json.loads(path.read_text(encoding="utf-8"))
            measured_path=path.parent/metadata["measurements_file"]
            truth_path=path.parent/metadata["truth_file"]
            if (hashlib.sha256(measured_path.read_bytes()).hexdigest()!=metadata["measurements_sha256"] or
                    hashlib.sha256(truth_path.read_bytes()).hexdigest()!=metadata["truth_sha256"]):
                raise ValueError(f"sample hash mismatch: {path}")
            measured=np.load(measured_path)
            truth=np.load(truth_path)
            if (any(not key.startswith("input__") for key in measured.files) or
                    any(not (key.startswith("truth__") or key.startswith("metadata__"))
                        for key in truth.files)):
                raise ValueError(f"physical truth and measured inputs are mixed: {path}")
            if metadata["split"]!=split or metadata["crystal_id"]!=path.parent.name:
                raise ValueError(f"split/crystal metadata mismatch: {path}")
            groups[split].add(metadata["crystal_id"])
            coverage[split].add((float(metadata["physical_parameters"]["yb_at_percent"]),
                                 metadata["selected_target"]))
            outcome=metadata["trial_outcome"]
            outcomes[split][outcome]=outcomes[split].get(outcome,0)+1
            sequence_rows.setdefault(metadata["crystal_id"],[]).append(metadata)
            frame=measured["input__camera_adu"]
            if frame.ndim!=3 or (not smoke and frame.shape[1:]!=(1080,1920)):
                raise ValueError(f"not a 1080p plane stack: {path}")
            clean=truth["truth__clean_camera_fluence_J_m2"]
            diversity=(float(np.sum(abs(clean[0]-clean[1]))/max(np.sum(clean[0]),1e-30))
                       if len(clean)>1 else None)
            if diversity is not None and diversity<=.01:
                raise ValueError(f"weak phase diversity: {path}")
            temp=truth["truth__temperature_K"]
            probes=truth["truth__probe_temperature_K"]
            if not np.all(np.isfinite(temp)) or not np.all(np.isfinite(probes)):
                raise ValueError(f"nonfinite physical temperature: {path}")
            if metadata.get("setup_npz_sha256"):
                setup=path.parent/"setup.npz"
                if hashlib.sha256(setup.read_bytes()).hexdigest()!=metadata["setup_npz_sha256"]:
                    raise ValueError(f"setup hash mismatch: {setup}")
            label_valid=bool(truth["truth__correction_label_valid"])
            correction_labels["verified" if label_valid else "rejected"]+=1
            label=truth["truth__verified_slm_correction_rad"]
            command=(truth["truth__verified_slm_command_rad"] if new_labels else label)
            if label_valid != metadata["correction_label_valid"]:
                raise ValueError(f"correction label validity mismatch: {path}")
            if label_valid:
                thresholds=metadata["correction_label_thresholds"]
                phase=metadata["wavefront_rms_rad"]
                shape=metadata["shape_overlap"]
                shape_ok=(shape["corrected"] >= thresholds["shape_overlap"]-1e-7
                          if new_labels else
                          shape["corrected"]-shape["uncorrected"] >= thresholds["shape_overlap"]-1e-7)
                if thresholds.get("objective")=="coherent_fidelity":
                    fidelity=metadata["coherent_fidelity"]
                    hold=metadata.get("correction_label_method")=="modal:hold"
                    objective_ok=(fidelity["corrected"] is not None and
                        fidelity["corrected"] >= thresholds["target_fidelity"]-1e-7 and
                        (hold or fidelity["corrected"]-fidelity["uncorrected"] >=
                         thresholds["min_coherent_gain"]-1e-7) and
                        metadata["corrected_energy_fraction"] >=
                        thresholds["min_energy_fraction"]-1e-7)
                    coefficients=truth["truth__modal_coefficients_rad"]
                    objective_ok=(objective_ok and coefficients.size ==
                        len(metadata["modal_basis_names"]) and
                        np.all(np.isfinite(coefficients)))
                else:
                    objective_ok=(phase["uncorrected"]-phase["corrected"] >=
                                  thresholds["phase_rad"]-1e-7)
                if (not np.all(np.isfinite(label)) or not np.all(np.isfinite(command)) or
                    not objective_ok or not shape_ok):
                    raise ValueError(f"unverified correction label: {path}")
            elif not np.all(np.isnan(label)) or not np.all(np.isnan(command)):
                raise ValueError(f"rejected correction label must be missing: {path}")
            if new_labels:
                for key in ("truth__input_fluence_J_m2","truth__disk_input_fluence_J_m2",
                            "truth__output_fluence_J_m2","truth__yb_at_percent_map"):
                    if not np.all(np.isfinite(truth[key])):
                        raise ValueError(f"nonfinite physical ground truth {key}: {path}")
                for field_key,fluence_key in (
                    ("truth__input_complex_field_sqrt_J_m","truth__input_fluence_J_m2"),
                    ("truth__disk_input_complex_field_sqrt_J_m","truth__disk_input_fluence_J_m2"),
                    ("truth__complex_field_sqrt_J_m","truth__output_fluence_J_m2"),
                ):
                    if not np.allclose(abs(truth[field_key])**2,truth[fluence_key],rtol=2e-5,atol=1e-9):
                        raise ValueError(f"complex field and fluence disagree: {path}")
                incoming=truth["truth__input_fluence_J_m2"]
                if (not np.allclose(measured["input__incoming_beam_shape"],
                                    incoming/max(float(incoming.sum()),1e-30),rtol=2e-5,atol=1e-8) or
                    not np.allclose(measured["input__yb_relative_map"],truth["truth__yb_scale"])):
                    raise ValueError(f"known input map and physical truth disagree: {path}")
            details.append(dict(path=name,split=split,crystal_id=metadata["crystal_id"],
                trial_index=metadata["trial_index"],trial_outcome=outcome,
                target=metadata["selected_target"],
                yb_at_percent=metadata["physical_parameters"]["yb_at_percent"],
                planes=int(frame.shape[0]),phase_diversity_relative_L1=diversity,
                saturated_fraction=[row["saturated_fraction"] for row in metadata["camera_diagnostics"]],
                wavefront_rms_rad=metadata.get("wavefront_rms_rad")))
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:
                raise ValueError(f"crystals shared between {a} and {b}")
    if not smoke and not all(groups[k] for k in ("train","validation","test")):
        raise ValueError("train/validation/test must each contain a complete setup")
    planned=manifest.get("setup_plan",{}).get("train") or []
    doping={float(r["yb_at_percent"]) for r in planned} or {5.,10.,15.}
    required={(float(d),t) for d in doping for t in
              ("Gaussian TEM00","Flattop super-Gaussian","Helical LG(0,+1)",
               "Needle Bessel-Gaussian")}
    if not smoke:
        for split in ("train","validation","test"):
            if not required <= coverage[split]:
                raise ValueError(f"missing doping/target coverage in {split}: {required-coverage[split]}")
    for group,rows in sequence_rows.items():
        try:
            check_trial_sequence(rows)
        except ValueError as exc:
            raise ValueError(f"{group}: {exc}") from exc
    return dict(schema="ybyag_nn_dataset_check_v3",
                status="smoke_checks_passed" if smoke else "software_checks_passed",
                dataset_ready=False,groups={k:sorted(v) for k,v in groups.items()},
                coverage={k:sorted([{"yb_at_percent":d,"target":t} for d,t in v],
                                   key=lambda row:(row["yb_at_percent"],row["target"]))
                          for k,v in coverage.items()},
                trial_outcomes=outcomes,samples=details,
                correction_labels=correction_labels,
                physical_validity_limits=manifest["limits"],
                camera_calibration=manifest["camera_calibration"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    report=check(args.root)
    path=args.root/"validation.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(path)


if __name__=="__main__":
    main()
