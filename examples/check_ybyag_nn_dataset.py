"""Check generated group separation, file integrity and observable diversity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    groups={}
    details=[]
    for split,files in manifest["splits"].items():
        groups[split]=set()
        for name in files:
            path=root/name
            arrays=np.load(path)
            metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata["npz_sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            if metadata["split"]!=split or metadata["crystal_id"]!=path.parent.name:
                raise ValueError(f"split/crystal metadata mismatch: {path}")
            groups[split].add(metadata["crystal_id"])
            frame=arrays["input__camera_adu"] if "input__camera_adu" in arrays else arrays["observable__camera_adu"]
            if frame.ndim!=3 or frame.shape[1:]!=(1080,1920):
                raise ValueError(f"not a 1080p plane stack: {path}")
            clean=arrays["truth__clean_camera_fluence_J_m2"]
            diversity=(float(np.sum(abs(clean[0]-clean[1]))/max(np.sum(clean[0]),1e-30))
                       if len(clean)>1 else None)
            if diversity is not None and diversity<=.01:
                raise ValueError(f"weak phase diversity: {path}")
            temp=arrays["truth__temperature_K"]
            probes=arrays["truth__probe_temperature_K"]
            if not np.all(np.isfinite(temp)) or not np.all(np.isfinite(probes)):
                raise ValueError(f"nonfinite physical temperature: {path}")
            if metadata.get("setup_npz_sha256"):
                setup=path.parent/"setup.npz"
                if hashlib.sha256(setup.read_bytes()).hexdigest()!=metadata["setup_npz_sha256"]:
                    raise ValueError(f"setup hash mismatch: {setup}")
            details.append(dict(path=name,split=split,crystal_id=metadata["crystal_id"],
                planes=int(frame.shape[0]),phase_diversity_relative_L1=diversity,
                saturated_fraction=[row["saturated_fraction"] for row in metadata["camera_diagnostics"]],
                wavefront_rms_rad=metadata.get("wavefront_rms_rad")))
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:
                raise ValueError(f"crystals shared between {a} and {b}")
    if not all(groups[k] for k in ("train","validation","test")):
        raise ValueError("train/validation/test must each contain a complete setup")
    return dict(schema="ybyag_nn_dataset_check_v1",status="software_checks_passed",
                dataset_ready=False,groups={k:sorted(v) for k,v in groups.items()},
                samples=details)


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
