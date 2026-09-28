"""Evaluate the saved initial model on complete, separate Yb:YAG setups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_ybyag_initial_nn import _complete_paths, _rms, build_model, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    root = args.dataset.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    paths = _complete_paths(root, manifest, args.split)
    if not paths:
        parser.error(f"no complete {args.split} setups")
    spatial, scalar, phase, correction, valid = read_split(paths)
    model = build_model(scalar.shape[1])
    model.load_weights(args.weights)
    phase_pred, correction_pred = model.predict((spatial, scalar),
                                                 batch_size=4, verbose=0)
    result = {"status": "offline_imitation_evaluation_only",
              "dataset": str(root), "split": args.split,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "setups": len(paths) // 3, "trials": len(paths),
              "verified_correction_labels": int(valid.sum()),
              "phase_rms_rad": _rms(phase_pred, phase),
              "zero_phase_rms_rad": _rms(np.zeros_like(phase_pred), phase),
              "correction_rms_rad": _rms(correction_pred, correction),
              "zero_correction_rms_rad": _rms(np.zeros_like(correction_pred),
                                                correction),
              "fresh_solver_verified": False,
              "dataset_ready": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
