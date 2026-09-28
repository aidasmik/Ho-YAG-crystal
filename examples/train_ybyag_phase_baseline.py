"""Small observable-to-phase baseline for the closed-loop Yb:YAG dataset.

This estimates low-order *output* phase error. It is not an SLM correction
policy: a measured response/forward model and a fresh acceptance check are
still needed before applying a command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ybyag_control.controller import correction_basis


TARGETS = (
    "Gaussian TEM00",
    "Flattop super-Gaussian",
    "Helical LG(0,+1)",
    "Needle Bessel-Gaussian",
)
MODES = 6


def _pool(array: np.ndarray, size: int = 8) -> np.ndarray:
    h, w = array.shape
    bh, bw = h // size, w // size
    return array[: bh * size, : bw * size].reshape(size, bh, size, bw).mean((1, 3))


def observable_features(measured: dict, metadata: dict) -> np.ndarray:
    """Read measurements and commands only; never read truth arrays here."""
    black = float(metadata["camera_settings"]["black_level_adu"])
    frames = np.asarray(measured["input__camera_adu"])
    if frames.shape[0] != 2:
        raise ValueError("two phase-diverse camera planes are required")
    parts = []
    for frame in frames:
        signal = np.maximum(frame.astype(np.float32) - black, 0)
        binned = _pool(signal)
        parts.extend((np.sqrt(binned / max(float(binned.sum()), 1)),
                      np.array([np.log1p(float(signal.sum()))])))
    for key in ("input__yb_relative_map", "input__incoming_beam_shape"):
        image = np.asarray(measured[key], np.float32)
        parts.append(_pool(image))
    command = np.asarray(measured["input__slm_command_rad"])
    target_phase = np.asarray(measured["input__target_phase_mask_rad"])
    for phase in (command, target_phase):
        parts.extend((_pool(np.sin(phase)), _pool(np.cos(phase))))
    probes = np.asarray(measured["input__disk_temperature_probes_K"], float)
    valid = np.asarray(measured["input__disk_temperature_probe_valid"], bool)
    room = float(measured["input__room_temperature_K"])
    parts.extend((np.where(valid, probes - room, 0), valid.astype(float)))
    for key, scale in (("input__pump_power_W", 1),
                       ("input__seed_energy_J", 1e-6),
                       ("input__pump_radius_m", 1e-3),
                       ("input__seed_waist_m", 1e-3),
                       ("input__yb_nominal_at_percent", 10)):
        parts.append(np.array([float(measured[key]) / scale]))
    target = metadata["selected_target"]
    if target not in TARGETS:
        raise ValueError(f"unexpected target: {target}")
    parts.append(np.array([float(target == name) for name in TARGETS]))
    features = np.concatenate([np.ravel(p) for p in parts]).astype(np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("nonfinite observable features")
    return features


def phase_basis(truth: dict, measured: dict) -> np.ndarray:
    n = np.asarray(truth["truth__unwanted_phase_rad"]).shape[0]
    width = float(truth["metadata__field_size_m"])
    coords = (np.arange(n) - (n - 1) / 2) * width / n
    waist = float(measured["input__seed_waist_m"])
    return correction_basis(coords, coords, waist, mode_count=MODES)


def phase_coefficients(truth: dict, measured: dict) -> np.ndarray:
    phase = np.asarray(truth["truth__unwanted_phase_rad"], float)
    mask = np.asarray(truth["truth__phase_valid_mask"], bool)
    fluence = np.asarray(truth["truth__output_fluence_J_m2"], float)
    basis = phase_basis(truth, measured)
    weight = np.where(mask, fluence, 0)
    weight /= max(weight.sum(), 1e-30)
    design = basis.reshape(MODES, -1)
    flat_weight = weight.ravel()
    gram = (design * flat_weight) @ design.T
    rhs = (design * flat_weight) @ phase.ravel()
    return np.linalg.solve(gram + 1e-8 * np.eye(MODES), rhs)


def _snapshot_rows(root: Path, manifest: dict, split: str) -> list[Path]:
    """Use complete setup groups, even while another process extends a split."""
    groups = {}
    for name in manifest["splits"][split]:
        path = root / name
        groups.setdefault(path.parent, []).append(path)
    rows = []
    for group in sorted(groups):
        paths = sorted(groups[group])
        if len(paths) != 3:
            continue
        if [p.stem for p in paths] != [f"point_{i:03d}" for i in range(3)]:
            continue
        rows.extend(paths)
    return rows


def read_examples(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.parent / metadata["measurements_file"]) as measured:
            x.append(observable_features(measured, metadata))
            with np.load(path.parent / metadata["truth_file"]) as truth:
                y.append(phase_coefficients(truth, measured))
    return np.stack(x), np.stack(y)


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float = 10.0) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-5, scale, 1.0)
    z = np.clip((x - mean) / scale, -10, 10)
    y_mean = y.mean(axis=0)
    # The dual solve remains small when there are few physical setups.
    dual = np.linalg.solve(z @ z.T + penalty * np.eye(len(z)), y - y_mean)
    return {"feature_mean": mean, "feature_scale": scale,
            "weights": z.T @ dual, "target_mean": y_mean,
            "penalty": np.asarray(penalty)}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.clip((x - model["feature_mean"]) / model["feature_scale"], -10, 10)
    return z @ model["weights"] + model["target_mean"]


def select_penalty_by_setup(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Tune only on training setups, keeping all three trials together."""
    if len(x) % 3:
        raise ValueError("expected three trials per complete setup")
    setups = len(x) // 3
    options = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
    scores = []
    for penalty in options:
        errors = []
        for held_out in range(setups):
            validation = np.arange(3 * held_out, 3 * held_out + 3)
            training = np.r_[0:3 * held_out, 3 * held_out + 3:len(x)]
            candidate = fit_ridge(x[training], y[training], penalty)
            errors.append((predict(candidate, x[validation]) - y[validation]) ** 2)
        scores.append(float(np.sqrt(np.mean(errors))))
    index = int(np.argmin(scores))
    return options[index], scores[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "ybyag_nn_closed_loop_v3":
        parser.error("a v3 dataset is required")
    train = _snapshot_rows(root, manifest, "train")
    validation = _snapshot_rows(root, manifest, "validation")
    if len(train) < 6 or len(validation) < 3:
        parser.error("need at least two complete training setups and one validation setup")
    x_train, y_train = read_examples(train)
    x_val, y_val = read_examples(validation)
    penalty, cross_validation_rmse = select_penalty_by_setup(x_train, y_train)
    model = fit_ridge(x_train, y_train, penalty)
    predicted = predict(model, x_val)
    rmse = float(np.sqrt(np.mean((predicted - y_val) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(y_val ** 2)))
    train_mean_rmse = float(np.sqrt(np.mean((model["target_mean"] - y_val) ** 2)))
    result = {"kind": "low_order_output_phase_estimator_not_slm_policy",
              "dataset": str(root),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "train_trials": len(train), "validation_trials": len(validation),
              "train_setups": len(train) // 3,
              "validation_setups": len(validation) // 3,
              "phase_modes": MODES, "ridge_penalty": penalty,
              "train_setup_cross_validation_rmse_rad": cross_validation_rmse,
              "validation_coefficient_rmse_rad": rmse,
              "zero_phase_coefficient_rmse_rad": zero_rmse,
              "train_mean_coefficient_rmse_rad": train_mean_rmse,
              "validation_improves_zero": bool(rmse < zero_rmse),
              "validation_improves_train_mean": bool(rmse < train_mean_rmse),
              "limitations": "Illustrative solver truth; no SLM command or fresh closed-loop verification. Dataset may be incomplete."}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "phase_baseline.npz", **model)
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
