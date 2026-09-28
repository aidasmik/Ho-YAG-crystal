"""Train a small spatial Yb:YAG phase/correction prototype on completed setups.

This is an offline imitation experiment, not a verified SLM controller. Inputs
come only from measurement files; physical truth is used only as training
targets. Complete train/validation setups remain disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


TARGETS = (
    "Gaussian TEM00", "Flattop super-Gaussian",
    "Helical LG(0,+1)", "Needle Bessel-Gaussian",
)
SIZE = 128


def pool(array: np.ndarray) -> np.ndarray:
    """Area average fixed 1080p camera or 384² optical maps to 128²."""
    array = np.asarray(array, np.float32)
    if array.shape == (1080, 1920):
        return array[28:1052].reshape(SIZE, 8, SIZE, 15).mean((1, 3))
    if array.shape == (384, 384):
        return array.reshape(SIZE, 3, SIZE, 3).mean((1, 3))
    raise ValueError(f"unexpected source image shape {array.shape}")


def _circular_target(phase: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weighted = weight * np.exp(1j * phase)
    coarse = pool(weighted.real) + 1j * pool(weighted.imag)
    mask = pool(weight)
    mask /= max(float(mask.sum()), 1e-30)
    return np.stack((np.angle(coarse), mask), axis=-1).astype(np.float32)


def observable_inputs(measured, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Construct model inputs without opening or referencing truth arrays."""
    camera = np.asarray(measured["input__camera_adu"])
    if camera.shape != (2, 1080, 1920):
        raise ValueError("two 1080p phase-diverse images are required")
    black = float(metadata["camera_settings"]["black_level_adu"])
    channels = []
    for frame in camera:
        image = pool(np.maximum(frame.astype(np.float32) - black, 0))
        scale = max(float(np.percentile(image, 99)), 1)
        channels.append(np.clip(np.log1p(image) / np.log1p(scale), 0, 2))
    yb = pool(measured["input__yb_relative_map"])
    channels.append(np.clip((yb - 1) / .1, -3, 3))
    incoming = pool(measured["input__incoming_beam_shape"])
    channels.append(incoming / max(float(incoming.max()), 1e-30))
    for name in ("input__target_phase_mask_rad", "input__slm_command_rad"):
        phase = np.asarray(measured[name], np.float32)
        channels.extend((pool(np.sin(phase)), pool(np.cos(phase))))
    spatial = np.stack(channels, axis=-1).astype(np.float32)
    room = float(measured["input__room_temperature_K"])
    probes = np.asarray(measured["input__disk_temperature_probes_K"], float)
    valid = np.asarray(measured["input__disk_temperature_probe_valid"], bool)
    target = metadata["selected_target"]
    if target not in TARGETS:
        raise ValueError(f"unknown beam target {target}")
    scalar = np.r_[np.where(valid, probes - room, 0), valid.astype(float),
        (room - 293.15) / 5,
        float(measured["input__pump_power_W"]) / .1,
        float(measured["input__seed_energy_J"]) / 1e-8,
        float(measured["input__pump_radius_m"]) / 1e-3,
        float(measured["input__seed_waist_m"]) / 1e-3,
        float(measured["input__yb_nominal_at_percent"]) / 10,
        [float(target == name) for name in TARGETS]].astype(np.float32)
    if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(scalar)):
        raise ValueError("nonfinite observable input")
    return spatial, scalar


def _complete_paths(root: Path, manifest: dict, split: str) -> list[Path]:
    groups = {}
    for name in manifest["splits"][split]:
        path = root / name
        groups.setdefault(path.parent, []).append(path)
    paths = []
    for group in sorted(groups):
        members = sorted(groups[group])
        if [p.stem for p in members] == [f"point_{i:03d}" for i in range(3)]:
            paths.extend(members)
    return paths


def read_split(paths: list[Path]):
    images, scalars, phases, corrections, valid_labels = [], [], [], [], []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.parent / row["measurements_file"]) as measured:
            spatial, scalar = observable_inputs(measured, row)
            input_weight = np.asarray(measured["input__incoming_beam_shape"], float)
            with np.load(path.parent / row["truth_file"]) as truth:
                output_weight = (np.asarray(truth["truth__output_fluence_J_m2"], float) *
                                 np.asarray(truth["truth__phase_valid_mask"], bool))
                phase = _circular_target(
                    np.asarray(truth["truth__unwanted_phase_rad"], float), output_weight)
                label_valid = bool(truth["truth__correction_label_valid"])
                if label_valid != bool(row["correction_label_valid"]):
                    raise ValueError(f"label validity mismatch: {path}")
                if label_valid:
                    correction_weight = (input_weight *
                        np.asarray(truth["truth__slm_valid_mask"], bool))
                    correction = _circular_target(np.asarray(
                        truth["truth__verified_slm_correction_rad"], float),
                        correction_weight)
                else:
                    correction = np.zeros((SIZE, SIZE, 2), np.float32)
        images.append(spatial)
        scalars.append(scalar)
        phases.append(phase)
        corrections.append(correction)
        valid_labels.append(label_valid)
    return (np.stack(images), np.stack(scalars), np.stack(phases),
            np.stack(corrections), np.asarray(valid_labels, bool))


@tf.keras.utils.register_keras_serializable(package="YbYAG")
def masked_circular_mse(y_true, y_pred):
    phase = y_true[..., :1]
    weight = y_true[..., 1:2]
    delta = tf.atan2(tf.sin(y_pred - phase), tf.cos(y_pred - phase))
    numerator = tf.reduce_sum(weight * tf.square(delta), axis=(1, 2, 3))
    denominator = tf.reduce_sum(weight, axis=(1, 2, 3))
    return tf.reduce_mean(tf.where(denominator > 0,
                                   numerator / tf.maximum(denominator, 1e-12), 0))


def build_model(scalar_width: int) -> tf.keras.Model:
    spatial = tf.keras.Input((SIZE, SIZE, 8), name="measured_maps")
    scalar = tf.keras.Input((scalar_width,), name="measured_scalars")

    def block(x, channels):
        x = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu")(x)

    skip1 = block(spatial, 12)
    skip2 = block(tf.keras.layers.AveragePooling2D(pool_size=2)(skip1), 20)
    core = block(tf.keras.layers.AveragePooling2D(pool_size=2)(skip2), 28)
    context = tf.keras.layers.Dense(12, activation="relu")(scalar)
    context = tf.keras.layers.Reshape((1, 1, 12))(context)
    context = tf.keras.layers.UpSampling2D((SIZE // 4, SIZE // 4))(context)
    core = tf.keras.layers.Concatenate()((core, context))
    up = tf.keras.layers.UpSampling2D()(core)
    up = block(tf.keras.layers.Concatenate()((up, skip2)), 20)
    up = tf.keras.layers.UpSampling2D()(up)
    up = block(tf.keras.layers.Concatenate()((up, skip1)), 12)
    phase = tf.keras.layers.Conv2D(1, 1, name="output_phase_rad")(up)
    correction = tf.keras.layers.Conv2D(1, 1, name="slm_delta_phase_rad")(up)
    return tf.keras.Model((spatial, scalar), (phase, correction))


def _rms(prediction: np.ndarray, target: np.ndarray) -> float:
    delta = np.angle(np.exp(1j * (prediction[..., 0] - target[..., 0])))
    weight = target[..., 1]
    good = weight.sum(axis=(1, 2)) > 0
    if not np.any(good):
        return float("nan")
    per = np.sum(weight * delta * delta, axis=(1, 2)) / np.maximum(
        weight.sum(axis=(1, 2)), 1e-30)
    return float(np.sqrt(np.mean(per[good])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(20260927)
    root = args.dataset.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "ybyag_nn_closed_loop_v3":
        parser.error("v3 closed-loop dataset required")
    train_paths = _complete_paths(root, manifest, "train")
    validation_paths = _complete_paths(root, manifest, "validation")
    if len(train_paths) < 36 or len(validation_paths) < 12:
        parser.error("need complete independent training and validation setups")
    train = read_split(train_paths)
    validation = read_split(validation_paths)
    if train[4].sum() < 10 or validation[4].sum() < 3:
        parser.error("too few verified correction labels for an initial fit")
    model = build_model(train[1].shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss={"output_phase_rad": masked_circular_mse,
                        "slm_delta_phase_rad": masked_circular_mse},
                  loss_weights={"output_phase_rad": .01,
                                "slm_delta_phase_rad": 1.})
    history = model.fit((train[0], train[1]),
        {"output_phase_rad": train[2], "slm_delta_phase_rad": train[3]},
        validation_data=((validation[0], validation[1]),
                         {"output_phase_rad": validation[2],
                          "slm_delta_phase_rad": validation[3]}),
        epochs=args.epochs, batch_size=4, verbose=2)
    predicted_phase, predicted_correction = model.predict(
        (validation[0], validation[1]), batch_size=4, verbose=0)
    zero_phase = np.zeros_like(predicted_phase)
    zero_correction = np.zeros_like(predicted_correction)
    report = {"status": "prototype_only", "dataset_ready": False,
        "dataset": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "train_setups": len(train_paths) // 3,
        "validation_setups": len(validation_paths) // 3,
        "train_trials": len(train_paths),
        "validation_trials": len(validation_paths),
        "train_verified_labels": int(train[4].sum()),
        "validation_verified_labels": int(validation[4].sum()),
        "validation_output_phase_rms_rad": _rms(predicted_phase, validation[2]),
        "validation_zero_output_phase_rms_rad": _rms(zero_phase, validation[2]),
        "validation_correction_rms_rad": _rms(predicted_correction, validation[3]),
        "validation_zero_correction_rms_rad": _rms(zero_correction, validation[3]),
        "epochs": args.epochs,
        "limitations": "Coarse 128² supervised imitation; no fresh solver verification of predicted SLM commands, no hardware calibration, no test-set evaluation."}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output / "initial.weights.h5")
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2),
                                                encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
