"""Train a measurement-only modal SLM correction model.

Pack on the dataset host (NumPy only), then fit on a TensorFlow host.  The
test split is evaluated only after validation-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

TARGETS = ("Gaussian TEM00", "Flattop super-Gaussian", "Helical LG(0,+1)",
           "Needle Bessel-Gaussian")
SIZE = 64
MODES = 23


def pool(array):
    array = np.asarray(array, np.float32)
    if array.shape == (1080, 1920):
        return array[28:1052].reshape(64, 16, 64, 30).mean((1, 3))
    if array.shape == (384, 384):
        return array.reshape(64, 6, 64, 6).mean((1, 3))
    raise ValueError(f"unexpected image shape {array.shape}")


def measured_inputs(measured, metadata):
    camera = np.asarray(measured["input__camera_adu"])
    if camera.shape != (2, 1080, 1920):
        raise ValueError("two 1080p camera planes are required")
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
    spatial = np.stack(channels, -1).astype(np.float32)
    room = float(measured["input__room_temperature_K"])
    probes = np.asarray(measured["input__disk_temperature_probes_K"], float)
    valid = np.asarray(measured["input__disk_temperature_probe_valid"], bool)
    target = metadata["selected_target"]
    if target not in TARGETS:
        raise ValueError(f"unknown target {target}")
    scalar = np.r_[np.where(valid, probes - room, 0), valid.astype(float),
        (room - 293.15) / 5,
        float(measured["input__pump_power_W"]) / .1,
        float(measured["input__seed_energy_J"]) / 1e-8,
        float(measured["input__pump_radius_m"]) / 1e-3,
        float(measured["input__seed_waist_m"]) / 1e-3,
        float(measured["input__yb_nominal_at_percent"]) / 10,
        [float(target == name) for name in TARGETS]].astype(np.float32)
    if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(scalar)):
        raise ValueError("nonfinite measured input")
    return spatial, scalar


def pack(dataset, output):
    dataset = Path(dataset)
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "ybyag_nn_closed_loop_v3":
        raise ValueError("v3 dataset required")
    arrays = {}
    summary = {"dataset": str(dataset.resolve()),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "feature_shape": [64, 64, 8], "modes": MODES,
               "splits": {}}
    ids_by_split = {}
    for split in ("train", "validation", "test"):
        spatial_rows, scalar_rows, coefficients, masks, valid_rows = [], [], [], [], []
        ids = []
        for name in manifest["splits"][split]:
            path = dataset / name
            meta = json.loads(path.read_text())
            ids.append(meta["crystal_id"])
            with np.load(path.parent / meta["measurements_file"]) as measured:
                spatial, scalar = measured_inputs(measured, meta)
            with np.load(path.parent / meta["truth_file"]) as truth:
                is_valid = bool(truth["truth__correction_label_valid"])
                if is_valid != bool(meta["correction_label_valid"]):
                    raise ValueError(f"label validity mismatch: {path}")
                raw = np.asarray(truth["truth__modal_coefficients_rad"], np.float32)
            if len(raw) != len(meta["modal_basis_names"]) or len(raw) > MODES:
                raise ValueError(f"bad modal label: {path}")
            if is_valid and not np.all(np.isfinite(raw)):
                raise ValueError(f"nonfinite verified label: {path}")
            label = np.zeros(MODES, np.float32)
            mask = np.zeros(MODES, np.float32)
            if is_valid:
                label[:len(raw)] = raw
                mask[:len(raw)] = 1
            spatial_rows.append(spatial.astype(np.float16))
            scalar_rows.append(scalar)
            coefficients.append(label)
            masks.append(mask)
            valid_rows.append(is_valid)
        if len(ids) != len(set(ids)) * 3:
            raise ValueError(f"incomplete three-trial setup sequence in {split}")
        ids_by_split[split] = set(ids)
        for key, values in (("spatial", spatial_rows), ("scalar", scalar_rows),
                            ("coefficients", coefficients), ("mask", masks),
                            ("valid", valid_rows)):
            arrays[f"{split}_{key}"] = np.asarray(values)
        summary["splits"][split] = {"setups": len(set(ids)), "trials": len(ids),
                                     "verified_labels": int(sum(valid_rows)),
                                     "nonzero_labels": int(sum(np.linalg.norm(x) > 1e-6
                                                               for x in coefficients))}
    if any(ids_by_split[a] & ids_by_split[b] for a in ids_by_split for b in ids_by_split if a != b):
        raise ValueError("setup leakage across splits")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    (output.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def build_model(scalar_width):
    import tensorflow as tf
    spatial = tf.keras.Input((SIZE, SIZE, 8), name="measured_maps")
    scalar = tf.keras.Input((scalar_width,), name="measured_scalars")
    x = spatial
    for channels in (16, 24, 32):
        skip = tf.keras.layers.Conv2D(channels, 1, strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2D(channels, 3, strides=2, padding="same",
                                   activation="swish")(x)
        x = tf.keras.layers.Conv2D(channels, 3, padding="same")(x)
        x = tf.keras.layers.Activation("swish")(tf.keras.layers.Add()([x, skip]))
    context = tf.keras.layers.Dense(24, activation="swish")(scalar)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, context])
    x = tf.keras.layers.Dense(96, activation="swish")(x)
    x = tf.keras.layers.Dropout(.15)(x)
    output = tf.keras.layers.Dense(MODES, name="modal_step_scaled")(x)
    return tf.keras.Model((spatial, scalar), output)


def metrics(pred, truth, mask, valid):
    active = valid & (np.linalg.norm(truth, axis=1) > 1e-6)
    def group(select):
        if not np.any(select):
            return {"count": 0}
        residual = (pred[select] - truth[select]) * mask[select]
        target = truth[select] * mask[select]
        return {"count": int(select.sum()),
                "prediction_phase_rms_rad": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                "zero_command_phase_rms_rad": float(np.sqrt(np.mean(np.sum(target**2, axis=1))))}
    return {"verified": group(valid), "nonzero_corrections": group(active),
            "hold": group(valid & ~active)}


def fit(packed, output, epochs):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(20260928)
    data = np.load(packed)
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = tuple(np.asarray(data[f"{split}_{key}"], np.float32)
                              for key in ("spatial", "scalar", "coefficients", "mask", "valid"))
    train = splits["train"]
    mean = train[1].mean(axis=0)
    std = np.maximum(train[1].std(axis=0), .05)
    def inputs(rows):
        return (rows[0], np.clip((rows[1] - mean) / std, -5, 5))
    def targets(rows):
        return np.concatenate((rows[2] / .25, rows[3]), axis=1)
    def sample_weights(rows):
        nonzero = np.linalg.norm(rows[2], axis=1) > 1e-6
        return np.where(rows[4] > .5, np.where(nonzero, 2., 1.), 0.).astype(np.float32)
    def masked_huber(y_true, y_pred):
        target = y_true[:, :MODES]
        mask = y_true[:, MODES:]
        delta = tf.abs(y_pred - target)
        huber = tf.where(delta < 1., .5 * tf.square(delta), delta - .5)
        return tf.reduce_sum(huber * mask, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1.)
    model = build_model(train[1].shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=masked_huber)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.weights.h5"
    history = model.fit(inputs(train), targets(train), sample_weight=sample_weights(train),
        validation_data=(inputs(splits["validation"]), targets(splits["validation"]),
                         sample_weights(splits["validation"])),
        epochs=epochs, batch_size=8, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                                    restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
                                                       save_best_only=True, save_weights_only=True)])
    model.load_weights(checkpoint)
    report = {"status": "offline_modal_imitation", "packed_dataset": str(Path(packed).resolve()),
              "epochs_completed": len(history.history["loss"]),
              "best_validation_loss": float(min(history.history["val_loss"])),
              "fresh_solver_verified": False, "splits": {}}
    for split in ("validation", "test"):
        rows = splits[split]
        pred = .25 * model.predict(inputs(rows), batch_size=8, verbose=0)
        report["splits"][split] = metrics(pred, rows[2], rows[3], rows[4] > .5)
    np.savez(output / "normalization.npz", scalar_mean=mean, scalar_std=std)
    (output / "history.json").write_text(json.dumps(history.history, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def ridge_features(spatial, scalar):
    """Keep coarse phase-diverse structure and every measured scalar."""
    spatial = np.asarray(spatial, np.float32)
    coarse = spatial.reshape(-1, 8, 8, 8, 8, 8).mean((2, 4)).reshape(-1, 512)
    return np.concatenate((coarse, np.asarray(scalar, np.float32)), axis=1)


def fit_ridge(packed, output):
    data = np.load(packed)
    parts = {}
    for split in ("train", "validation", "test"):
        parts[split] = {key: np.asarray(data[f"{split}_{key}"])
                        for key in ("spatial", "scalar", "coefficients", "mask", "valid")}
    train = parts["train"]
    x = ridge_features(train["spatial"], train["scalar"])
    mean = x.mean(0)
    std = np.maximum(x.std(0), .1)
    x = (x - mean) / std
    y = train["coefficients"]
    val = parts["validation"]
    xv = (ridge_features(val["spatial"], val["scalar"]) - mean) / std
    select = val["valid"].astype(bool)
    def objective(pred):
        delta = (pred - val["coefficients"]) * val["mask"]
        return float(np.sqrt(np.mean(np.sum(delta[select] ** 2, axis=1))))
    baseline = objective(np.zeros_like(val["coefficients"]))
    best = (baseline, None, 0., np.zeros((x.shape[1], MODES)))
    for alpha in (.1, 1., 10., 100., 1000.):
        coef = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
        for scale in (.25, .5, 1.):
            score = objective(scale * xv @ coef)
            if score < best[0]:
                best = (score, alpha, scale, coef)
    _, alpha, scale, coef = best
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "ridge_model.npz", mean=mean, std=std, coef=coef,
             scale=scale, alpha=0. if alpha is None else alpha)
    report = {"status": "offline_modal_imitation", "method": "regularized_linear_modal",
              "packed_dataset": str(Path(packed).resolve()), "selected_alpha": alpha,
              "selected_scale": scale, "fresh_solver_verified": False,
              "validation_selected_against_zero_baseline": bool(alpha is not None),
              "splits": {}}
    for split in ("validation", "test"):
        rows = parts[split]
        xs = (ridge_features(rows["spatial"], rows["scalar"]) - mean) / std
        pred = scale * xs @ coef
        split_report = metrics(pred, rows["coefficients"], rows["mask"],
                               rows["valid"].astype(bool))
        target_ids = np.argmax(rows["scalar"][:, -4:], axis=1)
        split_report["by_target"] = {
            name: metrics(pred[target_ids == index], rows["coefficients"][target_ids == index],
                          rows["mask"][target_ids == index],
                          rows["valid"][target_ids == index].astype(bool))["verified"]
            for index, name in enumerate(TARGETS)}
        report["splits"][split] = split_report
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def predict(model_file, trial_json, config_file, output):
    """Propose a phase-only SLM command from measured inputs only."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hoyag.propagation import Grid2D
    from ybyag_dataset.modal_teacher import modal_basis
    trial_json = Path(trial_json)
    meta = json.loads(trial_json.read_text())
    config = json.loads(Path(config_file).read_text())
    model = np.load(model_file)
    with np.load(trial_json.parent / meta["measurements_file"]) as measured:
        spatial, scalar = measured_inputs(measured, meta)
        feature = ridge_features(spatial[None], scalar[None])[0]
        coefficients = float(model["scale"]) * ((feature - model["mean"]) /
                         model["std"]) @ model["coef"]
        incoming = np.asarray(measured["input__incoming_beam_shape"], np.float64)
        current = np.asarray(measured["input__slm_command_rad"], np.float64)
        waist = float(measured["input__seed_waist_m"])
    n = current.shape[0]
    field_size_m = float(config["nominal"]["field_size_mm"]) * 1e-3
    grid = Grid2D.square(n, field_size_m)
    xx, yy = grid.mesh
    names, basis, weight = modal_basis(xx, yy, np.sqrt(incoming), waist,
                                       radial_order=4, residual_grid=0)
    step = np.einsum("k,kij->ij", coefficients[:len(names)], basis)
    command = np.mod(current + step, 2 * np.pi)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predicted_modal_coefficients_rad=coefficients,
                        proposed_slm_step_rad=step.astype(np.float32),
                        proposed_slm_command_rad=command.astype(np.float32))
    print(json.dumps({"candidate": str(output), "mode_count": len(names),
                      "weighted_step_rms_rad": float(np.sqrt(np.sum(weight * step**2))),
                      "fresh_solver_verified": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("fit")
    p.add_argument("packed", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p = sub.add_parser("fit-ridge")
    p.add_argument("packed", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("predict")
    p.add_argument("model", type=Path)
    p.add_argument("trial", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.dataset, args.output)
    elif args.command == "fit":
        fit(args.packed, args.output, args.epochs)
    elif args.command == "fit-ridge":
        fit_ridge(args.packed, args.output)
    else:
        predict(args.model, args.trial, args.config, args.output)


if __name__ == "__main__":
    main()
