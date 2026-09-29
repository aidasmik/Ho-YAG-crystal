"""Select, evaluate, and use a measured-input Yb:YAG modal correction ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train_ybyag_modal_nn import build_model, measured_inputs, metrics, ridge_features, TARGETS
from hoyag.propagation import Grid2D
from ybyag_dataset.modal_teacher import modal_basis


def load_models(work: Path, scalar_width: int):
    normal = np.load(work / 'spatial_model/normalization.npz')
    ridge = np.load(work / 'ridge_model/ridge_model.npz')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    neural = build_model(scalar_width)
    neural.load_weights(work / 'spatial_model/best.weights.h5')
    return normal, ridge, neural


def components(spatial, scalar, normal, ridge, neural):
    spatial = np.asarray(spatial, np.float32)
    scalar = np.asarray(scalar, np.float32)
    scaled = np.clip((scalar - normal['scalar_mean']) / normal['scalar_std'], -5, 5)
    cnn = .25 * neural.predict((spatial, scaled), batch_size=8, verbose=0)
    feature = ridge_features(spatial, scalar)
    linear = float(ridge['scale']) * ((feature - ridge['mean']) / ridge['std']) @ ridge['coef']
    return cnn, linear


def first_fourteen(prediction):
    result = np.asarray(prediction).copy()
    result[:, 14:] = 0
    return result


def split_metrics(pred, packed, split):
    truth = packed[f'{split}_coefficients']
    mask = packed[f'{split}_mask']
    valid = packed[f'{split}_valid'].astype(bool)
    report = metrics(pred, truth, mask, valid)
    targets = np.argmax(packed[f'{split}_scalar'][:, -4:], axis=1)
    report['by_target'] = {
        name: metrics(pred[targets == index], truth[targets == index],
                      mask[targets == index], valid[targets == index])['verified']
        for index, name in enumerate(TARGETS)}
    return report


def evaluate(work: Path):
    packed = np.load(work / 'packed.npz')
    normal, ridge, neural = load_models(work, packed['train_scalar'].shape[1])
    components_by_split = {}
    for split in ('validation', 'test'):
        components_by_split[split] = components(packed[f'{split}_spatial'],
            packed[f'{split}_scalar'], normal, ridge, neural)

    def options(cnn, linear):
        candidates = {'zero': np.zeros_like(cnn), 'ridge': linear}
        for scale in (.25, .5, .75, 1.):
            candidates[f'cnn_{scale:g}'] = scale * cnn
        for neural_share in (.25, .5, .75):
            candidates[f'blend_{neural_share:g}'] = neural_share * cnn + (1-neural_share) * linear
        return {name: first_fourteen(candidate) for name, candidate in candidates.items()}

    validation_options = options(*components_by_split['validation'])
    ranking = sorted((split_metrics(pred, packed, 'validation')['verified']['prediction_phase_rms_rad'],
                      name) for name, pred in validation_options.items())
    selected = ranking[0][1]
    results = {}
    for split in ('validation', 'test'):
        pred = options(*components_by_split[split])[selected]
        results[split] = split_metrics(pred, packed, split)
    provenance = json.loads((work / 'packed.json').read_text())
    report = {'status': 'offline_modal_imitation', 'selected_on': 'validation',
              'selection': selected, 'validation_ranking': ranking,
              'source_manifest_sha256': provenance['manifest_sha256'],
              'applied_modes': 14, 'fresh_solver_verified': False,
              'test_exploratory': True,
              'test_note': 'The test split was inspected during model development; use fresh unseen setups for final qualification.',
              'splits': results}
    (work / 'ensemble_selection.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def predict(work: Path, trial: Path, config: Path, output: Path):
    selection = json.loads((work / 'ensemble_selection.json').read_text())
    choice = selection['selection']
    meta = json.loads(trial.read_text())
    with np.load(trial.parent / meta['measurements_file']) as measured:
        spatial, scalar = measured_inputs(measured, meta)
        incoming = np.asarray(measured['input__incoming_beam_shape'], np.float64)
        current = np.asarray(measured['input__slm_command_rad'], np.float64)
        waist = float(measured['input__seed_waist_m'])
    normal, ridge, neural = load_models(work, len(scalar))
    cnn, linear = components(spatial[None], scalar[None], normal, ridge, neural)
    if choice == 'zero':
        coeff = np.zeros_like(cnn)
    elif choice == 'ridge':
        coeff = linear
    elif choice.startswith('cnn_'):
        coeff = float(choice.split('_')[1]) * cnn
    elif choice.startswith('blend_'):
        share = float(choice.split('_')[1])
        coeff = share * cnn + (1-share) * linear
    else:
        raise ValueError(f'unknown selected ensemble {choice}')
    coeff = first_fourteen(coeff)[0]
    nominal = json.loads(config.read_text())['nominal']
    grid = Grid2D.square(current.shape[0], float(nominal['field_size_mm']) * 1e-3)
    x, y = grid.mesh
    names, basis, weight = modal_basis(x, y, np.sqrt(incoming), waist,
                                       radial_order=4, residual_grid=0)
    if len(names) != 14:
        raise ValueError('unexpected modal basis')
    step = np.einsum('k,kij->ij', coeff[:14], basis)
    command = np.mod(current + step, 2*np.pi)
    if not np.all(np.isfinite(command)):
        raise ValueError('nonfinite proposed SLM command')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predicted_modal_coefficients_rad=coeff.astype(np.float32),
        proposed_slm_step_rad=step.astype(np.float32),
        proposed_slm_command_rad=command.astype(np.float32))
    report = {'candidate': str(output), 'selection': choice,
              'weighted_step_rms_rad': float(np.sqrt(np.sum(weight * step**2))),
              'fresh_solver_verified': False,
              'trial_sha256': hashlib.sha256(trial.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('training_dir', type=Path)
    p = sub.add_parser('predict')
    p.add_argument('training_dir', type=Path)
    p.add_argument('trial_json', type=Path)
    p.add_argument('config', type=Path)
    p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate(args.training_dir)
    else:
        predict(args.training_dir, args.trial_json, args.config, args.output)


if __name__ == '__main__':
    main()
