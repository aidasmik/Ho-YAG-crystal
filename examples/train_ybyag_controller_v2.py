"""Pack, train, calibrate, and propose with the initial multiscale Yb:YAG controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import hashlib
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from ybyag_control.nn_v2 import (SCHEMA, MODES, INPUTS, TARGETS, build_models,
                                features, pupil, command, choose, digest)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def calibration_setup(setup):
    """Reserve entire validation setups before fitting/early stopping."""
    return int(hashlib.sha256(setup.encode()).hexdigest()[:8], 16) % 3 != 0


def source_trials(index):
    return sorted((r['split'],r['setup'],r['trial_sha256']) for r in index['records'])


def pack(dataset, config_path, output, replays=None):
    """Per-trial cache, preserving setups; optionally import exact replayed actions."""
    dataset, output = Path(dataset).resolve(), Path(output)
    manifest = json.loads((dataset/'manifest.json').read_text())
    config = json.loads(Path(config_path).read_text())
    if manifest['schema'] != 'ybyag_nn_closed_loop_v3':
        raise ValueError('v3 closed-loop dataset required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty cache directory to preserve provenance')
    records, owners, replay_map = [], {}, {}
    if replays:
        for p in sorted((Path(replays)/'reports').glob('*.json')):
            r = json.loads(p.read_text())
            replay_map.setdefault(str(Path(r['trial']).resolve()), []).append((p, r))
    # Validate all split ownership before writing caches.
    for split in ('train', 'validation', 'test'):
        for name in manifest['splits'][split]:
            m = json.loads((dataset/name).read_text())
            setup = m['setup_id']
            if m['split'] != split or owners.get(setup, split) != split:
                raise ValueError('setup leakage or incorrect split metadata')
            owners[setup] = split
    for split in ('train', 'validation', 'test'):
        seen = set()
        for name in manifest['splits'][split]:
            path = dataset/name
            meta = json.loads(path.read_text())
            ident = (meta['setup_id'], int(meta['point_index']))
            if ident in seen:
                raise ValueError('duplicate trial')
            seen.add(ident)
            with np.load(path.parent/meta['measurements_file']) as m:
                measured = dict(m)
            f = features(measured, meta, config)
            baseline = np.array([meta['coherent_fidelity']['uncorrected'],
                                 meta['shape_overlap']['uncorrected']], np.float32)
            actions, outcomes, origins = [np.zeros(MODES)], [np.zeros(3)], ['hold']
            action_models = [None]
            label = np.zeros(MODES, np.float32)
            verified = False
            with np.load(path.parent/meta['truth_file']) as truth:
                raw = np.asarray(truth['truth__modal_coefficients_rad'], float)
                if bool(truth['truth__correction_label_valid']) != meta['correction_label_valid']:
                    raise ValueError('teacher validity mismatch')
                if meta['correction_label_valid'] and raw.shape == (MODES,) and np.all(np.isfinite(raw)):
                    _, names, _, weight = pupil(measured, config)
                    if names != meta['modal_basis_names']:
                        raise ValueError('modal basis mismatch')
                    actual, _ = command(measured, config, raw)
                    saved = truth['truth__verified_slm_command_rad']
                    mismatch = np.sum(weight*abs(np.exp(1j*actual)-np.exp(1j*saved))**2)
                    if mismatch > 1e-8:
                        raise ValueError(f'teacher action does not reproduce saved command: {path}')
                    label, verified = raw.astype(np.float32), True
                    if np.linalg.norm(raw) > 1e-8:
                        actions.append(raw)
                        outcomes.append([meta['coherent_fidelity']['corrected']-baseline[0],
                                         meta['shape_overlap']['corrected']-baseline[1],
                                         np.log(meta['corrected_energy_fraction'])])
                        origins.append('teacher')
                        action_models.append(None)
            for report_path, replay in replay_map.get(str(path.resolve()), []):
                cp = report_path.parent.parent/'candidates'/report_path.with_suffix('.npz').name
                provenance = json.loads(cp.with_suffix('.json').read_text())
                if replay.get('candidate_sha256') != digest(cp) or replay.get('trial_sha256') != digest(path):
                    raise ValueError('replay lacks exact candidate/trial fingerprints; rerun verification')
                if provenance['trial_sha256'] != digest(path):
                    raise ValueError('replay trial fingerprint mismatch')
                if replay['source_config_sha256'] != digest(config_path) or replay['baseline_reproduction_relative_l2'] > 1e-4:
                    raise ValueError('replay provenance or baseline mismatch')
                if not np.allclose(baseline, [replay['baseline']['fidelity'],replay['baseline']['shape']], atol=1e-5, rtol=0):
                    raise ValueError('replay from different physical state')
                with np.load(cp) as candidate:
                    a = np.asarray(candidate['predicted_modal_coefficients_rad'], float)
                    if a.size < MODES or np.any(abs(a[MODES:]) > 1e-8):
                        raise ValueError('replay uses unsupported higher modes')
                    a = a[:MODES]
                    cmd, _ = command(measured, config, a)
                    if np.max(abs(np.exp(1j*cmd)-np.exp(1j*candidate['proposed_slm_command_rad']))) > 1e-4:
                        raise ValueError('replay command/coefficients mismatch')
                actions.append(a)
                outcomes.append([replay['fidelity_gain'], replay['shape_change'],
                                 np.log(replay['candidate']['energy_fraction'])])
                origins.append('replay')
                action_models.append(provenance.get('proposal_sha256'))
            dest = output/split/f'{meta["setup_id"]}_{meta["point_index"]:03d}.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            arrays = dict(**f, label=label, verified=np.array(verified), baseline=baseline,
                          actions=np.asarray(actions, np.float32), outcomes=np.asarray(outcomes, np.float32))
            if not all(np.all(np.isfinite(a)) for a in arrays.values()):
                raise ValueError('nonfinite training record')
            np.savez_compressed(dest, **arrays)
            records.append(dict(file=str(dest.relative_to(output)), split=split,
                setup=meta['setup_id'], target=meta['selected_target'],
                doping=meta['physical_parameters']['yb_at_percent'], verified=verified,
                origins=origins, action_models=action_models, trial=str(path), trial_sha256=digest(path)))
            if len(seen) % 64 == 0:
                print(f'Packing {split}: {len(seen)} trials', flush=True)
        print(f'Packed {split}: {len(seen)} trials', flush=True)
    save_json(output/'index.json', dict(schema=SCHEMA, config=config,
        config_sha256=digest(config_path), manifest_sha256=digest(dataset/'manifest.json'),
        dataset=str(dataset), registration='nominal unless measured affine supplied', records=records))


def inputs(record, mean, std):
    return [np.asarray(record[k], np.float32)[None] if k != 'context' else
            ((np.asarray(record[k], np.float32)-mean)/std)[None] for k in INPUTS]


def initialize_tf():
    # The bundled CPU build segfaulted in oneDNN on the first real batch.
    # These flags must be set before importing TensorFlow.
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('YBYAG_TF_THREADS', '4')))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(20260930)
    return tf


def fit(cache, output, epochs=30, patience=5):
    tf = initialize_tf()
    cache, output = Path(cache), Path(output)
    index = json.loads((cache/'index.json').read_text())
    train = [r for r in index['records'] if r['split']=='train']
    validation = [r for r in index['records'] if r['split']=='validation' and not calibration_setup(r['setup'])]
    if not train or not validation or not any(r['verified'] for r in train):
        raise ValueError('training and independent validation with verified labels required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty model directory')
    output.mkdir(parents=True, exist_ok=True)
    contexts = []
    for r in train:
        with np.load(cache/r['file']) as d:
            contexts.append(d['context'])
    contexts = np.array(contexts)
    mean = contexts.mean(0)
    std = np.maximum(contexts.std(0), np.maximum(abs(mean)*1e-6, 1e-12))
    np.savez(output/'normalization.npz',mean=mean,std=std)
    save_json(output/'model.json',dict(schema=SCHEMA,trained=False,status='training_incomplete'))
    model, critic = build_models(len(mean))
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4, clipnorm=1.),
                  loss=['huber','mse'], loss_weights=[16.,1.])
    critic.compile(optimizer=tf.keras.optimizers.Adam(2e-4, clipnorm=1.), loss='mse')
    best, stale, history = float('inf'), 0, []
    rng = np.random.default_rng(20260930)
    started = time.monotonic()
    for epoch in range(epochs):
        for batch_index, i in enumerate(rng.permutation(len(train)), 1):
            r = train[i]
            with np.load(cache/r['file']) as d:
                x = inputs(d, mean, std)
                model.train_on_batch(x, [d['label'][None],d['baseline'][None]],
                    sample_weight=[np.array([float(d['verified'])]),np.ones(1)])
                n = len(d['actions'])
                critic.train_on_batch([*[np.repeat(v,n,axis=0) for v in x],d['actions']],d['outcomes'])
            if batch_index == 1 or batch_index % 32 == 0:
                progress = dict(status='training', epoch=epoch+1, max_epochs=epochs,
                    batch=batch_index, batches=len(train), elapsed_s=time.monotonic()-started)
                save_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        errors = []
        for r in validation:
            with np.load(cache/r['file']) as d:
                x = inputs(d, mean, std)
                c, b = [v.numpy()[0] for v in model(x, training=False)]
                n = len(d['actions'])
                o = critic([*[np.repeat(v,n,axis=0) for v in x],d['actions']],training=False).numpy()
                errors.append(16*float(d['verified'])*np.mean((c-d['label'])**2)
                              + np.mean((b-d['baseline'])**2)+np.mean((o-d['outcomes'])**2))
        loss = float(np.mean(errors))
        if not np.isfinite(loss):
            raise RuntimeError('nonfinite validation loss')
        history.append(dict(epoch=epoch+1,validation_loss=loss))
        print(json.dumps(history[-1]),flush=True)
        if loss < best:
            best, stale = loss, 0
            model.save_weights(output/'proposal.weights.h5')
            critic.save_weights(output/'critic.weights.h5')
            save_json(output/'model.json',dict(schema=SCHEMA,cache_index_sha256=digest(cache/'index.json'),
                config_sha256=index['config_sha256'],config=index['config'],context_width=len(mean),
                best_validation_loss=best,history=history,trained=True,calibrated=False,
                source_sha256=digest(ROOT/'src/ybyag_control/nn_v2.py'),
                tensorflow_cpu_backend='onednn_disabled',
                source_trials=source_trials(index),mode_count=MODES,test_used=False,
                physical_solver_verified=False,training_complete=False,status='best_checkpoint'))
        else:
            stale += 1
        if stale >= patience:
            break
    np.savez(output/'normalization.npz',mean=mean,std=std)
    save_json(output/'model.json',dict(schema=SCHEMA,cache_index_sha256=digest(cache/'index.json'),
        config_sha256=index['config_sha256'],config=index['config'],context_width=len(mean),
        best_validation_loss=best,history=history,trained=True,calibrated=False,
        source_sha256=digest(ROOT/'src/ybyag_control/nn_v2.py'),
        tensorflow_cpu_backend='onednn_disabled',
        source_trials=source_trials(index),
        mode_count=MODES,test_used=False,physical_solver_verified=False,training_complete=True))
    save_json(output/'progress.json',dict(status='completed',epochs_completed=len(history),
        best_validation_loss=best,elapsed_s=time.monotonic()-started))


def load_model(folder):
    folder = Path(folder)
    spec = json.loads((folder/'model.json').read_text())
    if spec['schema'] != SCHEMA or not spec['trained']:
        raise ValueError('incompatible/untrained model')
    p,c = build_models(spec['context_width'])
    p.load_weights(folder/'proposal.weights.h5')
    c.load_weights(folder/'critic.weights.h5')
    with np.load(folder/'normalization.npz') as n:
        mean,std = n['mean'],n['std']
    return spec,p,c,mean,std


def calibrate(cache, folder):
    initialize_tf()
    cache,folder = Path(cache),Path(folder)
    spec,p,c,mean,std = load_model(folder)
    index = json.loads((cache/'index.json').read_text())
    if (spec['config_sha256'] != index['config_sha256'] or
        spec['source_trials'] != [list(v) for v in source_trials(index)]):
        raise ValueError('calibration sources differ from fitted dataset')
    report = dict(schema=SCHEMA,proposal_sha256=digest(folder/'proposal.weights.h5'),
                  critic_sha256=digest(folder/'critic.weights.h5'),families={})
    for family in TARGETS:
        setup_errors, negative_setups, sampled_actions = {},set(),[]
        for r in index['records']:
            if r['split'] != 'validation' or r['target'] != family or not calibration_setup(r['setup']):
                continue
            with np.load(cache/r['file']) as d:
                ids = [i for i,s in enumerate(r['origins']) if s=='replay'
                       and r['action_models'][i] == report['proposal_sha256']
                       and np.linalg.norm(d['actions'][i]) > 1e-6]
                if not ids:
                    continue
                x = inputs(d,mean,std)
                _,b = p(x,training=False)
                predicted = c([*[np.repeat(v,len(ids),axis=0) for v in x],d['actions'][ids]],training=False).numpy()
                sampled_actions.extend(d['actions'][ids])
                err = np.r_[abs(b.numpy()[0]-d['baseline']),
                            abs(predicted-d['outcomes'][ids]).max(axis=0)]
                setup_errors[r['setup']] = np.maximum(setup_errors.get(r['setup'], np.zeros(5)),err)
                if np.any(d['outcomes'][ids,0] < -.005):
                    negative_setups.add(r['setup'])
        n = len(setup_errors)
        ready = n >= 20 and len(negative_setups) >= 5
        # Simultaneous conservative empirical envelope over observed validation
        # errors. This does not guarantee coverage on arbitrary new actions.
        bounds = np.max(list(setup_errors.values()),axis=0).tolist() if n else [1,1,1,1,10]
        report['families'][family] = dict(calibrated=ready,bounds=bounds,
            action_min=np.min(sampled_actions,axis=0).tolist() if sampled_actions else [0.]*MODES,
            action_max=np.max(sampled_actions,axis=0).tolist() if sampled_actions else [0.]*MODES,
            validation_setups=n,harmful_action_setups=len(negative_setups),
            reason='empirical_validation_envelope' if ready else 'insufficient_independent_counterfactuals')
    save_json(folder/'calibration.json',report)
    print(json.dumps(report,indent=2))


def predict(folder, trial, output, proposal_only=False):
    initialize_tf()
    folder,trial,output = Path(folder),Path(trial),Path(output)
    spec,p,c,mean,std = load_model(folder)
    meta = json.loads(trial.read_text())
    with np.load(trial.parent/meta['measurements_file']) as d:
        measured = dict(d)
    f = features(measured,meta,spec['config'])
    x = inputs(f,mean,std)
    coeff,baseline = [a.numpy()[0] for a in p(x,training=False)]
    actions = np.stack([np.zeros(MODES),coeff,.5*coeff])
    deltas = c([*[np.repeat(v,3,axis=0) for v in x],actions],training=False).numpy()
    calibration = dict(calibrated=False,bounds=[1,1,1,1,10])
    if (folder/'calibration.json').exists():
        cr = json.loads((folder/'calibration.json').read_text())
        if cr['proposal_sha256'] != digest(folder/'proposal.weights.h5') or cr['critic_sha256'] != digest(folder/'critic.weights.h5'):
            raise ValueError('stale calibration after weight changes')
        calibration = cr['families'][meta['selected_target']]
    supported = np.all((actions >= np.asarray(calibration.get('action_min',[0.]*MODES))) &
                       (actions <= np.asarray(calibration.get('action_max',[0.]*MODES))),axis=1)
    context_supported = bool(np.all(abs(x[3]) <= 6))
    selected,status = choose(actions,deltas,baseline,calibration['bounds'],
        calibrated=calibration['calibrated'] and context_supported,supported=supported)
    if proposal_only:
        selected,status = 1,'offline_proposal_only_not_approved_for_application'
    chosen,step = command(measured,spec['config'],actions[selected])
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,predicted_modal_coefficients_rad=actions[selected],
        proposed_slm_step_rad=step.astype(np.float32),proposed_slm_command_rad=chosen.astype(np.float32))
    report = dict(status=status,trial_sha256=digest(trial),selected_candidate=selected,
        proposal_sha256=digest(folder/'proposal.weights.h5'),critic_sha256=digest(folder/'critic.weights.h5'),
        source_config_sha256=spec['config_sha256'],context_supported=context_supported,
        predicted_baseline=baseline.tolist(),predicted_deltas=deltas.tolist(),
        calibrated=calibration['calibrated'],fresh_solver_verified=False,
        requires_measured_verification=selected != 0)
    save_json(output.with_suffix('.json'),report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('pack');p.add_argument('dataset',type=Path);p.add_argument('config',type=Path);p.add_argument('output',type=Path);p.add_argument('--replays',type=Path)
    p=sub.add_parser('fit');p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--epochs',type=int,default=30);p.add_argument('--seconds',type=int,default=900);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p=sub.add_parser('calibrate');p.add_argument('cache',type=Path);p.add_argument('model',type=Path)
    p=sub.add_parser('predict');p.add_argument('model',type=Path);p.add_argument('trial',type=Path);p.add_argument('output',type=Path);p.add_argument('--proposal-only',action='store_true')
    a=parser.parse_args()
    if a.cmd=='pack':pack(a.dataset,a.config,a.output,a.replays)
    elif a.cmd=='fit':
        if not 1 <= a.epochs <= 200:parser.error('epochs must be 1..200')
        if a.worker:
            fit(a.cache,a.output,a.epochs)
        else:
            from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
            if not 1 <= a.seconds <= 7200:parser.error('seconds must be 1..7200')
            result=run_bounded([sys.executable,str(Path(__file__).resolve()),'fit',
                str(a.cache.resolve()),str(a.output.resolve()),'--epochs',str(a.epochs),'--worker'],
                cwd=ROOT,log_path=a.output.parent/(a.output.name+'.training.log'),
                summary_path=a.output.parent/(a.output.name+'.runtime.json'),
                ledger=BudgetLedger(ROOT/'.local_runtime/budget.json',Limits(enforce_cumulative_budget=True)),
                label='ybyag_controller_v2_training',configured_seconds=a.seconds,category='coupled')
            print(json.dumps(result,indent=2))
            if result.get('status') != 'completed':raise SystemExit(1)
    elif a.cmd=='calibrate':calibrate(a.cache,a.model)
    else:predict(a.model,a.trial,a.output,a.proposal_only)


if __name__=='__main__':main()
