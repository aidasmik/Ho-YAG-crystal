"""Train, evaluate and apply the one-shot certified Yb:YAG controller (V3).

fit       label-free training on sampled passive states (bounded, resumable)
evaluate  held-out synthetic end-to-end check: one-shot gain, harm, coverage
calibrate fit per-target certificate margins on independent outcomes
predict   measured trial -> candidate SLM command + certificate (replay-ready)

Nothing here runs the pulsed amplifier/thermal solver. Use
examples/replay_ybyag_nn_candidate.py on a predicted candidate for that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def training_loss(v3, plant, net, batch, truth, weights):
    """Corrected fidelity through the true sampled plant + state likelihood."""
    import torch
    images, context = v3.network_inputs(plant.optics, batch)
    out = net(images, context)
    step = plant.optics.step_map(out['coefficients'], out['residual'])
    F, S, P = plant.outcome(batch, truth, step)
    required = torch.clamp(truth['hold_shape']-.012, min=.95)
    scale = torch.as_tensor(v3.STATE_SCALE, dtype=F.dtype, device=F.device)
    z = truth['theta']/scale
    var = torch.exp(out['state_logvar'])
    # beta-NLL (beta=0.5) keeps the variance head from freezing hard samples.
    nll = (.5*((z-out['state_mean'])**2/var+out['state_logvar'])*var.detach()**.5).mean()
    outcome = ((1-F).mean() + weights['shape']*torch.relu(required-S).square().mean()
               + weights['pump']*torch.relu(v3.CertificateLimits.min_pump_ratio-P/truth['hold_pump']).square().mean())
    loss = outcome+weights['state']*nll+weights['residual']*out['residual'].square().mean()
    return loss, dict(loss=float(loss.detach()), fidelity=float(F.mean()),
                      gain=float((F-truth['hold_fidelity']).mean()),
                      harm=float((F-truth['hold_fidelity'] < -.005).float().mean()),
                      shape_violation=float((S < required).float().mean()), nll=float(nll))


def fit(output, config_path, *, steps, batch, seed, device, members, lr, widen):
    import torch
    from ybyag_control import nn_v3 as v3
    output = Path(output)
    config = json.loads(Path(config_path).read_text())
    spec = v3.Spec.from_config(config)
    state_file = output/'train_state.json'
    state = (json.loads(state_file.read_text()) if state_file.exists() else
             dict(schema=v3.SCHEMA, members={}, spec=spec.to_json(),
                  config_sha256=digest(config_path), config=config))
    if state['config_sha256'] != digest(config_path) or state['spec'] != spec.to_json():
        raise ValueError('existing training directory used a different configuration')
    output.mkdir(parents=True, exist_ok=True)
    plant = v3.SyntheticPlant(spec, config, device, widen=widen)
    weights = dict(shape=50., pump=50., state=.05, residual=1e-3)
    started = time.monotonic()
    for member in range(members):
        folder = output/f'member_{member}'
        folder.mkdir(exist_ok=True)
        record = state['members'].setdefault(str(member), dict(step=0, best=None, history=[]))
        if record['step'] >= steps:
            continue
        torch.manual_seed(seed+1000*member)
        net = v3.OneShotNet(spec.crop).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.05)
        if (folder/'checkpoint.pt').exists():
            ck = torch.load(folder/'checkpoint.pt', map_location=device)
            net.load_state_dict(ck['net'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
        gen = torch.Generator(device=device).manual_seed(seed+1000*member+record['step'])
        # Fixed validation states never enter an optimizer step.
        vgen = torch.Generator(device=device).manual_seed(987654321)
        validation = [plant.sample(batch, vgen) for _ in range(8)]
        net.train()
        while record['step'] < steps:
            b, t = plant.sample(batch, gen)
            loss, stats = training_loss(v3, plant, net, b, t, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            sched.step()
            record['step'] += 1
            if record['step'] % 50 == 0:
                print(json.dumps(dict(member=member, step=record['step'], **stats,
                                      elapsed_s=round(time.monotonic()-started, 1))), flush=True)
            if record['step'] % 500 == 0 or record['step'] == steps:
                net.eval()
                with torch.no_grad():
                    rows = [training_loss(v3, plant, net, vb, vt, weights)[1] for vb, vt in validation]
                net.train()
                summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                summary['step'] = record['step']
                record['history'].append(summary)
                print(json.dumps(dict(member=member, validation=summary)), flush=True)
                if record['best'] is None or summary['loss'] < record['best']['loss']:
                    record['best'] = summary
                    torch.save(net.state_dict(), folder/'weights.pt')
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(),
                                sched=sched.state_dict()), folder/'checkpoint.pt')
                save_json(state_file, state)
        save_json(state_file, state)
    complete = all(state['members'].get(str(m), {}).get('step', 0) >= steps for m in range(members))
    save_json(output/'model.json', dict(schema=v3.SCHEMA, spec=spec.to_json(),
        config_sha256=state['config_sha256'], config=config, members=members,
        training_complete=complete, steps=steps, batch=batch, widen=widen,
        source_sha256=digest(ROOT/'src/ybyag_control/nn_v3.py'),
        validation={m: r['best'] for m, r in state['members'].items()},
        training_data='sampled frozen passive states; no solver-verified labels',
        physical_solver_verified=False, calibrated=False))


def load(folder, device, calibrated=True, precision='double'):
    import torch
    from ybyag_control import nn_v3 as v3
    folder = Path(folder)
    spec_json = json.loads((folder/'model.json').read_text())
    if spec_json['schema'] != v3.SCHEMA:
        raise ValueError('incompatible model schema')
    spec = v3.Spec.from_json(spec_json['spec'])
    nets = []
    for member in sorted(folder.glob('member_*/weights.pt')):
        net = v3.OneShotNet(spec.crop)
        net.load_state_dict(torch.load(member, map_location='cpu'))
        nets.append(net)
    limits = v3.CertificateLimits()
    if calibrated and (folder/'calibration.json').exists():
        calibration = json.loads((folder/'calibration.json').read_text())
        weights = {p.parent.name: digest(p) for p in sorted(folder.glob('member_*/weights.pt'))}
        if calibration['weights_sha256'] != weights:
            raise ValueError('stale calibration after weight changes; recalibrate')
        values = calibration['limits']
        limits = v3.CertificateLimits(**{**values, **{k: tuple(values[k]) for k in (
            'margin_base', 'margin_slope', 'shape_margin', 'planned_margin_base',
            'planned_margin_slope', 'planned_shape_margin') if k in values}})
    return spec_json, v3.OneShotController(spec, nets, device, limits, precision=precision)


def synthetic_cases(controller, plant, *, samples, seed, device, cache, allow_uncalibrated):
    """Run the full pipeline on fixed-seed synthetic states and score every
    candidate against that state's hidden truth. Rows are cached (resumable)."""
    import torch
    from ybyag_control import nn_v3 as v3
    cache = Path(cache)
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            if row['seed'] == seed:
                done[row['index']] = row
    gen = torch.Generator(device=device).manual_seed(seed)
    chunk = 16
    rows = []
    for first in range(0, samples, chunk):
        batch, truth = plant.sample(min(chunk, samples-first), gen)
        for j in range(len(batch['amp'])):
            i = first+j
            if i in done:
                rows.append(done[i])
                continue
            sub = {k: v[j:j+1] for k, v in batch.items()}
            subt = {k: (None if v is None else v[j:j+1]) for k, v in truth.items()}
            started = time.monotonic()
            result = controller.propose(plant.trial(batch, j), seed=i,
                                        allow_uncalibrated=allow_uncalibrated)
            elapsed = time.monotonic()-started
            cert = result['certificate']
            hold = float(subt['hold_fidelity'])
            shape_req = max(.95, float(subt['hold_shape'])-.012)
            candidates = {}
            for name, step in result['candidate_steps'].items():
                F, S, P = plant.outcome(sub, subt, torch.as_tensor(
                    step, device=device, dtype=plant.optics.real)[None])
                ratio = float(P/subt['hold_pump'])
                candidates[name] = dict(cert['candidates'][name], true_gain=float(F)-hold,
                    true_fidelity=float(F), true_shape=float(S), true_pump_ratio=ratio,
                    true_shape_slack=float(S)-shape_req,
                    guard_ok=bool(float(S) >= shape_req and ratio >= controller.limits.min_pump_ratio))
            row = dict(seed=seed, index=i, family=v3.TARGETS[int(batch['family'][j])],
                       decision=cert['decision'], status=cert['status'], seconds=elapsed,
                       hold_fidelity=hold, model_consistent=cert['model_consistent'],
                       relative_residual=cert['relative_amplitude_residual'],
                       reduced_chi2=cert['reduced_chi2'], candidates=candidates)
            with cache.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(dict(index=i, family=row['family'], decision=row['decision'],
                                  hold=round(hold, 4),
                                  chosen_gain=round(candidates[row['decision']]['true_gain'], 4),
                                  network_gain=round(candidates['network']['true_gain'], 4),
                                  planned_gain=round(candidates['planned']['true_gain'], 4),
                                  seconds=round(elapsed, 1))), flush=True)
            rows.append(row)
    return rows


def summarize(rows):
    from ybyag_control import nn_v3 as v3
    chosen = [r['candidates'][r['decision']] for r in rows]
    applied = [c for r, c in zip(rows, chosen) if r['decision'] != 'hold']
    summary = dict(samples=len(rows), applied_fraction=len(applied)/len(rows),
        mean_gain_all=float(np.mean([c['true_gain'] for c in chosen])),
        mean_gain_applied=float(np.mean([c['true_gain'] for c in applied])) if applied else None,
        harmful_applied=int(sum(c['true_gain'] < -.005 for c in applied)),
        guard_violations_applied=int(sum(not c['guard_ok'] for c in applied)),
        certified_coverage_applied=(float(np.mean([c['true_gain'] >= c['gain_certified'] for c in applied]))
                                    if applied else None),
        task_success_fraction=float(np.mean([c['true_fidelity'] >= .9 and c['guard_ok'] for c in chosen])),
        hold_task_success_fraction=float(np.mean([r['hold_fidelity'] >= .9 for r in rows])),
        network_only_mean_gain=float(np.mean([r['candidates']['network']['true_gain'] for r in rows])),
        network_only_harmful=int(sum(r['candidates']['network']['true_gain'] < -.005 for r in rows)),
        planned_full_step_mean_gain=float(np.mean([r['candidates']['planned']['true_gain'] for r in rows])),
        planned_full_step_harmful=int(sum(r['candidates']['planned']['true_gain'] < -.005 for r in rows)),
        released_by_candidate=dict(__import__('collections').Counter(
            'planned' if r['decision'].startswith('planned') else r['decision']
            for r in rows if r['decision'] != 'hold')),
        mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        scope='synthetic frozen passive plant; not a pulsed amplifier/thermal solver result')
    for family in v3.TARGETS:
        sel = [(r, c) for r, c in zip(rows, chosen) if r['family'] == family]
        if sel:
            summary[family] = dict(n=len(sel), mean_gain=float(np.mean([c['true_gain'] for _, c in sel])),
                applied=sum(r['decision'] != 'hold' for r, _ in sel),
                network_only_mean_gain=float(np.mean([r['candidates']['network']['true_gain'] for r, _ in sel])),
                planned_mean_gain=float(np.mean([r['candidates']['planned']['true_gain'] for r, _ in sel])))
    return summary


def evaluate(folder, output, *, samples, seed, device, widen, allow_uncalibrated, precision='double'):
    """Synthetic held-out states the network never trained on (fixed seed)."""
    from dataclasses import asdict
    from ybyag_control import nn_v3 as v3
    spec_json, controller = load(folder, device, precision=precision)
    plant = v3.SyntheticPlant(controller.spec, spec_json['config'], device, widen=widen)
    output = Path(output)
    rows = synthetic_cases(controller, plant, samples=samples, seed=seed, device=device,
                           cache=output.with_suffix('.rows.jsonl'),
                           allow_uncalibrated=allow_uncalibrated)
    summary = dict(summarize(rows), seed=seed, widen=widen, limits=asdict(controller.limits))
    save_json(output, summary)
    print(json.dumps(summary, indent=2))


def calibrate(folder, *, samples, seed, device, widen, coverage, replays=None, precision='double'):
    """Fit per-target certificate margins on independent outcomes.

    Full-solver replay reports (``REPLAYS/reports/*.json`` paired with the
    ``REPLAYS/candidates/*.json`` certificates written by ``predict``) replace
    the synthetic rows for any target that has at least 20 of them.
    """
    from dataclasses import asdict, replace
    from ybyag_control import nn_v3 as v3
    folder = Path(folder)
    spec_json, controller = load(folder, device, calibrated=False, precision=precision)
    plant = v3.SyntheticPlant(controller.spec, spec_json['config'], device, widen=widen)
    rows = synthetic_cases(controller, plant, samples=samples, seed=seed, device=device,
                           cache=folder/'calibration_rows.jsonl', allow_uncalibrated=True)
    # Consistency is re-evaluated with the current gate, so cached rows stay valid.
    lim = controller.limits
    flat = [dict(family=r['family'], candidate=name, **c) for r in rows
            if r['relative_residual'] <= lim.max_relative_residual
            and r['reduced_chi2'] <= lim.max_reduced_chi2
            for name, c in r['candidates'].items() if name != 'hold']
    sources = {f: 'synthetic_passive' for f in v3.TARGETS}
    if replays:
        solver = []
        for report_path in sorted((Path(replays)/'reports').glob('*.json')):
            report = json.loads(report_path.read_text())
            cert = json.loads((Path(replays)/'candidates'/report_path.name).read_text())
            if cert.get('schema') != v3.SCHEMA or cert['decision'] == 'hold':
                continue
            if report.get('candidate_sha256') != digest((Path(replays)/'candidates'/report_path.name).with_suffix('.npz')):
                raise ValueError(f'replay report does not match its candidate: {report_path}')
            chosen = cert['candidates'][cert['decision']]
            required = max(cert['limits']['min_shape'],
                           report['baseline']['shape']-cert['limits']['max_shape_drop'])
            solver.append(dict(family=report['target'], gain_lower=chosen['gain_lower'],
                               gain_mean=chosen['gain_mean'], shape_slack=chosen['shape_slack'],
                               pump_ok=True, true_gain=report['fidelity_gain'],
                               true_shape_slack=report['candidate']['shape']-required,
                               guard_ok=bool(report['safe_shape_energy'])))
        for family in v3.TARGETS:
            if sum(r['family'] == family for r in solver) >= 20:
                flat = [r for r in flat if r['family'] != family]+[r for r in solver if r['family'] == family]
                sources[family] = 'full_solver_replay'
    # Separate empirical margins for the network step and re-planned steps.
    network = [r for r in flat if not r.get('candidate', 'network').startswith('planned')]
    planned = [r for r in flat if r.get('candidate', '').startswith('planned')]
    base, slope, shape, report = v3.fit_margins(network, target_coverage=coverage,
                                                min_gain=controller.limits.min_gain)
    pbase, pslope, pshape, preport = v3.fit_margins(planned, target_coverage=coverage,
                                                    min_gain=controller.limits.min_gain)
    limits = replace(controller.limits, margin_base=base, margin_slope=slope, shape_margin=shape,
                     planned_margin_base=pbase, planned_margin_slope=pslope,
                     planned_shape_margin=pshape,
                     calibrated=all(report[f]['calibrated'] for f in v3.TARGETS))
    for family in v3.TARGETS:
        report[family]['source'] = sources[family]
        report[family]['planned'] = preport[family]
    weights = {p.parent.name: digest(p) for p in sorted(folder.glob('member_*/weights.pt'))}
    save_json(folder/'calibration.json', dict(schema=v3.SCHEMA, limits={**asdict(limits),
        'margin_base': list(base), 'margin_slope': list(slope), 'shape_margin': list(shape),
        'planned_margin_base': list(pbase), 'planned_margin_slope': list(pslope),
        'planned_shape_margin': list(pshape)},
        families=report,
        target_coverage=coverage, samples=samples, seed=seed, widen=widen, weights_sha256=weights,
        precision=precision,
        scope=('Coverage is empirical on the listed source. Synthetic calibration does not '
               'certify the pulsed amplifier; replace it with full-solver replay rows.')))
    print(json.dumps(report, indent=2))


def predict(folder, trial, output, *, device, seed, proposal_only=False):
    from ybyag_control import nn_v3 as v3
    trial, output = Path(trial), Path(output)
    spec_json, controller = load(folder, device)
    meta = json.loads(trial.read_text())
    with np.load(trial.parent/meta['measurements_file']) as data:
        measured = dict(data)
    inputs = v3.TrialInputs.from_trial(measured, meta, controller.spec)
    result = controller.propose(inputs, seed=seed, allow_uncalibrated=proposal_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, proposed_slm_command_rad=result['command'].astype(np.float32),
        proposed_slm_step_rad=result['step'].astype(np.float32),
        predicted_modal_coefficients_rad=result['coefficients'].astype(np.float32),
        predicted_residual_rad=result['residual'].astype(np.float32))
    report = dict(result['certificate'], trial=str(trial), trial_sha256=digest(trial),
        model_sha256={p.parent.name: digest(p) for p in sorted(Path(folder).glob('member_*/weights.pt'))},
        source_config_sha256=spec_json['config_sha256'], fresh_solver_verified=False,
        proposal_only=proposal_only)
    if proposal_only:
        report['status'] += '_offline_proposal_only_not_approved_for_application'
    save_json(output.with_suffix('.json'), report)
    print(json.dumps({k: report[k] for k in ('decision', 'status', 'model_consistent',
                                              'relative_amplitude_residual', 'reduced_chi2')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('fit')
    p.add_argument('output', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'config/ybyag_nn_dataset.json')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--members', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--widen', type=float, default=1.25)
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('evaluate')
    p.add_argument('model', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=424242)
    p.add_argument('--widen', type=float, default=1.)
    p.add_argument('--device', default='cuda')
    p.add_argument('--allow-uncalibrated', action='store_true')
    p.add_argument('--precision', choices=('single', 'double'), default='double')
    p = sub.add_parser('calibrate')
    p.add_argument('model', type=Path)
    p.add_argument('--samples', type=int, default=160)
    p.add_argument('--seed', type=int, default=13579)
    p.add_argument('--widen', type=float, default=1.)
    p.add_argument('--coverage', type=float, default=.95)
    p.add_argument('--replays', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=('single', 'double'), default='double')
    p = sub.add_parser('predict')
    p.add_argument('model', type=Path)
    p.add_argument('trial', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--proposal-only', action='store_true',
                   help='export the best candidate for offline solver replay even if uncalibrated')
    for name in ('fit', 'evaluate', 'calibrate'):
        sub.choices[name].add_argument('--seconds', type=int, default=900,
                                       help='supervised wall-time limit for this attempt')
        sub.choices[name].add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.cmd == 'fit' and (not 1 <= a.steps <= 10**6 or not 1 <= a.members <= 8):
        parser.error('steps must be 1..1e6 and members 1..8')
    if a.cmd != 'predict' and not a.worker:
        # Every expensive command runs as an owned worker under the shared
        # local budget ledger; interrupted runs resume from their caches.
        from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
        if not 1 <= a.seconds <= 43200:
            parser.error('seconds must be 1..43200')
        target = a.output if a.cmd != 'calibrate' else a.model/'calibration'
        result = run_bounded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
            cwd=Path.cwd(), log_path=target.parent/(target.name+f'.{a.cmd}.log'),
            summary_path=target.parent/(target.name+f'.{a.cmd}.runtime.json'),
            ledger=BudgetLedger(ROOT/'.local_runtime/budget.json', Limits(case_seconds=max(900, a.seconds))),
            label=f'ybyag_controller_v3_{a.cmd}', configured_seconds=a.seconds, category='coupled',
            env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        print(json.dumps(result, indent=2))
        if result.get('status') == 'timed_out':
            print('Time limit reached; rerun the same command to resume.')
        elif result.get('status') != 'completed':
            raise SystemExit(1)
    elif a.cmd == 'fit':
        fit(a.output, a.config, steps=a.steps, batch=a.batch, seed=a.seed, device=a.device,
            members=a.members, lr=a.lr, widen=a.widen)
    elif a.cmd == 'evaluate':
        evaluate(a.model, a.output, samples=a.samples, seed=a.seed, device=a.device, widen=a.widen,
                 allow_uncalibrated=a.allow_uncalibrated, precision=a.precision)
    elif a.cmd == 'calibrate':
        calibrate(a.model, samples=a.samples, seed=a.seed, device=a.device, widen=a.widen,
                  coverage=a.coverage, replays=a.replays, precision=a.precision)
    else:
        predict(a.model, a.trial, a.output, device=a.device, seed=a.seed,
                proposal_only=a.proposal_only)

if __name__ == '__main__':
    main()
