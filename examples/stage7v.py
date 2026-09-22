"""Run/review Stage 7V. Each case has an isolated deadline and verifiable cache.

python examples/stage7v.py plan --profile full
python examples/stage7v.py run --profile pilot --cases base256 opt384 opt512
python examples/stage7v.py report --profile pilot

The report command succeeds when a report is written, not when it qualifies the
model. --require-qualified is an explicit failing gate for unvalidated data.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from hoyag.validation_campaign import (make_plan,execute_case,compile_report,
          write_json,serializable,case_identity,verify_saved,source_digest)
ROOT=Path(__file__).resolve().parents[1]


def terminated(folder,case,status,reason):
    # Retain progress but never reuse any possibly stale/partial result as a pass.
    write_json(folder/'summary.json',{'status':status,'case':case,'reason':reason})


def run_isolated(case, output, *, timeout_override=None):
    folder=output/case['id'];folder.mkdir(parents=True,exist_ok=True)
    config=folder/'case.json';write_json(config,case)
    limit=case['timeout_s'] if timeout_override is None else timeout_override
    if not np.isfinite(limit) or limit<=0:raise ValueError('timeout must be positive')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    with (folder/'run.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker',
                  '--case-file',str(config),'--output',str(output)],
                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            terminated(folder,case,'TIMEOUT',f'per-case deadline {limit}s reached; no result qualified')
            return False
    if code:
        if not (folder/'summary.json').exists():terminated(folder,case,'ERROR',f'worker exit {code}')
        print((folder/'run.log').read_text()[-6000:],flush=True)
        return False
    summary=verify_saved(folder)
    if summary is None:
        terminated(folder,case,'ERROR','worker did not produce a verifiable state')
        return False
    print('STAGE7V_RESULT '+json.dumps({k:serializable(summary[k]) for k in
           ('case','status','scalars','mode_power_W','elapsed_s','mirror_sampling','edge_power_fraction')},allow_nan=False),flush=True)
    columns=('charge','double_pass_power_gain','vector_overlap_with_input','input_azimuthal_purity','output_azimuthal_purity','cross_polarized_fraction')
    print('STAGE7V_PROBE_TABLE '+json.dumps({'case':case['id'],'probes':[{k:p[k] for k in columns} for p in summary['probes']]},allow_nan=False),flush=True)
    return summary['status']=='COMPLETED'


def plot_report(report, output):
    import matplotlib.pyplot as plt
    from hoyag.validation_campaign import verify_saved
    out=Path(output)/'figures';out.mkdir(exist_ok=True);paths=[]
    def save(name):
        plt.tight_layout()
        for ext in ('png','svg'):
            file=out/f'{name}.{ext}';plt.savefig(file,dpi=180);paths.append(file)
        plt.close()
    completed={c['id']:c for c in report['cases'] if c['status']=='COMPLETED'}
    optical=[(n,completed[id]) for n,id in [(256,'base256'),(384,'opt384'),(512,'opt512')] if id in completed]
    if optical:
        plt.figure(figsize=(7,4.5));plt.plot([x[0] for x in optical],[x[1]['scalars']['output_W'] for x in optical],'o-')
        plt.xlabel('Optical grid points per axis');plt.ylabel('Cycle-averaged laser output (W)')
        plt.title('Optical refinement — other grids held fixed');save('optical_output_refinement')
    group=next((g for g in report['groups'] if g['id']=='optical'),None)
    if group:
        pairs=[p for p in group['pairs'] if 'differences' in p]
        if pairs:
            plt.figure(figsize=(7,4.5));plt.plot(range(len(pairs)),[p['differences']['opd']['rms_difference_m']*1e9 for p in pairs],'o-')
            plt.axhline(report['thresholds']['opd_rms_m']*1e9,linestyle='--',label='Proposed 2 nm acceptance limit')
            plt.xticks(range(len(pairs)),[p['a']+' → '+p['b'] for p in pairs]);plt.ylabel('Weighted OPD difference, piston removed (nm)')
            plt.title('Physical-coordinate wavefront refinement');plt.legend();save('opd_refinement')
    mounting=[id for id in ['mount_base','contact_low','contact_high','shear_soft','roller','coolant_low'] if id in completed]
    if mounting:
        plt.figure(figsize=(8,4.5));plt.bar(mounting,[completed[id]['scalars']['disk_peak_K']-293.15 for id in mounting])
        plt.xticks(rotation=25,ha='right');plt.ylabel('Maximum crystal temperature rise (K)')
        plt.title('Mount sensitivity at fixed heat — not new laser equilibria');save('mount_temperature_sensitivity')
    baseline=verify_saved(Path(output)/'base256')
    if baseline and baseline['status']=='COMPLETED':
        probes=baseline['probes']
        plt.figure(figsize=(7,4.5));plt.plot([p['charge'] for p in probes],[p['double_pass_power_gain'] for p in probes],'o-')
        plt.axhline(1,linestyle='--');plt.xlabel('Injected LG azimuthal order');plt.ylabel('Reflected double-pass weak-probe gain')
        plt.title('Seeded vortex amplification in frozen hot-disk state');save('seeded_vortex_gain')
        plt.figure(figsize=(7,4.5));plt.plot([p['charge'] for p in probes],[p['output_azimuthal_purity'] for p in probes],'o-',label='Output')
        plt.plot([p['charge'] for p in probes],[p['input_azimuthal_purity'] for p in probes],'s--',label='Sampled input')
        plt.xlabel('Injected LG azimuthal order');plt.ylabel('Fraction in injected azimuthal order')
        plt.title('Azimuthal purity, integrated over radius and polarization');plt.legend();save('seeded_oam_purity')
    return paths


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['plan','run','report','_worker'])
    ap.add_argument('--profile',choices=['pilot','full'],default='pilot')
    ap.add_argument('--cases',nargs='+')
    ap.add_argument('--case-file',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'results/stage7v/generated')
    ap.add_argument('--timeout',type=float)
    ap.add_argument('--require-qualified',action='store_true')
    ap.add_argument('--require-completed',action='store_true')
    args=ap.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.command=='_worker':
        case=json.loads(args.case_file.read_text());execute_case(case,ROOT,output);return
    plan=make_plan(args.profile)
    write_json(output/'plan.json',plan)
    if args.command=='plan':
        print(json.dumps(plan,indent=2));return
    if args.command=='run':
        selected=plan['cases']
        if args.cases:
            lookup={c['id']:c for c in selected}
            unknown=set(args.cases)-set(lookup)
            if unknown:raise SystemExit('Unknown cases: '+', '.join(sorted(unknown)))
            selected=[lookup[id] for id in args.cases]
        okay=True
        for case in selected:
            print('STAGE7V_START '+case['id'],flush=True)
            okay=run_isolated(case,output,timeout_override=args.timeout) and okay
        if args.require_completed and not okay:raise SystemExit('At least one selected case failed, timed out or did not converge')
    report=compile_report(plan,output)
    write_json(output/'report.json',report)
    print('STAGE7V_REPORT '+json.dumps(serializable(report),allow_nan=False),flush=True)
    for path in plot_report(report,output):print('STAGE7V_FIGURE '+str(path))
    if args.require_qualified and not report['numerical_qualification_passed']:
        raise SystemExit('Numerical qualification incomplete or failed; inspect report.json')

if __name__=='__main__':main()
