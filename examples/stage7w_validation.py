"""Stage 7W campaigns using the original Stage 7V scientific acceptance tests.

Examples:
  python examples/stage7w_validation.py plan --output config/stage7w_validation.json
  python examples/stage7w_validation.py run --plan config/stage7w_validation.json \
      --cases reference modes_4 modes_8 --seconds-per-case 3600 --output results/stage7w/run

Child process budgets never turn a timeout into a scientific pass. The parent
keeps partial iteration logs and rebuilds the full-plan report with missing cases.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    from hoyag.polarized_validation import make_polarized_plan,execute_polarized_case
    from hoyag.validation_plan import validate_plan
    from hoyag.validation_backend import source_manifest
    from hoyag.validation_campaign import write_json,build_report
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--output',type=Path,required=True)
    for name in ('run','worker'):
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,required=True)
        p.add_argument('--cases',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
        if name=='run':p.add_argument('--seconds-per-case',type=int,default=3600)
    p=sub.add_parser('report');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='plan':
        plan=make_polarized_plan(ROOT);write_json(args.output,plan)
        print(json.dumps({'cases_planned':len(plan['cases']),'dataset_ready':False,'plan':str(args.output)}))
        return
    plan=validate_plan(json.loads(args.plan.read_text()))
    if any(c['numerics'].get('closure')!='polarization_pair/1' for c in plan['cases']):
        raise ValueError('every case must explicitly select the Stage 7W closure')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=source_manifest(ROOT)
    source_path=out/'source_manifest.json'
    if source_path.exists() and json.loads(source_path.read_text())['source_hash']!=manifest['source_hash']:
        raise ValueError('source or runtime differs; do not mix evidence in this directory')
    write_json(source_path,manifest);write_json(out/'plan.json',plan)
    if args.command=='report':
        report=build_report(plan,out,manifest)
        print('STAGE7W_REPORT '+json.dumps({'status':report['status'],'case_status':report['case_status'],'dataset_ready':False}))
        return
    by_id={c['id']:c for c in plan['cases']}
    if len(set(args.cases))!=len(args.cases) or any(cid not in by_id for cid in args.cases):
        raise ValueError('unknown or duplicate case ID')
    if args.command=='worker':
        if len(args.cases)!=1:raise ValueError('one case per worker')
        result=execute_polarized_case(by_id[args.cases[0]],out,manifest)
        print('STAGE7W_CASE '+json.dumps({'id':result['id'],'status':result['status'],
                                       'solver_status':result.get('solver_status'),'metrics':result.get('metrics')}),flush=True)
        raise SystemExit(0 if result['status']=='completed' else 1)
    if not 30<=args.seconds_per_case<=21600:raise ValueError('case budget must be 30..21600 seconds')
    failed=[]
    for cid in args.cases:
        folder=out/cid;folder.mkdir(exist_ok=True)
        begin=time.perf_counter()
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--plan',str(args.plan.resolve()),
             '--cases',cid,'--output',str(out.resolve())]
        with (folder/'execution.log').open('a') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=args.seconds_per_case)
                timed_out=False
            except subprocess.TimeoutExpired:
                timed_out=True
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        path=folder/'summary.json'
        if path.exists():record=json.loads(path.read_text())
        else:
            from hoyag.validation_plan import SCHEMA
            c=by_id[cid]
            record={'schema':SCHEMA,'id':cid,'kind':'coupled','purpose':c['purpose'],
                    'spec_hash':c['spec_hash'],'physics_hash':c['physics_hash'],
                    'source_hash':manifest['source_hash'],'case':c,'dataset_ready':False}
        if timed_out:
            record.update(status='timed_out',reason='explicit wall-time limit; partial iterations are not a result')
        elif code!=0 and record.get('status')=='running':
            record.update(status='failed',reason='worker failed before producing a final record')
        record['execution']={'exit_code':code,'seconds_per_case':args.seconds_per_case,
                             'elapsed_s':time.perf_counter()-begin}
        write_json(path,record)
        print('BOUNDED_STAGE7W '+json.dumps({'id':cid,'status':record.get('status'),**record['execution']}),flush=True)
        if record.get('status')!='completed':failed.append(cid)
    report=build_report(plan,out,manifest)
    print('STAGE7W_REPORT '+json.dumps({'status':report['status'],'case_status':report['case_status'],'dataset_ready':False}),flush=True)
    if failed:raise SystemExit('Selected cases not converged: '+', '.join(failed))


if __name__=='__main__':main()
