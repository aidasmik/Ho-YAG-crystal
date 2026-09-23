"""Run cases with explicit wall-time limits and preserve interrupted evidence.

A resource timeout is NOT convergence, and a bounded attempt cannot qualify a
missing refinement family. Linux process groups are terminated, including child
workers. No automatic paid hardware, credentials or background subscriptions.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from hoyag.validation_backend import source_manifest
from hoyag.validation_campaign import write_json,build_report
from hoyag.validation_metrics import stable_hash
from hoyag.validation_plan import validate_plan,SCHEMA


def run_one(case,plan_path,out,seconds,ledger,reference=None):
    folder=out/case['id'];folder.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'examples/stage7_validation.py'),'run','--plan',str(plan_path),
         '--cases',case['id'],'--output',str(out)]
    if reference:cmd+=['--reference',str(reference)]
    execution=run_bounded(cmd,cwd=ROOT,log_path=folder/'execution.log',
        summary_path=folder/'execution_status.json',ledger=ledger,label=case['id'],
        configured_seconds=seconds)
    path=folder/'summary.json'
    try:record=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):record={}
    # Preserve a proper failed/nonconverged record written by the inner solver.
    # A stale cache does not become reusable merely because the subprocess ended.
    if execution['status']!='completed' or not record or record.get('status')=='running':
        manifest=source_manifest(ROOT)
        if record and record.get('spec_hash')!=case['spec_hash']:
            record={}
        record.update({'schema':SCHEMA,'id':case['id'],'kind':case['kind'],'purpose':case['purpose'],
            'case':case,'spec_hash':case['spec_hash'],'physics_hash':case['physics_hash'],
            'source_hash':manifest['source_hash'],
            'execution_key':stable_hash({'schema':SCHEMA,'spec_hash':case['spec_hash'],'source_hash':manifest['source_hash']}),
            'status':execution['status'],'exit_code':execution['exit_code'],
            'wall_time_limit_s':execution['effective_limit_s'],'wall_seconds':execution['elapsed_s'],'dataset_ready':False,
            'note':'Resource-limited attempt; no steady-state or numerical-qualification claim.'})
        write_json(path,record)
    print('BOUNDED_CASE '+json.dumps({'id':case['id'],**execution}),flush=True)
    return 0 if record.get('status')=='completed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    select=p.add_mutually_exclusive_group(required=True)
    select.add_argument('--cases',nargs='+');select.add_argument('--all',action='store_true')
    p.add_argument('--seconds-per-case',type=float,default=900.)
    p.add_argument('--ledger',type=Path,default=ROOT/'.local_runtime/budget.json')
    p.add_argument('--memory-mb',type=int,default=None)
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    if not 1<=args.seconds_per_case<=21600:p.error('time limit must be between 1 second and 6 hours')
    import psutil
    max_memory_mb=min(8192,int(psutil.virtual_memory().total*.7/1024**2))
    memory_mb=args.memory_mb or max_memory_mb
    if memory_mb<=0 or memory_mb>max_memory_mb:p.error('memory ceiling exceeds 8 GiB or 70% of RAM')
    ledger=BudgetLedger(args.ledger,Limits(memory_bytes=memory_mb*1024**2))
    plan=validate_plan(json.loads(args.plan.read_text()))
    known={c['id']:c for c in plan['cases']}
    ids=list(known) if args.all else args.cases
    if set(ids)-known.keys():p.error('unknown cases: '+', '.join(set(ids)-known.keys()))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=source_manifest(ROOT)
    write_json(out/'source_manifest.json',manifest);write_json(out/'plan.json',plan)
    codes=[run_one(known[cid],args.plan.resolve(),out,args.seconds_per_case,ledger,args.reference) for cid in ids]
    report=build_report(plan,out,manifest)
    print('BOUNDED_REPORT '+json.dumps({'status':report['status'],'dataset_ready':False}),flush=True)
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
