"""Bounded hot-operator retained-family diagnostic from a saved Stage 7W state."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def worker(case_directory, output, plan_output, candidates):
    if os.environ.get('HOYAG_SUPERVISED')!='1':
        raise RuntimeError('candidate diagnostic requires the shared budget supervisor')
    from hoyag.validation_backend import make_grid,sha256_file
    from hoyag.resonator import ThinDiskResonator
    from hoyag.vector_cavity import VectorRoundTrip
    from hoyag.family_planning import solve_candidate_diagnostic
    from hoyag.polarized_validation import make_polarized_plan,add_six_mode_candidate
    from hoyag.local_supervisor import atomic_json
    summary=json.loads((case_directory/'summary.json').read_text())
    state=case_directory/'state.npz'
    if summary.get('status')!='completed' or sha256_file(state)!=summary.get('state_sha256'):
        raise ValueError('diagnostic requires a hash-verified completed state')
    case=summary['case']
    with np.load(state,allow_pickle=False) as data:
        required=('inward_jones','outward_jones','geometry_roundtrip_opd_m',
                  'single_pass_log_gain','fields_used')
        if any(name not in data for name in required):
            raise ValueError('saved state lacks the complete hot optical operator')
        screens=SimpleNamespace(wavelength_m=case['physics']['cavity']['wavelength_m'],
            inward_jones=data['inward_jones'],outward_jones=data['outward_jones'],
            geometry_roundtrip_opd_m=data['geometry_roundtrip_opd_m'])
        operator=VectorRoundTrip(make_grid(case),ThinDiskResonator(**case['physics']['cavity']),
                                 screens,data['single_pass_log_gain'])
        result=solve_candidate_diagnostic(operator,candidate_count=candidates,
                                          previous=data['fields_used'])
    result.update(reference_case=str(case_directory.resolve()),
                  reference_state_sha256=summary['state_sha256'],
                  source_hash=summary['source_hash'],
                  scope='one saved hot operator; coupled solver must independently guard every iteration')
    atomic_json(output,result)
    if result.get('modes_6_complete_family_candidate') and plan_output:
        plan=add_six_mode_candidate(make_polarized_plan(ROOT),result)
        atomic_json(plan_output,plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','worker'))
    p.add_argument('--case-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-output',type=Path)
    p.add_argument('--candidates',type=int,default=10)
    p.add_argument('--ledger',type=Path,default=ROOT/'.local_runtime/budget.json')
    args=p.parse_args()
    if args.command=='worker':
        worker(args.case_directory,args.output,args.plan_output,args.candidates)
        return
    from hoyag.local_supervisor import BudgetLedger,Limits,run_bounded
    cmd=[sys.executable,str(Path(__file__).resolve()),'worker',
         '--case-directory',str(args.case_directory.resolve()),'--output',str(args.output.resolve()),
         '--candidates',str(args.candidates)]
    if args.plan_output:cmd+=['--plan-output',str(args.plan_output.resolve())]
    result=run_bounded(cmd,cwd=ROOT,log_path=args.output.parent/'diagnostic.log',
        summary_path=args.output.parent/'diagnostic_execution.json',
        ledger=BudgetLedger(args.ledger,Limits()),label='hot_family_diagnostic',
        configured_seconds=180,category='profile',env={**os.environ,'HOYAG_SUPERVISED':'1'})
    print(json.dumps(result))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
