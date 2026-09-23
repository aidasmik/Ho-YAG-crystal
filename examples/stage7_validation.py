"""Stage 7V: plan, execute selected cases, then report all required evidence.

Planning is cheap. `run` executes only named cases; `--all` must be explicit.
A frozen reference study cannot qualify the complete coupled model. Use a fresh
output directory when code or physical parameters change: stale caches fail.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from hoyag.validation_backend import FrozenReference, source_manifest
from hoyag.validation_plan import make_plan, validate_plan, physics_from_repository
from hoyag.validation_campaign import write_json, execute_case, build_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("plan")
    p.add_argument("--kind",choices=["coupled","frozen"],default="coupled")
    p.add_argument("--reference",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"results/stage7v/plan.json")
    p=sub.add_parser("run")
    p.add_argument("--plan",type=Path,required=True)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cases",nargs="+")
    selection.add_argument("--all",action="store_true")
    p.add_argument("--reference",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--no-resume",action="store_true")
    p.add_argument("--stop-on-failure",action="store_true")
    p=sub.add_parser("report")
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--require-qualified",action="store_true")
    args=parser.parse_args()
    if args.command=="plan":
        if args.kind=="frozen":
            if args.reference is None:parser.error("--reference is required for a frozen plan")
            ref=FrozenReference(args.reference)
            # Physical assumptions come from the archived case, not current defaults.
            s=ref.summary
            physics={"cavity":{k:v for k,v in s['metadata']['cavity'].items() if k in (
                'disk_diameter_m','disk_thickness_m','air_gap_m','output_mirror_radius_m',
                'output_transmission','disk_hr_reflectivity','other_roundtrip_loss','wavelength_m',
                'host_index','host_group_index','pump_hr_reflectivity')},
                "assembly":s['assembly_configuration'].copy(),
                "pump":{"energy_J":s['metadata']['pump_energy_J'],"repetition_rate_Hz":s['metadata']['repetition_rate_Hz'],
                        "duration_s":s['metadata']['pump_duration_s'],"waist_m":s['metadata']['pump_waist_m']},
                "density":{"mean_m3":1.52e26,"radius_m":.005,"thickness_m":.001,"contrast_bound":0.,
                           "correlation_m":.0006,"seed":17,"terms":16},
                "probe":{"waist_m":.000408,"charges":[0,1,-1,2],
                         "model":"weak seeded double-pass probe of frozen mean populations"}}
            physics['assembly'].pop('numerics',None)
            plan=make_plan(physics,kind='frozen',reference=ref.provenance())
        else:
            plan=make_plan(physics_from_repository(ROOT))
        validate_plan(plan);write_json(args.output,plan)
        print(json.dumps({"plan":str(args.output),"cases_planned":len(plan['cases']),
                          "cases_executed":0,"dataset_ready":False},indent=2))
        return
    plan=validate_plan(json.loads(args.plan.read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=source_manifest(ROOT)
    if args.command=="run":
        chosen=plan['cases'] if args.all else [c for c in plan['cases'] if c['id'] in args.cases]
        if args.cases and set(args.cases)-{c['id'] for c in chosen}:
            parser.error('unknown case identifiers: '+', '.join(set(args.cases)-{c['id'] for c in chosen}))
        write_json(args.output/'source_manifest.json',manifest)
        write_json(args.output/'plan.json',plan)
        failures=[]
        for case in chosen:
            print('STAGE7V_START '+case['id'],flush=True)
            record=execute_case(case,args.output,manifest,reference_directory=args.reference,
                                resume=not args.no_resume,
                                progress=lambda row: print('STAGE7V_INNER '+json.dumps(row),flush=True))
            print('STAGE7V_CASE '+json.dumps({k:record.get(k) for k in ('id','status','metrics','wall_seconds')}),flush=True)
            if record['status']!='completed':
                failures.append(case['id'])
                if args.stop_on_failure:break
        report=build_report(plan,args.output,manifest)
        print('STAGE7V_REPORT '+json.dumps({'status':report['status'],'dataset_ready':False}),flush=True)
        if failures:raise SystemExit('Failed or nonconverged requested cases: '+', '.join(failures))
    else:
        report=build_report(plan,args.output,manifest)
        print(json.dumps({'status':report['status'],'coupled_numerically_qualified':report['coupled_numerically_qualified'],
                          'dataset_ready':False},indent=2))
        if args.require_qualified and not report['coupled_numerically_qualified']:
            raise SystemExit(2)

if __name__=='__main__':main()
