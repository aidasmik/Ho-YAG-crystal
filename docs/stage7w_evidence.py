"""Collect immutable Stage 7W artifacts into compact, hash-verified evidence.

Reads results only; does not rerun a numerical model or reinterpret failures.
GitHub CLI must be authorized for the given repository. No token is persisted.
"""
from __future__ import annotations
import argparse,hashlib,io,json,subprocess,zipfile
from pathlib import Path


def api(path,binary=False):
    data=subprocess.check_output(['gh','api',path])
    return data if binary else json.loads(data)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='aidasmik/Ho-YAG-crystal')
    ap.add_argument('--run-id',type=int,default=35854247190)
    ap.add_argument('--output',type=Path,default=Path('results/stage7w'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run=api(f'repos/{args.repo}/actions/runs/{args.run_id}')
    artifacts=[];page=1
    while True:
        batch=api(f'repos/{args.repo}/actions/runs/{args.run_id}/artifacts?per_page=100&page={page}')['artifacts']
        artifacts+=batch
        if len(batch)<100:break
        page+=1
    cases=[];testing=None
    for art in artifacts:
        if not art['name'].startswith('stage7w-'):continue
        raw=api(f'repos/{args.repo}/actions/artifacts/{art["id"]}/zip',binary=True)
        digest=hashlib.sha256(raw).hexdigest()
        if art.get('digest') and art['digest']!='sha256:'+digest:
            raise ValueError('artifact checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if art['name']=='stage7w-tests':
                import xml.etree.ElementTree as ET
                names=[n for n in z.namelist() if n.endswith('junit.xml')]
                suites=ET.fromstring(z.read(names[0])).findall('.//testsuite')
                audit=json.loads(z.read(next(n for n in z.namelist() if n.endswith('audit.json'))))
                testing={'new_tests':sum(int(s.attrib.get('tests',0)) for s in suites),
                         'failures':sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites),
                         'independent_audit_failures':audit['failed_checks'],'artifact_id':art['id'],'sha256':digest}
                continue
            if not art['name'].startswith('stage7w-case-'):continue
            names=[n for n in z.namelist() if n.endswith('/summary.json')]
            if len(names)!=1:raise ValueError(f'ambiguous summary members {names}')
            s=json.loads(z.read(names[0]));folder=args.output/'cases';folder.mkdir(exist_ok=True)
            (folder/(s['id']+'.json')).write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
            last=(s.get('history') or [{}])[-1]
            row={'id':s['id'],'status':s.get('status'),'solver_status':s.get('solver_status'),
                 'wall_seconds':s.get('wall_seconds'),'iteration_count':len(s.get('history',[])),
                 'metrics':s.get('metrics'),'last_iteration':last,
                 'artifact_id':art['id'],'artifact_sha256':digest,'source_hash':s['source_hash'],
                 'source_revision':run['head_sha']}
            cases.append(row)
            print('CASE_EVIDENCE '+json.dumps(row,allow_nan=False),flush=True)
    out={'source_revision':run['head_sha'],'run_id':args.run_id,'run_status':run['status'],
         'workflow_conclusion':run['conclusion'],'testing':testing,'cases':cases,
         'expected_case_ids':['reference','modes_4','modes_8','sensitivity_support_free'],
         'missing_case_ids':sorted({'reference','modes_4','modes_8','sensitivity_support_free'}-{r['id'] for r in cases}),
         'full_stage7v_qualification':False,'dataset_ready':False,
         'note':'A representative result does not establish grid, mode-count, hardware or dataset qualification.'}
    (args.output/'evidence_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('STAGE7W_EVIDENCE '+json.dumps({k:v for k,v in out.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
