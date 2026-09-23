"""Inspect saved Stage 7V execution evidence; never infer convergence from a timeout.

This reads JSON/checkpoint/log members directly from the immutable artifact ZIP.
It does not extract source, load pickle, modify tolerances, or run a new laser.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def inspect_archive(archive: Path, output: Path, expected_sha256: str) -> dict:
    h = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        raise ValueError('Evidence archive digest mismatch')
    records = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for name in sorted(names):
            p = PurePosixPath(name)
            if p.name != 'summary.json' or 'coupled' not in p.parts:
                continue
            s = json.loads(z.read(name))
            if s.get('kind') != 'coupled' or 'id' not in s:
                continue
            prefix = str(p.parent) + '/'
            checkpoint = prefix + 'iterations.jsonl'
            rows, malformed = [], []
            if checkpoint in names:
                for index, line in enumerate(z.read(checkpoint).decode().splitlines()):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed.append(index + 1)
            row = {'id': s['id'], 'status': s.get('status'),
                   'status_reason': s.get('reason', s.get('error', s.get('message'))),
                   'numerics': s.get('case', {}).get('numerics'),
                   'metrics': s.get('metrics'), 'checkpoint_count': len(rows),
                   'first_iteration': rows[0] if rows else None,
                   'last_iterations': rows[-6:], 'malformed_checkpoint_lines': malformed,
                   'artifact_member': name, 'source_hash': s.get('source_hash'),
                   'wall_seconds': s.get('wall_seconds'),
                   'execution': {k:s.get(k) for k in ('elapsed_s','wall_time_limit_s','exit_code','timeout_seconds') if k in s}}
            logs = [n for n in names if n.startswith(prefix) and PurePosixPath(n).suffix == '.log']
            row['log_tails'] = {n:z.read(n).decode('utf-8',errors='replace')[-1800:] for n in sorted(logs)}
            records.append(row)
        report = {'evidence_sha256': h.hexdigest(), 'cases': records,
                  'status_counts': dict(Counter(r['status'] for r in records)),
                  'source_run_id':35835101412, 'source_artifact_id':10740066509,
                  'note':'A status=completed record is still subject to its scientific qualification gates.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'stall_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('STALL_COUNTS '+json.dumps(report['status_counts']),flush=True)
    selected = {'reference','optical_384','optical_512','material_1','mechanical_1',
                'modes_2','modes_4','modes_8','start_lg_plus','sensitivity_support_free'}
    for r in records:
        compact = {k:r[k] for k in ('id','status','checkpoint_count','wall_seconds','execution')}
        if r['last_iterations']:
            compact['last_iteration'] = r['last_iterations'][-1]
        print('STALL_CASE '+json.dumps(compact,allow_nan=False),flush=True)
        if r['id'] in selected:
            print('STALL_DETAIL '+json.dumps(r,allow_nan=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    args=ap.parse_args()
    inspect_archive(args.archive,args.output,args.sha256)


if __name__=='__main__':
    main()
