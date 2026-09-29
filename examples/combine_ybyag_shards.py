"""Combine two completed local Yb:YAG shards and validate the whole dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'examples'))
from check_ybyag_nn_dataset import check


def wait_for_shards(paths: list[Path], interval_s: int = 60) -> None:
    while True:
        records = []
        for path in paths:
            report = path / 'execution.json'
            records.append(json.loads(report.read_text()) if report.exists() else None)
        for path, record in zip(paths, records):
            if record is not None and (record['status'] != 'completed' or record['exit_code'] != 0):
                raise RuntimeError(f'{path.name} ended {record["status"]}; resume it before combining')
        if all(record is not None for record in records):
            return
        time.sleep(interval_s)


def combine(root: Path) -> Path:
    sources = [root / 'shard_local_0', root / 'shard_local_1']
    out = root / 'combined'
    stage = root / 'combined.staging'
    if out.exists() or stage.exists():
        raise FileExistsError('combined output or staging directory already exists')
    manifests = [json.loads((source / 'manifest.json').read_text()) for source in sources]
    left, right = manifests
    for key in ('schema', 'source_config_sha256', 'source_generator_sha256',
                'numerical_source_sha256', 'camera_calibration'):
        if left[key] != right[key]:
            raise ValueError(f'incompatible shards: {key}')
    if {(m['shard_index'], m['shard_count']) for m in manifests} != {(0, 2), (1, 2)}:
        raise ValueError('expected both halves of the two-way setup plan')
    merged = dict(left)
    merged.pop('shard_index')
    merged.pop('shard_count')
    merged['created_unix_s'] = min(m['created_unix_s'] for m in manifests)
    merged['limits'] = list(dict.fromkeys(x for m in manifests for x in m['limits']))
    merged['merged_shards'] = [
        {'directory': source.name, 'shard_index': manifest['shard_index'],
         'runtime_versions': manifest['runtime_versions']}
        for source, manifest in zip(sources, manifests)]
    merged['splits'] = {}
    merged['setup_plan'] = {}
    merged['coverage'] = {}
    for split in left['splits']:
        names = [name for manifest in manifests for name in manifest['splits'][split]]
        if len(names) != len(set(names)):
            raise ValueError(f'duplicate trial paths in {split}')
        plan = [item for manifest in manifests for item in manifest['setup_plan'][split]]
        ids = [item['setup_id'] for item in plan]
        if len(ids) != len(set(ids)) or len(names) != 3 * len(plan):
            raise ValueError(f'incomplete or overlapping setups in {split}')
        merged['splits'][split] = sorted(names)
        merged['setup_plan'][split] = sorted(plan, key=lambda item: item['setup_id'])
        merged['coverage'][split] = dict(left['coverage'][split])
        merged['coverage'][split]['setups'] = len(plan)
    stage.mkdir()
    for source in sources:
        for split in merged['splits']:
            directory = source / split
            if not directory.exists():
                continue
            for original in directory.rglob('*'):
                if not original.is_file():
                    continue
                destination = stage / original.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f'duplicate payload: {destination}')
                os.link(original, destination)
    (stage / 'manifest.json').write_text(json.dumps(merged, indent=2))
    validation = check(stage)
    (stage / 'validation.json').write_text(json.dumps(validation, indent=2))
    if validation['status'] != 'software_checks_passed':
        raise RuntimeError(f'combined dataset check: {validation["status"]}')
    stage.rename(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    sources = [root / 'shard_local_0', root / 'shard_local_1']
    if args.wait:
        wait_for_shards(sources)
    out = combine(root)
    report = json.loads((out / 'validation.json').read_text())
    print(json.dumps({'combined': str(out), 'status': report['status'],
                      'trials': len(report['samples']),
                      'correction_labels': report['correction_labels']}), flush=True)


if __name__ == '__main__':
    main()
