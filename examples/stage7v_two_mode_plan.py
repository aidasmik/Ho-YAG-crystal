"""Explicit two-retained-mode qualification plan after the single-mode failure.

This changes numerical modal truncation, not physical material/mirror inputs or
acceptance tolerances. The original one-mode plan and failed evidence remain
unchanged. No timeout is reclassified as a completed calculation.
"""
from __future__ import annotations
from copy import deepcopy
import argparse
import json
from pathlib import Path
from hoyag.validation_metrics import stable_hash
from hoyag.validation_plan import make_plan, physics_from_repository, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def two_mode_plan(original):
    validate_plan(original)
    if original['kind'] != 'coupled':
        raise ValueError('the two-mode closure is for coupled campaigns only')
    plan = deepcopy(original)
    changed = []
    for case in plan['cases']:
        if case['numerics']['mode_count'] == 1:
            case['numerics']['mode_count'] = 2
            case['numerics']['settings']['eigen_candidates'] = max(
                4, case['numerics']['settings']['eigen_candidates'])
            case['spec_hash'] = stable_hash({k:v for k,v in case.items() if k != 'spec_hash'})
            changed.append(case['id'])
    for group in plan['groups']:
        if group['name'] == 'retained_modes':
            group['case_ids'] = ['reference', 'modes_4', 'modes_8']
            group['required_successive_pairs'] = 2
    plan['campaign_revision'] = {
        'name': 'two-retained-mode spatial refinement',
        'original_plan_hash': stable_hash(original),
        'changed_cases': changed,
        'reason': 'The original single-retained-vector-mode runs did not converge; '
                  'two/four/eight retained-mode results converged to nearly equal output. '
                  'This plan retains both leading vector branches without declaring '
                  'the original one-mode truncation valid.',
        'physical_inputs_changed': False,
        'acceptance_thresholds_changed': False,
        'original_one_mode_status': 'unqualified; archived separately',
    }
    return validate_plan(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.input.read_text()) if args.input else make_plan(physics_from_repository(ROOT))
    plan = two_mode_plan(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'planned':len(plan['cases']), 'changed':plan['campaign_revision']['changed_cases'],
                      'dataset_ready':False}))

if __name__ == '__main__':
    main()
