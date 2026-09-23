from copy import deepcopy
from pathlib import Path
import runpy
import pytest
from hoyag.validation_plan import make_plan, physics_from_repository, validate_plan
from hoyag.validation_metrics import stable_hash

ROOT=Path(__file__).resolve().parents[1]
transform=runpy.run_path(str(ROOT/'examples/stage7v_two_mode_plan.py'))['two_mode_plan']


def test_two_mode_plan_preserves_physics_tolerances_and_original():
    original=make_plan(physics_from_repository(ROOT));before=deepcopy(original)
    plan=transform(original)
    assert original==before
    assert plan['acceptance']==original['acceptance']
    assert plan['comparison_domain']==original['comparison_domain']
    assert plan['physics_hash']==original['physics_hash']
    assert len(plan['cases'])==len(original['cases'])==33
    for old,new in zip(original['cases'],plan['cases']):
        assert old['physics']==new['physics'] and old['physics_hash']==new['physics_hash']
        assert new['numerics']['mode_count']>=2
        assert old['numerics']['settings']==new['numerics']['settings']
        assert new['spec_hash']==stable_hash({k:v for k,v in new.items() if k!='spec_hash'})
    validate_plan(plan)


def test_previously_completed_modal_cases_remain_identical():
    original=make_plan(physics_from_repository(ROOT));plan=transform(original)
    for old,new in zip(original['cases'],plan['cases']):
        if old['numerics']['mode_count']>1:
            assert old==new
        else:
            assert old['spec_hash']!=new['spec_hash']
    group=next(g for g in plan['groups'] if g['name']=='retained_modes')
    assert group['case_ids']==['reference','modes_4','modes_8']
    assert group['required_successive_pairs']==2
    initialization=next(g for g in plan['groups'] if g['name']=='initialization')
    assert initialization['required_successive_pairs']==len(initialization['case_ids'])-1


def test_frozen_plan_cannot_be_relabeled_as_coupled():
    p=make_plan(physics_from_repository(ROOT),kind='frozen',reference={'state_sha256':'a','summary_sha256':'b'})
    with pytest.raises(ValueError):transform(p)
