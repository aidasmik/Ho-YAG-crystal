from pathlib import Path
from copy import deepcopy
import pytest
from hoyag.polarized_validation import make_polarized_plan,coupled_case_polarized
from hoyag.validation_plan import make_plan,physics_from_repository,validate_plan
from hoyag.validation_campaign import build_report
from hoyag.validation_backend import source_manifest
ROOT=Path(__file__).resolve().parents[1]


def test_stage7w_plan_preserves_all_physics_and_acceptance_thresholds():
    legacy=make_plan(physics_from_repository(ROOT),kind='coupled')
    paired=make_polarized_plan(ROOT)
    assert paired['acceptance']==legacy['acceptance']
    assert paired['physics_hash']==legacy['physics_hash']
    old={c['id']:c for c in legacy['cases']}
    for c in paired['cases']:
        assert c['physics_hash']==old[c['id']]['physics_hash']
        assert c['physics']==old[c['id']]['physics']
        assert c['numerics']['mode_count']>=2
        assert c['numerics']['settings']['eigen_candidates']>=c['numerics']['mode_count']+2
        assert c['numerics']['closure']=='polarization_pair/1'
    assert len(paired['cases'])==len(legacy['cases'])-1
    groups={g['name']:g for g in paired['groups']}
    assert groups['retained_modes']['case_ids']==['reference','modes_4','modes_8']
    assert groups['retained_modes']['required_successive_pairs']==2
    assert groups['initialization']['required_successive_pairs']==4
    validate_plan(paired)


def test_legacy_one_mode_controls_not_mutated():
    legacy=make_plan(physics_from_repository(ROOT),kind='coupled')
    before=deepcopy(legacy)
    make_polarized_plan(ROOT)
    assert legacy==before
    assert legacy['cases'][0]['numerics']['mode_count']==1


def test_missing_stage7w_cases_do_not_become_qualified(tmp_path):
    plan=make_polarized_plan(ROOT)
    report=build_report(plan,tmp_path,source_manifest(ROOT))
    assert report['status']=='incomplete'
    assert not report['dataset_ready']
    assert set(report['case_status'].values())=={'not_run'}


def test_legacy_case_must_be_explicitly_replanned_for_stage7w():
    c=make_plan(physics_from_repository(ROOT),kind='coupled')['cases'][0]
    with pytest.raises(ValueError,match='explicit Stage 7W'):
        coupled_case_polarized(c)
