"""Acceptance/timeout bookkeeping tests; not simulated physical results."""
from pathlib import Path
from types import SimpleNamespace
import json
import runpy
from hoyag.validation_plan import make_plan,physics_from_repository,Acceptance
from hoyag.validation_campaign import quality_gates
ROOT=Path(__file__).resolve().parents[1]


def test_all_initial_guesses_are_required_not_only_the_final_two():
    plan=make_plan(physics_from_repository(ROOT))
    group=next(g for g in plan['groups'] if g['name']=='initialization')
    assert group['required_successive_pairs']==len(group['case_ids'])-1


def test_timeout_is_persisted_and_never_passes_quality(tmp_path,monkeypatch):
    fn=runpy.run_path(str(ROOT/'examples/stage7v_bounded.py'))['run_one']
    globals_=fn.__globals__
    def limited(*args,**kwargs):
        return {'status':'timed_out','exit_code':-15,'effective_limit_s':1.,
                'elapsed_s':1.,'peak_rss_bytes':0,'label':'reference','log_path':'execution.log'}
    monkeypatch.setitem(globals_,'run_bounded',limited)
    globals_['source_manifest']=lambda root:{'source_hash':'synthetic-unit-test'}
    case=make_plan(physics_from_repository(ROOT))['cases'][0]
    code=fn(case,tmp_path/'plan.json',tmp_path,1.,None)
    record=json.loads((tmp_path/case['id']/'summary.json').read_text())
    assert code!=0
    assert record['status']=='timed_out'
    assert not record['dataset_ready']
    assert not quality_gates(record,Acceptance())['execution_completed']
