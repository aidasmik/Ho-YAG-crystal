"""Acceptance/timeout bookkeeping tests; not simulated physical results."""
from pathlib import Path
from types import SimpleNamespace
import json
import runpy
import subprocess
from hoyag.validation_plan import make_plan,physics_from_repository,Acceptance
from hoyag.validation_campaign import quality_gates
ROOT=Path(__file__).resolve().parents[1]


def test_all_initial_guesses_are_required_not_only_the_final_two():
    plan=make_plan(physics_from_repository(ROOT))
    group=next(g for g in plan['groups'] if g['name']=='initialization')
    assert group['required_successive_pairs']==len(group['case_ids'])-1


def test_timeout_is_persisted_and_never_passes_quality(tmp_path,monkeypatch):
    fn=runpy.run_path(str(ROOT/'examples/stage7v_bounded.py'))['run_one']
    globals_=fn.__globals__;calls=[]
    class Process:
        pid=987654
        count=0
        def wait(self,timeout=None):
            self.count+=1
            if self.count==1:raise subprocess.TimeoutExpired('synthetic process',timeout)
            return -15
    monkeypatch.setattr(globals_['subprocess'],'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(globals_['os'],'killpg',lambda pid,sig:calls.append((pid,sig)))
    globals_['source_manifest']=lambda root:{'source_hash':'synthetic-unit-test'}
    case=make_plan(physics_from_repository(ROOT))['cases'][0]
    code=fn(case,tmp_path/'plan.json',tmp_path,1.)
    record=json.loads((tmp_path/case['id']/'summary.json').read_text())
    assert code==124 and calls
    assert record['status']=='timed_out'
    assert not record['dataset_ready']
    assert not quality_gates(record,Acceptance())['execution_completed']
