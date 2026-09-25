import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil
import pytest

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded


def _run(tmp_path, code, *, seconds=2, memory=256*1024**2, cancel=None):
    ledger = BudgetLedger(tmp_path/'ledger.json', Limits(total_seconds=10,
        case_seconds=3, max_attempts=4, memory_bytes=memory,
        poll_seconds=.02, termination_grace_seconds=.1))
    return run_bounded([sys.executable, '-c', code], cwd=tmp_path,
        log_path=tmp_path/'run.log', summary_path=tmp_path/'execution.json',
        ledger=ledger, label='fixture', configured_seconds=seconds, cancel=cancel)


def test_normal_failure_and_persistent_ledger(tmp_path):
    good = _run(tmp_path, "print('accepted', flush=True)")
    bad = _run(tmp_path, 'raise RuntimeError("probe")')
    assert good['status']=='completed' and good['exit_code']==0
    assert bad['status']=='failed' and bad['exit_code']!=0
    assert 'accepted' in (tmp_path/'run.log').read_text()
    assert 'RuntimeError: probe' in (tmp_path/'run.log').read_text()
    assert len(json.loads((tmp_path/'ledger.json').read_text())['attempts'])==2


def test_timeout_kills_owned_descendant_and_keeps_checkpoint(tmp_path):
    code = ("import subprocess,sys,time,pathlib; "
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "pathlib.Path('child.pid').write_text(str(child.pid)); "
            "pathlib.Path('checkpoint.txt').write_text('partial'); time.sleep(30)")
    result = _run(tmp_path, code, seconds=.3)
    assert result['status']=='timed_out'
    assert (tmp_path/'checkpoint.txt').read_text()=='partial'
    pid = int((tmp_path/'child.pid').read_text())
    for _ in range(50):
        if not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:
            break
        time.sleep(.02)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE


def test_memory_ceiling_and_cancellation(tmp_path):
    result = _run(tmp_path, 'import time; a=bytearray(80_000_000); time.sleep(30)',
                  memory=60_000_000)
    assert result['status']=='resource_limit'
    event = threading.Event()
    timer = threading.Timer(.15, event.set)
    timer.start()
    try:
        cancelled = _run(tmp_path, 'import time; time.sleep(30)', cancel=event)
    finally:
        timer.join()
    assert cancelled['status']=='cancelled'


def test_budget_exhaustion_persists_across_instances(tmp_path):
    limits=Limits(total_seconds=.05, case_seconds=1, max_attempts=1,
                  memory_bytes=256*1024**2, poll_seconds=.01,
                  enforce_cumulative_budget=True)
    ledger=BudgetLedger(tmp_path/'ledger.json', limits)
    run_bounded([sys.executable,'-c','pass'],cwd=tmp_path,
        log_path=tmp_path/'log',summary_path=tmp_path/'summary',ledger=ledger,
        label='first',configured_seconds=1)
    with pytest.raises(RuntimeError,match='budget_exhausted'):
        BudgetLedger(tmp_path/'ledger.json',limits).begin('second',1)


def test_default_ledger_has_no_cumulative_attempt_or_time_cap(tmp_path):
    limits=Limits(total_seconds=.01, case_seconds=1, max_attempts=1)
    ledger=BudgetLedger(tmp_path/'ledger.json',limits)
    for attempt in range(2):
        assert ledger.begin(str(attempt), 2) == 1
        ledger.finish('completed', .1)
    assert len(json.loads((tmp_path/'ledger.json').read_text())['attempts']) == 2
