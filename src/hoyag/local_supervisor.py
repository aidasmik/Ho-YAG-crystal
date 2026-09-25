"""Persistent run ledger and owned process-tree supervision.

The ledger is shared by all bounded entry points. A run lock prevents concurrent
expensive attempts; a crashed owner is recorded before reuse. Cumulative caps
are optional; individual run time and memory limits always remain in force.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Sequence

import psutil


@dataclass(frozen=True)
class Limits:
    total_seconds: float = 7200
    case_seconds: float = 900
    profile_seconds: float = 180
    regression_seconds: float = 600
    audit_seconds: float = 300
    viewer_seconds: float = 120
    max_attempts: int = 4
    enforce_cumulative_budget: bool = False
    memory_bytes: int = 8 * 1024**3
    poll_seconds: float = .1
    termination_grace_seconds: float = 5

    def __post_init__(self):
        if any(x <= 0 for x in (self.total_seconds, self.case_seconds,
                                 self.profile_seconds,self.regression_seconds,
                                 self.audit_seconds,self.viewer_seconds,
                                 self.max_attempts, self.memory_bytes,
                                 self.poll_seconds, self.termination_grace_seconds)):
            raise ValueError('all resource limits must be positive')


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


class BudgetLedger:
    def __init__(self, path: Path, limits: Limits = Limits()):
        self.path = Path(path)
        self.limits = limits
        self.lock = self.path.with_suffix('.run.lock')
        self.owner = False

    def _read(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('schema') != 1:
                raise ValueError('unknown budget ledger schema')
            return data
        return {'schema': 1, 'total_seconds': self.limits.total_seconds,
                'max_attempts': self.limits.max_attempts, 'attempts': [], 'active': None}

    def _acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(str(os.getpid()))
                self.owner = True
                return
            except FileExistsError:
                try:
                    pid = int(self.lock.read_text())
                except (ValueError, OSError):
                    raise RuntimeError('budget lock unreadable; inspect it manually')
                if _alive(pid):
                    raise RuntimeError(f'another supervised run is active (PID {pid})')
                self.lock.unlink()
        raise RuntimeError('could not acquire budget lock')

    def begin(self, label: str, configured_seconds: float, category: str = 'coupled') -> float:
        self._acquire()
        try:
            data = self._read()
            if (self.limits.enforce_cumulative_budget and
                    (data['total_seconds'] != self.limits.total_seconds or
                     data['max_attempts'] != self.limits.max_attempts)):
                raise ValueError('ledger limits differ; use the original limits or a new ledger')
            if data['active'] is not None:
                active = data['active']
                data['active'] = None
                active.update(status='interrupted', elapsed_s=min(active['limit_s']+self.limits.termination_grace_seconds,
                              max(0., time.time() - active['started_wall'])))
                data['attempts'].append(active)
            spent = sum(a['elapsed_s'] for a in data['attempts'])
            remaining = max(0., self.limits.total_seconds - spent)
            expensive=sum(a.get('category','coupled')=='coupled' for a in data['attempts'])
            if (self.limits.enforce_cumulative_budget and
                    (remaining <= 0 or
                     (category=='coupled' and expensive >= self.limits.max_attempts))):
                atomic_json(self.path, data)
                raise RuntimeError('budget_exhausted')
            if configured_seconds <= 0:
                raise ValueError('configured case limit must be positive')
            effective = min(configured_seconds, self.category_limit(category))
            if self.limits.enforce_cumulative_budget:
                effective = min(effective, remaining)
            data['active'] = {'label': label, 'category':category, 'pid': os.getpid(),
                              'started_wall': time.time(), 'limit_s': effective}
            atomic_json(self.path, data)
            return effective
        except BaseException:
            self.release()
            raise

    def category_limit(self, category: str) -> float:
        limits={'coupled':self.limits.case_seconds,'profile':self.limits.profile_seconds,
                'regression':self.limits.regression_seconds,'audit':self.limits.audit_seconds,
                'viewer':self.limits.viewer_seconds}
        if category not in limits:
            raise ValueError('unknown budget category')
        return limits[category]

    def finish(self, status: str, elapsed: float) -> None:
        try:
            data = self._read()
            active = data['active']
            data['active'] = None
            if active is None or active['pid'] != os.getpid():
                raise RuntimeError('budget reservation is not owned by this process')
            active.update(status=status, elapsed_s=max(0., elapsed))
            data['attempts'].append(active)
            atomic_json(self.path, data)
        finally:
            self.release()

    def release(self) -> None:
        if self.owner:
            self.lock.unlink(missing_ok=True)
            self.owner = False


class _WindowsJob:
    """Kill-on-close Job Object; descendants inherit membership on Windows."""
    def __init__(self, process: subprocess.Popen):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), 'CreateJobObjectW failed')
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION starts with BasicLimitInformation:
        # limit flags are DWORD at offset 16 (32-bit) or 16 (64-bit).
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                        ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]
        class Io(ctypes.Structure):
            _fields_ = [('ReadOperationCount', ctypes.c_uint64), ('WriteOperationCount', ctypes.c_uint64),
                        ('OtherOperationCount', ctypes.c_uint64), ('ReadTransferCount', ctypes.c_uint64),
                        ('WriteTransferCount', ctypes.c_uint64), ('OtherTransferCount', ctypes.c_uint64)]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', Io),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), 'SetInformationJobObject failed')
        if not kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process._handle)):
            raise OSError(ctypes.get_last_error(), 'AssignProcessToJobObject failed')

    def terminate(self):
        self.kernel.TerminateJobObject(self.handle, 1)

    def close(self):
        self.kernel.CloseHandle(self.handle)


def _tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            pass
    return total


def _terminate(process: subprocess.Popen, job, grace: float) -> None:
    if os.name == 'nt' and process.poll() is None:
        # CTRL_BREAK is cooperative for console-aware workers; the Job Object
        # closes the full process tree after the bounded grace period.
        try: process.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError: pass
    elif os.name != 'nt':
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    if process.poll() is None:
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    if os.name == 'nt':
        if job is not None:
            job.terminate()
        elif process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    if process.poll() is None:
        process.wait()


def run_bounded(command: Sequence[str], *, cwd: Path, log_path: Path,
                summary_path: Path, ledger: BudgetLedger, label: str,
                configured_seconds: float, cancel=None, env=None,
                category: str = 'coupled') -> dict:
    """Run one owned worker; return and persist an execution record."""
    limit = ledger.begin(label, configured_seconds, category)
    began = time.monotonic()
    process = None
    job = None
    peak_rss = 0
    status = 'failed'
    code = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', buffering=1) as stream:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            process = subprocess.Popen(list(command), cwd=cwd, stdout=stream,
                                       stderr=subprocess.STDOUT, env=env,
                                       creationflags=flags, start_new_session=os.name != 'nt')
            if os.name == 'nt':
                job = _WindowsJob(process)
            while process.poll() is None:
                elapsed = time.monotonic() - began
                peak_rss = max(peak_rss, _tree_rss(process.pid))
                if cancel is not None and cancel.is_set():
                    status = 'cancelled'; break
                if peak_rss > ledger.limits.memory_bytes:
                    status = 'resource_limit'; break
                if elapsed >= limit:
                    status = ('timed_out' if limit >= min(configured_seconds, ledger.category_limit(category))
                              else 'budget_exhausted')
                    break
                time.sleep(min(ledger.limits.poll_seconds, max(.001, limit - elapsed)))
            if status in ('cancelled', 'resource_limit', 'timed_out', 'budget_exhausted'):
                _terminate(process, job, ledger.limits.termination_grace_seconds)
            code = process.wait()
            if status == 'failed':
                status = 'completed' if code == 0 else 'failed'
            # A worker that exits can still leave descendants behind.
            _terminate(process, job, 0)
    finally:
        if process is not None and process.poll() is None:
            _terminate(process, job, ledger.limits.termination_grace_seconds)
        if job is not None:
            job.close()
        elapsed = time.monotonic() - began
        record = {'label': label, 'status': status, 'exit_code': code,
                  'elapsed_s': elapsed, 'effective_limit_s': limit,
                  'peak_rss_bytes': peak_rss, 'log_path': str(log_path)}
        atomic_json(summary_path, record)
        ledger.finish(status, elapsed)
    return record
