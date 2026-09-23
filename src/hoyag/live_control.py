"""Local, timestamped controls applied only at accepted solver boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .local_supervisor import atomic_json


class UserCancelled(Exception):
    pass


def request_control(case_directory: Path, command: str) -> dict:
    if command not in ('pause','resume','step','cancel'):
        raise ValueError('unsupported solver control')
    path=Path(case_directory)/'control.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    record={'sequence':int(previous.get('sequence',0))+1,
            'command':command,'requested_wall':time.time()}
    atomic_json(path,record)
    return record


class ControlState:
    def __init__(self, case_directory: Path):
        self.path=Path(case_directory)/'control.json'
        self.sequence=0
        self.paused=False
        self.step_permit=False

    def _poll(self):
        if not self.path.exists():return
        record=json.loads(self.path.read_text())
        if record['sequence']<=self.sequence:return
        self.sequence=record['sequence']
        command=record['command']
        if command=='cancel':raise UserCancelled('user cancelled bounded live worker')
        if command=='pause':self.paused=True
        elif command=='resume':self.paused=False;self.step_permit=False
        elif command=='step':self.paused=True;self.step_permit=True

    def after_iteration(self):
        """Pause before the next iteration, or permit exactly one solver step."""
        while True:
            self._poll()
            if not self.paused:return
            if self.step_permit:
                self.step_permit=False
                return
            time.sleep(.1)


class CancellationFlag:
    def __init__(self, case_directory: Path):
        self.path=Path(case_directory)/'control.json'

    def is_set(self):
        try:
            return json.loads(self.path.read_text()).get('command')=='cancel'
        except FileNotFoundError:
            return False
