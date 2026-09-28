"""Resume a timed-out bounded Yb:YAG dataset run after its owner exits.

Only the named supervised run is followed. A completed run is left alone; a
failed or resource-limited run is recorded and never retried automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hoyag.local_supervisor import atomic_json


def _owner_alive(pid: int, output: Path) -> bool:
    try:
        command = " ".join(psutil.Process(pid).cmdline())
        return "ybyag_nn_dataset.py" in command and output.name in command
    except psutil.Error:
        return False


def _saved(output: Path) -> int:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return sum(map(len, manifest["splits"].values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--setups-per-combination", type=int, required=True)
    parser.add_argument("--points-per-setup", type=int, default=3)
    parser.add_argument("--max-resume-segments", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.max_resume_segments < 1:
        parser.error("workers and max resume segments must be positive")
    output = args.output.resolve()
    config = args.config.resolve()
    status_path = output / "continuation_status.json"
    log_path = output / "continuation.log"
    target = 36 * args.setups_per_combination * args.points_per_setup
    status = {"schema": 1, "target_trials": target,
              "next_workers": args.workers, "segments": [],
              "state": "waiting_for_current_supervised_run"}
    atomic_json(status_path, status)
    while _owner_alive(args.owner_pid, output):
        time.sleep(10)
    # The owner writes execution.json and releases the persistent ledger
    # before exiting. Do not start a second generator during that run.
    for segment in range(args.max_resume_segments + 1):
        execution_file = output / "execution.json"
        if not execution_file.is_file():
            status.update(state="stopped_without_execution_record")
            atomic_json(status_path, status)
            return
        execution = json.loads(execution_file.read_text(encoding="utf-8"))
        saved = _saved(output)
        status.update(saved_trials=saved, previous_run_status=execution.get("status"))
        if execution.get("status") == "completed" and saved == target:
            status.update(state="completed")
            atomic_json(status_path, status)
            return
        if execution.get("status") != "timed_out":
            status.update(state="stopped_after_non_timeout")
            atomic_json(status_path, status)
            return
        if segment == args.max_resume_segments:
            status.update(state="resume_segment_limit_reached")
            atomic_json(status_path, status)
            return
        command = [sys.executable, str(ROOT / "examples" / "ybyag_nn_dataset.py"),
                   "--config", str(config), "--output", str(output),
                   "--workers", str(args.workers),
                   "--points-per-setup", str(args.points_per_setup),
                   "--setups-per-combination", str(args.setups_per_combination),
                   "--time-limit-s", "10800", "--resume"]
        status.update(state="running_resume", active_segment=segment + 1)
        atomic_json(status_path, status)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nResume segment {segment + 1}: {time.ctime()}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        status["segments"].append({"index": segment + 1,
                                   "exit_code": result.returncode,
                                   "saved_trials": _saved(output),
                                   "finished_wall": time.time()})
        atomic_json(status_path, status)


if __name__ == "__main__":
    main()
