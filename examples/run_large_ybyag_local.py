"""Run one resumable Yb:YAG dataset shard with its own compute ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--time-limit-s', type=float, default=43200)
    parser.add_argument('--memory-limit-gib', type=float, default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'examples/ybyag_nn_dataset.py'),
               '--config', str(ROOT / 'config/ybyag_nn_dataset.json'),
               '--output', str(output), '--points-per-setup', '3',
               '--workers', str(args.workers),
               '--setups-per-combination', '12',
               '--shard-index', str(args.shard_index), '--shard-count', '2',
               '--worker']
    if args.resume:
        command.append('--resume')
    limits = Limits(case_seconds=max(900, args.time_limit_s),
                    memory_bytes=int(args.memory_limit_gib * 2**30))
    record = run_bounded(command, cwd=ROOT, log_path=output / 'execution.log',
                         summary_path=output / 'execution.json',
                         ledger=BudgetLedger(output / 'budget.json', limits),
                         label='ybyag_nn_dataset_local_shard',
                         configured_seconds=args.time_limit_s, category='coupled')
    print(json.dumps(record), flush=True)
    return 0 if record['status'] == 'completed' and record['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
