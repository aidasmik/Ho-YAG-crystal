"""Bounded native Yb:YAG phase-reconstruction dataset generator (no server)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from ybyag_dataset.generator import generate
from check_ybyag_nn_dataset import check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config"/"ybyag_nn_dataset.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"ybyag_nn_dataset")
    parser.add_argument("--points-per-setup",type=int,default=1)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    count=sum(config["split_counts"].values())*args.points_per_setup
    if not 1 <= count <= 8:
        parser.error("configure 1–8 total examples; train, validation and test each need one")
    if args.worker:
        path=generate(args.config,args.output,points_per_setup=args.points_per_setup)
        validation=check(args.output)
        (args.output/"validation.json").write_text(
            json.dumps(validation,indent=2),encoding="utf-8")
        print(path)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),
             "--output",str(args.output.resolve()),"--points-per-setup",
             str(args.points_per_setup),"--worker"]
    record=run_bounded(command,cwd=ROOT,log_path=args.output/"execution.log",
        summary_path=args.output/"execution.json",
        ledger=BudgetLedger(ROOT/".local_runtime"/"budget.json",Limits()),
        label="ybyag_nn_dataset_bounded",configured_seconds=900,category="coupled")
    if record["status"]!="completed" or record["exit_code"]!=0:
        log=(args.output/"execution.log").read_text(encoding="utf-8")
        raise RuntimeError(f"{record['status']}: {log[-3000:]}")
    print(args.output/"manifest.json")


if __name__=="__main__":
    main()
