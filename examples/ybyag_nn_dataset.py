"""Bounded native Yb:YAG phase-reconstruction dataset generator (no server)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from ybyag.material_data import ApproximationWarning
from ybyag_dataset.generator import generate, setup_plan, shard_setup_plan
from check_ybyag_nn_dataset import check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config"/"ybyag_nn_dataset.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"ybyag_nn_dataset")
    parser.add_argument("--points-per-setup",type=int,default=3)
    parser.add_argument("--workers",type=int,default=6,
                        help="independent setup processes; trials within a setup stay sequential")
    parser.add_argument("--shard-index",type=int,default=0,
                        help="zero-based machine shard; whole setups stay together")
    parser.add_argument("--shard-count",type=int,default=1,
                        help="number of machine shards")
    parser.add_argument("--setups-per-combination",type=int,default=None,
                        help="independent setups per Yb concentration and beam target in each normal split")
    parser.add_argument("--time-limit-s",type=float,default=10800)
    parser.add_argument("--resume",action="store_true",help="continue an interrupted dataset in the output directory")
    parser.add_argument("--plan",action="store_true",help="report split coverage without running physics")
    parser.add_argument("--smoke",action="store_true",help="one unqualified setup for bounded integration testing")
    parser.add_argument("--smoke-setup-index",type=int,default=0,
                        help="index of the planned training setup to use with --smoke")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    if args.workers<1:
        parser.error("workers must be positive")
    if args.setups_per_combination is not None and args.setups_per_combination<1:
        parser.error("setups per combination must be positive")
    counts=(None if args.setups_per_combination is None else
            {**config["split_counts"],
             **{split:len(set(config["nominal"]["yb_at_percent_candidates"]))*
                      len(config["nominal"]["target_candidates"])*args.setups_per_combination
                for split in ("train","validation","test")}})
    plan=shard_setup_plan(setup_plan(config,counts),args.shard_index,args.shard_count)
    if args.plan:
        print(json.dumps({"shard_index":args.shard_index,
                          "shard_count":args.shard_count,
                          "setups":{k:len(v) for k,v in plan.items()},
                          "trials_per_setup":args.points_per_setup,
                          "total_measured_trials":sum(map(len,plan.values()))*args.points_per_setup,
                          "coverage":{k:sorted({(row["yb_at_percent"],row["target"])
                              for row in v}) for k,v in plan.items()}},indent=2))
        return
    if args.points_per_setup<3 or args.time_limit_s<=0:
        parser.error("use at least three trials per setup and a positive time limit")
    if args.worker:
        warnings.filterwarnings("ignore",message="Interpolating thermal resistivity across different dopings.",
                                category=ApproximationWarning)
        path=generate(args.config,args.output,points_per_setup=args.points_per_setup,
                      smoke=args.smoke,resume=args.resume,split_counts=counts,
                      workers=args.workers,
                      shard_index=args.shard_index,shard_count=args.shard_count,
                      smoke_setup_index=args.smoke_setup_index)
        validation=check(args.output)
        (args.output/"validation.json").write_text(
            json.dumps(validation,indent=2),encoding="utf-8")
        print(path)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),
             "--output",str(args.output.resolve()),"--points-per-setup",
             str(args.points_per_setup),"--workers",str(args.workers),
             "--shard-index",str(args.shard_index),
             "--shard-count",str(args.shard_count),"--worker"]
    if args.smoke:
        command.extend(("--smoke","--smoke-setup-index",str(args.smoke_setup_index)))
    if args.resume:
        command.append("--resume")
    if args.setups_per_combination is not None:
        command.extend(("--setups-per-combination",str(args.setups_per_combination)))
    record=run_bounded(command,cwd=ROOT,log_path=args.output/"execution.log",
        summary_path=args.output/"execution.json",
        ledger=BudgetLedger(ROOT/".local_runtime"/"budget.json",
                            Limits(case_seconds=max(900,args.time_limit_s))),
        label="ybyag_nn_dataset_bounded",configured_seconds=args.time_limit_s,category="coupled")
    if record["status"]!="completed" or record["exit_code"]!=0:
        log=(args.output/"execution.log").read_text(encoding="utf-8")
        if record["status"]=="timed_out":
            raise RuntimeError(f"Dataset reached its {record['effective_limit_s']:.0f} s run limit. "
                               f"Completed trials remain in {args.output}. "
                               "Resume with --resume --output pointing to that directory. "
                               f"Execution details: {args.output/'execution.json'}")
        raise RuntimeError(f"{record['status']}: {log[-3000:]}")
    print(args.output/"manifest.json")


if __name__=="__main__":
    main()
