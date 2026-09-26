"""Run one Yb:YAG correction episode inside the persistent local supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded


def run_bounded_episode(
    config_path,
    output_dir,
    *,
    smoke=False,
    mixed=False,
    in_situ=False,
    grid_n=None,
    phase_only=False,
    hardware=False,
    cancel=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if smoke:
        config["episode"].update(
            pump_W=0.01,
            grid_n=32,
            camera_width=128,
            camera_height=72,
            thermal_nr=4,
            thermal_nphi=4,
            thermal_nz=1,
            enable_material=False,
            enable_slm_error=False,
            enable_camera_noise=False,
        )
        config["controller"].update(
            mode_count=2, iterations=1, evaluation_limit=24, target_loss=0.0
        )
    if mixed:
        config["episode"].update(enable_material=True, enable_slm_error=True)
    if phase_only:
        config["episode"].update(
            enable_material=False, enable_slm_error=False, enable_external_optics=True
        )
    if in_situ:
        config["episode"]["mode"] = "in_situ"
    if hardware:
        config["episode"].update(
            mode="in_situ",
            enable_thermal_variation=True,
            enable_camera_noise=True,
            control_period_s=0.1,
            slm_drift_fraction_per_sqrt_s=0.0005,
        )
        config["controller"].update(
            method="spgd",
            iterations=4 if smoke else 60,
            evaluation_limit=40 if smoke else 400,
        )
    if grid_n is not None:
        config["episode"]["grid_n"] = int(grid_n)
    selected = output_dir / "request.json"
    selected.write_text(json.dumps(config, indent=2), encoding="utf-8")
    source_hash = hashlib.sha256()
    paths = sorted(
        p
        for directory in (
            ROOT / "src/hoyag",
            ROOT / "src/ybluag",
            ROOT / "src/ybyag",
            ROOT / "src/ybyag_dataset",
            ROOT / "src/ybyag_control",
        )
        for p in directory.rglob("*.py")
    )
    paths.extend(
        (ROOT / "examples/ybyag_control.py", ROOT / "config/ybyag_nn_dataset.json")
    )
    for path in paths:
        source_hash.update(path.relative_to(ROOT).as_posix().encode())
        source_hash.update(path.read_bytes())
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_sha256": hashlib.sha256(
                    selected.read_bytes()
                ).hexdigest(),
                "numerical_source_sha256": source_hash.hexdigest(),
                "source_files": len(paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    record = run_bounded(
        [
            sys.executable,
            str(ROOT / "examples/ybyag_control.py"),
            "--config",
            str(selected),
            "--output",
            str(output_dir / "result.json"),
            "--progress",
            str(output_dir / "progress.json"),
        ],
        cwd=ROOT,
        log_path=output_dir / "execution.log",
        summary_path=output_dir / "execution.json",
        ledger=BudgetLedger(ROOT / ".local_runtime/budget.json", Limits()),
        label="ybyag_closed_loop_smoke" if smoke else "ybyag_closed_loop",
        configured_seconds=180 if smoke else 900,
        category="coupled",
        cancel=cancel,
    )
    if record["status"] != "completed" or record["exit_code"] != 0:
        raise RuntimeError(
            f"{record['status']}: "
            + (output_dir / "execution.log").read_text(encoding="utf-8")[-2500:]
        )
    return output_dir / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/ybyag_control.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="include fixed material and SLM imperfections",
    )
    parser.add_argument(
        "--in-situ",
        action="store_true",
        help="carry supported thermal and sensor state forward",
    )
    parser.add_argument("--grid", type=int, help="optical points per transverse axis")
    parser.add_argument(
        "--phase-only",
        action="store_true",
        help="isolate the external optical phase disturbance",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="chronological noisy SPGD correction with about 60 command updates",
    )
    args = parser.parse_args()
    result = run_bounded_episode(
        args.config,
        args.output,
        smoke=args.smoke,
        mixed=args.mixed,
        in_situ=args.in_situ,
        grid_n=args.grid,
        phase_only=args.phase_only,
        hardware=args.hardware,
    )
    print(result)


if __name__ == "__main__":
    main()
