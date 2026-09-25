"""Export bounded exploratory camera frames from saved Yb desktop calculations.

Example: python examples/yb_camera_dataset.py --runs results/desktop_runs/RUN \
    --output results/camera_datasets/example/camera
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ybluag.camera_dataset import (CameraSettings, export_camera_dataset,
                                   load_saved_state, suggest_optical_throughput)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Saved, independently solved Yb desktop run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-state", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-mm", type=float, default=10.0)
    parser.add_argument("--qe", type=float, default=.05)
    parser.add_argument("--throughput", type=float, default=None,
                        help="Fixed optical throughput; default chooses 50%% full well on first state")
    parser.add_argument("--pulses-per-exposure", type=int, default=100)
    parser.add_argument("--read-noise-e", type=float, default=3.0)
    parser.add_argument("--sensor-temp-jitter-C", type=float, default=.1)
    args = parser.parse_args(argv)
    states = [load_saved_state(path) for path in args.runs]
    base_settings = CameraSettings(
        width=args.width, height=args.height, object_fov_width_mm=args.fov_mm,
        qe_at_signal=args.qe,
        pulses_per_exposure=args.pulses_per_exposure,
        read_noise_e=args.read_noise_e,
        sensor_temperature_jitter_C=args.sensor_temp_jitter_C)
    throughput = (args.throughput if args.throughput is not None else
                  suggest_optical_throughput(states[0]["result"], base_settings))
    settings = CameraSettings(**{**vars(base_settings), "optical_throughput": throughput})
    npz, metadata = export_camera_dataset(
        args.output, states, settings, seed=args.seed,
        frames_per_state=args.frames_per_state)
    print(f"Camera frames: {npz}")
    print(f"Metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
