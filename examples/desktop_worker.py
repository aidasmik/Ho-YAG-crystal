"""Bounded Yb:LuAG calculation worker for the Tkinter application."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ybluag_app import calculate, calculate_pulsed, calculate_structured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("cw", "structured", "pulsed", "pump_sweep"))
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    if args.kind == "pump_sweep":
        top = float(payload["pump_W"])
        result = {"points": [calculate_pulsed(
            {**payload, "pump_W": max(0.001, top * fraction),
             "operation_duration_s": 0},
            compute_thermal=False, summary_only=True)
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)]}
    else:
        functions = {"cw": calculate, "structured": calculate_structured,
                     "pulsed": calculate_pulsed}
        result = functions[args.kind](payload)
    args.result.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
