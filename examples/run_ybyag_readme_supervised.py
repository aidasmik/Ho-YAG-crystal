"""Run the README Yb:YAG cases under the repository's persistent local ledger."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded


def main():
    out = ROOT / "Yb-YAG" / "readme_figures"
    out.mkdir(parents=True, exist_ok=True)
    record = run_bounded(
        [sys.executable, str(ROOT / "examples/ybyag_readme_figures.py")],
        cwd=ROOT, log_path=out / "execution.log",
        summary_path=out / "execution.json",
        ledger=BudgetLedger(ROOT / ".local_runtime/budget.json", Limits()),
        label="ybyag_readme_examples", configured_seconds=180, category="coupled")
    print(record)
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
