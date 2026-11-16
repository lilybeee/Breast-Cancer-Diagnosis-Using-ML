"""
export_to_dashboard.py - copy the WDBC reports into the Streamlit dashboard.

The dashboard page reads only from dashboard/data/wdbc/, so the dashboard folder
stays deployable on its own.  This script writes ONLY inside that new wdbc/
sub-folder - it never touches any existing dashboard file.

Run:   python src/export_to_dashboard.py      (from the wdbc/ project root)
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # .../wdbc
PROJECT = ROOT.parent                                # .../dashboard_project
REPORTS = ROOT / "reports"
DEST = PROJECT / "dashboard" / "data" / "wdbc"

CSVS = [
    "model_comparison.csv", "cv_results.csv", "test_results.csv",
    "confusion_matrices.csv", "cv_paired_vs_best.csv", "threshold_analysis.csv",
    "threshold_analysis_test.csv", "feature_importance.csv", "lr_coefficients.csv",
    "feature_glossary.csv", "eda_summary.csv",
]
JSONS = ["split_info.json", "model_selection.json", "selected_threshold.json",
         "best_model_report.json"]


def main():
    if not REPORTS.exists():
        sys.exit("reports/ not found - run src/train.py and src/evaluate.py first")

    (DEST / "figures").mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for name in CSVS + JSONS:
        src = REPORTS / name
        if src.exists():
            shutil.copy2(src, DEST / name)
            copied.append(name)
        else:
            missing.append(name)

    figs = sorted((REPORTS / "figures").glob("*.png"))
    for f in figs:
        shutil.copy2(f, DEST / "figures" / f.name)

    (DEST / "wdbc_run_info.json").write_text(json.dumps({
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "wdbc/reports",
        "n_csv": len(copied), "n_figures": len(figs),
        "selected_model": json.loads((REPORTS / "model_selection.json").read_text())["selected_model"],
    }, indent=2))

    print(f"-> {DEST}")
    print(f"   {len(copied)} data files, {len(figs)} figures")
    if missing:
        print(f"   MISSING (not copied): {', '.join(missing)}")


if __name__ == "__main__":
    main()
