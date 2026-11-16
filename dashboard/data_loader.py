"""
data_loader.py - the ONLY place that reads files.  Pages never open files themselves.

Contract
  DATASETS : registry  key -> dict(file=<csv name inside data/>, read=<pandas.read_csv kwargs>)
  load(key)           -> DataFrame, or None if the file is missing (pages call ui.need(key))
  load_csv(name, **kw)-> DataFrame for an ad-hoc CSV in data/ (or None)
  run_info()          -> dict from run_info.json ({} if missing)
  primary_model()     -> name of the primary model
  image(name)         -> path string of a PNG in data/ (or None)
  available()         -> {key: file exists?}
Add a dataset in one line:  python setup_dashboard.py --add-dataset KEY FILE.csv [--index-col]
"""
import json
import os
from pathlib import Path

import pandas as pd

try:                                    # works with or without Streamlit installed
    import streamlit as st
    _cache = st.cache_data(show_spinner=False)
except Exception:                       # e.g. when running `setup_dashboard.py --check`
    def _cache(f):
        return f

DATA_DIR = Path(os.environ.get("DASH_DATA_DIR", Path(__file__).resolve().parent / "data"))

DATASETS = {
    "cv_summary":     dict(file="cv_endterm_summary.csv", read=dict(index_col=0)),
    "cv_mid":         dict(file="cv_midterm_summary.csv", read=dict(index_col=0)),
    "test":           dict(file="test_endterm_results.csv", read=dict(index_col=0)),
    "stats":          dict(file="stat_tests_vs_LR.csv"),
    "paired_auc":     dict(file="paired_bootstrap_auc_vs_LR.csv"),
    "odds_ratios":    dict(file="lr_odds_ratios.csv", read=dict(index_col=0)),
    "perm_imp":       dict(file="permutation_importance.csv", read=dict(index_col=0)),
    "error_analysis": dict(file="error_analysis_primary.csv", read=dict(index_col=0)),
    "calc_cleaned": dict(file="breast_cancer_cleaned_final_calc__.csv"),
    "mass_cleaned": dict(file="breast_cancer_cleaned_final_mass__.csv"),
    # <<NEW_DATASETS>>  (setup_dashboard.py --add-dataset inserts lines above this marker)
}

FAMILY = {
    "Logistic Regression": "Linear", "LR + Interactions": "Linear", "SVM": "Kernel",
    "k-NN": "Instance-based", "Decision Tree": "Tree", "Bernoulli NB": "Probabilistic",
    "Random Forest": "Ensemble (bagging)", "HistGradientBoosting": "Ensemble (boosting)",
    "AdaBoost": "Ensemble (boosting)", "Soft-Vote Ensemble": "Ensemble (voting)",
}


def path(filename):
    return DATA_DIR / filename


def available():
    return {k: path(v["file"]).exists() for k, v in DATASETS.items()}


@_cache
def _read_csv(p, read_items, mtime):          # mtime in the key -> cache refreshes when the file changes
    return pd.read_csv(p, **dict(read_items))


def load_csv(filename, **read):
    p = path(filename)
    if not p.exists():
        return None
    return _read_csv(str(p), tuple(sorted(read.items())), p.stat().st_mtime)


def load(key):
    spec = DATASETS[key]
    return load_csv(spec["file"], **spec.get("read", {}))


def run_info():
    p = path("run_info.json")
    return json.loads(p.read_text()) if p.exists() else {}


def primary_model():
    info = run_info()
    if info.get("primary"):
        return info["primary"]
    cv = load("cv_summary")
    return None if cv is None else cv["specificity_mean"].idxmax()


def image(name):
    p = path(name)
    return str(p) if p.exists() else None
