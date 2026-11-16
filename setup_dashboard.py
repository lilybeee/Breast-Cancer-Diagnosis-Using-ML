"""
setup_dashboard.py  -  generates (and extends) the Streamlit results dashboard

One file creates the whole project, so you can share just this file with teammates.

  python setup_dashboard.py                              # create ./dashboard (existing files are kept)
  python setup_dashboard.py --dir my_dash --force        # create elsewhere / overwrite generated files
  python setup_dashboard.py --new-page "Calcification"   # add pages/<n>_Calcification.py from the template
  python setup_dashboard.py --add-dataset cal_cv cal_cv_summary.csv --index-col
                                                         # register a new CSV in data_loader.DATASETS
  python setup_dashboard.py --check                      # list which data files are found / missing

Then:  cd dashboard && pip install -r requirements.txt && streamlit run app.py
Copy the notebook's outputs/ files into dashboard/data/ (see DASHBOARD_GUIDE.md).
"""
import argparse
import importlib
import re
import sys
from pathlib import Path

FILES = {}

# ----------------------------------------------------------------------------------------------
FILES["requirements.txt"] = r'''streamlit>=1.30
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.2
'''

FILES[".streamlit/config.toml"] = r'''[theme]
primaryColor = "#C2185B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F9EEF3"
textColor = "#2B1A33"
'''

FILES["data/README.txt"] = r'''Put the notebook's outputs/ files in this folder (names must match data_loader.DATASETS).

From Breast_Mass_EndTerm_Final.ipynb (Part C):
  cv_endterm_summary.csv  cv_endterm_per_fold.csv  cv_midterm_summary.csv
  test_endterm_results.csv  test_midterm_results.csv
  midterm_vs_endterm_test.csv  midterm_vs_wide_tuning_cv.csv
  stat_tests_vs_LR.csv  paired_bootstrap_auc_vs_LR.csv
  lr_odds_ratios.csv  permutation_importance.csv  error_analysis_primary.csv
  final_hyperparameters.csv  test_scores.csv  dev_oof_scores.csv  run_info.json
  B4_calibration.png  A_decision_tree.png  A_lr_coefficients.png   (optional images)

calcification_summary.csv is filled from the team's calcification slides (model, cv_auc, test_auc).
Replace it with the calcification notebook's real export if you have one.
'''

FILES["data/calcification_summary.csv"] = r'''model,cv_auc,test_auc
Logistic Regression,0.8067,0.749
Logistic Regression (poly),0.8125,0.781
SVM,0.7955,0.792
k-NN,0.7810,0.778
Decision Tree,0.7788,0.763
Bernoulli NB,0.8054,0.754
Soft-Vote Ensemble,,0.778
'''

# ----------------------------------------------------------------------------------------------
FILES["data_loader.py"] = r'''"""
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
    "cv_folds":       dict(file="cv_endterm_per_fold.csv"),
    "test":           dict(file="test_endterm_results.csv", read=dict(index_col=0)),
    "test_mid":       dict(file="test_midterm_results.csv", read=dict(index_col=0)),
    "mid_vs_end":     dict(file="midterm_vs_endterm_test.csv", read=dict(index_col=0)),
    "mid_vs_wide":    dict(file="midterm_vs_wide_tuning_cv.csv", read=dict(index_col=0)),
    "stats":          dict(file="stat_tests_vs_LR.csv"),
    "paired_auc":     dict(file="paired_bootstrap_auc_vs_LR.csv"),
    "odds_ratios":    dict(file="lr_odds_ratios.csv", read=dict(index_col=0)),
    "perm_imp":       dict(file="permutation_importance.csv", read=dict(index_col=0)),
    "error_analysis": dict(file="error_analysis_primary.csv", read=dict(index_col=0)),
    "hyperparams":    dict(file="final_hyperparameters.csv"),
    "test_scores":    dict(file="test_scores.csv"),
    "dev_oof":        dict(file="dev_oof_scores.csv"),
    "calcification":  dict(file="calcification_summary.csv"),
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
'''

# ----------------------------------------------------------------------------------------------
FILES["ui.py"] = r'''"""ui.py - shared page setup, banner, data-file status and missing-data handling."""
import streamlit as st

import data_loader as dl


def setup(title):
    """Call first on every page."""
    st.set_page_config(page_title=f"{title} | Breast Mass Dashboard", layout="wide")
    st.caption("Research prototype on a curated dataset (CBIS-DDSM masses). "
               "Not a clinical tool; no external validation.")
    if dl.run_info().get("fast_mode"):
        st.error("These files come from a FAST_MODE dry run. Do not report these numbers.")
    with st.sidebar:
        st.markdown("**Data files**")
        for key, ok in dl.available().items():
            st.write(("[x] " if ok else "[ ] ") + key)


def need(*keys):
    """Return the requested datasets, or show a clear message and stop the page."""
    missing = [k for k in keys if dl.load(k) is None]
    if missing:
        names = ", ".join(dl.DATASETS[k]["file"] for k in missing)
        st.warning(f"Missing data file(s): {names}. Copy them from the notebook's outputs/ folder into {dl.DATA_DIR}.")
        st.stop()
    return [dl.load(k) for k in keys]


def fmt_ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}-{hi:.3f}]"


def fmt_ms(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"
'''

# ----------------------------------------------------------------------------------------------
FILES["charts.py"] = r'''"""charts.py - metric maths and matplotlib figures used by the pages (no file access here)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score, roc_curve)

PINK, PURPLE, MID = "#C2185B", "#5E4B6B", "#7B4B94"


# ---------- metrics (same logic as the notebook) ----------
def thr_for_recall(y, s, target):
    """Highest threshold that still reaches the recall target (identical to the notebook)."""
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr >= target)]


def op_metrics(y, s, thr):
    y = np.asarray(y); pred = np.asarray(s) >= thr
    tp = int((pred & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                precision=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1))


def ppv_npv(sens, spec, prev):
    prev = np.asarray(prev, dtype=float)
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    return ppv, npv


# ---------- figures ----------
def dot_plot(end, mid, metric, models, title):
    """Mean +/- std per model; filled = end-term, hollow = midterm (if given)."""
    order = sorted(models, key=lambda m: end.loc[m, f"{metric}_mean"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.6))
    for i, m in enumerate(order):
        ax.errorbar(end.loc[m, f"{metric}_mean"], i, xerr=end.loc[m, f"{metric}_std"], fmt="o", color=PINK, capsize=3)
        if mid is not None and m in mid.index:
            ax.errorbar(mid.loc[m, f"{metric}_mean"], i + 0.2, xerr=mid.loc[m, f"{metric}_std"], fmt="o",
                        mfc="white", color=PURPLE, capsize=3, alpha=.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_title(title); ax.grid(axis="x", alpha=.3)
    ax.plot([], [], "o", color=PINK, label="end-term")
    if mid is not None:
        ax.plot([], [], "o", mfc="white", color=PURPLE, label="midterm")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    return fig


def roc_pr(te, models, ops, highlight=None):
    """Test ROC (left) and precision-recall (right) curves; dots mark each model's selected operating point."""
    y = te["y"].values
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for m in models:
        s = te[m].values; lw = 2.6 if m == highlight else 1.3
        fpr, tpr, _ = roc_curve(y, s); line, = axs[0].plot(fpr, tpr, lw=lw, label=f"{m} ({roc_auc_score(y, s):.3f})")
        pr, rc, _ = precision_recall_curve(y, s); axs[1].plot(rc, pr, lw=lw, color=line.get_color(), label=f"{m} ({average_precision_score(y, s):.3f})")
        o = ops[m]
        axs[0].plot(1 - o["specificity"], o["recall"], "o", color=line.get_color(), mec="k")
        axs[1].plot(o["recall"], o["precision"], "o", color=line.get_color(), mec="k")
    axs[0].plot([0, 1], [0, 1], "k--", lw=.8); axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate")
    axs[0].set_title("Test ROC (AUC); dots = selected operating point"); axs[0].legend(fontsize=7, loc="lower right")
    axs[1].axhline(y.mean(), color="k", ls="--", lw=.8); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_title("Test precision-recall (PR-AUC)"); axs[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def prevalence_fig(sens, spec, marks, current):
    prev = np.linspace(0.05, 0.8, 100); ppv, npv = ppv_npv(sens, spec, prev)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prev, ppv, color=PINK, label="PPV (precision)"); ax.plot(prev, npv, color=PURPLE, label="NPV")
    for pv, lab in marks.items():
        ax.axvline(pv, color="gray", ls=":"); ax.text(pv, 0.02, lab, rotation=90, fontsize=8)
    ax.axvline(current, color=MID, lw=2)
    ax.set_xlabel("Malignant prevalence"); ax.set_ylim(0, 1.02); ax.legend(); fig.tight_layout()
    return fig


def odds_ratio_fig(orx, top_n=16):
    d = orx.copy(); d["strength"] = np.abs(np.log(d["odds_ratio"]))
    d = d.sort_values("strength", ascending=False).head(top_n).sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(d) + 1.4)); yy = np.arange(len(d))
    cols = [PINK if bool(v) else "#9A8AA5" for v in d["ci_excludes_1"]]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=cols[i], lw=1.5)
        ax.plot(r.odds_ratio, i, "o", color=cols[i])
    ax.axvline(1, color="k", ls="--", lw=.8); ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8]); ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.minorticks_off()
    ax.set_yticks(yy); ax.set_yticklabels(d.index, fontsize=8)
    ax.set_title("LR odds ratios (95% patient-bootstrap CI); pink = CI excludes 1"); fig.tight_layout()
    return fig


def hbar(series, title, color=PURPLE):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(s) + 1.4))
    ax.barh(s.index, s.values, color=color); ax.set_title(title); fig.tight_layout()
    return fig


def grouped_bar(df, a_col, b_col, a_label, b_label, title):
    fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(df)); w = .38
    ax.bar(x - w / 2, df[a_col] - .5, w, bottom=.5, color=PINK, label=a_label)
    ax.bar(x + w / 2, df[b_col] - .5, w, bottom=.5, color=PURPLE, label=b_label)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(.5, max(df[a_col].max(), df[b_col].max()) + .05); ax.set_ylabel("ROC-AUC (axis starts at 0.5 = chance)")
    ax.set_title(title); ax.legend(); fig.tight_layout()
    return fig
'''

# ----------------------------------------------------------------------------------------------
FILES["app.py"] = r'''"""
app.py - dashboard entry point = page 1 (Overview).   Run:  streamlit run app.py
Other pages live in pages/ (Streamlit lists them in the sidebar automatically).
"""
import streamlit as st

import data_loader as dl
import ui

ui.setup("Overview")
st.title("Breast Mass Lesion Classification: Results Dashboard")
st.write("Benign vs malignant prediction from radiologist-described features (CBIS-DDSM masses). "
         "Primary metric: **specificity at >= 90% sensitivity**, because a missed cancer costs more than a false alarm.")

cv, test = ui.need("cv_summary", "test")
info = dl.run_info(); primary = dl.primary_model()
if primary not in test.index:
    st.error(f"Primary model '{primary}' not found in test results."); st.stop()

st.subheader(f"Primary model: {primary}")
st.caption("Chosen on development cross-validation only; the test set was scored once.")
t, c = test.loc[primary], cv.loc[primary]

k = st.columns(5)
k[0].metric("Test ROC-AUC", f"{t.roc_auc:.3f}"); k[0].caption(f"95% CI {t.auc_lo:.3f}-{t.auc_hi:.3f}")
k[1].metric("Test sensitivity", f"{t.recall:.3f}"); k[1].caption(f"95% CI {t.recall_lo:.3f}-{t.recall_hi:.3f}")
k[2].metric("Test specificity", f"{t.specificity:.3f}"); k[2].caption(f"95% CI {t.spec_lo:.3f}-{t.spec_hi:.3f}")
k[3].metric("Test precision (PPV)", f"{t.precision:.3f}"); k[3].caption("depends on prevalence")
k[4].metric("Test NPV", f"{t.npv:.3f}")

k = st.columns(4)
k[0].metric("CV ROC-AUC", f"{c.roc_auc_mean:.3f}"); k[0].caption(f"std {c.roc_auc_std:.3f} over folds")
k[1].metric("CV specificity at recall >= 0.90", f"{c.specificity_mean:.3f}"); k[1].caption(f"std {c.specificity_std:.3f}")
if info:
    k[2].metric("Train / test rows", f"{info['n_train']} / {info['n_test']}")
    k[3].metric("Malignant share train / test", f"{info['prev_train']:.1%} / {info['prev_test']:.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Protocol")
    st.markdown(
        "- Official train/test split; **no patient appears in both**\n"
        "- Patient-grouped **nested** cross-validation (5 outer x 3 inner folds, several seeds)\n"
        "- Threshold set on **training** data so that recall >= 0.90, then applied unchanged\n"
        "- Test set evaluated **once**; 95% CIs by resampling whole patients\n"
        + (f"- {info['n_features']} features ({info['n_numeric']} numeric + {info['n_binary']} binary flags)\n" if info else ""))
    st.code("data -> grouped nested CV -> pick primary model (dev CV only)\n"
            "     -> refit on all training rows -> ONE test evaluation", language="text")
with right:
    st.subheader("What to keep in mind")
    st.markdown(
        "- Models may be statistically **tied**; see *Model Comparison* before calling any model best\n"
        "- Precision depends on prevalence (train and test differ); see *Operating Point*\n"
        "- Curated dataset of already-detected lesions; **not** a screening tool\n"
        "- Next steps: ROI-mask shape features, CNN on ROI crops, lesion-level scoring, external validation")
'''

# ----------------------------------------------------------------------------------------------
FILES["pages/1_Model_Comparison.py"] = r'''"""Page 2 - Model comparison: results table, CV comparison, statistics, midterm vs end-term, mass vs calcification."""
import pandas as pd
import streamlit as st

import charts
import data_loader as dl
import ui

ui.setup("Model Comparison")
st.title("Model comparison")
cv, test = ui.need("cv_summary", "test")
mid = dl.load("cv_mid")

fam = pd.Series({m: dl.FAMILY.get(m, "Other") for m in cv.index})
with st.sidebar:
    fams = st.multiselect("Model family", sorted(fam.unique()), default=sorted(fam.unique()))
    metric = st.selectbox("CV metric", ["roc_auc", "specificity", "pr_auc", "precision", "npv"])
models = [m for m in cv.index if fam[m] in fams]
if not models:
    st.info("Select at least one model family."); st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Results table", "CV comparison", "Is the difference real?", "Midterm vs end-term and calcification"])

with tab1:
    rows = []
    for m in models:
        c, t = cv.loc[m], (test.loc[m] if m in test.index else None)
        row = {"Model": m, "Family": fam[m],
               "CV AUC": ui.fmt_ms(c.roc_auc_mean, c.roc_auc_std),
               "CV specificity@recall0.90": ui.fmt_ms(c.specificity_mean, c.specificity_std)}
        if t is not None:
            row.update({"Test AUC [95% CI]": ui.fmt_ci(t.roc_auc, t.auc_lo, t.auc_hi), "Test recall": round(t.recall, 3),
                        "Test specificity [95% CI]": ui.fmt_ci(t.specificity, t.spec_lo, t.spec_hi),
                        "Test precision": round(t.precision, 3), "Test NPV": round(t.npv, 3)})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table)
    st.download_button("Download table (CSV)", table.to_csv().encode(), "model_comparison.csv", "text/csv")
    st.caption("No model is marked 'best' here on purpose: overlapping intervals mean models are tied unless the statistics tab says otherwise.")

with tab2:
    st.pyplot(charts.dot_plot(cv, mid, metric, models, f"Nested-CV {metric} (mean +/- std over folds)"))
    st.caption("Filled = end-term (wider tuning), hollow = midterm. Overlapping bars = no clear winner.")

with tab3:
    stats, paired = dl.load("stats"), dl.load("paired_auc")
    if stats is None and paired is None:
        st.info("Statistics files not found (stat_tests_vs_LR.csv, paired_bootstrap_auc_vs_LR.csv).")
    if stats is not None:
        s = stats[stats["model"].isin(models)].copy()
        s["conclusion"] = s["p_holm"].apply(lambda p: "no significant difference" if p >= 0.05 else "significantly different")
        st.markdown("**Cross-validation: corrected resampled t-test vs Logistic Regression (Holm-adjusted)**")
        st.dataframe(s.round(4).set_index(["metric", "model"]))
    if paired is not None:
        p = paired[paired["model"].isin(models)].copy()
        p["conclusion"] = p["CI excludes 0"].apply(lambda v: "AUC differs from LR" if bool(v) else "no demonstrated difference")
        st.markdown("**Test set: paired patient-bootstrap AUC difference vs Logistic Regression**")
        st.dataframe(p.round(4).set_index("model"))
    st.caption("A model is only better than the baseline if the adjusted p-value is < 0.05 or the bootstrap CI excludes 0.")

with tab4:
    wide, mve = dl.load("mid_vs_wide"), dl.load("mid_vs_end")
    if wide is not None:
        st.markdown("**Effect of wider hyperparameter tuning (nested-CV means)**"); st.dataframe(wide.round(3))
    if mve is not None:
        st.markdown("**Midterm vs end-term on the test set**"); st.dataframe(mve.round(3))
    cal = dl.load("calcification")
    if cal is not None:
        cal = cal.set_index("model")
        shared = [m for m in cv.index if m in cal.index]
        if shared:
            comp = pd.DataFrame({"Mass CV AUC": cv.loc[shared, "roc_auc_mean"], "Calcification CV AUC": cal.loc[shared, "cv_auc"],
                                 "Mass test AUC": test.loc[[m for m in shared if m in test.index], "roc_auc"],
                                 "Calcification test AUC": cal.loc[shared, "test_auc"]})
            st.markdown("**Mass vs calcification (models both projects trained)**")
            st.pyplot(charts.grouped_bar(comp.dropna(subset=["Mass CV AUC", "Calcification CV AUC"]), "Mass CV AUC",
                                         "Calcification CV AUC", "Mass", "Calcification", "Nested-CV ROC-AUC"))
            st.dataframe(comp.round(3))
            st.caption("Indicative only: the two projects used different features, tuning metrics and threshold protocols.")
'''

# ----------------------------------------------------------------------------------------------
FILES["pages/2_Operating_Point.py"] = r'''"""Page 3 - Operating point: ROC/PR curves, target-sensitivity slider, confusion matrix, prevalence effect, calibration."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

import charts
import data_loader as dl
import ui

ui.setup("Operating Point")
st.title("Operating point explorer")
dev, te = ui.need("dev_oof", "test_scores")
info = dl.run_info(); primary = dl.primary_model()
models = [c for c in te.columns if c != "y" and c in dev.columns]
default = [m for m in dict.fromkeys([primary, "Logistic Regression"]) if m in models]

with st.sidebar:
    sel = st.multiselect("Models", models, default=default)
    target = st.slider("Target sensitivity", 0.80, 0.99, float(info.get("recall_target", 0.90)), 0.01,
                       help="Threshold is chosen on training out-of-fold scores, then applied once to the test set.")
if not sel:
    st.info("Select at least one model."); st.stop()

ops, rows = {}, []
for m in sel:
    thr = charts.thr_for_recall(dev["y"].values, dev[m].values, target)
    o = charts.op_metrics(te["y"].values, te[m].values, thr); o["threshold"] = thr; ops[m] = o
    rows.append({"Model": m, "Test recall": o["recall"], "Test specificity": o["specificity"], "Precision": o["precision"],
                 "NPV": o["npv"], "Test ROC-AUC": roc_auc_score(te["y"], te[m]), "Threshold": o["threshold"]})
st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))
st.pyplot(charts.roc_pr(te, sel, ops, highlight=primary))

ref = dl.load("test")
if ref is not None and abs(target - float(info.get("recall_target", 0.90))) < 1e-9:
    ok = all(m in ref.index and abs(ref.loc[m, "specificity"] - ops[m]["specificity"]) < 1e-3 for m in sel)
    st.caption("Consistency check against the notebook table: " + ("matches." if ok else "DIFFERS - re-export the notebook outputs."))

st.subheader("Confusion matrix and what it means")
cm_model = st.selectbox("Model", sel, index=sel.index(primary) if primary in sel else 0)
o = ops[cm_model]
cm = pd.DataFrame([[o["tn"], o["fp"]], [o["fn"], o["tp"]]], index=["Actual benign", "Actual malignant"],
                  columns=["Predicted benign", "Predicted malignant"])
c1, c2 = st.columns([1, 2])
c1.dataframe(cm)
c2.markdown(f"Per 100 malignant lesions, **{o['recall'] * 100:.0f}** are caught. "
            f"Per 100 benign lesions, **{(1 - o['specificity']) * 100:.0f}** are flagged as a false alarm.")

st.subheader("Why precision depends on prevalence")
prev_default = float(info.get("prev_test", 0.39))
prev = st.slider("Malignant prevalence of the population", 0.05, 0.80, prev_default, 0.01)
ppv, npv = charts.ppv_npv(o["recall"], o["specificity"], prev)
m1, m2 = st.columns(2); m1.metric("PPV at this prevalence", f"{float(ppv):.3f}"); m2.metric("NPV at this prevalence", f"{float(npv):.3f}")
marks = {info.get("prev_train", 0.483): "train", prev_default: "test"}
st.pyplot(charts.prevalence_fig(o["recall"], o["specificity"], marks, prev))
st.caption("Sensitivity and specificity are held fixed; only the share of malignant lesions changes.")

cal_img = dl.image("B4_calibration.png")
if cal_img:
    with st.expander("Calibration of the primary model (from the notebook)"):
        st.image(cal_img)
        st.caption("Points near the diagonal mean predicted probabilities match observed malignancy rates.")
'''

# ----------------------------------------------------------------------------------------------
FILES["pages/3_Explainability.py"] = r'''"""Page 4 - Explainability: odds ratios, permutation importance, error analysis."""
import streamlit as st

import charts
import data_loader as dl
import ui

ui.setup("Explainability")
st.title("Which features drive the predictions?")
orx, perm = dl.load("odds_ratios"), dl.load("perm_imp")
if orx is None and perm is None:
    ui.need("odds_ratios", "perm_imp")

left, right = st.columns(2)
with left:
    if orx is not None:
        st.subheader("Logistic Regression odds ratios")
        top_n = st.slider("Features shown", 5, len(orx), min(16, len(orx)))
        st.pyplot(charts.odds_ratio_fig(orx, top_n))
        st.caption("Odds ratio > 1 raises malignancy odds (numeric features: per 1 SD). If the interval crosses 1 the effect is not reliable.")
with right:
    if perm is not None:
        st.subheader("Permutation importance (test set)")
        model = st.selectbox("Model", list(perm.columns))
        st.pyplot(charts.hbar(perm[model].sort_values(ascending=False).head(10), f"{model}: drop in test ROC-AUC when shuffled"))
        st.caption("Shows what the fitted model actually uses on unseen data.")

ea = dl.load("error_analysis")
if ea is not None:
    st.subheader("Missed vs caught malignant lesions")
    st.dataframe(ea.reindex(ea["diff (FN-TP)"].abs().sort_values(ascending=False).index).head(8).round(3))
    st.caption("FN = missed malignant lesions, TP = caught. Large differences show where the features give the model little signal.")

tree = dl.image("A_decision_tree.png")
if tree:
    with st.expander("Midterm decision tree (depth 3)"):
        st.image(tree)
'''

# ----------------------------------------------------------------------------------------------
PAGE_TEMPLATE = r'''"""__TITLE__  (page created by setup_dashboard.py --new-page)

How a page works
  1. ui.setup()  sets the title, banner and the sidebar data-file status.   (always first)
  2. ui.need("key", ...)  loads datasets registered in data_loader.DATASETS and stops with a
     clear message if a file is missing.  Pages never read files directly.
  3. Build the page with st.* widgets and charts.* helpers (or plain matplotlib + st.pyplot).
Use a NEW csv:  python setup_dashboard.py --add-dataset my_key my_file.csv [--index-col]
                then call ui.need("my_key") here.
"""
import pandas as pd
import streamlit as st

import charts
import data_loader as dl
import ui

ui.setup("__TITLE__")
st.title("__TITLE__")

(summary,) = ui.need("cv_summary")        # <- replace with the dataset key(s) this page needs

st.subheader("Table")
st.dataframe(summary.round(3))

st.subheader("Chart")
st.pyplot(charts.dot_plot(summary, None, "roc_auc", list(summary.index), "Nested-CV ROC-AUC"))
'''


# ----------------------------------------------------------------------------------------------
def write_project(root, force):
    created, kept = [], []
    for rel, content in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not force:
            kept.append(rel)
            continue
        p.write_text(content, encoding="utf-8")
        created.append(rel)
    (root / "data").mkdir(exist_ok=True)
    print(f"created {len(created)} file(s), kept {len(kept)} existing file(s) in {root}")
    for rel in created:
        print("  +", rel)
    print(f"\nNext: copy the notebook outputs into {root / 'data'}, then\n"
          f"  cd {root} && pip install -r requirements.txt && streamlit run app.py")


def new_page(root, title):
    pages = root / "pages"
    if not (root / "data_loader.py").exists():
        sys.exit("Project not found. Run `python setup_dashboard.py` first (or pass --dir).")
    pages.mkdir(exist_ok=True)
    nums = [int(m.group(1)) for f in pages.glob("*.py") if (m := re.match(r"^(\d+)_", f.name))]
    slug = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_") or "Page"
    path = pages / f"{max(nums, default=0) + 1}_{slug}.py"
    if path.exists():
        sys.exit(f"{path} already exists")
    path.write_text(PAGE_TEMPLATE.replace("__TITLE__", title), encoding="utf-8")
    print(f"created {path}\nEdit it, then refresh the browser (Streamlit lists it in the sidebar automatically).")


def add_dataset(root, key, filename, index_col):
    loader = root / "data_loader.py"
    if not loader.exists():
        sys.exit("Project not found. Run `python setup_dashboard.py` first (or pass --dir).")
    text = loader.read_text(encoding="utf-8")
    if f'"{key}":' in text:
        sys.exit(f'dataset key "{key}" already exists in data_loader.py')
    marker = "    # <<NEW_DATASETS>>"
    if marker not in text:
        sys.exit("marker '# <<NEW_DATASETS>>' not found in data_loader.py; add the entry manually.")
    read = ", read=dict(index_col=0)" if index_col else ""
    line = f'    "{key}": dict(file="{filename}"{read}),\n'
    loader.write_text(text.replace(marker, line + marker), encoding="utf-8")
    print(f'registered dataset "{key}" -> data/{filename}. Use ui.need("{key}") in a page.')


def check(root):
    if not (root / "data_loader.py").exists():
        sys.exit("Project not found.")
    sys.path.insert(0, str(root.resolve()))
    dl = importlib.import_module("data_loader")
    ok = dl.available()
    for k, v in ok.items():
        print(("[x] " if v else "[ ] ") + f"{k:15s} {dl.DATASETS[k]['file']}")
    print(f"\n{sum(ok.values())}/{len(ok)} files found in {dl.DATA_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="dashboard", help="project folder (default: dashboard)")
    ap.add_argument("--force", action="store_true", help="overwrite generated files")
    ap.add_argument("--new-page", metavar="TITLE")
    ap.add_argument("--add-dataset", nargs=2, metavar=("KEY", "FILE"))
    ap.add_argument("--index-col", action="store_true", help="with --add-dataset: first CSV column is the index")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.dir)
    if a.new_page:
        new_page(root, a.new_page)
    elif a.add_dataset:
        add_dataset(root, a.add_dataset[0], a.add_dataset[1], a.index_col)
    elif a.check:
        check(root)
    else:
        write_project(root, a.force)


if __name__ == "__main__":
    main()
