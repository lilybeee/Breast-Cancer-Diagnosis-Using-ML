"""
evaluate.py - WDBC evaluation, figures and threshold analysis.

Consumes the artefacts written by train.py, scores every tuned model on the
held-out test set, builds the publication figures and selects a recommended
decision threshold from TRAINING-ONLY cross-validated predictions.

Run:   python src/evaluate.py          (after src/train.py, from wdbc/)

Writes
  reports/test_results.csv             per-model test metrics
  reports/model_comparison.csv         the consolidated CV + test comparison table
  reports/confusion_matrices.csv       tn/fp/fn/tp per model on the test set
  reports/threshold_analysis.csv       metrics vs threshold from training CV predictions
  reports/threshold_analysis_test.csv  the same sweep on the test set (reporting only)
  reports/selected_threshold.json      the recommended threshold and why
  reports/feature_importance.csv       tree/ensemble importances + LR coefficients
  reports/best_model_report.json       classification report of the selected model
  reports/eda_summary.csv              per-feature summary split by diagnosis
  reports/feature_glossary.csv         plain-language description of the 30 features
  reports/figures/*.png                the eight figure groups
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import wdbc_style as S
from train import (FEATURE_GLOSSARY, SEED, SUFFIX_GLOSSARY, load_data,
                   load_raw, split)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

METRICS = ["accuracy", "precision", "recall", "specificity", "f1",
           "roc_auc", "pr_auc", "balanced_accuracy"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def scores_of(pipe, X):
    """Probability of the malignant class, falling back to a decision function."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    d = pipe.decision_function(X)
    return (d - d.min()) / (d.max() - d.min() + 1e-12)


def point_metrics(y, proba, thr=0.5):
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "npv": tn / max(tn + fn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_cv_accuracy_bars(cv: pd.DataFrame):
    """1. Mean CV accuracy per model, error bars = standard deviation across folds."""
    d = cv.sort_values("cv_accuracy_mean")
    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(d) + 1.9))
    y = np.arange(len(d))
    ax.barh(y, d["cv_accuracy_mean"], xerr=d["cv_accuracy_std"],
            color=S.ACCENT, height=0.62, error_kw=dict(ecolor=S.INK_SOFT, capsize=3, lw=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.set_xlim(0.85, 1.005)
    for i, (m, s) in enumerate(zip(d["cv_accuracy_mean"], d["cv_accuracy_std"])):
        ax.text(m + s + 0.002, i, f"{m:.3f}", va="center", fontsize=8.5, color=S.INK)
    S.tidy(ax, xlabel="Cross-validated accuracy (5-fold x 3 repeats, training data only)",
           title="Mean CV accuracy by model, +/- 1 SD across the 15 folds", grid_axis="x")
    fig.tight_layout()
    return S.save(fig, FIGURES / "01_cv_accuracy_comparison.png")


def fig_roc(test_scores: pd.DataFrame, styles):
    """2. ROC curves, all models, one panel.  Every entry is directly labelled with its AUC."""
    y = test_scores["y"].to_numpy()
    models = [c for c in test_scores.columns if c != "y"]
    aucs = {m: roc_auc_score(y, test_scores[m]) for m in models}
    # Every curve hugs the top-left corner on this dataset, so the full view is
    # paired with a zoom on the region where the models actually differ.
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.9))
    for m in sorted(models, key=lambda k: -aucs[k]):
        c, ls = styles[m]
        fpr, tpr, _ = roc_curve(y, test_scores[m])
        lab = f"{m} - AUC {aucs[m]:.4f}"
        axs[0].plot(fpr, tpr, color=c, linestyle=ls, lw=2.0, label=lab)
        axs[1].plot(fpr, tpr, color=c, linestyle=ls, lw=2.0, label=lab)
    axs[0].plot([0, 1], [0, 1], color=S.INK_SOFT, ls=":", lw=1.0,
                label="chance - AUC 0.5000")
    axs[0].set_xlim(-0.01, 1.01)
    axs[0].set_ylim(-0.01, 1.01)
    S.tidy(axs[0], xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (malignant recall)",
           title="Full view", grid_axis="both")
    axs[1].set_xlim(-0.005, 0.25)
    axs[1].set_ylim(0.70, 1.005)
    S.tidy(axs[1], xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (malignant recall)",
           title="Zoom on the top-left corner", grid_axis="both")
    axs[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Test-set ROC curves", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.save(fig, FIGURES / "02_roc_curves.png")


def fig_pr(test_scores: pd.DataFrame, styles):
    """3. Precision-recall curves, all models, directly labelled with PR-AUC."""
    y = test_scores["y"].to_numpy()
    models = [c for c in test_scores.columns if c != "y"]
    aps = {m: average_precision_score(y, test_scores[m]) for m in models}
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for m in sorted(models, key=lambda k: -aps[k]):
        c, ls = styles[m]
        pr, rc, _ = precision_recall_curve(y, test_scores[m])
        ax.plot(rc, pr, color=c, linestyle=ls, lw=2.0,
                label=f"{m} - PR-AUC {aps[m]:.4f}")
    ax.axhline(y.mean(), color=S.INK_SOFT, ls=":", lw=1.0,
               label=f"prevalence - {y.mean():.4f}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.4, 1.01)
    S.tidy(ax, xlabel="Recall (malignant)", ylabel="Precision (malignant)",
           title="Test-set precision-recall curves", grid_axis="both")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return S.save(fig, FIGURES / "03_pr_curves.png")


def fig_confusions(cms: pd.DataFrame, best: str):
    """4. Confusion matrix per model on the held-out test set (small multiples)."""
    models = list(cms["model"])
    ncol, nrow = 5, int(np.ceil(len(models) / 5))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.75 * nrow))
    axs = np.atleast_1d(axs).ravel()
    vmax = cms[["tn", "fp", "fn", "tp"]].to_numpy().max()
    cmap = mpl_cmap()
    for ax, m in zip(axs, models):
        r = cms[cms["model"] == m].iloc[0]
        mat = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=13,
                        fontweight="bold",
                        color="#FFFFFF" if v > 0.55 * vmax else S.INK)
        ax.set_xticks([0, 1], ["pred B", "pred M"], fontsize=8.5)
        ax.set_yticks([0, 1], ["true B", "true M"], fontsize=8.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        title = f"{m}\nFN {int(r.fn)} | FP {int(r.fp)}"
        ax.set_title(title + ("  <- selected" if m == best else ""),
                     fontsize=9.5, loc="center",
                     color=S.ACCENT if m == best else S.INK)
    for ax in axs[len(models):]:
        ax.axis("off")
    fig.suptitle("Test-set confusion matrices (FN = missed malignancy)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.4)
    return S.save(fig, FIGURES / "04_confusion_matrices.png")


def mpl_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wdbc_blue", S.SEQ_BLUE)


def fig_metric_heatmap(table: pd.DataFrame):
    """5. All CV metrics x all models as a single-hue heat-map with the value in each cell."""
    cols = [f"cv_{m}_mean" for m in METRICS]
    d = table.set_index("model")[cols]
    d.columns = [m.replace("_", " ") for m in METRICS]
    d = d.loc[table.sort_values("cv_roc_auc_mean", ascending=False)["model"]]
    fig, ax = plt.subplots(figsize=(1.08 * len(d.columns) + 3.6, 0.46 * len(d) + 2.1))
    vals = d.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=mpl_cmap(), vmin=np.nanmin(vals), vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(d.columns)), d.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(d.index)), d.index, fontsize=9)
    lo, hi = np.nanmin(vals), 1.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="#FFFFFF" if (v - lo) / max(hi - lo, 1e-9) > 0.6 else S.INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cross-validated metrics by model (training data only)",
                 loc="left", color=S.INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="score")
    fig.tight_layout()
    return S.save(fig, FIGURES / "05_metric_heatmap.png")


def fig_feature_importance(imp: pd.DataFrame, top_n=15):
    """6. Impurity-based feature importance for the tree-based models."""
    models = [c for c in imp.columns if c != "feature"]
    fig, axs = plt.subplots(1, len(models), figsize=(4.4 * len(models), 0.3 * top_n + 2.3),
                            squeeze=False)
    for ax, m in zip(axs[0], models):
        d = imp[["feature", m]].sort_values(m, ascending=False).head(top_n).iloc[::-1]
        ax.barh(d["feature"], d[m], color=S.ACCENT_SOFT, height=0.66)
        for i, v in enumerate(d[m]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7.6, color=S.INK_SOFT)
        ax.set_xlim(0, d[m].max() * 1.26)
        ax.tick_params(axis="y", labelsize=8.2)
        S.tidy(ax, xlabel="importance", title=m, grid_axis="x")
    fig.suptitle(f"Impurity-based feature importance, top {top_n} (tree-based models)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return S.save(fig, FIGURES / "06_feature_importance.png")


def fig_lr_coefficients(coefs: pd.Series, top_n=20):
    """7. Standardised logistic-regression coefficients, signed."""
    d = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(d) + 2.0))
    cols = [S.BAD if v > 0 else S.SERIES[0] for v in d.values]
    ax.barh(d.index, d.values, color=cols, height=0.66)
    ax.axvline(0, color=S.INK_SOFT, lw=0.9)
    for i, v in enumerate(d.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.8, color=S.INK_SOFT)
    pad = max(abs(d.values).max() * 0.22, 0.1)
    ax.set_xlim(d.values.min() - pad, d.values.max() + pad)
    ax.tick_params(axis="y", labelsize=8.4)
    S.tidy(ax, xlabel="coefficient on standardised features (log-odds of malignant)",
           title=f"Logistic-regression coefficients, {top_n} largest by magnitude",
           grid_axis="x")
    ax.text(0.99, 0.02, "red = pushes toward malignant   blue = toward benign",
            transform=ax.transAxes, ha="right", fontsize=8, color=S.INK_SOFT)
    fig.tight_layout()
    return S.save(fig, FIGURES / "07_lr_coefficients.png")


def fig_training_time(cv: pd.DataFrame):
    """8. Fit time and grid-search time per model (log scale, both directly labelled)."""
    d = cv.sort_values("fit_seconds")
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 0.44 * len(d) + 2.1))
    for ax, col, lab, color in (
            (axs[0], "fit_seconds", "Fit time of the tuned model (s)", S.SERIES[0]),
            (axs[1], "tune_seconds", "Total grid-search time (s)", S.SERIES[1])):
        dd = d.sort_values(col)
        y = np.arange(len(dd))
        ax.barh(y, dd[col], color=color, height=0.62)
        ax.set_yticks(y, dd["model"], fontsize=9)
        ax.set_xscale("log")
        for i, v in enumerate(dd[col]):
            ax.text(v * 1.1, i, f"{v:.3g}s", va="center", fontsize=8.2, color=S.INK)
        ax.set_xlim(right=dd[col].max() * 3.2)
        S.tidy(ax, xlabel=lab + "  (log scale)", grid_axis="x")
    axs[1].set_title("Grid-search cost", loc="left")
    axs[0].set_title("Single-fit cost", loc="left")
    fig.suptitle("Computational cost by model", fontsize=12, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return S.save(fig, FIGURES / "08_training_time.png")


def fig_threshold(cvthr: pd.DataFrame, chosen: float, best: str):
    """9. Threshold trade-off, from training-only cross-validated predictions."""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.7))
    ax = axs[0]
    ax.plot(cvthr["threshold"], cvthr["recall"], color=S.SERIES[0], label="recall (sensitivity)")
    ax.plot(cvthr["threshold"], cvthr["precision"], color=S.SERIES[1], label="precision")
    ax.plot(cvthr["threshold"], cvthr["f1"], color=S.SERIES[2], ls="--", label="F1")
    ax.axvline(chosen, color=S.ACCENT, lw=1.6)
    ax.text(chosen, 0.42, f" chosen = {chosen:.2f}", color=S.ACCENT, fontsize=9, rotation=90)
    ax.axvline(0.5, color=S.INK_SOFT, ls=":", lw=1.0)
    ax.text(0.5, 0.42, " default 0.50", color=S.INK_SOFT, fontsize=9, rotation=90)
    ax.set_ylim(0.4, 1.02)
    S.tidy(ax, xlabel="decision threshold on P(malignant)", ylabel="score",
           title="Precision / recall trade-off", grid_axis="both")
    ax.legend(loc="lower right")

    ax = axs[1]
    ax.plot(cvthr["threshold"], cvthr["fn"], color=S.BAD, label="false negatives (missed malignancy)")
    ax.plot(cvthr["threshold"], cvthr["fp"], color=S.SERIES[0], label="false positives (false alarm)")
    ax.axvline(chosen, color=S.ACCENT, lw=1.6)
    ax.axvline(0.5, color=S.INK_SOFT, ls=":", lw=1.0)
    S.tidy(ax, xlabel="decision threshold on P(malignant)", ylabel="count (455 training cases)",
           title="Error counts", grid_axis="both")
    ax.legend(loc="upper center")
    fig.suptitle(f"Decision threshold for {best}, from 5-fold CV predictions on the "
                 f"training split only", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return S.save(fig, FIGURES / "09_threshold_tradeoff.png")


def fig_eda(df: pd.DataFrame):
    """Supporting EDA figure for the dashboard: class balance + the clearest separators."""
    fig, axs = plt.subplots(1, 4, figsize=(15.2, 3.9))
    counts = df["diagnosis"].value_counts().reindex(["B", "M"])
    axs[0].bar(["Benign", "Malignant"], counts.to_numpy(),
               color=[S.SERIES[0], S.ACCENT], width=0.55)
    for i, v in enumerate(counts.to_numpy()):
        axs[0].text(i, v + 6, f"{v}  ({v / len(df):.1%})", ha="center", fontsize=9)
    axs[0].set_ylim(0, counts.max() * 1.18)
    S.tidy(axs[0], ylabel="cases", title=f"Class balance (n = {len(df)})")

    for ax, feat in zip(axs[1:], ["concave_points_worst", "area_worst", "texture_mean"]):
        data = [df.loc[df["diagnosis"] == g, feat].to_numpy() for g in ("B", "M")]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        tick_labels=["Benign", "Malignant"],
                        medianprops=dict(color=S.INK, lw=1.6),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=S.INK_SOFT,
                                        markeredgecolor="none", alpha=0.5))
        for patch, c in zip(bp["boxes"], [S.SERIES[0], S.ACCENT]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor(S.SURFACE)
            patch.set_linewidth(2)
        S.tidy(ax, ylabel=feat.replace("_", " "), title=feat)
    fig.suptitle("WDBC exploratory view: class balance and the strongest single separators",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return S.save(fig, FIGURES / "00_eda_overview.png")


def fig_correlation(X: pd.DataFrame):
    """Supporting EDA figure: the feature correlation structure (diverging, grey midpoint)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "wdbc_div", ["#184f95", "#6da7ec", "#EDEDEA", "#f0a075", "#b8431a"])
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10.4, 9.0))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7.4)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7.4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature correlation matrix (Pearson) - size features are near-duplicates",
                 loc="left", color=S.INK)
    fig.colorbar(im, ax=ax, shrink=0.72, label="correlation")
    fig.tight_layout()
    return S.save(fig, FIGURES / "00_correlation_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    S.use_style()

    cv = pd.read_csv(REPORTS / "cv_results.csv")
    selection = json.loads((REPORTS / "model_selection.json").read_text())
    best_name = selection["selected_model"]

    fitted = joblib.load(MODELS / "tuned_models.joblib")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    raw = load_raw()

    models = list(cv["model"])
    styles = S.model_styles(models)

    # ---------------- test-set scoring ----------------
    test_scores = pd.DataFrame({"y": y_test.to_numpy()}, index=y_test.index)
    rows, cms = [], []
    for m in models:
        proba = scores_of(fitted[m], X_test)
        test_scores[m] = proba
        pm = point_metrics(y_test, proba, 0.5)
        rows.append({"model": m, **{k: pm[k] for k in METRICS + ["npv"]}})
        cms.append({"model": m, "tn": pm["tn"], "fp": pm["fp"],
                    "fn": pm["fn"], "tp": pm["tp"]})
    test_df = pd.DataFrame(rows)
    cm_df = pd.DataFrame(cms)
    test_df.to_csv(REPORTS / "test_results.csv", index=False)
    cm_df.to_csv(REPORTS / "confusion_matrices.csv", index=False)
    test_scores.to_csv(REPORTS / "test_scores.csv", index=False)

    # ---------------- consolidated comparison table ----------------
    comp = cv.merge(test_df.add_prefix("test_").rename(columns={"test_model": "model"}),
                    on="model", how="left")
    comp = comp.merge(cm_df, on="model", how="left")
    comp["selected"] = comp["model"] == best_name
    ordered = (["model", "family", "scaled", "best_params", "selection_score", "selected"]
               + [f"cv_{m}_{s}" for m in METRICS for s in ("mean", "std")]
               + [f"test_{m}" for m in METRICS + ["npv"]]
               + ["tn", "fp", "fn", "tp", "fit_seconds", "tune_seconds", "n_candidates"])
    comp = comp[[c for c in ordered if c in comp.columns]]
    comp.to_csv(REPORTS / "model_comparison.csv", index=False)

    # ---------------- threshold analysis (TRAINING DATA ONLY) ----------------
    thr_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    if hasattr(fitted[best_name], "predict_proba"):
        oof = cross_val_predict(fitted[best_name], X_train, y_train, cv=thr_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
    else:                                   # score-only estimator: min-max to [0, 1]
        d = cross_val_predict(fitted[best_name], X_train, y_train, cv=thr_cv,
                              method="decision_function", n_jobs=-1)
        oof = (d - d.min()) / (d.max() - d.min() + 1e-12)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    cv_thr = pd.DataFrame([{"threshold": t, **point_metrics(y_train, oof, t)} for t in grid])
    cv_thr.to_csv(REPORTS / "threshold_analysis.csv", index=False)

    # Recommended operating point: the highest threshold whose cross-validated
    # recall is still at or above the target.  A high threshold is preferred
    # among the qualifying ones because it keeps precision as high as possible.
    TARGET_RECALL = 0.98
    ok = cv_thr[cv_thr["recall"] >= TARGET_RECALL]
    chosen = float(ok["threshold"].max()) if len(ok) else 0.5
    at = cv_thr[cv_thr["threshold"] == chosen].iloc[0]
    default = cv_thr[cv_thr["threshold"] == 0.50].iloc[0]

    best_proba_test = test_scores[best_name].to_numpy()
    test_thr = pd.DataFrame([{"threshold": t, **point_metrics(y_test, best_proba_test, t)}
                             for t in grid])
    test_thr.to_csv(REPORTS / "threshold_analysis_test.csv", index=False)
    at_test = point_metrics(y_test, best_proba_test, chosen)

    (REPORTS / "selected_threshold.json").write_text(json.dumps({
        "model": best_name,
        "threshold": chosen,
        "target_recall": TARGET_RECALL,
        "selection_rule": ("highest threshold whose 5-fold cross-validated recall on the "
                           "TRAINING split is >= the target; the held-out test set plays "
                           "no part in this choice"),
        "training_cv_at_threshold": {k: float(at[k]) for k in
                                     ["recall", "precision", "specificity", "f1", "fn", "fp"]},
        "training_cv_at_default_0.5": {k: float(default[k]) for k in
                                       ["recall", "precision", "specificity", "f1", "fn", "fp"]},
        "test_at_threshold_reporting_only": {k: float(at_test[k]) for k in
                                             ["recall", "precision", "specificity", "f1", "fn", "fp"]},
    }, indent=2))

    # ---------------- interpretation artefacts ----------------
    imp = pd.DataFrame({"feature": X.columns})
    for m in models:
        est = fitted[m].named_steps["model"]
        if hasattr(est, "feature_importances_"):
            imp[m] = est.feature_importances_
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)

    lr = fitted["Logistic Regression"].named_steps["model"]   # always present in model_zoo()
    coefs = pd.Series(lr.coef_[0], index=X.columns, name="coefficient")
    coefs.rename_axis("feature").reset_index().assign(
        odds_ratio=np.exp(coefs.to_numpy())
    ).to_csv(REPORTS / "lr_coefficients.csv", index=False)

    best_pipe = fitted[best_name]
    rep = classification_report(y_test, (best_proba_test >= 0.5).astype(int),
                                target_names=["Benign", "Malignant"],
                                output_dict=True, zero_division=0)
    (REPORTS / "best_model_report.json").write_text(json.dumps({
        "model": best_name,
        "best_params": selection["best_params"],
        "threshold_used": 0.5,
        "classification_report": rep,
        "confusion_matrix": cm_df[cm_df["model"] == best_name].iloc[0].to_dict(),
    }, indent=2, default=str))

    # EDA tables for the dashboard page
    eda = (raw.drop(columns=["id"]).groupby("diagnosis").agg(["mean", "std"])
           .T.unstack(level=-1))
    eda.columns = ["_".join(map(str, c)) for c in eda.columns]
    eda.rename_axis("feature").reset_index().to_csv(REPORTS / "eda_summary.csv", index=False)

    gloss = []
    for col in X.columns:
        base, suffix = col.rsplit("_", 1)
        gloss.append({"feature": col, "measurement": base.replace("_", " "),
                      "statistic": suffix,
                      "what_it_measures": FEATURE_GLOSSARY[base],
                      "how_it_is_aggregated": SUFFIX_GLOSSARY[suffix]})
    pd.DataFrame(gloss).to_csv(REPORTS / "feature_glossary.csv", index=False)

    # ---------------- figures ----------------
    made = [
        fig_eda(raw.drop(columns=["id"])),
        fig_correlation(X),
        fig_cv_accuracy_bars(cv),
        fig_roc(test_scores, styles),
        fig_pr(test_scores, styles),
        fig_confusions(cm_df, best_name),
        fig_metric_heatmap(cv),
        fig_feature_importance(imp),
        fig_lr_coefficients(coefs),
        fig_training_time(cv),
        fig_threshold(cv_thr, chosen, best_name),
    ]

    print(f"selected model : {best_name}")
    print(f"test accuracy  : {test_df.loc[test_df['model'] == best_name, 'accuracy'].iloc[0]:.4f}")
    print(f"test recall    : {test_df.loc[test_df['model'] == best_name, 'recall'].iloc[0]:.4f}")
    print(f"test ROC-AUC   : {test_df.loc[test_df['model'] == best_name, 'roc_auc'].iloc[0]:.4f}")
    print(f"threshold      : {chosen:.2f} (training CV recall {at['recall']:.4f})")
    print(f"figures written: {len(made)}")
    for p in made:
        print("  ", Path(p).name)


if __name__ == "__main__":
    main()
