"""
WDBC nuclear-feature analysis - Streamlit page.

Wisconsin Diagnostic Breast Cancer: benign vs malignant from 30 numeric features
computed from digitised fine-needle-aspirate images of cell nuclei.

This page is deliberately SELF-CONTAINED.  It imports none of the shared dashboard
modules (data_loader, ui, charts) and touches no existing dashboard file, so the
mammogram and calcification work is completely unaffected by it.  Streamlit picks
the page up automatically from pages/ - no navigation change is needed.

Inputs: dashboard/data/wdbc/, written by  wdbc/src/export_to_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wdbc"
FIG_DIR = DATA_DIR / "figures"

st.set_page_config(page_title="WDBC Nuclear Feature Analysis | Breast Cancer Dashboard",
                   page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.45rem; padding-bottom: 2.5rem; max-width: 1450px;}
      .wd-eyebrow {color:#64748b;font-size:.75rem;font-weight:700;letter-spacing:.12em;
                   text-transform:uppercase;margin-bottom:.35rem;}
      .wd-hero {padding:1.1rem 1.3rem;border:1px solid rgba(128,128,128,.22);
                border-radius:13px;
                background:linear-gradient(115deg,rgba(194,24,91,.07),rgba(94,75,107,.03));}
      .wd-hero h3 {margin:0 0 .35rem 0;font-weight:650;}
      .wd-hero p {margin:0;color:#64748b;font-size:.94rem;}
      .wd-card {border:1px solid rgba(128,128,128,.23);border-radius:12px;
                padding:.8rem .95rem;min-height:104px;height:100%;}
      .wd-label {color:#64748b;font-size:.8rem;line-height:1.3;margin-bottom:.45rem;}
      .wd-value {font-size:clamp(1.05rem,1.6vw,1.6rem);font-weight:650;line-height:1.25;}
      .wd-help {color:#64748b;font-size:.75rem;margin-top:.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loading (this page reads only its own folder)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(name: str, mtime: float | None = None):
    p = DATA_DIR / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:                       # pragma: no cover - defensive
        st.caption(f"Could not read {name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_json(name: str, mtime: float | None = None):
    p = DATA_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:                       # pragma: no cover - defensive
        st.caption(f"Could not read {name}: {exc}")
        return None


def mtime(name: str):
    p = DATA_DIR / name
    return p.stat().st_mtime if p.exists() else None


def csv(name):
    return load_csv(name, mtime(name))


def js(name):
    return load_json(name, mtime(name))


def figure(name: str, caption: str | None = None, width: str = "stretch"):
    p = FIG_DIR / name
    if not p.exists():
        st.info(f"Figure not found: {name}. "
                "Run `python src/evaluate.py` then `python src/export_to_dashboard.py` "
                "in the wdbc/ folder.")
        return
    st.image(str(p), caption=caption, use_container_width=(width == "stretch"))


def card(label, value, help_text=""):
    st.markdown(
        f'<div class="wd-card"><div class="wd-label">{label}</div>'
        f'<div class="wd-value">{value}</div>'
        f'<div class="wd-help">{help_text}</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Guard: nothing exported yet
# ---------------------------------------------------------------------------
comparison = csv("model_comparison.csv")
split_info = js("split_info.json")

st.markdown('<div class="wd-eyebrow">Dataset 3 of 3 &mdash; independent analysis</div>',
            unsafe_allow_html=True)
st.title("WDBC nuclear feature analysis")

if comparison is None or split_info is None:
    st.warning(
        f"No WDBC results found in `{DATA_DIR}`.\n\n"
        "Generate them from the `wdbc/` project folder:\n\n"
        "```\npython src/train.py\npython src/evaluate.py\n"
        "python src/export_to_dashboard.py\n```")
    st.stop()

selection = js("model_selection.json") or {}
threshold = js("selected_threshold.json") or {}
best_name = selection.get("selected_model", comparison.iloc[0]["model"])
best_row = comparison[comparison["model"] == best_name].iloc[0]

st.markdown(
    f"""
    <div class="wd-hero">
      <h3>Benign vs malignant from 30 cell-nucleus measurements</h3>
      <p>Ten tuned classifiers compared on identical cross-validation folds, using the
      Wisconsin Diagnostic Breast Cancer dataset ({split_info['n_samples']} fine-needle
      aspirates). Separate from the mammogram and calcification work elsewhere in this
      dashboard &mdash; a different data source, different features and its own pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
st.caption("Educational classification experiment on a public research dataset. "
           "Not a clinically validated diagnostic system.")
st.write("")

tabs = st.tabs([
    "Overview", "EDA", "The 30 features", "Model comparison",
    "ROC & PR", "Confusion matrices", "Best model", "Interpretation",
    "Threshold", "Conclusions",
])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset overview")
    cc = split_info["class_counts"]
    c = st.columns(5)
    with c[0]:
        card("Cases", f"{split_info['n_samples']}", "fine-needle aspirate images")
    with c[1]:
        card("Features", f"{split_info['n_features']}", "10 measurements x 3 statistics")
    with c[2]:
        card("Benign / Malignant", f"{cc['benign']} / {cc['malignant']}",
             f"{split_info['malignant_rate']:.1%} malignant")
    with c[3]:
        card("Missing values", f"{split_info['missing_values']}",
             f"duplicate rows: {split_info['duplicate_rows']}")
    with c[4]:
        card("Train / Test", f"{split_info['n_train']} / {split_info['n_test']}",
             "80:20 stratified split")

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Sample distribution**")
        dist = pd.DataFrame({
            "Split": ["Train", "Train", "Test", "Test"],
            "Diagnosis": ["Benign", "Malignant", "Benign", "Malignant"],
            "Cases": [split_info["train_class_counts"]["benign"],
                      split_info["train_class_counts"]["malignant"],
                      split_info["test_class_counts"]["benign"],
                      split_info["test_class_counts"]["malignant"]],
        })
        st.bar_chart(dist.pivot(index="Split", columns="Diagnosis", values="Cases"),
                     height=260)
        st.caption("Stratified, so the malignant rate is the same in both splits.")
    with right:
        st.markdown("**Method**")
        st.markdown(
            f"""
- **Target** &mdash; diagnosis, with **malignant (M) as the positive class**.
- **Preprocessing** &mdash; the ID column is dropped; no missing values and no duplicate
  rows are present. `StandardScaler` is applied **inside a `Pipeline`**, so it is
  re-fitted on each training fold and never sees validation rows. Tree-based and
  Naive Bayes models are left unscaled.
- **Tuning** &mdash; `GridSearchCV` on `{split_info['inner_cv']}`, scored on
  `{split_info['tuning_metric']}`, **on the training split only**.
- **Comparison** &mdash; every tuned model re-scored on `{split_info['outer_cv']}`:
  15 folds, identical for all ten models.
- **Test set** &mdash; {split_info['n_test']} cases, read once, after selection.
            """)

    st.divider()
    c = st.columns(4)
    with c[0]:
        card("Selected model", best_name, "chosen on CV, not on test results")
    with c[1]:
        card("Test accuracy", f"{best_row['test_accuracy']:.1%}", "held-out split")
    with c[2]:
        card("Test recall (malignant)", f"{best_row['test_recall']:.1%}",
             f"{int(best_row['fn'])} missed of "
             f"{split_info['test_class_counts']['malignant']}")
    with c[3]:
        card("Test ROC-AUC", f"{best_row['test_roc_auc']:.4f}",
             f"CV {best_row['cv_roc_auc_mean']:.4f} +/- {best_row['cv_roc_auc_std']:.4f}")

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory data analysis")
    figure("00_eda_overview.png",
           "Class balance and the three features that separate the classes most cleanly.")
    st.markdown(
        "Malignant nuclei are larger (`area`, `radius`), more irregular in outline "
        "(`concave_points`, `concavity`) and more variable in grey-scale texture. "
        "The separation on `concave_points_worst` alone is already substantial, which "
        "is why even simple models do well on this dataset.")

    st.divider()
    figure("00_correlation_matrix.png",
           "Pearson correlation across the 30 features.")
    st.markdown(
        "The block structure matters for interpretation: `radius`, `perimeter` and "
        "`area` are near-perfect duplicates of one another (they are geometric "
        "functions of the same boundary), and each measurement's mean and worst "
        "variants are strongly correlated too. Individual coefficients should "
        "therefore be read as **groups** of related measurements rather than as "
        "independent effects.")

    eda = csv("eda_summary.csv")
    if eda is not None:
        st.divider()
        st.markdown("**Per-feature summary by diagnosis**")
        st.dataframe(eda.round(4), use_container_width=True, hide_index=True)
        st.download_button("Download EDA summary (CSV)",
                           eda.to_csv(index=False).encode(),
                           "wdbc_eda_summary.csv", "text/csv")

# ---------------------------------------------------------------------------
# 3. The 30 features
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("What the 30 features measure")
    st.markdown(
        "Each image is segmented into individual cell nuclei. **Ten** measurements are "
        "taken on every nucleus, and each is then summarised **three** ways across the "
        "nuclei in the image &mdash; giving 10 x 3 = 30 features.")

    gloss = csv("feature_glossary.csv")
    if gloss is None:
        st.info("feature_glossary.csv not found.")
    else:
        base = (gloss.drop_duplicates("measurement")[["measurement", "what_it_measures"]]
                .reset_index(drop=True))
        base.index = np.arange(1, len(base) + 1)
        stat = (gloss.drop_duplicates("statistic")[["statistic", "how_it_is_aggregated"]]
                .reset_index(drop=True))

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("**The ten nuclear measurements**")
            st.dataframe(base.rename(columns={"measurement": "Measurement",
                                              "what_it_measures": "What it measures"}),
                         use_container_width=True)
        with right:
            st.markdown("**The three summary statistics**")
            st.dataframe(stat.rename(columns={"statistic": "Suffix",
                                              "how_it_is_aggregated": "Meaning"}),
                         use_container_width=True, hide_index=True)
            st.caption("So `area_worst` is the mean area of the three largest nuclei in "
                       "the image, and `texture_se` is the standard error of the "
                       "grey-scale spread across nuclei.")

        st.divider()
        pick = st.multiselect("Filter the full list by measurement",
                              sorted(base["measurement"]), default=[])
        view = gloss if not pick else gloss[gloss["measurement"].isin(pick)]
        st.dataframe(view[["feature", "measurement", "statistic", "what_it_measures"]]
                     .rename(columns={"feature": "Feature", "measurement": "Measurement",
                                      "statistic": "Statistic",
                                      "what_it_measures": "What it measures"}),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Model comparison")
    st.caption("Cross-validated on the training split (15 shared folds). "
               "Test columns are reported, never used for selection.")

    headline = comparison[[
        "model", "cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
        "cv_f1_mean", "cv_roc_auc_mean", "cv_pr_auc_mean",
        "test_accuracy", "test_recall", "test_roc_auc"]].copy()
    headline.columns = ["Model", "CV Accuracy", "CV Precision", "CV Recall", "CV F1",
                        "CV ROC-AUC", "CV PR-AUC", "Test Accuracy", "Test Recall",
                        "Test ROC-AUC"]
    st.dataframe(headline.round(4), use_container_width=True, hide_index=True)
    st.download_button("Download comparison table (CSV)",
                       comparison.to_csv(index=False).encode(),
                       "wdbc_model_comparison.csv", "text/csv")

    st.markdown("**Best hyperparameters found by grid search**")
    hp = comparison[["model", "family", "scaled", "n_candidates", "best_params"]].copy()
    hp.columns = ["Model", "Family", "Scaled", "Grid size", "Best parameters"]
    st.dataframe(hp, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Interactive metric comparison**")
    metric_names = {"accuracy": "Accuracy", "precision": "Precision",
                    "recall": "Recall / sensitivity", "specificity": "Specificity",
                    "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                    "balanced_accuracy": "Balanced accuracy"}
    c1, c2 = st.columns([1, 1])
    with c1:
        metric = st.selectbox("Metric", list(metric_names),
                              format_func=lambda k: metric_names[k], index=5)
    with c2:
        source = st.radio("Source", ["Cross-validation (training)", "Held-out test"],
                          horizontal=True)

    col = f"cv_{metric}_mean" if source.startswith("Cross") else f"test_{metric}"
    if col not in comparison.columns:
        st.info(f"{metric_names[metric]} is not available for this source.")
    else:
        chart = comparison[["model", col]].set_index("model").sort_values(col)
        chart.columns = [metric_names[metric]]
        st.bar_chart(chart, height=330)
        if source.startswith("Cross"):
            sd = comparison[["model", f"cv_{metric}_std"]].set_index("model")
            st.caption(f"Fold-to-fold standard deviation ranges from "
                       f"{sd.min().iloc[0]:.4f} to {sd.max().iloc[0]:.4f}. "
                       "Differences smaller than that are not meaningful.")

    st.divider()
    left, right = st.columns(2)
    with left:
        figure("01_cv_accuracy_comparison.png")
    with right:
        figure("05_metric_heatmap.png")

    st.divider()
    st.markdown("**Are the differences meaningful?**")
    paired = csv("cv_paired_vs_best.csv")
    if paired is None:
        st.info("cv_paired_vs_best.csv not found.")
    else:
        pm = st.selectbox("Compare on", sorted(paired["metric"].unique()), index=1,
                          key="paired_metric")
        view = paired[paired["metric"] == pm][
            ["model", "reference", "mean_diff", "ci_lo", "ci_hi",
             "ci_excludes_0", "conclusion"]]
        view.columns = [f"Model", "Reference", "Mean difference", "CI low", "CI high",
                        "CI excludes 0", "Conclusion"]
        st.dataframe(view.round(4), use_container_width=True, hide_index=True)
        st.caption(
            "Every model saw the identical folds, so scores are differenced fold by "
            "fold (reference minus model). An interval containing 0 means the data do "
            "not demonstrate a difference. Repeated-CV folds overlap heavily, so the "
            "nominal 95% coverage is optimistic - read small gaps as ties.")

    st.divider()
    figure("08_training_time.png")
    st.caption("Accuracy is not the only axis: the linear models fit in milliseconds, "
               "the ensembles and the MLP cost orders of magnitude more for no "
               "measurable gain on this dataset.")

# ---------------------------------------------------------------------------
# 5. ROC & PR
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("ROC and precision-recall comparison")
    figure("02_roc_curves.png")
    st.caption("All curves hug the top-left corner on this dataset, so the full view "
               "is paired with a zoom on the region where the models actually differ.")
    st.divider()
    figure("03_pr_curves.png")
    st.caption("Precision-recall is the more informative view when the positive class "
               "is the minority (37% malignant); the dotted line is the prevalence "
               "baseline a random ranker would achieve.")

    st.divider()
    auc_tbl = comparison[["model", "cv_roc_auc_mean", "cv_pr_auc_mean",
                          "test_roc_auc", "test_pr_auc"]].copy()
    auc_tbl.columns = ["Model", "CV ROC-AUC", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC"]
    st.dataframe(auc_tbl.round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 6. Confusion matrices
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Confusion matrices on the held-out test set")
    cms = csv("confusion_matrices.csv")
    n_mal = split_info["test_class_counts"]["malignant"]
    st.caption(f"{split_info['n_test']} cases, {n_mal} of them malignant. "
               f"One extra miss moves recall by {1 / n_mal:.1%}, so differences of "
               "one or two cases between models are noise.")
    figure("04_confusion_matrices.png")

    if cms is not None:
        st.divider()
        pick = st.selectbox("Inspect one model", list(cms["model"]),
                            index=list(cms["model"]).index(best_name)
                            if best_name in list(cms["model"]) else 0)
        r = cms[cms["model"] == pick].iloc[0]
        t = comparison[comparison["model"] == pick].iloc[0]
        c = st.columns(4)
        with c[0]:
            card("True negatives", int(r["tn"]), "benign, called benign")
        with c[1]:
            card("False positives", int(r["fp"]), "benign, called malignant")
        with c[2]:
            card("False negatives", int(r["fn"]), "malignant, MISSED")
        with c[3]:
            card("True positives", int(r["tp"]), "malignant, caught")
        st.write("")
        c = st.columns(4)
        for col, (lab, key, hint) in zip(c, [
                ("Recall (sensitivity)", "test_recall", "share of malignant cases caught"),
                ("Specificity", "test_specificity", "share of benign cases cleared"),
                ("Precision", "test_precision", "share of malignant calls that were right"),
                ("NPV", "test_npv", "share of benign calls that were right")]):
            with col:
                card(lab, f"{t[key]:.4f}", hint)

        st.divider()
        st.markdown("**Missed malignancies by model (lower is better)**")
        fn_chart = cms.set_index("model")[["fn", "fp"]].rename(
            columns={"fn": "False negatives", "fp": "False positives"})
        st.bar_chart(fn_chart.sort_values("False negatives"), height=330)

# ---------------------------------------------------------------------------
# 7. Best model
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader(f"Selected model: {best_name}")
    if selection.get("selection_rule"):
        st.caption("Selection rule: " + selection["selection_rule"])

    c = st.columns(4)
    with c[0]:
        card("Hyperparameters",
             ", ".join(f"{k}={v}" for k, v in selection.get("best_params", {}).items())
             or "defaults", "chosen by grid search on the training split")
    with c[1]:
        card("CV ROC-AUC",
             f"{best_row['cv_roc_auc_mean']:.4f}",
             f"+/- {best_row['cv_roc_auc_std']:.4f} over 15 folds")
    with c[2]:
        card("CV recall",
             f"{best_row['cv_recall_mean']:.4f}",
             f"+/- {best_row['cv_recall_std']:.4f}")
    with c[3]:
        card("Fit time", f"{best_row['fit_seconds']:.4f}s",
             "single fit on the full training split")

    st.write("")
    ranking = pd.DataFrame(selection.get("ranking", []))
    if not ranking.empty:
        st.markdown("**Ranking that produced the selection** (lower score = better)")
        ranking.columns = [c.replace("cv_", "CV ").replace("_mean", "")
                           .replace("_", " ").title() for c in ranking.columns]
        st.dataframe(ranking.round(4), use_container_width=True, hide_index=True)
    n_tied = selection.get("n_ties_within_1se")
    if n_tied:
        st.info(f"**{n_tied} of {len(comparison)} models** sit within one standard "
                f"deviation of the leader's CV ROC-AUC. The selection is a choice among "
                f"near-ties, not a clear win &mdash; and among ties, the cheapest and "
                f"most interpretable model is the reasonable pick.")

    report = js("best_model_report.json")
    if report:
        st.divider()
        st.markdown("**Classification report on the held-out test set** (threshold 0.50)")
        rep = pd.DataFrame(report["classification_report"]).T
        st.dataframe(rep.round(4), use_container_width=True)

    st.divider()
    st.markdown("**Why this model performed well**")
    st.markdown(
        """
The WDBC features were engineered for exactly this task, and the two classes are close
to linearly separable in the 30-dimensional standardised space &mdash; the dataset
documentation notes they are linearly separable using all 30 features. A regularised
linear boundary is therefore already the right shape for the problem. With 455 training
rows and 30 correlated predictors, the L2 penalty also does real work: it is the main
defence against the collinearity among the radius / perimeter / area group.

On ROC-AUC, the paired fold-wise intervals for LDA, the MLP and the RBF SVM all contain
zero &mdash; they are statistically indistinguishable from it. LDA ties on AUC but reaches
it with markedly lower malignant recall (0.882 against 0.957), which is exactly why recall
carries weight in the selection rule. The tree ensembles and KNN are measurably behind.
**More complex did not mean better** on a small, well-separated, near-linear dataset.
        """)

# ---------------------------------------------------------------------------
# 8. Interpretation
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Feature importance and interpretation")
    st.markdown("**Logistic-regression coefficients** (standardised features)")
    figure("07_lr_coefficients.png")

    coefs = csv("lr_coefficients.csv")
    if coefs is not None:
        top = st.slider("Features to show", 5, 30, 12, key="coef_n")
        d = coefs.assign(magnitude=coefs["coefficient"].abs()) \
                 .sort_values("magnitude", ascending=False).head(top)
        st.bar_chart(d.set_index("feature")[["coefficient"]], height=360)
        st.dataframe(d[["feature", "coefficient", "odds_ratio"]].round(4),
                     use_container_width=True, hide_index=True)
        st.caption("A positive coefficient pushes the prediction toward malignant. "
                   "The odds ratio is the multiplicative change in the odds of "
                   "malignancy per one standard deviation of that feature.")

    st.divider()
    st.markdown("**Tree-based feature importance**")
    figure("06_feature_importance.png")
    imp = csv("feature_importance.csv")
    if imp is not None and len(imp.columns) > 1:
        tree_models = [c for c in imp.columns if c != "feature"]
        pick = st.multiselect("Models", tree_models, default=tree_models,
                              key="imp_models")
        if pick:
            d = imp[["feature"] + pick].copy()
            d["mean"] = d[pick].mean(axis=1)
            d = d.sort_values("mean", ascending=False).head(15)
            st.bar_chart(d.set_index("feature")[pick], height=400)
            st.dataframe(d.round(4), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
**How to read these.** The two model families agree on the broad story &mdash; the
*worst* (largest-three-nuclei) concavity and size measurements carry most of the
signal &mdash; which is reassuring, since they arrive at it by completely different
routes.

**But treat individual rankings with care.** `radius`, `perimeter` and `area` are
near-duplicates, so the penalty and the trees both split credit among them more or
less arbitrarily. Read the importances as **groups of related measurements**
(size, texture, concavity, symmetry), not as a ranking of independent effects.
        """)

# ---------------------------------------------------------------------------
# 9. Threshold
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Decision threshold analysis")
    st.markdown(
        "A false negative &mdash; a malignant case called benign &mdash; is far more "
        "costly here than a false positive, which leads to a follow-up test. The "
        "default 0.50 cut-off optimises neither.")

    if threshold:
        st.success(
            f"**Recommended threshold: {threshold['threshold']:.2f}** &mdash; "
            f"{threshold['selection_rule']}. "
            f"Target: cross-validated recall >= {threshold['target_recall']}.")

    figure("09_threshold_tradeoff.png")

    thr = csv("threshold_analysis.csv")
    if thr is not None:
        st.divider()
        st.markdown("**Explore the trade-off** (training-split CV predictions)")
        t = st.slider("Decision threshold on P(malignant)",
                      float(thr["threshold"].min()), float(thr["threshold"].max()),
                      float(threshold.get("threshold", 0.5)), step=0.01)
        row = thr.iloc[(thr["threshold"] - t).abs().argmin()]
        c = st.columns(5)
        for col, (lab, key, fmt) in zip(c, [
                ("Recall", "recall", "{:.4f}"), ("Precision", "precision", "{:.4f}"),
                ("Specificity", "specificity", "{:.4f}"), ("F1", "f1", "{:.4f}"),
                ("Missed malignancies", "fn", "{:.0f}")]):
            with col:
                card(lab, fmt.format(row[key]),
                     "of 455 training cases" if key == "fn" else "")
        st.write("")
        st.line_chart(thr.set_index("threshold")[["recall", "precision", "f1"]],
                      height=300)
        st.line_chart(thr.set_index("threshold")[["fn", "fp"]].rename(
            columns={"fn": "False negatives", "fp": "False positives"}), height=260)

        st.markdown(
            "**The threshold is chosen from cross-validated predictions on the "
            "training split only.** Picking it by looking at test-set results would "
            "be tuning on the test set, which is exactly what the held-out split "
            "exists to prevent.")

        tt = csv("threshold_analysis_test.csv")
        if tt is not None:
            with st.expander("The same sweep on the test set (reporting only)"):
                st.dataframe(
                    tt[tt["threshold"].isin([0.20, 0.30, 0.40, 0.50])][
                        ["threshold", "recall", "precision", "specificity",
                         "f1", "fn", "fp"]].round(4),
                    use_container_width=True, hide_index=True)
                st.caption("Shown for completeness. These numbers did not inform the "
                           "threshold choice above.")

    st.warning(
        "**This is an educational classification experiment, not a clinically validated "
        "diagnostic system.** The threshold is chosen against a recall target picked for "
        "illustration, not against measured clinical costs. The model has had no external "
        "validation, no prospective evaluation and no regulatory assessment. Nothing on "
        "this page should be used to inform a diagnosis or a clinical decision.")

# ---------------------------------------------------------------------------
# 10. Conclusions
# ---------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Conclusions and limitations")
    st.markdown(
        f"""
### What the comparison showed

**Selected model:** {best_name}
{(" (" + ", ".join(f"`{k}={v}`" for k, v in selection.get("best_params", {}).items()) + ")")
 if selection.get("best_params") else ""}, chosen on cross-validated ROC-AUC and
malignant recall &mdash; **not on accuracy alone**.

- Four of the ten models are statistically indistinguishable on the 15 shared folds.
  The tree ensembles, KNN, Naive Bayes and especially the single Decision Tree sit
  measurably behind.
- On a small, well-separated, near-linear dataset the simplest model wins, and it
  trains two to three orders of magnitude faster than the ensembles.
- **Test performance:** {best_row['test_accuracy']:.1%} accuracy,
  {best_row['test_recall']:.1%} recall on malignant cases, ROC-AUC
  {best_row['test_roc_auc']:.4f} &mdash; {int(best_row['fn'])} false negatives and
  {int(best_row['fp'])} false positives out of {split_info['n_test']} cases.
- **Threshold:** lowering the cut-off from 0.50 to
  {threshold.get('threshold', 0.5):.2f} trades a few extra false alarms for fewer
  missed malignancies, which is the right direction for a screening-style task.

### Methodology

All preprocessing lives inside `Pipeline` objects, so the scaler is re-fitted per fold
and there is no train/test leakage. Hyperparameters were tuned on the training split
only; all ten models were compared on identical folds; the test set was read once,
after selection; and the decision threshold was chosen from training-split
cross-validated predictions.

### Limitations

- **One small dataset, one source.** 569 cases from a single institution, digitised
  with one imaging and segmentation pipeline. Nothing here speaks to how it transfers.
- **Features, not images.** The 30 inputs are the output of a hand-built nuclear
  segmentation step. A deployed system would have to reproduce that step, and its errors.
- **The test set is small.** {split_info['n_test']} cases,
  {split_info['test_class_counts']['malignant']} of them malignant &mdash; the intervals
  around every test number are wide.
- **Correlated features** make individual coefficients and importances hard to read;
  they should be interpreted as groups.
- **Probabilities are uncalibrated.** They are used for thresholding, which only needs
  the ranking, but they are not literal risks.
- **No external validation**, no patient-level grouping, no prospective evaluation.
        """)

    run_info = js("wdbc_run_info.json")
    if run_info:
        st.caption(f"Results exported {run_info['exported_at']} from `wdbc/reports` "
                   f"({run_info['n_csv']} data files, {run_info['n_figures']} figures).")

    st.divider()
    st.markdown("**Source**")
    st.caption(
        "Wolberg, Street & Mangasarian, Wisconsin Diagnostic Breast Cancer (WDBC), "
        "University of Wisconsin, November 1995. Features are computed from digitised "
        "images of fine-needle aspirates of breast masses.")
