"""Breast calcification classification — concise Streamlit project page.

Place this file in dashboard/pages/4_Calcification_Project.py.
Place notebook exports directly in dashboard/data/.
"""
from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Calcification Classification",
    page_icon="🩺",
    layout="wide",
)

DASH_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DASH_DIR / "data"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.45rem; padding-bottom: 2rem; max-width: 1450px;}
      .eyebrow {color:#64748b;font-size:.75rem;font-weight:700;letter-spacing:.12em;
                text-transform:uppercase;margin-bottom:.35rem;}
      .hero {padding:1.1rem 1.3rem;border:1px solid rgba(128,128,128,.22);
             border-radius:13px;background:linear-gradient(115deg,rgba(37,99,235,.07),rgba(14,165,233,.025));}
      .hero h3 {margin:0 0 .35rem 0;font-weight:650;}
      .hero p {margin:0;color:#64748b;font-size:.94rem;}
      .metric-card {border:1px solid rgba(128,128,128,.23);border-radius:12px;
                    padding:.8rem .95rem;min-height:112px;height:100%;}
      .metric-label {color:#64748b;font-size:.82rem;line-height:1.3;margin-bottom:.5rem;}
      .metric-value {font-size:clamp(1.1rem,1.7vw,1.65rem);font-weight:650;line-height:1.25;
                     white-space:normal;overflow-wrap:anywhere;word-break:normal;}
      .metric-help {color:#64748b;font-size:.76rem;margin-top:.35rem;}
      h2 {padding-top:.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.caption(f"Could not read {filename}: {exc}")
        return None


def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_text(filename):
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else None


def show_image(filename, caption=None):
    path = DATA_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def number(value, digits=3):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def metric_card(column, label, value, help_text=None):
    help_html = f'<div class="metric-help">{help_text}</div>' if help_text else ""
    column.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{help_html}</div>',
        unsafe_allow_html=True,
    )


def without_soft_vote(frame):
    """Exclude soft-voting ensemble rows from every displayed result."""
    if frame is None or frame.empty or "model" not in frame.columns:
        return frame
    names = frame["model"].astype(str).str.lower()
    mask = ~names.str.contains(r"soft[\s_-]*vote|soft[\s_-]*voting", regex=True, na=False)
    return frame.loc[mask].copy()


def display_table(frame, height=280):
    if frame is not None and not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True, height=height)


# Load the notebook exports. Files are expected directly inside dashboard/data/.
cv = without_soft_vote(load_csv("nested_cv_summary.csv"))
holdout = without_soft_vote(load_csv("holdout_endterm_table.csv"))
ops = without_soft_vote(load_csv("holdout_operating_points.csv"))
ablation = load_csv("ablation_nested_cv1.csv")
odds = load_csv("lr_odds_ratios1.csv")
permutation = load_csv("permutation_importance_groups.csv")
lesion = load_csv("lesion_level_results.csv")
eda = load_csv("eda_descriptor_rates1.csv")
info = load_json("run_info.json")
findings = load_text("endterm_findings.txt")

# -----------------------------------------------------------------------------
# Validate core exports and identify the strongest end-term candidate.
# -----------------------------------------------------------------------------
required = ["nested_cv_summary.csv", "holdout_endterm_table.csv"]
missing = [name for name in required if not (DATA_DIR / name).exists()]
if missing:
    st.error("Required result files are missing from dashboard/data/: " + ", ".join(missing))
    st.stop()
if cv is None or holdout is None or cv.empty or holdout.empty:
    st.error("The required result files are empty or could not be read. Re-export them from the final notebook.")
    st.stop()

cv_required = {"model", "stage", "roc_auc_mean"}
ho_required = {"model", "stage", "roc_auc"}
if not cv_required.issubset(cv.columns) or not ho_required.issubset(holdout.columns):
    st.error("The required CSV columns do not match the expected notebook export format.")
    st.stop()

stage_text = cv["stage"].astype(str).str.lower()
mid_rows = cv[stage_text.eq("midterm")].copy()
end_rows = cv[~stage_text.eq("midterm")].copy()
if end_rows.empty:
    st.error("No end-term rows were found in nested_cv_summary.csv.")
    st.stop()

best_end = end_rows.sort_values("roc_auc_mean", ascending=False).iloc[0]
best_end_name = str(best_end["model"])
end_holdout_rows = holdout[holdout["model"].astype(str).eq(best_end_name)]
end_test = end_holdout_rows.iloc[0] if not end_holdout_rows.empty else None

# -----------------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Machine learning project · Radiologist-described features</div>', unsafe_allow_html=True)
st.title("Breast Calcification Classification")
st.markdown(
    "<div class='hero'><h3>Can calcification descriptors help distinguish benign from malignant cases?</h3>"
    "<p>A patient-aware modelling study: develop the classifier, evaluate its decisions, and interpret its predictions.</p></div>",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------------------------------
# 1. Research objective
# -----------------------------------------------------------------------------
st.markdown("## 1. Research objective")
st.write(
    "The task is to classify breast calcification observations as **benign or malignant** using "
    "radiologist-described features. The modelling pipeline keeps observations from the same patient "
    "together during cross-validation and reserves a separate hold-out set for final evaluation."
)

# -----------------------------------------------------------------------------
# 2. Modelling approach and final performance — main focus of the page
# -----------------------------------------------------------------------------
st.markdown("## 2. Model development and final performance")
st.write(
    "The original candidate models were re-tuned using ROC-AUC. The end-term experiments also examined "
    "tree-based models, class weighting and interaction features. Patient-grouped nested cross-validation "
    "was used for model selection, followed by evaluation on a separate hold-out set."
)

m1, m2, m3, m4 = st.columns(4)
metric_card(m1, "Selected end-term model", best_end_name)
if end_test is not None:
    metric_card(m2, "Hold-out ROC-AUC", number(end_test.get("roc_auc")), "Discrimination on the held-out set")
    metric_card(m3, "Sensitivity", number(end_test.get("sensitivity")), "At the saved operating threshold")
    metric_card(m4, "Specificity", number(end_test.get("specificity")), "At the saved operating threshold")
else:
    metric_card(m2, "Hold-out ROC-AUC", "—")
    metric_card(m3, "Sensitivity", "—")
    metric_card(m4, "Specificity", "—")

if end_test is not None and ("auc_lo" in end_test.index or "auc_hi" in end_test.index):
    st.caption(
        f"Patient-bootstrap 95% confidence interval for hold-out ROC-AUC: "
        f"{number(end_test.get('auc_lo'))}–{number(end_test.get('auc_hi'))}."
    )

if info:
    run_details = []
    for key, label in [
        ("n_train", "Training observations"),
        ("n_test", "Hold-out observations"),
        ("patients_train", "Training patients"),
        ("patients_test", "Hold-out patients"),
    ]:
        if info.get(key) is not None:
            run_details.append(f"{label}: {info[key]}")
    if run_details:
        st.caption(" · ".join(run_details))
    if info.get("quick_run") is True:
        st.caption("These exports were generated with QUICK_RUN=True; use full-run results for final reporting.")

if (DATA_DIR / "fig_holdout_roc_pr_calibration.png").exists():
    show_image("fig_holdout_roc_pr_calibration.png", "Hold-out ROC, precision–recall and calibration assessment")

# -----------------------------------------------------------------------------
# 3. From predicted probability to a classification decision
# -----------------------------------------------------------------------------
st.markdown("## 3. From probability to classification decision")
st.write(
    "The model outputs a probability. A decision threshold converts that probability into a benign or "
    "malignant classification. Thresholds were selected using training out-of-fold predictions and then "
    "evaluated on the hold-out set."
)

if ops is not None and not ops.empty and {"model", "rule"}.issubset(ops.columns):
    model_options = sorted(ops["model"].dropna().astype(str).unique().tolist())
    if model_options:
        default_model = best_end_name if best_end_name in model_options else model_options[0]
        selected_model = st.selectbox(
            "Model", model_options, index=model_options.index(default_model), key="calc_model_select"
        )
        selected_ops = ops[ops["model"].astype(str).eq(selected_model)].copy()
        rule_values = selected_ops["rule"].dropna().astype(str).tolist()
        preferred_rules = [rule for rule in ["recall90", "recall85", "recall95", "youden"] if rule in rule_values]
        rule_options = preferred_rules or list(dict.fromkeys(rule_values))
        if rule_options:
            chosen_rule = st.selectbox("Threshold strategy", rule_options, key="calc_rule_select")
            chosen = selected_ops[selected_ops["rule"].astype(str).eq(chosen_rule)].iloc[0]
            threshold_cols = st.columns(5)
            for col, label, key in [
                (threshold_cols[0], "Sensitivity", "sensitivity"),
                (threshold_cols[1], "Specificity", "specificity"),
                (threshold_cols[2], "Precision (PPV)", "ppv"),
                (threshold_cols[3], "Negative predictive value", "npv"),
                (threshold_cols[4], "Probability threshold", "thr"),
            ]:
                col.metric(label, number(chosen.get(key)))
            st.caption("Recall-target thresholds are derived from training out-of-fold predictions, not hold-out labels.")
        else:
            st.write("No recognised threshold strategies are present in holdout_operating_points.csv.")
else:
    st.write("Operating-threshold results are unavailable because holdout_operating_points.csv was not found or is empty.")

if (DATA_DIR / "fig_confusion_matrices.png").exists():
    show_image("fig_confusion_matrices.png", "Hold-out confusion matrices at the saved operating points")

# -----------------------------------------------------------------------------
# 4. Interpretation
# -----------------------------------------------------------------------------
st.markdown("## 4. What influences the model's predictions?")
st.write(
    "Two complementary views help interpret the fitted models: Logistic Regression odds ratios describe "
    "direction of association, while grouped permutation importance measures how much predictive performance "
    "changes when a feature group is shuffled. These are model-interpretation measures, not causal effects."
)

left, right = st.columns(2)
with left:
    st.markdown("### Logistic Regression: odds ratios")
    if odds is not None and not odds.empty:
        display_table(odds.head(8), height=260)
        show_image("fig_lr_odds_ratios.png")
    else:
        st.caption("Odds-ratio results are unavailable.")
with right:
    st.markdown("### Grouped permutation importance")
    if permutation is not None and not permutation.empty:
        display_table(permutation.head(8), height=260)
        show_image("fig_permutation_importance.png")
    else:
        st.caption("Permutation-importance results are unavailable.")

with st.expander("Explore descriptor patterns and lesion-proxy analysis"):
    if eda is not None and not eda.empty:
        st.markdown("**Descriptor-level malignancy rates**")
        display_table(eda, height=280)
        show_image("eda_descriptors.png")
    if lesion is not None and not lesion.empty:
        st.markdown("**Lesion-proxy evaluation**")
        display_table(lesion, height=260)
        st.caption(
            "A true lesion identifier was unavailable. Grouping used patient, breast side, and calcification "
            "type/distribution flags as a proxy; these groups are not confirmed lesion identities."
        )

# -----------------------------------------------------------------------------
# 5. Development progress — one compact comparison plot, deliberately last
# -----------------------------------------------------------------------------
st.markdown("## 5. Development progress")
st.write(
    "The plot gives a compact reference point for the modelling work: the strongest midterm model versus "
    "the strongest end-term candidate. The main evaluation above remains focused on the final end-term model."
)

if not mid_rows.empty:
    best_mid = mid_rows.sort_values("roc_auc_mean", ascending=False).iloc[0]
    comparison = pd.DataFrame(
        {
            "Stage": ["Midterm reference", "End-term candidate"],
            "Model": [str(best_mid["model"]), best_end_name],
            "Nested-CV ROC-AUC": [float(best_mid["roc_auc_mean"]), float(best_end["roc_auc_mean"])],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = np.arange(len(comparison))
    bars = ax.barh(y, comparison["Nested-CV ROC-AUC"], color=["#94a3b8", "#2563eb"], height=0.48)
    labels = [
        f"{stage}\n{textwrap.fill(model, width=34)}"
        for stage, model in zip(comparison["Stage"], comparison["Model"])
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean nested cross-validation ROC-AUC")
    ax.set_title("Best model at each project stage", loc="left", pad=12, weight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, comparison["Nested-CV ROC-AUC"]):
        ax.text(min(value + .015, 1.01), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=10, weight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    st.caption("Midterm reference rows were not present, so the stage-comparison plot cannot be shown.")

with st.expander("View ablation experiments and statistical checks"):
    if ablation is not None and not ablation.empty:
        display_table(ablation, height=320)
    else:
        st.caption("Ablation results are unavailable.")

if findings:
    with st.expander("View notebook-generated findings"):
        st.text(findings)

# -----------------------------------------------------------------------------
# Limitations — concise and visible, not highlighted in a callout box
# -----------------------------------------------------------------------------
st.markdown("## Study limitations")
st.write(
    "The analysis uses descriptor-based inputs, has no external validation, and does not contain a true lesion ID. "
    "Lesion-level grouping is therefore a proxy based on available patient, breast-side and calcification flags. "
    "These results describe an academic modelling study, not a clinically validated diagnostic or screening tool."
)

st.divider()
st.caption("Breast Calcification Classification · End-term contribution")
