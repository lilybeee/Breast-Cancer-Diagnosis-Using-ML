"""Page 2 - Model comparison: results table, CV comparison, statistics, midterm vs end-term, mass vs calcification."""
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

tab1, tab2, tab3= st.tabs(["Results table", "CV comparison", "Is the difference real?"])

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

