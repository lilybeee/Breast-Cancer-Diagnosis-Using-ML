"""Page 4 - Explainability: odds ratios, permutation importance, error analysis."""
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
