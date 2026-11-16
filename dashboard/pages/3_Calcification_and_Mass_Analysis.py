"""Page 3 - Calcification & Mass Lesion Data Analysis (EDA, Descriptors & Data Quality Audit)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import charts
import data_loader as dl
import ui

ui.setup("Calcification & Mass Analysis")
st.title("Calcification & Mass Lesion Data Analysis")
st.caption(
    "Exploratory data analysis, radiologist morphological descriptors, cohort distributions, "
    "and data quality / leakage safeguards for CBIS-DDSM calcification and mass cases."
)

# ---------------------------------------------------------------------------
# Load cleaned feature datasets via data_loader
# ---------------------------------------------------------------------------
calc_clean = dl.load("calc_cleaned")
mass_clean = dl.load("mass_cleaned")

if calc_clean is None and mass_clean is None:
    st.warning(
        f"Missing cleaned datasets in `{dl.DATA_DIR}`. "
        "Ensure `breast_cancer_cleaned_final_calc__.csv` and `breast_cancer_cleaned_final_mass__.csv` "
        "are inside `dashboard/data/`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("**Data Analysis Controls**")
    cohort = st.radio(
        "Cohort View",
        ["Side-by-Side Comparison", "Calcification Only", "Mass Only"],
        index=0,
    )
    split_filter = st.selectbox("Dataset Split", ["All (Train + Test)", "train", "test"])
    min_samples = st.slider("Min lesions per descriptor", 5, 100, 15, step=5)


def filter_split(df):
    if df is None or split_filter == "All (Train + Test)":
        return df
    return df[df["split"] == split_filter]


calc_df = filter_split(calc_clean)
mass_df = filter_split(mass_clean)

# ---------------------------------------------------------------------------
# Top KPI Strip (Cohort Overview)
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

if calc_df is not None:
    k1.metric(
        "Calcification Lesions",
        f"{len(calc_df):,}",
        f"{calc_df['patient_id'].nunique():,} unique patients",
    )
    k2.metric(
        "Calcification Malignancy Rate",
        f"{calc_df['malignant'].mean():.1%}",
        f"{int(calc_df['malignant'].sum()):,} malignant / {len(calc_df):,} total",
    )

if mass_df is not None:
    k3.metric(
        "Mass Lesions",
        f"{len(mass_df):,}",
        f"{mass_df['patient_id'].nunique():,} unique patients",
    )
    k4.metric(
        "Mass Malignancy Rate",
        f"{mass_df['malignant'].mean():.1%}",
        f"{int(mass_df['malignant'].sum()):,} malignant / {len(mass_df):,} total",
    )

# ---------------------------------------------------------------------------
# Main EDA Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Radiologist Descriptors & Risk",
    "2. Clinical & Imaging Covariates",
    "3. Data Quality & Leakage Audit",
    "4. Diagnostic EDA Figures",
])

# === TAB 1: DESCRIPTOR MALIGNANCY RATES ===
with tab1:
    def descriptor_stats(df, prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        rows = []
        for c in cols:
            active = df[df[c] == 1]
            n = len(active)
            if n >= min_samples:
                rows.append({
                    "Descriptor": c.replace(prefix, ""),
                    "Lesions (n)": n,
                    "Cohort Share (%)": round(n / len(df) * 100, 1),
                    "Malignant Rate": active["malignant"].mean(),
                })
        return pd.DataFrame(rows).sort_values("Malignant Rate", ascending=True)

    def render_cohort_descriptors(df, label, prefixes):
        st.subheader(f"{label} Descriptors ({len(df):,} lesions)")
        cols = st.columns(len(prefixes))
        for col_ui, (title, pref) in zip(cols, prefixes):
            with col_ui:
                stats_df = descriptor_stats(df, pref)
                if stats_df.empty:
                    st.info(f"No {title} descriptors with >= {min_samples} lesions.")
                    continue
                fig, ax = plt.subplots(figsize=(6, max(3.2, 0.38 * len(stats_df) + 1.2)))
                colors = [
                    charts.PINK if r >= df["malignant"].mean() else charts.PURPLE
                    for r in stats_df["Malignant Rate"]
                ]
                ax.barh(stats_df["Descriptor"], stats_df["Malignant Rate"], color=colors)
                ax.axvline(
                    df["malignant"].mean(),
                    color="k",
                    ls="--",
                    lw=1,
                    label=f"Cohort mean ({df['malignant'].mean():.1%})",
                )
                for i, (_, r) in enumerate(stats_df.iterrows()):
                    ax.text(
                        r["Malignant Rate"] + 0.01,
                        i,
                        f"{r['Malignant Rate']:.1%} (n={int(r['Lesions (n)'])})",
                        va="center",
                        fontsize=8,
                    )
                ax.set_xlim(0, 1.18)
                ax.set_xlabel("Malignant Share")
                ax.set_title(title)
                ax.legend(fontsize=8, loc="lower right")
                fig.tight_layout()
                st.pyplot(fig)

                with st.expander(f"View {title} table"):
                    tbl = stats_df.sort_values("Malignant Rate", ascending=False).copy()
                    tbl["Malignant Rate"] = (tbl["Malignant Rate"] * 100).round(1).astype(str) + "%"
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if cohort in ("Side-by-Side Comparison", "Calcification Only") and calc_df is not None:
        render_cohort_descriptors(
            calc_df,
            "Calcification",
            [("Calcification Type", "calc_type__"), ("Calcification Distribution", "calc_distribution__")],
        )

    if cohort == "Side-by-Side Comparison" and calc_df is not None and mass_df is not None:
        st.markdown("---")

    if cohort in ("Side-by-Side Comparison", "Mass Only") and mass_df is not None:
        render_cohort_descriptors(
            mass_df,
            "Mass",
            [("Mass Shape", "mass_shape__"), ("Mass Margins", "mass_margins__")],
        )

# === TAB 2: CLINICAL & IMAGING COVARIATES ===
with tab2:
    st.subheader("Malignancy Rate by Breast Density, Subtlety & Imaging View")
    cov_col = st.selectbox(
        "Select Covariate",
        ["breast_density", "subtlety", "view_mlo", "breast_right"],
        format_func=lambda x: {
            "breast_density": "Breast Density (ACR 1–4)",
            "subtlety": "Radiologist Subtlety Rating (1–5)",
            "view_mlo": "Mammogram View (0 = CC, 1 = MLO)",
            "breast_right": "Laterality (0 = Left, 1 = Right)",
        }[x],
    )

    c1, c2 = st.columns(2)
    for ui_col, label, df in [(c1, "Calcification", calc_df), (c2, "Mass", mass_df)]:
        with ui_col:
            if df is None:
                continue
            grp = (
                df.groupby(cov_col)["malignant"]
                .agg(Count="count", Malignant_Rate="mean")
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            bars = ax.bar(grp[cov_col].astype(str), grp["Malignant_Rate"], color=charts.PINK, width=0.55)
            ax.axhline(df["malignant"].mean(), color="k", ls="--", lw=1, label=f"Mean ({df['malignant'].mean():.1%})")
            for bar, (_, r) in zip(bars, grp.iterrows()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{r['Malignant_Rate']:.1%}\n(n={int(r['Count'])})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(cov_col)
            ax.set_ylabel("Malignant Rate")
            ax.set_title(f"{label}: Malignancy by {cov_col}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("Lesions per Patient Distribution (Why Patient-Grouped Splitting is Required)")
    p1, p2 = st.columns(2)
    for ui_col, label, df in [(p1, "Calcification", calc_df), (p2, "Mass", mass_df)]:
        with ui_col:
            if df is None:
                continue
            per_pt = df.groupby("patient_id").size()
            summary = pd.DataFrame({
                "Metric": ["Unique Patients", "Total Lesion Rows", "Mean Rows / Patient", "Median Rows / Patient", "Max Rows / Patient"],
                "Value": [
                    f"{len(per_pt):,}",
                    f"{len(df):,}",
                    f"{per_pt.mean():.2f}",
                    f"{per_pt.median():.0f}",
                    f"{per_pt.max():,}",
                ],
            })
            st.markdown(f"**{label} Patient Multiplicity**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

# === TAB 3: DATA QUALITY & LEAKAGE AUDIT ===
with tab3:
    st.subheader("Data Cleaning, Integrity & Leakage Safeguards")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**1. Informative Missingness (`UNKNOWN` Encoding)**\n"
            "- In calcification training cases, `calc_distribution` is missing in **376 / 1,546 rows (24.3%)**.\n"
            "- **0 of those 376 lesions are malignant** (371 are `BENIGN_WITHOUT_CALLBACK`).\n"
            "- Missing descriptors cannot be reliably recovered from other clinical columns (patient-grouped RF macro-F1 ~0.13), "
            "so missingness is preserved explicitly as an `UNKNOWN` indicator."
        )
        st.markdown(
            "**2. BI-RADS `assessment` Excluded (Target Leakage)**\n"
            "- `assessment` shows strong association with pathology (Cramér's $V = 0.604$: BI-RADS 2 is 0.2% malignant vs. BI-RADS 5 at 98.7%).\n"
            "- Because BI-RADS reflects the radiologist's diagnostic conclusion rather than lesion morphology, it is dropped from the cleaned feature matrix."
        )
    with right:
        st.markdown(
            "**3. Vocabulary & Range Standardization (Train-Only Fit)**\n"
            "- Descriptor synonyms (e.g., `LUCENT_CENTERED` $\\rightarrow$ `LUCENT_CENTER`) are normalized before multi-hot encoding.\n"
            "- Out-of-range values (`breast_density` outside 1–4, `subtlety` outside 1–5) are set to `NaN` and imputed using **training-set medians only**.\n"
            "- **Zero patient overlap** exists between the official `train` and `test` splits."
        )
        split_rows = []
        for lbl, d in [("Calcification", calc_clean), ("Mass", mass_clean)]:
            if d is not None:
                for sp, grp in d.groupby("split"):
                    split_rows.append({
                        "Cohort": lbl,
                        "Split": sp,
                        "Lesions": len(grp),
                        "Patients": grp["patient_id"].nunique(),
                        "Malignant Rate": f"{grp['malignant'].mean():.1%}",
                    })
        if split_rows:
            st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

# === TAB 4: SAVED EDA FIGURES ===
with tab4:
    st.subheader("Diagnostic Figures from Cleaning & EDA Pipeline")
    fig_names = [
        ("Missing Values & Malignancy", "01_missing_values.png"),
        ("Target Class Balance", "02_target_balance.png"),
        ("BI-RADS Assessment vs. Pathology", "03_assessment_vs_pathology.png"),
        ("Ordinal Spearman Correlation", "04_spearman_ordinal.png"),
        ("Malignant Rate by Calc Type", "05_malignant_rate_by_calc_type.png"),
        ("Malignant Rate by Calc Distribution", "05_malignant_rate_by_calc_distribution.png"),
        ("Malignant Rate by Mass Shape", "05_malignant_rate_by_mass_shape.png"),
        ("Malignant Rate by Mass Margins", "05_malignant_rate_by_mass_margins.png"),
    ]
    found = [(title, dl.image(fname)) for title, fname in fig_names if dl.image(fname)]
    if not found:
        st.info(
            f"No figure PNGs found in `{dl.DATA_DIR}`. Copy the PNGs from `dashboard/project/outputs/figures/` into `dashboard/data/`."
        )
    else:
        grid = st.columns(2)
        for idx, (title, img_path) in enumerate(found):
            with grid[idx % 2]:
                st.markdown(f"**{title}**")
                st.image(img_path, use_container_width=True)