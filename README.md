# Breast Cancer Analysis Dashboard — three independent dataset components

A Streamlit dashboard over three **independent** breast-cancer analyses. Each component
has its own data, its own preprocessing, its own models and its own results folder; they
share only the dashboard shell.

| # | Component | Data | Modality | Pages |
|---|-----------|------|----------|-------|
| 1 | **Mass analysis** | CBIS-DDSM mass cases | mammography (BI-RADS descriptors) | 1, 2, 3 |
| 2 | **Calcification analysis** | CBIS-DDSM calcification cases | mammography (BI-RADS descriptors) | 3, 4 |
| 3 | **WDBC nuclear features** | Wisconsin Diagnostic Breast Cancer | fine-needle aspirate cytology | 5 |

> All three are **educational classification experiments on public research datasets.
> None is a clinically validated diagnostic system.**

---

## Quick start

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Streamlit discovers everything in `pages/` automatically, so all five pages appear in the
sidebar with no configuration.

To regenerate the WDBC results (component 3):

```bash
cd wdbc
python src/train.py                 # tune + cross-validate 10 models, save the best
python src/evaluate.py              # test metrics, figures, threshold analysis
python src/export_to_dashboard.py   # copy results into dashboard/data/wdbc/
```

---

## Layout

```
dashboard_project/
├── dashboard/                      # the Streamlit app
│   ├── app.py                      # entry point
│   ├── data_loader.py              # shared file access (components 1–2)
│   ├── ui.py, charts.py            # shared helpers (components 1–2)
│   ├── pages/
│   │   ├── 1_Model_Comparison.py
│   │   ├── 2_Explainability.py
│   │   ├── 3_Calcification_and_Mass_Analysis.py
│   │   ├── 4_Calcification_Project.py
│   │   └── 5_WDBC_Nuclear_Feature_Analysis.py     ← added
│   └── data/
│       ├── …                       # mammogram exports (components 1–2)
│       └── wdbc/                   ← added: WDBC exports, read only by page 5
├── wdbc/                           ← added: the WDBC component, self-contained
│   ├── data/        wdbc.data, wdbc.names
│   ├── src/         train.py, evaluate.py, export_to_dashboard.py, wdbc_style.py
│   ├── notebooks/   breast_cancer_diagnosis_new.ipynb
│   ├── reports/     CSVs, JSON and figures/
│   └── models/      best_model.joblib, tuned_models.joblib
├── setup_dashboard.py
└── DASHBOARD_GUIDE.md
```

**Component 3 adds files only.** Every pre-existing file is byte-identical to the original
project: no shared module was edited, no navigation was rewired, `requirements.txt` is
unchanged, and page 5 imports none of the shared dashboard modules. It reads only
`dashboard/data/wdbc/`.

---

## Component 1 — Mass analysis (CBIS-DDSM)

Malignant-vs-benign classification of mammographic **masses** from BI-RADS descriptors
(shape, margins, density, assessment). Nested cross-validation across ten model families
with a corrected resampled *t*-test and paired patient-bootstrap AUC comparisons against a
logistic-regression baseline, plus odds ratios and permutation importance for
explainability.

Pages 1 (model comparison), 2 (explainability), 3 (combined mass/calcification view).
Results live in `dashboard/data/` and are read through `data_loader.py`.

## Component 2 — Calcification analysis (CBIS-DDSM)

The same descriptor-based framing applied to mammographic **calcifications** (type,
distribution, assessment). Page 4 is a standalone project page; page 3 places the mass and
calcification results side by side.

## Component 3 — WDBC nuclear features

Benign-vs-malignant classification from **30 numeric features of cell nuclei**, computed
from digitised fine-needle-aspirate images: ten measurements (radius, texture, perimeter,
area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension),
each summarised three ways (mean, standard error, and "worst" = mean of the three largest).
569 cases, 357 benign and 212 malignant, no missing values and no duplicate rows.

### Models compared

Logistic Regression · LDA · KNN · Decision Tree · Random Forest · Gaussian Naive Bayes ·
SVM (linear) · SVM (RBF) · Neural network (MLPClassifier) · Gradient Boosting

### Protocol

- **Target** — diagnosis, with **malignant (M = 1) as the positive class**.
- **Split** — 80:20 stratified, `random_state=42`: 455 train / 114 test.
- **Pipelines** — `StandardScaler` lives inside each `Pipeline`, so it is re-fitted per
  fold and never sees validation rows. Tree-based and Naive Bayes models are left unscaled.
- **Tuning** — `GridSearchCV` over `StratifiedKFold(5, shuffle=True, random_state=42)`,
  scored on ROC-AUC, **on the training split only**.
- **Comparison** — each tuned model re-scored on
  `RepeatedStratifiedKFold(5 splits × 3 repeats, random_state=42)` — 15 folds, **identical
  for every model**, so scores can be differenced fold by fold.
- **Metrics** — accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC and balanced
  accuracy, each stored as a CV mean and standard deviation.
- **Selection** — lowest weighted average rank over CV ROC-AUC (×2), malignant recall (×2),
  F1, PR-AUC and balanced accuracy. **Accuracy alone is never the criterion.**
- **Test set** — read once, after selection.
- **Threshold** — chosen from `cross_val_predict` on the **training split only**.

### Results

Selected model: **Logistic Regression** (L2, `C=1`) on standardised features.

| | CV (15 folds) | Test (114 cases) |
|---|---|---|
| Accuracy | 0.9758 ± 0.0145 | 0.9649 |
| Recall (malignant) | 0.9569 ± 0.0312 | 0.9286 |
| ROC-AUC | 0.9960 ± 0.0047 | 0.9960 |

Three false negatives and one false positive at the default 0.50 threshold.

**Six of the ten models sit within one standard deviation of the leader's CV ROC-AUC**, and on
ROC-AUC the paired fold-wise intervals for LDA, the MLP and the RBF SVM contain zero. The
selection is a choice among near-ties rather than a clear win — so the cheapest and most
interpretable of them is the reasonable pick. (LDA ties on AUC but reaches it with markedly
lower malignant recall, 0.882 against 0.957, which is why recall carries weight in the
selection rule.) The tree ensembles, KNN, Naive Bayes and especially the single Decision
Tree are measurably behind. **More complex did not mean
better** on a small, well-separated, near-linear dataset.

Lowering the decision threshold from 0.50 to **0.20** — chosen on training-split CV
predictions alone — raises cross-validated recall from 0.953 to 0.982, trading ten extra
false alarms for five fewer missed malignancies across the 455 training cases.

### Outputs

`wdbc/reports/` (mirrored into `dashboard/data/wdbc/`):

| File | Contents |
|---|---|
| `model_comparison.csv` | the consolidated CV + test table, all ten models |
| `cv_results.csv`, `cv_per_fold.csv` | CV means/SDs, best params, timings; raw fold scores |
| `cv_paired_vs_best.csv` | paired fold-wise differences vs the selected model |
| `test_results.csv`, `confusion_matrices.csv` | held-out test metrics and error counts |
| `threshold_analysis.csv` | the threshold sweep on training CV predictions |
| `threshold_analysis_test.csv` | the same sweep on the test set — **reporting only** |
| `selected_threshold.json`, `model_selection.json` | the chosen threshold and model, with the rule used |
| `feature_importance.csv`, `lr_coefficients.csv` | tree importances; LR coefficients and odds ratios |
| `eda_summary.csv`, `feature_glossary.csv`, `split_info.json` | EDA tables and the reproducibility record |
| `figures/*.png` | 11 figures (EDA, correlation, CV bars, ROC, PR, confusion matrices, metric heat-map, importances, coefficients, timing, threshold) |

### Limitations

One small single-source dataset; pre-extracted features rather than raw images; a 114-case
test set where one extra miss moves recall by 2.4 points; heavily correlated predictors that
make individual coefficients unreliable in isolation; uncalibrated probabilities; and no
external validation. **An educational project, not a clinical tool.**

---

## Reproducibility

`random_state=42` throughout — splits, folds, and every stochastic estimator. `train.py`
records the exact split sizes, class balance and CV specifications in
`reports/split_info.json`.

Requirements are unchanged from the original project (`dashboard/requirements.txt`):
streamlit ≥1.30, pandas ≥2.0, numpy ≥1.24, matplotlib ≥3.7, scikit-learn ≥1.2. The WDBC
scripts additionally use `joblib`, which ships with scikit-learn.

---

## Sources

- **CBIS-DDSM** — Curated Breast Imaging Subset of DDSM (components 1–2).
- **WDBC** — Wolberg, Street & Mangasarian, *Wisconsin Diagnostic Breast Cancer*,
  University of Wisconsin, November 1995. Full attribution in `wdbc/data/wdbc.names`.
