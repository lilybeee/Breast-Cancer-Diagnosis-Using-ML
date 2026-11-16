"""
train.py - WDBC (Wisconsin Diagnostic Breast Cancer) comparative classification study.

Tunes ten classifier configurations with GridSearchCV on the TRAINING SPLIT ONLY,
then compares them with the *same* RepeatedStratifiedKFold folds, again on the
training split only.  The held-out test set is never touched here.

Run:   python src/train.py            (from the wdbc/ project root)

Writes
  reports/cv_results.csv           per-model CV mean/std for 8 metrics + best params + timings
  reports/cv_per_fold.csv          raw per-fold scores (used for paired comparisons)
  reports/cv_paired_vs_best.csv    paired fold-wise difference of each model vs the selected one
  reports/model_selection.json     selected model + the ranking that produced it
  reports/split_info.json          split sizes / class balance (reproducibility record)
  models/tuned_models.joblib       every tuned pipeline, fitted on the training split
  models/best_model.joblib         the selected pipeline, fitted on the training split
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
# scikit-learn >= 1.9 deprecates SVC(probability=True) in favour of
# CalibratedClassifierCV.  probability=True still works through 1.10 and keeps
# the hyperparameter names short, so it is kept here and the notice silenced.
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")

SEED = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "wdbc.data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE_FEATURES = ["radius", "texture", "perimeter", "area", "smoothness",
                 "compactness", "concavity", "concave_points", "symmetry",
                 "fractal_dimension"]
SUFFIXES = ["mean", "se", "worst"]
FEATURE_NAMES = [f"{b}_{s}" for s in SUFFIXES for b in BASE_FEATURES]
COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES

# Human-readable notes used by the notebook and the dashboard page.
FEATURE_GLOSSARY = {
    "radius": "mean distance from the nucleus centre to points on its perimeter",
    "texture": "standard deviation of the grey-scale values inside the nucleus",
    "perimeter": "length of the nuclear boundary",
    "area": "area enclosed by the nuclear boundary",
    "smoothness": "local variation in radius lengths along the boundary",
    "compactness": "perimeter^2 / area - 1.0",
    "concavity": "severity of concave portions of the contour",
    "concave_points": "number of concave portions of the contour",
    "symmetry": "symmetry of the nucleus about its major axis",
    "fractal_dimension": "'coastline approximation' of the boundary, minus 1",
}
SUFFIX_GLOSSARY = {
    "mean": "mean of the measurement over all nuclei in the image",
    "se": "standard error of the measurement over all nuclei in the image",
    "worst": "mean of the three largest values in the image",
}


def load_raw() -> pd.DataFrame:
    """The raw CSV with named columns (id still present, diagnosis still B/M)."""
    return pd.read_csv(DATA_FILE, header=None, names=COLUMNS)


def load_data():
    """Return (X, y) with the ID column removed and y encoded as M=1 (positive), B=0."""
    df = load_raw().drop(columns=["id"])
    y = df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    X = df.drop(columns=["diagnosis"])
    return X, y


def split(X, y):
    """The one and only 80:20 stratified split used across the whole project."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Malignant (M = 1) is the positive class throughout.
SPECIFICITY = make_scorer(recall_score, pos_label=0)
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "specificity": SPECIFICITY,
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
}
METRIC_ORDER = list(SCORING)

# The grid search optimises ROC-AUC: it is threshold-free and insensitive to the
# mild class imbalance (37% malignant), so it ranks hyperparameters on the
# quality of the score ranking rather than on an arbitrary 0.5 cut-off.
TUNING_METRIC = "roc_auc"


# ---------------------------------------------------------------------------
# Models.  needs_scaling=False for the tree/ensemble/NB models, which are
# invariant to monotone feature rescaling - no scaler is added for them.
# Grids are deliberately small (a few dozen fits each) to stay reproducible.
# ---------------------------------------------------------------------------
def model_zoo():
    return {
        "Logistic Regression": dict(
            estimator=LogisticRegression(max_iter=5000, random_state=SEED),
            grid={"model__C": [0.01, 0.1, 1, 10, 100],
                  "model__penalty": ["l2"],
                  "model__solver": ["lbfgs"]},
            needs_scaling=True,
            family="Linear",
        ),
        "LDA": dict(
            estimator=LinearDiscriminantAnalysis(),
            grid={"model__solver": ["svd", "lsqr"]},
            needs_scaling=True,
            family="Linear",
        ),
        "KNN": dict(
            estimator=KNeighborsClassifier(),
            grid={"model__n_neighbors": [3, 5, 7, 9, 11, 15],
                  "model__weights": ["uniform", "distance"],
                  "model__p": [1, 2]},
            needs_scaling=True,
            family="Instance-based",
        ),
        "Decision Tree": dict(
            estimator=DecisionTreeClassifier(random_state=SEED),
            grid={"model__max_depth": [3, 4, 5, 7, None],
                  "model__min_samples_leaf": [1, 3, 5],
                  "model__criterion": ["gini", "entropy"]},
            needs_scaling=False,
            family="Tree",
        ),
        "Random Forest": dict(
            estimator=RandomForestClassifier(random_state=SEED, n_jobs=-1),
            grid={"model__n_estimators": [300, 600],
                  "model__max_depth": [None, 6],
                  "model__max_features": ["sqrt", 0.3],
                  "model__min_samples_leaf": [1, 3]},
            needs_scaling=False,
            family="Ensemble (bagging)",
        ),
        "Gaussian NB": dict(
            estimator=GaussianNB(),
            grid={"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]},
            needs_scaling=False,
            family="Probabilistic",
        ),
        "SVM (linear)": dict(
            estimator=SVC(kernel="linear", probability=True, random_state=SEED),
            grid={"model__C": [0.01, 0.1, 1, 10, 100]},
            needs_scaling=True,
            family="Kernel",
        ),
        "SVM (RBF)": dict(
            estimator=SVC(kernel="rbf", probability=True, random_state=SEED),
            grid={"model__C": [0.1, 1, 10, 100],
                  "model__gamma": ["scale", 0.01, 0.1]},
            needs_scaling=True,
            family="Kernel",
        ),
        "Neural Net (MLP)": dict(
            estimator=MLPClassifier(max_iter=3000, random_state=SEED,
                                    early_stopping=False),
            grid={"model__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                  "model__alpha": [1e-4, 1e-2, 1e-1],
                  "model__activation": ["relu"]},
            needs_scaling=True,
            family="Neural network",
        ),
        "Gradient Boosting": dict(
            estimator=GradientBoostingClassifier(random_state=SEED),
            grid={"model__n_estimators": [200, 400],
                  "model__learning_rate": [0.05, 0.1],
                  "model__max_depth": [2, 3]},
            needs_scaling=False,
            family="Ensemble (boosting)",
        ),
    }


def build_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    """Scaling lives inside the pipeline, so it is re-fitted per CV fold (no leakage)."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    n_dup = int(raw.drop(columns=["id"]).duplicated().sum())
    n_dup_id = int(raw["id"].duplicated().sum())
    split_info = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "missing_values": int(X.isnull().sum().sum()),
        "duplicate_rows": n_dup,
        "duplicate_ids": n_dup_id,
        "class_counts": {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())},
        "malignant_rate": float(y.mean()),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_class_counts": {"benign": int((y_train == 0).sum()),
                               "malignant": int((y_train == 1).sum())},
        "test_class_counts": {"benign": int((y_test == 0).sum()),
                              "malignant": int((y_test == 1).sum())},
        "positive_class": "Malignant (M = 1)",
        "inner_cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "outer_cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
        "tuning_metric": TUNING_METRIC,
    }
    (REPORTS / "split_info.json").write_text(json.dumps(split_info, indent=2))

    print(f"samples={len(X)}  features={X.shape[1]}  "
          f"missing={split_info['missing_values']}  duplicate_rows={n_dup}")
    print(f"train={len(X_train)}  test={len(X_test)}  "
          f"malignant rate={y.mean():.3f}\n")

    # Same folds for every model: identical seeds -> identical partitions.
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    zoo = model_zoo()
    rows, per_fold, fitted = [], [], {}

    for name, spec in zoo.items():
        pipe = build_pipeline(spec["estimator"], spec["needs_scaling"])

        t0 = time.perf_counter()
        gs = GridSearchCV(pipe, spec["grid"], scoring=TUNING_METRIC,
                          cv=inner_cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)          # tuning: training data only
        tune_seconds = time.perf_counter() - t0

        tuned = gs.best_estimator_

        # Fresh clone of the tuned configuration, scored on the shared outer folds.
        cvres = cross_validate(tuned, X_train, y_train, cv=outer_cv,
                               scoring=SCORING, n_jobs=-1, return_train_score=False)

        # Fit-time of the tuned configuration on a full training fold.
        t0 = time.perf_counter()
        tuned.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        fitted[name] = tuned

        row = {"model": name, "family": spec["family"],
               "scaled": spec["needs_scaling"],
               "best_params": json.dumps(
                   {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                   default=str),
               "n_candidates": int(len(gs.cv_results_["params"])),
               "tune_seconds": round(tune_seconds, 3),
               "fit_seconds": round(fit_seconds, 4)}
        for m in METRIC_ORDER:
            s = cvres[f"test_{m}"]
            row[f"cv_{m}_mean"] = float(np.mean(s))
            row[f"cv_{m}_std"] = float(np.std(s, ddof=1))
            for i, v in enumerate(s):
                per_fold.append({"model": name, "metric": m, "fold": i, "score": float(v)})
        rows.append(row)

        print(f"{name:20s} CV AUC={row['cv_roc_auc_mean']:.4f}"
              f" +/-{row['cv_roc_auc_std']:.4f}  recall={row['cv_recall_mean']:.4f}"
              f"  f1={row['cv_f1_mean']:.4f}  ({tune_seconds:5.1f}s tuning)")

    cv_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Selection: average rank over five criteria, NOT accuracy alone.
    # ROC-AUC and malignant recall are weighted double because a missed
    # malignancy is the costly error in this problem.
    # -----------------------------------------------------------------------
    criteria = {"cv_roc_auc_mean": 2.0, "cv_recall_mean": 2.0, "cv_f1_mean": 1.0,
                "cv_pr_auc_mean": 1.0, "cv_balanced_accuracy_mean": 1.0}
    ranks = pd.DataFrame(index=cv_df["model"])
    for col in criteria:
        ranks[col] = cv_df.set_index("model")[col].rank(ascending=False, method="average")
    weights = pd.Series(criteria)
    cv_df["selection_score"] = ((ranks * weights).sum(axis=1) / weights.sum()).values
    cv_df = cv_df.sort_values("selection_score").reset_index(drop=True)
    best_name = cv_df.loc[0, "model"]

    cv_df.to_csv(REPORTS / "cv_results.csv", index=False)
    pf = pd.DataFrame(per_fold)
    pf.to_csv(REPORTS / "cv_per_fold.csv", index=False)

    # -----------------------------------------------------------------------
    # Are the differences meaningful?  Because every model saw the identical
    # folds, differences can be compared fold-by-fold (paired).
    # -----------------------------------------------------------------------
    paired = []
    for metric in ("roc_auc", "recall", "f1"):
        wide = pf[pf["metric"] == metric].pivot(index="fold", columns="model", values="score")
        base = wide[best_name]
        for m in wide.columns:
            if m == best_name:
                continue
            d = (base - wide[m]).to_numpy()
            n = len(d)
            mean_d = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            lo, hi = mean_d - 1.96 * se, mean_d + 1.96 * se
            paired.append({"metric": metric, "model": m, "reference": best_name,
                           "mean_diff": mean_d, "ci_lo": lo, "ci_hi": hi,
                           "ci_excludes_0": bool(lo > 0 or hi < 0),
                           "conclusion": ("reference is better" if lo > 0 else
                                          "model is better" if hi < 0 else
                                          "no demonstrated difference")})
    pd.DataFrame(paired).to_csv(REPORTS / "cv_paired_vs_best.csv", index=False)

    joblib.dump(fitted, MODELS / "tuned_models.joblib")
    joblib.dump(fitted[best_name], MODELS / "best_model.joblib")

    selection = {
        "selected_model": best_name,
        "selection_rule": ("lowest weighted average rank over CV ROC-AUC (w=2), "
                           "malignant recall (w=2), F1, PR-AUC and balanced "
                           "accuracy (w=1 each); accuracy alone is not used"),
        "ranking": cv_df[["model", "selection_score", "cv_roc_auc_mean",
                          "cv_recall_mean", "cv_f1_mean",
                          "cv_accuracy_mean"]].to_dict(orient="records"),
        "best_params": json.loads(cv_df.loc[0, "best_params"]),
        "n_ties_within_1se": int(
            (cv_df["cv_roc_auc_mean"] >=
             cv_df.loc[0, "cv_roc_auc_mean"] - cv_df.loc[0, "cv_roc_auc_std"]).sum()),
    }
    (REPORTS / "model_selection.json").write_text(json.dumps(selection, indent=2))

    print(f"\nselected: {best_name}  ({cv_df.loc[0, 'best_params']})")
    print(f"models within 1 SD of its CV ROC-AUC: {selection['n_ties_within_1se']}")
    print("wrote reports/cv_results.csv, cv_per_fold.csv, cv_paired_vs_best.csv, "
          "model_selection.json, split_info.json")
    print("wrote models/tuned_models.joblib, models/best_model.joblib")


if __name__ == "__main__":
    main()
