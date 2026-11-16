#!/usr/bin/env python3
"""
ML_Project_Data_Analysis_for_calc.py
====================================
Script version of ML_Project_Data_Analysis_for_calc.ipynb (CBIS-DDSM calcification cases).

Steps (same order as the notebook)
  1. Load + basic cleaning
  2. Duplicate / patient-overlap checks
  3. Missing-value analysis (+ mask-and-recover test)
  4. Multi-hot encoding of the descriptor columns
  5. Leakage checks, correlations, low-variance check
  6. Write the final cleaned CSV                       -> outputs/breast_cancer_cleaned_final_calc__.csv
  7. (added for the dashboard) patient-grouped CV + one test evaluation
                                                       -> outputs/calcification_summary.csv

The dashboard (dashboard/data/calcification_summary.csv, columns: model, cv_auc, test_auc) reads
file 7. Everything the notebook showed with plt.show() is saved as a PNG in outputs/figures/.

Usage
  python ML_Project_Data_Analysis_for_calc.py --data-dir ./data --out-dir ./outputs
  python ML_Project_Data_Analysis_for_calc.py --name mass        # mass descriptors instead
  python ML_Project_Data_Analysis_for_calc.py --skip-models      # cleaning/EDA only
  python ML_Project_Data_Analysis_for_calc.py --dashboard-dir dashboard/data   # also copy the CSV there

--data-dir must contain calc_case_description_train_set.csv and calc_case_description_test_set.csv
(or the mass_* files when --name mass). DATA_DIR / OUT_DIR env variables work too.
"""
import argparse
import os
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display needed; figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# ----------------------------------------------------------------------------- config
CONFIG = {
    "calc": {"train": "calc_case_description_train_set.csv",
             "test": "calc_case_description_test_set.csv",
             "desc": ["calc_type", "calc_distribution"]},
    "mass": {"train": "mass_case_description_train_set.csv",
             "test": "mass_case_description_test_set.csv",
             "desc": ["mass_shape", "mass_margins"]},
}
KEEP_VIEW = True          # False drops the CC/MLO flag from the final CSV and the models
SEED = 0
N_SPLITS = 5

FIG_DIR = Path(".")       # set in main()


def section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def save_fig(name):
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"[figure saved] {path}")


# ----------------------------------------------------------------------------- 1. load + clean
def basic_clean(df):
    df = df.copy()
    # 1. standardise column names: 'breast density' and 'breast_density' become the same
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    # 2. strip whitespace and newlines in every text column (fixes the path columns)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].str.strip()
    # 3. impossible values become NaN (ranges: density 1-4, subtlety 1-5)
    df["breast_density"] = df["breast_density"].where(df["breast_density"].between(1, 4))
    df["subtlety"] = df["subtlety"].where(df["subtlety"].between(1, 5))
    # 4. binary target: MALIGNANT = 1; BENIGN and BENIGN_WITHOUT_CALLBACK = 0
    df["malignant"] = (df["pathology"] == "MALIGNANT").astype(int)
    return df


def load_raw(data_dir, cfg):
    train_p, test_p = Path(data_dir) / cfg["train"], Path(data_dir) / cfg["test"]
    for p in (train_p, test_p):
        if not p.exists():
            raise SystemExit(f"Missing input file: {p}\nUse --data-dir to point at the folder with the CBIS-DDSM CSVs.")
    return pd.read_csv(train_p), pd.read_csv(test_p)


# ----------------------------------------------------------------------------- 2. integrity checks
def integrity_checks(train, test, desc):
    print("Exact duplicate rows:", train.duplicated().sum())
    key = ["patient_id", "left_or_right_breast", "image_view", "abnormality_id"]
    print("Duplicate lesion keys:", train.duplicated(key).sum())

    # Patient overlap between train and test MUST be empty
    overlap = set(train.patient_id) & set(test.patient_id)
    print("Patient overlap train/test:", len(overlap))

    # Rows per patient (why you must split by patient)
    print(train.groupby("patient_id").size().describe())

    # Image-path duplicates are NORMAL (one mammogram, several lesions) ...
    print("Rows sharing a full-image path:", train["image_file_path"].duplicated(keep=False).sum())
    # ... but check for conflicting labels on the same full image
    conf = train.groupby("image_file_path")["malignant"].nunique()
    print("Full images with conflicting labels:", (conf > 1).sum())

    # Patients with mixed labels
    pl = train.groupby("patient_id")["malignant"].agg(["mean", "count"])
    print("Patients with mixed labels:", ((pl["mean"] > 0) & (pl["mean"] < 1)).sum())

    # Unique values in each categorical column
    for c in ["breast_density", "left_or_right_breast", "image_view", "subtlety", "assessment", "pathology"] + desc:
        print("\n---", c)
        print(train[c].value_counts(dropna=False).head(15))


# ----------------------------------------------------------------------------- 3. missing values
def missing_table(df):
    m = df.isna().sum()
    t = pd.DataFrame({"missing": m, "percent": (m / len(df) * 100).round(2), "dtype": df.dtypes.astype(str)})
    return t[t.missing > 0]


def missing_analysis(train, test, desc):
    print("TRAIN"); print(missing_table(train))
    print("TEST");  print(missing_table(test))

    for c in desc:
        miss = train[c].isna()
        if miss.sum() == 0:
            continue
        print(f"\n=== {c}: {miss.sum()} missing ===")
        print("malignant rate | missing:", round(train.loc[miss, "malignant"].mean(), 3),
              "| present:", round(train.loc[~miss, "malignant"].mean(), 3))
        print("chi2 p vs target:", stats.chi2_contingency(pd.crosstab(miss, train["malignant"]))[1])
        print(pd.crosstab(miss, train["pathology"]))      # 3-class view
        for other in ["breast_density", "image_view", "subtlety", "assessment"]:
            tab = pd.crosstab(miss, train[other].fillna(-1))
            print(f"   vs {other}: p={stats.chi2_contingency(tab)[1]:.4f}")

    mt = missing_table(train)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if len(mt):
        mt["percent"].plot.bar(ax=ax[0], title="Missing % per column (train)")
    rates = {c: [train.loc[train[c].isna(), "malignant"].mean(), train.loc[train[c].notna(), "malignant"].mean()]
             for c in desc if train[c].isna().any()}
    if rates:
        pd.DataFrame(rates, index=["missing", "present"]).T.plot.bar(ax=ax[1], title="Malignant rate: missing vs present")
    save_fig("01_missing_values")


def mask_and_recover(train, desc, col, n_splits=5):
    """Can the missing descriptor be recovered from the other columns? (patient-grouped CV)"""
    obs = train[train[col].notna()].reset_index(drop=True)
    others = [c for c in desc if c != col]
    X = pd.get_dummies(obs[["breast_density", "left_or_right_breast", "image_view", "subtlety"] + others]
                       .fillna("MISSING").astype(str))
    y = obs[col]
    res = {"mode": [], "rf": []}
    for a, b in GroupKFold(n_splits).split(X, y, obs["patient_id"]):      # patient-grouped
        mode = y.iloc[a].mode()[0]                                        # learned on train part only
        p_mode = np.repeat(mode, len(b))
        rf = RandomForestClassifier(200, random_state=0).fit(X.iloc[a], y.iloc[a])
        p_rf = rf.predict(X.iloc[b])
        for k, p in (("mode", p_mode), ("rf", p_rf)):
            res[k].append((accuracy_score(y.iloc[b], p), f1_score(y.iloc[b], p, average="macro")))
    return {k: np.mean(v, axis=0).round(3) for k, v in res.items()}   # (accuracy, macro-F1)


# ----------------------------------------------------------------------------- 4. encoding
def tokenise(v):
    if pd.isna(v):
        return ["UNKNOWN"]
    v = v.replace("LUCENT_CENTERED", "LUCENT_CENTER")
    return sorted(set(v.split("-")))       # de-duplicate and sort


def build_features(fit_df, apply_df, desc):
    """Learn encodings on fit_df ONLY, apply to apply_df. Returns (X_fit, X_apply)."""
    out_fit, out_apply = [], []
    for col in desc:
        mlb = MultiLabelBinarizer()
        a = mlb.fit_transform(fit_df[col].map(tokenise))
        b = mlb.transform(apply_df[col].map(lambda v: [t for t in tokenise(v) if t in mlb.classes_]))
        names = [f"{col}__{c}" for c in mlb.classes_]
        out_fit.append(pd.DataFrame(a, columns=names, index=fit_df.index))
        out_apply.append(pd.DataFrame(b, columns=names, index=apply_df.index))
    base = ["breast_density", "subtlety"]
    med = fit_df[base].median()                               # learned on fit_df only
    Xf = pd.concat([fit_df[base].fillna(med)] + out_fit, axis=1)
    Xa = pd.concat([apply_df[base].fillna(med)] + out_apply, axis=1)
    for X, d in ((Xf, fit_df), (Xa, apply_df)):
        X["view_mlo"] = (d["image_view"] == "MLO").astype(int)
        X["breast_right"] = (d["left_or_right_breast"] == "RIGHT").astype(int)
    return Xf, Xa


def target_balance(train, test):
    print(train["pathology"].value_counts())
    print(train["malignant"].value_counts(normalize=True).round(3))
    print(test["malignant"].value_counts(normalize=True).round(3))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    train["pathology"].value_counts().plot.bar(ax=ax[0], title="Pathology (3 classes, train)")
    pd.concat([train["malignant"].value_counts().rename("train"),
               test["malignant"].value_counts().rename("test")], axis=1).plot.bar(ax=ax[1], title="Malignant(1) share")
    save_fig("02_target_balance")


# ----------------------------------------------------------------------------- 5. leakage checks
def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


def leakage_checks(train, X_train, desc):
    # 1) assessment vs pathology: why assessment must not be a feature
    print(pd.crosstab(train["assessment"], train["pathology"]))
    print(train.groupby("assessment")["malignant"].agg(["mean", "count"]).round(3))

    # 2) association of every feature with the target (Cramer's V)
    tmp = train.copy()
    tmp[desc] = tmp[desc].fillna("MISSING")
    for c in ["assessment", "breast_density", "subtlety", "image_view", "left_or_right_breast"] + desc:
        print(c, round(cramers_v(tmp[c], tmp["malignant"]), 3))

    # 3) heatmap of assessment vs pathology
    sns.heatmap(pd.crosstab(train["assessment"], train["pathology"]), annot=True, fmt="d", cmap="Blues")
    plt.title("Assessment (BI-RADS) vs pathology")
    save_fig("03_assessment_vs_pathology")

    # Correlation among ordinal columns
    sns.heatmap(train[["breast_density", "subtlety", "assessment"]].corr(method="spearman"),
                annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman correlation (ordinal columns)")
    save_fig("04_spearman_ordinal")

    # Low-variance check on the final feature matrix
    var = X_train.var().sort_values()
    print("Near-constant columns:", var[var < 0.01].index.tolist())

    # Malignant rate per descriptor category (top levels)
    for c in desc:
        top = train[c].fillna("UNKNOWN").value_counts().head(8).index
        r = train.assign(cat=train[c].fillna("UNKNOWN"))
        r[r.cat.isin(top)].groupby("cat")["malignant"].mean().sort_values().plot.barh(title=f"Malignant rate by {c}")
        save_fig(f"05_malignant_rate_by_{c}")


# ----------------------------------------------------------------------------- 6. final cleaned CSV
def build_final_csv(data_dir, name, cfg, keep_view, out_path):
    """Self-contained rebuild of the cleaned dataset (train-only statistics), as in the notebook's last cell."""
    desc = cfg["desc"]

    def load(split):
        d = pd.read_csv(Path(data_dir) / cfg[split])
        d.columns = [c.strip().replace(" ", "_").lower() for c in d.columns]
        d["split"] = split
        return d

    train, test = load("train"), load("test")
    assert "pathology" in train.columns and "pathology" in test.columns   # target source must exist

    for d in (train, test):
        d["breast_density"] = d["breast_density"].where(d["breast_density"].between(1, 4))  # invalid 0 -> NaN
        d["subtlety"] = d["subtlety"].where(d["subtlety"].between(1, 5))
        d["malignant"] = (d["pathology"] == "MALIGNANT").astype(int)  # BENIGN + BENIGN_WITHOUT_CALLBACK = 0

    med = train[["breast_density", "subtlety"]].median()                          # learned on TRAIN only
    mlbs = {c: MultiLabelBinarizer().fit(train[c].map(tokenise)) for c in desc}   # vocabulary from TRAIN only

    def build(d):
        out = d[["patient_id", "split", "malignant"]].copy()
        out["breast_density"] = d["breast_density"].fillna(med["breast_density"]).astype(int)
        out["subtlety"] = d["subtlety"].fillna(med["subtlety"]).astype(int)
        out["breast_right"] = (d["left_or_right_breast"] == "RIGHT").astype(int)   # 1 = RIGHT, 0 = LEFT
        if keep_view:
            out["view_mlo"] = (d["image_view"] == "MLO").astype(int)
        for c, m in mlbs.items():
            arr = m.transform(d[c].map(lambda v: [t for t in tokenise(v) if t in m.classes_]))
            out = pd.concat([out, pd.DataFrame(arr, columns=[f"{c}__{k}" for k in m.classes_], index=d.index)], axis=1)
        return out

    final = pd.concat([build(train), build(test)], ignore_index=True)

    # ---- validation ----
    assert len(final) == len(train) + len(test)                       # no rows lost
    assert set(final["malignant"].unique()) <= {0, 1}
    assert final.isna().sum().sum() == 0
    print(final.shape)
    print(final.groupby("split")["malignant"].agg(["mean", "count"]).round(3))
    print("Duplicate columns:", final.columns[final.T.duplicated()].tolist())
    print("Leakage/path columns left:", [c for c in final.columns if any(k in c for k in ["assess", "patholog", "path"])])
    final.to_csv(out_path, index=False)
    print("Saved", out_path)
    return final


# ----------------------------------------------------------------------------- 7. dashboard export
def model_zoo():
    """Default-hyperparameter models (no tuning) for a quick, honest AUC comparison."""
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    svm = lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))
    knn = lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    return {
        "Logistic Regression": lr,
        "SVM": svm,
        "k-NN": knn,
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
        "Bernoulli NB": lambda: BernoulliNB(),
        "Random Forest": lambda: RandomForestClassifier(300, random_state=SEED),
        "Soft-Vote Ensemble": lambda: VotingClassifier(
            [("lr", lr()), ("svm", svm()), ("nb", BernoulliNB())], voting="soft"),
    }


def evaluate_models(final, out_path):
    """Patient-grouped CV AUC on train + ONE test AUC per model -> calcification_summary.csv."""
    feats = [c for c in final.columns if c not in ("patient_id", "split", "malignant")]
    tr, te = final[final.split == "train"], final[final.split == "test"]
    Xtr, ytr, gtr = tr[feats], tr["malignant"], tr["patient_id"]
    Xte, yte = te[feats], te["malignant"]
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(Xtr, ytr, gtr))

    rows = []
    for name, make in model_zoo().items():
        aucs = []
        for a, b in folds:
            m = make().fit(Xtr.iloc[a], ytr.iloc[a])
            aucs.append(roc_auc_score(ytr.iloc[b], m.predict_proba(Xtr.iloc[b])[:, 1]))
        m = make().fit(Xtr, ytr)                                      # refit on all training rows
        test_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])     # test set used once
        rows.append({"model": name, "cv_auc": round(float(np.mean(aucs)), 4), "test_auc": round(float(test_auc), 4)})
        print(f"{name:22s} CV AUC {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})   test AUC {test_auc:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print("Saved", out_path)
    return summary


# ----------------------------------------------------------------------------- main
def main():
    global FIG_DIR, KEEP_VIEW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./outputs"))
    ap.add_argument("--name", choices=list(CONFIG), default="calc", help="run calc and mass separately, never merge")
    ap.add_argument("--drop-view", action="store_true", help="drop the CC/MLO flag")
    ap.add_argument("--skip-models", action="store_true", help="skip step 7 (CV + test AUC export)")
    ap.add_argument("--dashboard-dir", default=None, help="also copy calcification_summary.csv here (dashboard/data)")
    args = ap.parse_args()

    NAME, cfg = args.name, CONFIG[args.name]
    DESC = cfg["desc"]
    KEEP_VIEW = not args.drop_view
    out_dir = Path(args.out_dir)
    FIG_DIR = out_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    section(f"Working on: {NAME}")
    print("Train file:", cfg["train"]); print("Test file:", cfg["test"]); print("Descriptor columns:", DESC)

    section("1. Load and basic clean")
    raw_train, raw_test = load_raw(args.data_dir, cfg)
    print("Shapes:", raw_train.shape, raw_test.shape)
    print(raw_train.dtypes)
    print(raw_train.head())
    train, test = basic_clean(raw_train), basic_clean(raw_test)
    print("density outside 1-4 (now NaN):", train["breast_density"].isna().sum(), test["breast_density"].isna().sum())
    print("subtlety NaN:", train["subtlety"].isna().sum(), test["subtlety"].isna().sum())
    print("path whitespace left:", (train["image_file_path"] != train["image_file_path"].str.strip()).sum())

    section("2. Duplicates, patient overlap, value counts")
    integrity_checks(train, test, DESC)

    section("3. Missing-value analysis")
    missing_analysis(train, test, DESC)
    for c in DESC:
        if train[c].isna().any():
            print(c, mask_and_recover(train, DESC, c))

    section("4. Categorical cleaning and multi-hot encoding")
    X_train, X_test = build_features(train, test, DESC)
    print(X_train.shape, X_test.shape)
    print("Unseen test tokens are ignored; columns match:", list(X_train.columns) == list(X_test.columns))
    target_balance(train, test)

    section("5. Leakage checks")
    leakage_checks(train, X_train, DESC)

    section("6. Final cleaned dataset")
    final = build_final_csv(args.data_dir, NAME, cfg, KEEP_VIEW, out_dir / f"breast_cancer_cleaned_final_{NAME}__.csv")

    if args.skip_models:
        return
    section("7. Model comparison export for the dashboard")
    summary_path = out_dir / "calcification_summary.csv"
    evaluate_models(final, summary_path)
    if args.dashboard_dir:
        dest = Path(args.dashboard_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(summary_path, dest / "calcification_summary.csv")
        print("Copied to", dest / "calcification_summary.csv")


if __name__ == "__main__":
    main()
