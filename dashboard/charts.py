"""charts.py - metric maths and matplotlib figures used by the pages (no file access here)."""
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
