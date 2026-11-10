"""
STEP 9 - Paper-grade metrics with 95% confidence intervals.

Reads the test-set probability files written by steps 5, 6 and 7 and writes:
  results/metrics_ci.csv           macro metrics + 95% CI for every model
  results/per_class_ci.csv         per-class metrics + 95% CI for the main model
  results/subgroup_ci.csv          main model by body-surface-area group
  results/seg_ci.csv               Dice / HD95 with 95% CI per structure
  results/cohort_table.csv         cohort description
  figures/roc_pr.png, figures/calibration.png

The confidence intervals come from a percentile bootstrap over PATIENTS
(one row = one patient, so a resample never splits a patient).
No model is trained or tuned here. This script only reads predictions.

    python src/09_report_metrics.py [--main ensemble] [--boot 2000]
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, f1_score, accuracy_score)

from common import CLASSES, RESULTS_DIR, FIG_DIR, SEED

P = [f"p_{c}" for c in CLASSES]
SOURCES = {
    "rf_pred": ("rf_test_probs_pred.csv", "Random forest baseline (U-Net features)"),
    "cnn": ("cnn_test_probs.csv", "CNN (image)"),
    "xgb_pred": ("xgb_test_probs_pred.csv", "XGBoost (U-Net features)"),
    "ensemble": ("ensemble_test_probs.csv", "Ensemble (mean probability)"),
    "xgb_gt": ("xgb_test_probs_gt.csv", "XGBoost (expert-mask features)"),
}


# --------------------------------------------------------------------------- metrics
def onehot(y):
    return np.eye(len(CLASSES))[y]


def per_class_metrics(y, probs):
    """One-vs-rest metrics for each class. Returns dict name -> array of length 5.
    A class that is absent from y (or the only class) gives NaN for rank metrics."""
    pred = probs.argmax(1)
    out = {k: np.full(len(CLASSES), np.nan) for k in
           ("auroc", "auprc", "sensitivity", "specificity", "f1")}
    for k in range(len(CLASSES)):
        t = (y == k).astype(int)
        p = (pred == k).astype(int)
        tp, fn = ((p == 1) & (t == 1)).sum(), ((p == 0) & (t == 1)).sum()
        tn, fp = ((p == 0) & (t == 0)).sum(), ((p == 1) & (t == 0)).sum()
        if t.sum() > 0:
            out["sensitivity"][k] = tp / (tp + fn)
            out["f1"][k] = 2 * tp / (2 * tp + fp + fn)
        if t.sum() < len(t):
            out["specificity"][k] = tn / (tn + fp)
        if 0 < t.sum() < len(t):
            out["auroc"][k] = roc_auc_score(t, probs[:, k])
            out["auprc"][k] = average_precision_score(t, probs[:, k])
    return out


def brier(y, probs):
    return float(np.mean(np.sum((probs - onehot(y)) ** 2, axis=1)))


def ece(y, probs, bins=10):
    """Expected calibration error on the top-label confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def macro_metrics(y, probs):
    pc = per_class_metrics(y, probs)
    m = {k.upper() if k in ("auroc", "auprc") else k: float(np.nanmean(v)) for k, v in pc.items()}
    m["accuracy"] = float(accuracy_score(y, probs.argmax(1)))
    m["brier"] = brier(y, probs)
    m["ece"] = ece(y, probs)
    return m


def bootstrap(y, probs, fn, n_boot, rng):
    """Percentile bootstrap over patients. fn(y, probs) -> dict of floats."""
    point = fn(y, probs)
    draws = {k: [] for k in point}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = fn(y[idx], probs[idx])
        for k, v in r.items():
            draws[k].append(v)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
          for k, v in draws.items()}
    return point, ci


def fmt(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


# --------------------------------------------------------------------------- loaders
def load_probs(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "pid" not in df.columns:
        df = df.rename(columns={df.columns[0]: "pid"})
    return df.sort_values("pid").reset_index(drop=True)


# --------------------------------------------------------------------------- sections
def model_table(n_boot, rng):
    rows, store = [], {}
    for key, (fname, label) in SOURCES.items():
        df = load_probs(fname)
        if df is None:
            print(f"[skip] {fname} not found")
            continue
        y = df["true"].map(CLASSES.index).values
        probs = df[P].values
        point, ci = bootstrap(y, probs, macro_metrics, n_boot, rng)
        row = {"model": label, "n_patients": len(df)}
        for k in point:
            row[k] = round(point[k], 4)
            row[f"{k}_lo"], row[f"{k}_hi"] = round(ci[k][0], 4), round(ci[k][1], 4)
        rows.append(row)
        store[key] = (df, y, probs)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "metrics_ci.csv"), index=False)
    return out, store


def per_class_table(y, probs, n_boot, rng):
    def fn(yy, pp):
        pc = per_class_metrics(yy, pp)
        return {f"{m}|{c}": pc[m][i] for m in pc for i, c in enumerate(CLASSES)}
    point, ci = bootstrap(y, probs, fn, n_boot, rng)
    rows = []
    for c in CLASSES:
        row = {"class": c}
        for m in ("auroc", "auprc", "sensitivity", "specificity", "f1"):
            k = f"{m}|{c}"
            row[m] = round(point[k], 4)
            row[f"{m}_lo"], row[f"{m}_hi"] = round(ci[k][0], 4), round(ci[k][1], 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "per_class_ci.csv"), index=False)
    return out


def subgroup_table(df, y, probs, n_boot, rng):
    feat_path = os.path.join(RESULTS_DIR, "features_test_pred.csv")
    if not os.path.exists(feat_path):
        print("[skip] subgroup: features_test_pred.csv missing")
        return None
    f = pd.read_csv(feat_path).set_index("pid").loc[df["pid"]]
    bsa = f["BSA"].values
    groups = {"BSA below median": bsa < np.median(bsa), "BSA at or above median": bsa >= np.median(bsa)}
    rows = []
    for name, m in groups.items():
        if m.sum() < 5:
            continue
        point, ci = bootstrap(y[m], probs[m], macro_metrics, n_boot, rng)
        rows.append({"subgroup": name, "n": int(m.sum()),
                     **{k: fmt(point[k], *ci[k]) for k in ("AUROC", "AUPRC", "sensitivity",
                                                           "specificity", "f1", "accuracy")}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "subgroup_ci.csv"), index=False)
    return out


def seg_table(n_boot, rng):
    import glob
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "seg_metrics_*_per_patient.csv")))
    if not paths:
        print("[skip] segmentation: per-patient csv missing")
        return None
    df = pd.read_csv(paths[0])
    df = df[df["split"] == "test"]
    rows = []
    for (phase, st), g in df.groupby(["phase", "structure"]):
        per_pat = g.groupby("pid")[["dice", "hd95_mm"]].mean()
        row = {"phase": phase, "structure": st, "n": len(per_pat)}
        for col in ("dice", "hd95_mm"):
            v = per_pat[col].values
            bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
            row[col] = fmt(np.nanmean(v), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5),
                           3 if col == "dice" else 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "seg_ci.csv"), index=False)
    return out


def cohort_table():
    parts = []
    for name, fn in (("Training (n=100)", "features_train_gt.csv"),
                     ("Test (n=50)", "features_test_gt.csv")):
        p = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        for c in CLASSES + ["All"]:
            g = df if c == "All" else df[df["group"] == c]
            if len(g) == 0:
                continue
            parts.append({"split": name, "group": c, "n": len(g),
                          "height_cm": f"{g.Height.mean():.0f} +/- {g.Height.std():.0f}",
                          "weight_kg": f"{g.Weight.mean():.0f} +/- {g.Weight.std():.0f}",
                          "LV_EF_pct": f"{g.LV_EF.mean():.0f} +/- {g.LV_EF.std():.0f}",
                          "LV_EDV_ml": f"{g.LV_EDV.mean():.0f} +/- {g.LV_EDV.std():.0f}"})
    out = pd.DataFrame(parts)
    out.to_csv(os.path.join(RESULTS_DIR, "cohort_table.csv"), index=False)
    return out


# --------------------------------------------------------------------------- figures
def curve_figures(y, probs, title):
    colors = plt.cm.tab10(np.arange(len(CLASSES)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for k, c in enumerate(CLASSES):
        t = (y == k).astype(int)
        if not 0 < t.sum() < len(t):
            continue
        fpr, tpr, _ = roc_curve(t, probs[:, k])
        pr, rc, _ = precision_recall_curve(t, probs[:, k])
        ax[0].plot(fpr, tpr, color=colors[k], label=f"{c} (AUC {roc_auc_score(t, probs[:, k]):.2f})")
        ax[1].plot(rc, pr, color=colors[k], label=f"{c} (AP {average_precision_score(t, probs[:, k]):.2f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC (one vs rest)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (one vs rest)")
    for a in ax:
        a.legend(fontsize=7, loc="lower right" if a is ax[0] else "lower left")
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_pr.png"), dpi=200)
    plt.close(fig)


def calibration_figure(y, probs, title, bins=5):
    """Reliability diagram on the pooled one-vs-rest probabilities."""
    p = probs.ravel()
    t = onehot(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() > 0:
            xs.append(p[m].mean())
            ys.append(t[m].mean())
            ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="#3b6e8f", label="model")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yv), textcoords="offset points", xytext=(4, -10), fontsize=6)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title=title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="ensemble", choices=list(SOURCES))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    table, store = model_table(args.boot, rng)
    if table.empty:
        raise SystemExit("No probability files found. Run steps 5, 6 and 7 first.")
    key = args.main if args.main in store else next(iter(store))
    df, y, probs = store[key]
    label = SOURCES[key][1]
    print(f"Main model for per-class, subgroup and curves: {label}")

    show = table[["model", "AUROC", "AUPRC", "sensitivity", "specificity", "f1", "accuracy", "brier", "ece"]]
    print(show.to_string(index=False))
    print(per_class_table(y, probs, args.boot, rng).to_string(index=False))
    sg = subgroup_table(df, y, probs, args.boot, rng)
    if sg is not None:
        print(sg.to_string(index=False))
    sc = seg_table(args.boot, rng)
    if sc is not None:
        print(sc.to_string(index=False))
    print(cohort_table().to_string(index=False))
    curve_figures(y, probs, label)
    calibration_figure(y, probs, label)
    print("saved tables to results/ and figures to figures/")


if __name__ == "__main__":
    main()
