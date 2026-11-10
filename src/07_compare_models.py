"""
STEP 7 - Compare CNN (raw image) vs XGBoost (clinical features) and their ensemble
on the test patients. Everything here uses U-Net predicted masks (end-to-end),
plus XGBoost on expert masks as an upper bound.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from common import CLASSES, RESULTS_DIR, FIG_DIR

P = [f"p_{c}" for c in CLASSES]


def main():
    xgb = pd.read_csv(os.path.join(RESULTS_DIR, "xgb_test_probs_pred.csv")).set_index("pid")
    cnn = pd.read_csv(os.path.join(RESULTS_DIR, "cnn_test_probs.csv")).set_index("pid")
    cnn = cnn.loc[xgb.index]
    ens = xgb[P].values * 0.5 + cnn[P].values * 0.5
    y = xgb["true"].values

    preds = {
        "CNN (image)": cnn["pred"].values,
        "XGBoost (U-Net features)": xgb["pred"].values,
        "Ensemble (mean prob.)": np.array([CLASSES[i] for i in ens.argmax(1)]),
    }
    gt_path = os.path.join(RESULTS_DIR, "xgb_test_probs_gt.csv")
    if os.path.exists(gt_path):
        preds["XGBoost (expert-mask features, upper bound)"] = \
            pd.read_csv(gt_path).set_index("pid").loc[xgb.index, "pred"].values

    table = pd.DataFrame([{"model": k, "accuracy": accuracy_score(y, v),
                           "macro_f1": f1_score(y, v, average="macro")} for k, v in preds.items()]).round(3)
    table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(table.to_string(index=False))

    out = pd.DataFrame(ens, columns=P, index=xgb.index)
    out.insert(0, "true", y)
    out["pred"] = preds["Ensemble (mean prob.)"]
    out.to_csv(os.path.join(RESULTS_DIR, "ensemble_test_probs.csv"))

    fig, ax = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.3))
    for a, (name, p) in zip(np.atleast_1d(ax), preds.items()):
        cm = confusion_matrix(y, p, labels=CLASSES)
        a.imshow(cm, cmap="Blues")
        for i in range(5):
            for j in range(5):
                a.text(j, i, cm[i, j], ha="center", va="center")
        a.set_xticks(range(5), CLASSES)
        a.set_yticks(range(5), CLASSES)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(f"{name}\nacc {accuracy_score(y, p):.2f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=120)
    print("saved results/model_comparison.csv and figures/confusion_matrices.png")


if __name__ == "__main__":
    main()
