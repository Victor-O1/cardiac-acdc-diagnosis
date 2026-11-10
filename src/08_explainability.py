"""
STEP 8 - Explainability figures.
  * SHAP (XGBoost): global mean |SHAP| per feature, and per-class importance
  * Grad-CAM (CNN): one test patient per diagnosis, mid heart slice
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_cached, load_splits, get_device, pred_mask_path,
                    CLASSES, MODEL_DIR, RESULTS_DIR, FIG_DIR)
from features import FEATURE_COLS
from xai_utils import load_xgb, shap_values
from cnn_utils import load_cnn, GradCAM, heart_roi, heart_slices, make_input, to_tensor


def shap_figures():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "features_test_pred.csv"))
    model = load_xgb(os.path.join(MODEL_DIR, "xgb.json"))
    sv = shap_values(model, df[FEATURE_COLS].values, FEATURE_COLS)       # n, f, c

    glob_imp = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(glob_imp)
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(FEATURE_COLS)[order], glob_imp[order], color="#3b6e8f")
    plt.xlabel("mean |SHAP value| (all classes)")
    plt.title("XGBoost global feature importance (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_global.png"), dpi=120)
    plt.close()

    per_class = np.abs(sv).mean(axis=0)                                   # f, c
    pd.DataFrame(per_class, index=FEATURE_COLS, columns=CLASSES).round(4) \
        .to_csv(os.path.join(RESULTS_DIR, "shap_per_class.csv"))
    fig, ax = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    for k, c in enumerate(CLASSES):
        ax[k].barh(FEATURE_COLS, per_class[:, k], color="#8f3b4a")
        ax[k].set_title(c)
    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_per_class.png"), dpi=110)
    plt.close()
    print("saved SHAP figures")


def gradcam_figures():
    device = get_device()
    model = load_cnn(os.path.join(MODEL_DIR, "cnn_best.pt"), device)
    cam = GradCAM(model)
    s = load_splits()
    picks = {}
    for pid in s["test"]:
        g = load_cached(pid)["group"]
        picks.setdefault(g, pid)
    fig, ax = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 7.2))
    ax = np.array(ax).reshape(2, -1)
    for k, (g, pid) in enumerate(sorted(picks.items(), key=lambda t: CLASSES.index(t[0]))):
        d = load_cached(pid)
        mask = np.load(pred_mask_path(pid))["ED"]
        roi = heart_roi(mask, d["spacing"])
        zs = heart_slices(mask) or [d["ED_img"].shape[2] // 2]
        z = zs[len(zs) // 2]
        x = make_input(d["ED_img"], d["ES_img"], z, roi)
        heat, cls = cam(to_tensor(x).to(device))
        ax[0, k].imshow(x[..., 0].T, cmap="gray", origin="lower")
        ax[0, k].set_title(f"{pid} true {g}")
        ax[1, k].imshow(x[..., 0].T, cmap="gray", origin="lower")
        ax[1, k].imshow(heat.T, cmap="jet", alpha=0.4, origin="lower")
        ax[1, k].set_title(f"Grad-CAM -> {CLASSES[cls]}")
        for a in ax[:, k]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gradcam_examples.png"), dpi=110)
    plt.close()
    print("saved Grad-CAM figure")


if __name__ == "__main__":
    shap_figures()
    gradcam_figures()
