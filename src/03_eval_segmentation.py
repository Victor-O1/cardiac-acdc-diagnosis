"""
STEP 3 - Evaluate segmentation on the held-out test patients (Dice, IoU, HD, HD95)
and save predicted ED/ES masks (used later for end-to-end diagnosis).

    python src/03_eval_segmentation.py --arch unet
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_cached, load_splits, get_device, pred_mask_path,
                    MODEL_DIR, RESULTS_DIR, FIG_DIR, LABEL_NAMES)
from seg_utils import load_seg_model, predict_volume, dice, iou, hausdorff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp"])
    args = ap.parse_args()

    device = get_device()
    model = load_seg_model(os.path.join(MODEL_DIR, f"{args.arch}_best.pt"), device)
    splits = load_splits()
    rows = []
    # predict val too (not used for training) so the CNN script can use predicted ROIs if you want
    for split in ("val", "test"):
        for pid in splits[split]:
            d = load_cached(pid)
            preds = {}
            for ph in ("ED", "ES"):
                pred = predict_volume(model, d[f"{ph}_img"], device)
                preds[ph] = pred
                gt = d[f"{ph}_gt"]
                for c, name in LABEL_NAMES.items():
                    p, g = pred == c, gt == c
                    rows.append({"split": split, "pid": pid, "phase": ph, "structure": name,
                                 "dice": dice(p, g), "iou": iou(p, g),
                                 "hd_mm": hausdorff(p, g, d["spacing"]),
                                 "hd95_mm": hausdorff(p, g, d["spacing"], 95)})
            np.savez_compressed(pred_mask_path(pid), ED=preds["ED"], ES=preds["ES"])
            print(split, pid, "done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"seg_metrics_{args.arch}_per_patient.csv"), index=False)
    summ = df.groupby(["split", "phase", "structure"])[["dice", "iou", "hd_mm", "hd95_mm"]].mean().round(4)
    summ.to_csv(os.path.join(RESULTS_DIR, f"seg_metrics_{args.arch}_summary.csv"))
    print(summ)

    # visual check for 4 test patients
    pids = splits["test"][:4]
    fig, ax = plt.subplots(len(pids), 3, figsize=(10, 3.3 * len(pids)))
    ax = np.atleast_2d(ax)
    for r, pid in enumerate(pids):
        d = load_cached(pid)
        pred = np.load(pred_mask_path(pid))["ED"]
        z = d["ED_img"].shape[2] // 2
        ax[r, 0].imshow(d["ED_img"][:, :, z].T, cmap="gray", origin="lower")
        ax[r, 1].imshow(d["ED_gt"][:, :, z].T, vmin=0, vmax=3, origin="lower")
        ax[r, 2].imshow(pred[:, :, z].T, vmin=0, vmax=3, origin="lower")
        for a, t in zip(ax[r], (f"{pid} ED", "ground truth", "prediction")):
            a.set_title(t)
            a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"seg_examples_{args.arch}.png"), dpi=110)
    print("saved figure + metrics")


if __name__ == "__main__":
    main()
