"""
STEP 4 - Build feature tables.
  features_train_gt.csv : 100 training patients, features from EXPERT masks (classifier training)
  features_test_gt.csv  : test patients, expert masks (upper bound)
  features_test_pred.csv: test patients, U-Net masks (true end-to-end pipeline)
"""
import os

import numpy as np
import pandas as pd

from common import load_cached, load_splits, pred_mask_path, RESULTS_DIR
from features import compute_features


def table(pids, use_pred):
    rows = []
    for pid in pids:
        d = load_cached(pid)
        if use_pred:
            p = np.load(pred_mask_path(pid))
            ed, es = p["ED"], p["ES"]
        else:
            ed, es = d["ED_gt"], d["ES_gt"]
        f = compute_features(ed, es, d["spacing"], d["height"], d["weight"])
        rows.append({"pid": pid, "group": d["group"], **f})
    return pd.DataFrame(rows)


def main():
    s = load_splits()
    table(s["all_training"], False).to_csv(os.path.join(RESULTS_DIR, "features_train_gt.csv"), index=False)
    table(s["test"], False).to_csv(os.path.join(RESULTS_DIR, "features_test_gt.csv"), index=False)
    missing = [p for p in s["test"] if not os.path.exists(pred_mask_path(p))]
    if missing:
        print(f"[warn] {len(missing)} test patients have no predicted masks - run 03 first. Skipping pred table.")
    else:
        table(s["test"], True).to_csv(os.path.join(RESULTS_DIR, "features_test_pred.csv"), index=False)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "features_train_gt.csv"))
    print(df.groupby("group")[["LV_EDV", "LV_EF", "RV_EDV", "RV_EF", "MYO_mass_ED",
                                "MYO_thick_max_ED"]].mean().round(1))


if __name__ == "__main__":
    main()
