"""
STEP 1 - Cache normalized ED/ES volumes + masks and create patient-level splits.

Splits (by PATIENT, never by slice, so no leakage):
  train : 80 of the 100 'training' patients (stratified by Group)
  val   : remaining 20 'training' patients
  test  : all 'testing' patients (101-150) - only touched for final evaluation
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from common import (find_split_dirs, list_patients, load_patient_raw, normalize_volume,
                    cache_path, SPLITS_JSON, SEED)


def cache_split(split_dir, split_name):
    pids = list_patients(split_dir)
    groups = []
    for i, pid in enumerate(pids):
        p = load_patient_raw(split_dir, pid)
        if p["ED_gt"] is None or p["ES_gt"] is None:
            raise RuntimeError(f"{pid} has no ground-truth masks; 00_inspect_dataset.py should have flagged this.")
        np.savez_compressed(
            cache_path(pid),
            pid=pid, group=p["group"], split=split_name,
            height=p["height"], weight=p["weight"],
            spacing=np.array(p["spacing"], dtype=np.float32),
            ED_img=normalize_volume(p["ED_img"]), ES_img=normalize_volume(p["ES_img"]),
            ED_gt=p["ED_gt"], ES_gt=p["ES_gt"],
        )
        groups.append(p["group"])
        print(f"[{split_name}] {i + 1}/{len(pids)} {pid} group={p['group']} shape={p['ED_img'].shape} "
              f"spacing={tuple(round(s, 3) for s in p['spacing'])}")
    return pids, groups


def main():
    train_dir, test_dir = find_split_dirs()
    tr_pids, tr_groups = cache_split(train_dir, "training")
    te_pids, _ = cache_split(test_dir, "testing") if test_dir else ([], [])

    train, val = train_test_split(tr_pids, test_size=0.2, stratify=tr_groups, random_state=SEED)
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(te_pids),
              "all_training": sorted(tr_pids)}
    with open(SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=2)
    print({k: len(v) for k, v in splits.items()}, "-> saved", SPLITS_JSON)


if __name__ == "__main__":
    main()
