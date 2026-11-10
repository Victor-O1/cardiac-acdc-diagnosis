"""
STEP 0 - Look at the dataset before building anything.

Checks, on YOUR copy of ACDC:
  * where training/ and testing/ are and how many patients each has
  * every Info.cfg key, the Group (diagnosis) counts per split
  * that ED/ES images and *_gt masks exist for every patient
  * image shapes, voxel spacing, which axis is the slice axis
  * which label values appear in the masks
  * that ED and ES volumes of a patient have identical shape/spacing
Saves outputs/results/dataset_summary.json and a picture of one patient.
The later scripts rely on these checks, so fix anything it flags first.
"""
import os
import json
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (find_split_dirs, list_patients, read_info, find_frame_file, load_nii,
                    RESULTS_DIR, FIG_DIR, CLASSES)


def inspect_split(name, split_dir):
    pids = list_patients(split_dir)
    rep = {"dir": split_dir, "n_patients": len(pids)}
    if not pids:
        return rep
    keys, groups = Counter(), Counter()
    shapes, spacings, label_vals, dtypes = Counter(), [], set(), Counter()
    slice_axis = Counter()
    problems = []
    has_4d = 0
    frames_per_patient = defaultdict(list)
    for pid in pids:
        pdir = os.path.join(split_dir, pid)
        info = read_info(os.path.join(pdir, "Info.cfg"))
        keys.update(info.keys())
        groups[info.get("Group", "<missing>")] += 1
        if any(f.startswith(f"{pid}_4d") for f in os.listdir(pdir)):
            has_4d += 1
        try:
            ed, es = int(info["ED"]), int(info["ES"])
        except Exception as e:
            problems.append(f"{pid}: bad ED/ES in Info.cfg ({e})")
            continue
        frames_per_patient[pid] = [ed, es]
        per_phase = {}
        for phase, fr in (("ED", ed), ("ES", es)):
            ip = find_frame_file(pdir, pid, fr, required=False)
            gp = find_frame_file(pdir, pid, fr, gt=True, required=False)
            if ip is None:
                problems.append(f"{pid}: missing {phase} image (frame{fr:02d})")
                continue
            if gp is None:
                problems.append(f"{pid}: missing {phase} ground-truth mask (frame{fr:02d}_gt)")
            img, sp = load_nii(ip)
            shapes[img.shape] += 1
            spacings.append(sp)
            dtypes[str(img.dtype)] += 1
            slice_axis[int(np.argmax(sp))] += 1
            per_phase[phase] = (img.shape, sp)
            if gp is not None:
                m, msp = load_nii(gp)
                if m.shape != img.shape:
                    problems.append(f"{pid} {phase}: mask shape {m.shape} != image {img.shape}")
                if not np.allclose(msp, sp):
                    problems.append(f"{pid} {phase}: mask spacing {msp} != image {sp}")
                label_vals.update(np.unique(np.rint(m)).astype(int).tolist())
        if "ED" in per_phase and "ES" in per_phase:
            if per_phase["ED"][0] != per_phase["ES"][0]:
                problems.append(f"{pid}: ED shape {per_phase['ED'][0]} != ES shape {per_phase['ES'][0]}")
            if not np.allclose(per_phase["ED"][1], per_phase["ES"][1]):
                problems.append(f"{pid}: ED/ES spacing differ")
    sp = np.array(spacings)
    rep.update({
        "info_cfg_keys": dict(keys),
        "group_counts": dict(groups),
        "patients_with_4d_file": has_4d,
        "n_distinct_shapes": len(shapes),
        "most_common_shapes": [[list(k), v] for k, v in shapes.most_common(8)],
        "n_slices_range": [int(min(s[2] for s in shapes)), int(max(s[2] for s in shapes))],
        "spacing_min_xyz": sp.min(0).round(3).tolist(),
        "spacing_max_xyz": sp.max(0).round(3).tolist(),
        "axis_with_largest_spacing_counts": dict(slice_axis),
        "image_dtypes": dict(dtypes),
        "mask_label_values": sorted(label_vals),
        "problems": problems,
    })
    return rep


def save_example(split_dir, pid):
    pdir = os.path.join(split_dir, pid)
    info = read_info(os.path.join(pdir, "Info.cfg"))
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    for r, (phase, fr) in enumerate((("ED", int(info["ED"])), ("ES", int(info["ES"])))):
        img, _ = load_nii(find_frame_file(pdir, pid, fr))
        gp = find_frame_file(pdir, pid, fr, gt=True, required=False)
        m = np.rint(load_nii(gp)[0]).astype(int) if gp else np.zeros(img.shape, int)
        z = img.shape[2] // 2
        ax[r, 0].imshow(img[:, :, z].T, cmap="gray", origin="lower")
        ax[r, 0].set_title(f"{pid} {phase} slice {z}")
        ax[r, 1].imshow(m[:, :, z].T, cmap="viridis", vmin=0, vmax=3, origin="lower")
        ax[r, 1].set_title("mask (0 bg,1 RV,2 MYO,3 LV)")
        ax[r, 2].imshow(img[:, :, z].T, cmap="gray", origin="lower")
        ax[r, 2].imshow(np.ma.masked_where(m[:, :, z].T == 0, m[:, :, z].T),
                        cmap="jet", alpha=0.45, vmin=0, vmax=3, origin="lower")
        ax[r, 2].set_title("overlay")
    for a in ax.ravel():
        a.axis("off")
    plt.suptitle(f"{pid}  Group={info.get('Group')}  Height={info.get('Height')}  Weight={info.get('Weight')}")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"example_{pid}.png")
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def main():
    train_dir, test_dir = find_split_dirs()
    summary = {"training": inspect_split("training", train_dir)}
    summary["testing"] = inspect_split("testing", test_dir) if test_dir else {"dir": None, "n_patients": 0}

    print(json.dumps(summary, indent=2, default=str))
    with open(os.path.join(RESULTS_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Example figure:", save_example(train_dir, list_patients(train_dir)[0]))

    # ---- hard checks the rest of the pipeline depends on
    print("\n================ CHECKS ================")
    ok = True
    for split in ("training", "testing"):
        s = summary[split]
        if s.get("n_patients", 0) == 0:
            print(f"[FAIL] {split}: no patients found")
            ok = False
            continue
        unknown = set(s["group_counts"]) - set(CLASSES)
        if unknown:
            print(f"[FAIL] {split}: unexpected Group values {unknown} (expected {CLASSES})")
            ok = False
        if not set(s["mask_label_values"]) <= {0, 1, 2, 3}:
            print(f"[FAIL] {split}: unexpected mask values {s['mask_label_values']}")
            ok = False
        if set(s["axis_with_largest_spacing_counts"]) != {2}:
            print(f"[FAIL] {split}: slice axis is not always axis 2 -> {s['axis_with_largest_spacing_counts']}")
            ok = False
        for k in ("ED", "ES", "Group", "Height", "Weight"):
            if s["info_cfg_keys"].get(k, 0) != s["n_patients"]:
                print(f"[FAIL] {split}: Info.cfg key '{k}' missing for some patients")
                ok = False
        if s["problems"]:
            print(f"[FAIL] {split}: {len(s['problems'])} problems, first ones: {s['problems'][:5]}")
            ok = False
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - read the messages above before continuing")


if __name__ == "__main__":
    main()
