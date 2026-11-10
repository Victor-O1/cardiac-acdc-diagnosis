"""
Shared configuration and ACDC data utilities.

Every fact about the dataset that this file relies on is re-checked on YOUR copy
by 00_inspect_dataset.py. Run that first.

Verified ACDC layout (official release / Hugging Face mirror):
    <root>/training/patient001 ... patient100
    <root>/testing/patient101  ... patient150
    each patient folder:
        Info.cfg                      -> ED, ES, Group, Height, NbFrame, Weight
        patientXXX_4d.nii.gz          -> full cine sequence (NOT used here)
        patientXXX_frameYY.nii.gz     -> ED or ES 3D volume (YY = ED/ES number, 2 digits)
        patientXXX_frameYY_gt.nii.gz  -> expert mask: 0=background, 1=RV, 2=MYO, 3=LV
"""
import os
import re
import glob
import json

import cv2
import numpy as np
import nibabel as nib

# --------------------------------------------------------------------------- paths
# Project root = parent folder of src/. Override with env var CARDIAC_ROOT if needed.
PROJECT_ROOT = os.environ.get(
    "CARDIAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
PRED_DIR = os.path.join(OUT_DIR, "pred_masks")
MODEL_DIR = os.path.join(OUT_DIR, "models")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for _d in (OUT_DIR, CACHE_DIR, PRED_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

SPLITS_JSON = os.path.join(OUT_DIR, "splits.json")

# --------------------------------------------------------------------------- constants
CLASSES = ["NOR", "MINF", "DCM", "HCM", "RV"]          # Info.cfg "Group" values
CLASS_FULL = {
    "NOR": "Normal",
    "MINF": "Previous myocardial infarction",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}
LABEL_NAMES = {1: "RV", 2: "MYO", 3: "LV"}             # mask values in *_gt files
IMG_SIZE = 224          # 2D network input size
CROP_MM = 160.0         # physical size of the heart ROI crop for the CNN classifier
MYO_DENSITY = 1.05      # g/ml, used for myocardial mass
SEED = 42


# --------------------------------------------------------------------------- file helpers
def find_split_dirs():
    """Locate training/ and testing/ by searching for patient001/Info.cfg under data/."""
    hits = sorted(glob.glob(os.path.join(DATA_DIR, "**", "patient001", "Info.cfg"), recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"Could not find patient001/Info.cfg anywhere under {DATA_DIR}. "
            "Extract the ACDC 'training' and 'testing' folders inside data/."
        )
    if len(hits) > 1:
        print(f"[warn] several copies of patient001 found, using: {hits[0]}")
    train_dir = os.path.dirname(os.path.dirname(hits[0]))
    test_dir = os.path.join(os.path.dirname(train_dir), "testing")
    if not os.path.isdir(test_dir):
        test_dir = None
    return train_dir, test_dir


def list_patients(split_dir):
    if split_dir is None:
        return []
    return sorted(d for d in os.listdir(split_dir) if re.fullmatch(r"patient\d{3}", d))


def read_info(path):
    info = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def find_frame_file(pdir, pid, frame, gt=False, required=True):
    suffix = "_gt" if gt else ""
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(pdir, f"{pid}_frame{frame:02d}{suffix}{ext}")
        if os.path.exists(p):
            return p
    if required:
        raise FileNotFoundError(f"Missing {pid}_frame{frame:02d}{suffix}.nii(.gz) in {pdir}")
    return None


def load_nii(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing


def load_patient_raw(split_dir, pid):
    """Load ED/ES images (+ masks if present) and Info.cfg for one patient."""
    pdir = os.path.join(split_dir, pid)
    info = read_info(os.path.join(pdir, "Info.cfg"))
    ed, es = int(info["ED"]), int(info["ES"])
    out = {
        "pid": pid,
        "group": info.get("Group"),
        "height": float(info["Height"]),
        "weight": float(info["Weight"]),
        "ed_frame": ed,
        "es_frame": es,
    }
    for phase, fr in (("ED", ed), ("ES", es)):
        img, sp = load_nii(find_frame_file(pdir, pid, fr))
        out[f"{phase}_img"] = img.astype(np.float32)
        out[f"{phase}_spacing"] = sp
        gtp = find_frame_file(pdir, pid, fr, gt=True, required=False)
        out[f"{phase}_gt"] = np.rint(load_nii(gtp)[0]).astype(np.uint8) if gtp else None
    out["spacing"] = out["ED_spacing"]
    return out


# --------------------------------------------------------------------------- preprocessing
def normalize_volume(v):
    """Per-volume percentile clipping (1-99) then scaling to [0, 1]."""
    v = v.astype(np.float32)
    p1, p99 = np.percentile(v, [1, 99])
    v = np.clip(v, p1, p99)
    return (v - p1) / (p99 - p1 + 1e-8)


def cache_path(pid):
    return os.path.join(CACHE_DIR, f"{pid}.npz")


def load_cached(pid):
    d = np.load(cache_path(pid), allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["pid"] = str(out["pid"])
    out["group"] = str(out["group"])
    out["spacing"] = tuple(float(s) for s in out["spacing"])
    out["height"] = float(out["height"])
    out["weight"] = float(out["weight"])
    return out


def load_splits():
    if not os.path.exists(SPLITS_JSON):
        raise FileNotFoundError("splits.json missing - run 01_preprocess.py first.")
    with open(SPLITS_JSON) as f:
        return json.load(f)


def pred_mask_path(pid):
    return os.path.join(PRED_DIR, f"{pid}.npz")


# --------------------------------------------------------------------------- augmentation
def random_affine(img, mask, rng, max_rot=15.0, scale=(0.9, 1.1), max_shift=0.05):
    """Rotation/scale/shift. Image: bilinear. Mask: nearest (labels stay valid).
    No flips: flipping would put the RV on the wrong side of the heart."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-max_rot, max_rot), rng.uniform(*scale))
    M[:, 2] += rng.uniform(-max_shift, max_shift, 2) * np.array([w, h])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if mask is not None:
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


def get_device():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[warn] No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    return dev


def set_seed(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
