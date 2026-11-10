"""
STEP 6 - Image-based diagnosis with a CNN (ResNet18, ImageNet-pretrained).

Input per slice = [ED, ES, |ED-ES|] cropped around the heart; patient prediction = mean
softmax over the heart slices.
  train : heart ROI from EXPERT masks
  val/test : heart ROI from U-Net PREDICTED masks (run 03 first) -> honest end-to-end
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from common import (load_cached, load_splits, random_affine, get_device, set_seed,
                    pred_mask_path, CLASSES, MODEL_DIR, RESULTS_DIR)
from cnn_utils import (heart_roi, heart_slices, make_input, to_tensor, build_cnn,
                       load_cnn, patient_probs)


class CNNSliceDataset(Dataset):
    def __init__(self, pids, seed=0):
        self.x, self.y = [], []
        for pid in pids:
            d = load_cached(pid)
            roi = heart_roi(d["ED_gt"], d["spacing"])
            for z in heart_slices(d["ED_gt"]):
                self.x.append(make_input(d["ED_img"], d["ES_img"], z, roi))
                self.y.append(CLASSES.index(d["group"]))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, _ = random_affine(self.x[i], None, self.rng)
        x = np.clip(x * self.rng.uniform(0.9, 1.1), 0, 1)
        return to_tensor(x)[0], self.y[i]


def eval_patients(model, pids, device):
    rows = []
    for pid in pids:
        d = load_cached(pid)
        mp = pred_mask_path(pid)
        mask = np.load(mp)["ED"] if os.path.exists(mp) else d["ED_gt"]
        pr = patient_probs(model, d["ED_img"], d["ES_img"], mask, d["spacing"], device)
        rows.append({"pid": pid, "true": d["group"], **{f"p_{c}": pr[i] for i, c in enumerate(CLASSES)},
                     "pred": CLASSES[int(pr.argmax())]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    s = load_splits()
    for pid in s["val"] + s["test"]:
        if not os.path.exists(pred_mask_path(pid)):
            print(f"[warn] no predicted mask for {pid}; falling back to expert mask for its ROI. Run 03 first.")
            break
    ds = CNNSliceDataset(s["train"])
    print("train slices:", len(ds))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    model = build_cnn(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt = os.path.join(MODEL_DIR, "cnn_best.pt")
    best = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()
        model.eval()
        v = eval_patients(model, s["val"], device)
        acc = accuracy_score(v["true"], v["pred"])
        flag = ""
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_acc": acc}, ckpt)
            flag = "  <- saved"
        print(f"ep {ep:03d} loss {tot / len(ds):.4f} | val patient acc {acc:.3f}{flag}")

    model = load_cnn(ckpt, device)
    te = eval_patients(model, s["test"], device)
    te.to_csv(os.path.join(RESULTS_DIR, "cnn_test_probs.csv"), index=False)
    print(f"\nTEST CNN accuracy={accuracy_score(te['true'], te['pred']):.3f} "
          f"macroF1={f1_score(te['true'], te['pred'], average='macro'):.3f}")
    print(classification_report(te["true"], te["pred"], labels=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()
