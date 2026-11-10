"""
STEP 2 - Train the segmentation model (U-Net or U-Net++) on 2D slices.

    python src/02_train_unet.py --arch unet   --epochs 40
    python src/02_train_unet.py --arch unetpp --epochs 40   (optional comparison)

Validation = mean 3D Dice (RV, MYO, LV) over the 20 val patients, ED + ES.
Best checkpoint -> outputs/models/{arch}_best.pt
"""
import os
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from common import (load_cached, load_splits, random_affine, get_device, set_seed,
                    IMG_SIZE, MODEL_DIR, RESULTS_DIR)
from seg_utils import build_seg_model, predict_volume, dice


class SliceDataset(Dataset):
    def __init__(self, pids, augment, seed=0):
        self.imgs, self.masks = [], []
        for pid in pids:
            d = load_cached(pid)
            for ph in ("ED", "ES"):
                img, gt = d[f"{ph}_img"], d[f"{ph}_gt"]
                for z in range(img.shape[2]):
                    self.imgs.append(cv2.resize(img[:, :, z], (IMG_SIZE, IMG_SIZE),
                                                interpolation=cv2.INTER_LINEAR).astype(np.float32))
                    self.masks.append(cv2.resize(gt[:, :, z], (IMG_SIZE, IMG_SIZE),
                                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, m = self.imgs[i], self.masks[i]
        if self.augment:
            img, m = random_affine(img, m, self.rng)
            img = np.clip(img, 0, 1) ** self.rng.uniform(0.8, 1.2)       # gamma jitter
        return torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(m.astype(np.int64))


def val_dice(model, pids, device):
    model.eval()
    scores = []
    for pid in pids:
        d = load_cached(pid)
        for ph in ("ED", "ES"):
            pred = predict_volume(model, d[f"{ph}_img"], device)
            scores.append([dice(pred == c, d[f"{ph}_gt"] == c) for c in (1, 2, 3)])
    return np.mean(scores, axis=0)          # RV, MYO, LV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    splits = load_splits()
    train_ds = SliceDataset(splits["train"], augment=True)
    print(f"train slices: {len(train_ds)} | val patients: {len(splits['val'])}")
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

    model = build_seg_model(args.arch, args.encoder, pretrained=not args.no_pretrained).to(device)
    ce = nn.CrossEntropyLoss()
    dl_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, hist = -1, []
    ckpt = os.path.join(MODEL_DIR, f"{args.arch}_best.pt")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y) + dl_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        rv, myo, lv = val_dice(model, splits["val"], device)
        mean = (rv + myo + lv) / 3
        hist.append({"epoch": ep, "loss": tot / len(train_ds), "dice_RV": rv, "dice_MYO": myo,
                     "dice_LV": lv, "dice_mean": mean})
        flag = ""
        if mean > best:
            best = mean
            torch.save({"arch": args.arch, "encoder": args.encoder,
                        "state_dict": model.state_dict(), "epoch": ep, "val_dice": mean}, ckpt)
            flag = "  <- saved"
        print(f"ep {ep:03d} loss {tot / len(train_ds):.4f} | val Dice RV {rv:.3f} MYO {myo:.3f} "
              f"LV {lv:.3f} mean {mean:.3f}{flag}")
        pd.DataFrame(hist).to_csv(os.path.join(RESULTS_DIR, f"{args.arch}_history.csv"), index=False)
    print("best val mean Dice:", round(best, 4), "->", ckpt)


if __name__ == "__main__":
    main()
