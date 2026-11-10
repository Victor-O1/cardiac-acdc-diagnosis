"""CNN classifier helpers: heart-ROI crop, 3-channel input, ResNet18, Grad-CAM."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from common import CLASSES, IMG_SIZE, CROP_MM

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def heart_roi(mask_ed, spacing):
    """Centre of the heart (from a mask) and half-size of a CROP_MM box, in pixels per axis."""
    ys, xs, _ = np.where(mask_ed > 0)
    if len(ys) == 0:                                   # fallback: image centre
        cy, cx = mask_ed.shape[0] / 2, mask_ed.shape[1] / 2
    else:
        cy, cx = ys.mean(), xs.mean()
    return cy, cx, CROP_MM / spacing[0] / 2, CROP_MM / spacing[1] / 2


def crop2d(img, roi):
    cy, cx, hy, hx = roi
    y0, y1 = int(round(cy - hy)), int(round(cy + hy))
    x0, x1 = int(round(cx - hx)), int(round(cx + hx))
    pad = max(0, -y0, -x0, y1 - img.shape[0], x1 - img.shape[1])
    if pad:
        img = np.pad(img, pad)
        y0, y1, x0, x1 = y0 + pad, y1 + pad, x0 + pad, x1 + pad
    return cv2.resize(img[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def heart_slices(mask_ed):
    return [z for z in range(mask_ed.shape[2]) if (mask_ed[:, :, z] > 0).any()]


def make_input(ed_img, es_img, z, roi):
    """Channels: ED slice, ES slice (same position), |ED-ES| (encodes contraction)."""
    ed = crop2d(ed_img[:, :, z], roi)
    es = crop2d(es_img[:, :, z], roi)
    return np.stack([ed, es, np.abs(ed - es)], axis=-1).astype(np.float32)   # H,W,3


def to_tensor(x):
    """x: (H,W,3) or (N,H,W,3) in [0,1] -> ImageNet-normalized NCHW tensor."""
    x = (x - IMNET_MEAN) / IMNET_STD
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.permute(2, 0, 1).unsqueeze(0) if t.ndim == 3 else t.permute(0, 3, 1, 2)


def build_cnn(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def load_cnn(path, device):
    m = build_cnn(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def patient_probs(model, ed_img, es_img, mask_ed, spacing, device):
    """Average softmax over all slices that contain heart -> patient-level probabilities."""
    roi = heart_roi(mask_ed, spacing)
    zs = heart_slices(mask_ed) or list(range(ed_img.shape[2]))
    x = to_tensor(np.stack([make_input(ed_img, es_img, z, roi) for z in zs])).to(device)
    return torch.softmax(model(x), 1).mean(0).cpu().numpy()


class GradCAM:
    """Grad-CAM on the last conv block (layer4) of ResNet18."""

    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        model.layer4.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, cls=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        cls = int(logits.argmax(1)) if cls is None else cls
        self.model.zero_grad()
        logits[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), cls
