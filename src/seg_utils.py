"""U-Net / U-Net++ building, whole-volume inference, post-processing and metrics."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import segmentation_models_pytorch as smp

from common import IMG_SIZE

ARCHS = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}


def build_seg_model(arch="unet", encoder="resnet34", pretrained=True):
    return ARCHS[arch](encoder_name=encoder,
                       encoder_weights="imagenet" if pretrained else None,
                       in_channels=1, classes=4)


def load_seg_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_seg_model(ck["arch"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def keep_largest_cc(mask):
    """Keep the largest 3D connected component of each structure (removes stray blobs)."""
    out = np.zeros_like(mask)
    for lab in (1, 2, 3):
        comp, n = ndimage.label(mask == lab)
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(comp), comp, index=range(1, n + 1))
        out[comp == (int(np.argmax(sizes)) + 1)] = lab
    return out


@torch.no_grad()
def predict_volume(model, vol, device, batch=16, postprocess=True):
    """vol: normalized (H, W, Z) -> predicted label volume (H, W, Z) at ORIGINAL resolution."""
    H, W, Z = vol.shape
    slices = np.stack([cv2.resize(vol[:, :, z], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                       for z in range(Z)]).astype(np.float32)
    preds = []
    for i in range(0, Z, batch):
        x = torch.from_numpy(slices[i:i + batch]).unsqueeze(1).to(device)
        logits = model(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    pred = np.concatenate(preds).transpose(1, 2, 0)   # Z,H,W -> H,W,Z
    return keep_largest_cc(pred) if postprocess else pred


# --------------------------------------------------------------------------- metrics
def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / s) if s > 0 else 1.0


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u > 0 else 1.0


def hausdorff(pred, gt, spacing, percentile=100):
    """Symmetric (percentile) Hausdorff distance between surfaces, in mm."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    ps = pred ^ ndimage.binary_erosion(pred)
    gs = gt ^ ndimage.binary_erosion(gt)
    dt_g = ndimage.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~ps, sampling=spacing)
    return float(max(np.percentile(dt_g[ps], percentile), np.percentile(dt_p[gs], percentile)))
