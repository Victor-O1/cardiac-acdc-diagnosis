"""Clinical features computed from ED/ES label masks (0 bg, 1 RV, 2 MYO, 3 LV)."""
import numpy as np
from scipy import ndimage

from common import MYO_DENSITY

FEATURE_COLS = [
    "LV_EDV", "LV_ESV", "LV_EF", "RV_EDV", "RV_ESV", "RV_EF",
    "MYO_mass_ED", "MYO_mass_ES",
    "LV_EDV_idx", "RV_EDV_idx", "MYO_mass_idx",
    "RV_LV_EDV_ratio", "MYO_LV_ratio_ED",
    "MYO_thick_max_ED", "MYO_thick_mean_ED",
    "Height", "Weight", "BSA",
]


def myo_thickness(myo, spacing):
    """Approximate wall thickness per slice = 2 x max in-plane distance-to-boundary inside MYO (mm)."""
    th = []
    for z in range(myo.shape[2]):
        s = myo[:, :, z]
        if s.any():
            th.append(2.0 * ndimage.distance_transform_edt(s, sampling=spacing[:2]).max())
    return (float(max(th)), float(np.mean(th))) if th else (0.0, 0.0)


def compute_features(ed, es, spacing, height, weight):
    vox_ml = float(np.prod(spacing)) / 1000.0          # mm^3 -> ml
    vol = lambda m, c: float((m == c).sum()) * vox_ml
    lv_edv, lv_esv = vol(ed, 3), vol(es, 3)
    rv_edv, rv_esv = vol(ed, 1), vol(es, 1)
    myo_ed, myo_es = vol(ed, 2), vol(es, 2)
    bsa = float(np.sqrt(height * weight / 3600.0))     # Mosteller, m^2
    ef = lambda edv, esv: 100.0 * (edv - esv) / edv if edv > 0 else 0.0
    tmax, tmean = myo_thickness(ed == 2, spacing)
    return {
        "LV_EDV": lv_edv, "LV_ESV": lv_esv, "LV_EF": ef(lv_edv, lv_esv),
        "RV_EDV": rv_edv, "RV_ESV": rv_esv, "RV_EF": ef(rv_edv, rv_esv),
        "MYO_mass_ED": myo_ed * MYO_DENSITY, "MYO_mass_ES": myo_es * MYO_DENSITY,
        "LV_EDV_idx": lv_edv / bsa, "RV_EDV_idx": rv_edv / bsa,
        "MYO_mass_idx": myo_ed * MYO_DENSITY / bsa,
        "RV_LV_EDV_ratio": rv_edv / lv_edv if lv_edv > 0 else 0.0,
        "MYO_LV_ratio_ED": myo_ed / lv_edv if lv_edv > 0 else 0.0,
        "MYO_thick_max_ED": tmax, "MYO_thick_mean_ED": tmean,
        "Height": height, "Weight": weight, "BSA": bsa,
    }
