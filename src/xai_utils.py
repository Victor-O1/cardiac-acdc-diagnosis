"""SHAP values via XGBoost's built-in exact TreeSHAP (pred_contribs=True).
Avoids version clashes between the `shap` package and new XGBoost releases."""
import numpy as np
import xgboost as xgb

from common import CLASSES


def load_xgb(path):
    m = xgb.XGBClassifier()
    m.load_model(path)
    return m


def shap_values(model, X, feature_names):
    """Returns array (n_samples, n_features, n_classes); bias column dropped."""
    dm = xgb.DMatrix(np.asarray(X, dtype=np.float32), feature_names=list(feature_names))
    c = model.get_booster().predict(dm, pred_contribs=True)
    c = np.asarray(c)
    n_f, n_c = len(feature_names), len(CLASSES)
    if c.ndim == 3 and c.shape[1] == n_c:           # (n, classes, features+1)
        c = c[:, :, :n_f].transpose(0, 2, 1)
    elif c.ndim == 3 and c.shape[2] == n_c:         # (n, features+1, classes)
        c = c[:, :n_f, :]
    else:
        raise ValueError(f"Unexpected pred_contribs shape {c.shape}")
    return c


def fig_to_array(fig):
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return arr
