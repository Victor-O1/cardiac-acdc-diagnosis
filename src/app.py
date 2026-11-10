"""
STEP 9 - Analytics dashboard + automated report (Gradio).

    python src/app.py            (in Colab it prints a public *.gradio.live link)

Either pick a test patient, or upload an ED + ES .nii/.nii.gz pair with height/weight.
Pipeline: U-Net segmentation -> clinical features -> XGBoost + CNN -> ensemble
          -> SHAP + Grad-CAM -> HTML report (downloadable).
Research prototype, not for clinical use.
"""
import os
import io
import base64
import datetime

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (find_split_dirs, list_patients, load_patient_raw, normalize_volume, load_nii,
                    get_device, CLASSES, CLASS_FULL, MODEL_DIR, RESULTS_DIR, OUT_DIR)
from seg_utils import load_seg_model, predict_volume
from features import compute_features, FEATURE_COLS
from xai_utils import load_xgb, shap_values
from cnn_utils import load_cnn, patient_probs, GradCAM, heart_roi, heart_slices, make_input, to_tensor

DEVICE = get_device()
SEG_ARCH = os.environ.get("SEG_ARCH", "unet")
SEG = load_seg_model(os.path.join(MODEL_DIR, f"{SEG_ARCH}_best.pt"), DEVICE)
XGB = load_xgb(os.path.join(MODEL_DIR, "xgb.json"))
CNN = load_cnn(os.path.join(MODEL_DIR, "cnn_best.pt"), DEVICE)
CAM = GradCAM(CNN)
_, TEST_DIR = find_split_dirs()
TEST_PIDS = list_patients(TEST_DIR)


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def seg_figure(ed, es, m_ed, m_es):
    zs = heart_slices(m_ed) or [ed.shape[2] // 2]
    z = zs[len(zs) // 2]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, img, m, t in ((ax[0], ed, None, "ED"), (ax[1], ed, m_ed, "ED segmentation"),
                         (ax[2], es, None, "ES"), (ax[3], es, m_es, "ES segmentation")):
        a.imshow(img[:, :, z].T, cmap="gray", origin="lower")
        if m is not None:
            a.imshow(np.ma.masked_where(m[:, :, z].T == 0, m[:, :, z].T), cmap="jet",
                     vmin=0, vmax=3, alpha=0.45, origin="lower")
        a.set_title(f"{t} (slice {z})", fontsize=9)
        a.axis("off")
    fig.text(0.5, 0.01, "blue-green: RV   yellow-green: myocardium   red: LV", ha="center", fontsize=8)
    return fig


def cam_figure(ed, es, m_ed, spacing):
    roi = heart_roi(m_ed, spacing)
    zs = heart_slices(m_ed) or [ed.shape[2] // 2]
    z = zs[len(zs) // 2]
    x = make_input(ed, es, z, roi)
    heat, cls = CAM(to_tensor(x).to(DEVICE))
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(x[..., 0].T, cmap="gray", origin="lower")
    ax[0].set_title("Heart ROI (ED)", fontsize=9)
    ax[1].imshow(x[..., 0].T, cmap="gray", origin="lower")
    ax[1].imshow(heat.T, cmap="jet", alpha=0.4, origin="lower")
    ax[1].set_title(f"Grad-CAM for {CLASSES[cls]}", fontsize=9)
    for a in ax:
        a.axis("off")
    return fig


def shap_figure(feat_row, cls):
    sv = shap_values(XGB, feat_row[None], FEATURE_COLS)[0, :, cls]
    order = np.argsort(np.abs(sv))[-10:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(np.array(FEATURE_COLS)[order], sv[order],
            color=["#b23a48" if v > 0 else "#3b6e8f" for v in sv[order]])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel(f"SHAP value toward {CLASSES[cls]} (red pushes toward, blue away)")
    ax.set_title("Top 10 features for this patient", fontsize=10)
    return fig


def build_report(pid, true_group, f, p_cnn, p_xgb, p_ens, imgs):
    cls = int(p_ens.argmax())
    rows = "".join(f"<tr><td>{c}</td><td>{p_cnn[i]*100:.1f}%</td><td>{p_xgb[i]*100:.1f}%</td>"
                   f"<td><b>{p_ens[i]*100:.1f}%</b></td></tr>" for i, c in enumerate(CLASSES))
    meas = [("LV end-diastolic volume", f"{f['LV_EDV']:.1f} ml"),
            ("LV end-systolic volume", f"{f['LV_ESV']:.1f} ml"),
            ("LV ejection fraction", f"{f['LV_EF']:.1f} %"),
            ("RV end-diastolic volume", f"{f['RV_EDV']:.1f} ml"),
            ("RV end-systolic volume", f"{f['RV_ESV']:.1f} ml"),
            ("RV ejection fraction", f"{f['RV_EF']:.1f} %"),
            ("Myocardial mass (ED)", f"{f['MYO_mass_ED']:.1f} g"),
            ("LV EDV / BSA", f"{f['LV_EDV_idx']:.1f} ml/m²"),
            ("Myocardial mass / BSA", f"{f['MYO_mass_idx']:.1f} g/m²"),
            ("Max wall thickness (approx.)", f"{f['MYO_thick_max_ED']:.1f} mm"),
            ("Body surface area", f"{f['BSA']:.2f} m²")]
    mrows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in meas)
    truth = f"<p class='muted'>Reference diagnosis in dataset: {true_group}</p>" if true_group else ""
    img = lambda k: f"<img src='data:image/png;base64,{imgs[k]}'/>"
    return f"""
<div class="rep">
<style>
.rep{{font-family:Georgia,'Times New Roman',serif;color:#1f2a30;max-width:980px;line-height:1.5}}
.rep h1{{font-size:1.6rem;margin:0 0 .2rem;border-bottom:3px solid #8f2d3c;padding-bottom:.3rem}}
.rep h2{{font-size:1.1rem;margin:1.4rem 0 .4rem;color:#8f2d3c}}
.rep .dx{{font-size:1.35rem;margin:.6rem 0}}
.rep table{{border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:.9rem}}
.rep td,.rep th{{border-bottom:1px solid #d8dde0;padding:.3rem .8rem;text-align:left}}
.rep img{{max-width:100%;margin:.3rem 0}}
.rep .muted{{color:#5d6a70;font-size:.85rem}}
</style>
<h1>Cardiac MRI analysis report</h1>
<p class="muted">Case {pid} · generated {datetime.datetime.now():%Y-%m-%d %H:%M} · research prototype, not for clinical use</p>
<p class="dx">Predicted: <b>{CLASS_FULL[CLASSES[cls]]} ({CLASSES[cls]})</b> — ensemble confidence {p_ens[cls]*100:.1f}%</p>
{truth}
<h2>Cardiac measurements (from automatic segmentation)</h2>
<table>{mrows}</table>
<h2>Segmentation</h2>{img('seg')}
<h2>Model results</h2>
<table><tr><th>Class</th><th>CNN (image)</th><th>XGBoost (features)</th><th>Ensemble</th></tr>{rows}</table>
<h2>Explainability</h2>{img('cam')}{img('shap')}
</div>"""


def analyze(test_pid, ed_file, es_file, height, weight):
    fpath = lambda f: None if f is None else getattr(f, "name", f)
    if fpath(ed_file) and fpath(es_file):
        if not height or not weight:
            raise gr.Error("Enter height (cm) and weight (kg) for uploaded scans.")
        ed_raw, sp = load_nii(fpath(ed_file))
        es_raw, _ = load_nii(fpath(es_file))
        pid, true_group, height, weight = "uploaded case", None, float(height), float(weight)
    elif test_pid:
        p = load_patient_raw(TEST_DIR, test_pid)
        ed_raw, es_raw, sp = p["ED_img"], p["ES_img"], p["spacing"]
        pid, true_group, height, weight = test_pid, p["group"], p["height"], p["weight"]
    else:
        raise gr.Error("Pick a test patient or upload both ED and ES volumes.")

    ed, es = normalize_volume(ed_raw), normalize_volume(es_raw)
    m_ed, m_es = predict_volume(SEG, ed, DEVICE), predict_volume(SEG, es, DEVICE)
    f = compute_features(m_ed, m_es, sp, height, weight)
    x = np.array([f[c] for c in FEATURE_COLS], dtype=np.float32)
    p_xgb = XGB.predict_proba(x[None])[0]
    p_cnn = patient_probs(CNN, ed, es, m_ed, sp, DEVICE)
    p_ens = (p_xgb + p_cnn) / 2
    cls = int(p_ens.argmax())

    figs = {"seg": seg_figure(ed, es, m_ed, m_es), "cam": cam_figure(ed, es, m_ed, sp),
            "shap": shap_figure(x, cls)}
    imgs = {k: fig_b64(v) for k, v in figs.items()}
    html = build_report(pid, true_group, f, p_cnn, p_xgb, p_ens, imgs)
    out = os.path.join(OUT_DIR, f"report_{pid.replace(' ', '_')}.html")
    with open(out, "w") as fh:
        fh.write(f"<html><head><meta charset='utf-8'></head><body>{html}</body></html>")
    return html, out


def overall_metrics_md():
    p = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(p):
        return "Run 07_compare_models.py to show test-set metrics here."
    return "**Test-set performance**\n\n" + pd.read_csv(p).to_markdown(index=False)


with gr.Blocks(title="Cardiac MRI analysis") as demo:
    gr.Markdown("## Cardiac MRI analysis and diagnosis\nSegmentation, clinical measurements, "
                "two classifiers, explanations and a report.")
    with gr.Row():
        with gr.Column(scale=1):
            pid_in = gr.Dropdown(TEST_PIDS, label="Test patient", value=TEST_PIDS[0] if TEST_PIDS else None)
            gr.Markdown("or upload your own case")
            ed_in = gr.File(label="End-diastole volume (.nii / .nii.gz)")
            es_in = gr.File(label="End-systole volume (.nii / .nii.gz)")
            h_in = gr.Number(label="Height (cm)")
            w_in = gr.Number(label="Weight (kg)")
            btn = gr.Button("Analyze scan", variant="primary")
            gr.Markdown(overall_metrics_md())
        with gr.Column(scale=3):
            report = gr.HTML()
            dl = gr.File(label="Download report (HTML)")
    btn.click(analyze, [pid_in, ed_in, es_in, h_in, w_in], [report, dl])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
