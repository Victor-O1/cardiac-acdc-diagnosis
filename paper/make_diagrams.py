"""Draws the cohort flow diagram and the pipeline diagram used in the paper and slides.
Numbers come from the ACDC design (150 patients, 5 groups). Step 0 re-checks them on your copy."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#eaf1f6", ec="#3b6e8f", fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1))


def flow():
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .5, .92, .8, .11, "ACDC dataset\n150 patients, 5 diagnostic groups\n(30 per group)")
    box(ax, .5, .74, .8, .09, "Exclusions: 0 patients\n(all have ED/ES images and masks)")
    box(ax, .26, .56, .44, .11, "Training set\n100 patients\n(patient001-100)")
    box(ax, .76, .56, .44, .11, "Test set\n50 patients\n(patient101-150)", fc="#f6ebea", ec="#8f3b4a")
    box(ax, .13, .34, .24, .12, "Train\n80 patients\n(16 per group)", fs=7)
    box(ax, .39, .34, .24, .12, "Validation\n20 patients\n(4 per group)", fs=7)
    box(ax, .76, .34, .44, .12, "Held out\nused once for\nfinal evaluation", fc="#f6ebea", ec="#8f3b4a", fs=7)
    box(ax, .5, .10, .9, .13, "Patient-level bootstrap, 2000 resamples\nAUROC, AUPRC, sensitivity, specificity,\nF1, calibration with 95% CI", fc="#eef6ea", ec="#4a8f3b")
    arrow(ax, .5, .865, .5, .785); arrow(ax, .5, .695, .26, .615); arrow(ax, .5, .695, .76, .615)
    arrow(ax, .2, .50, .13, .40); arrow(ax, .32, .50, .39, .40); arrow(ax, .76, .50, .76, .40)
    arrow(ax, .26, .28, .4, .165); arrow(ax, .76, .28, .62, .165)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cohort_flow.png"), dpi=220); plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .07, .5, .12, .5, "Cine MRI\nED and ES\nvolumes", fs=7)
    box(ax, .24, .5, .13, .5, "U-Net\n(ResNet34)\nRV, MYO, LV\nmasks", fs=7)
    box(ax, .44, .78, .17, .34, "18 clinical\nfeatures\n(volumes, EF,\nmass, thickness)", fs=7)
    box(ax, .44, .24, .17, .34, "Heart crop\n[ED, ES, |ED-ES|]", fs=7)
    box(ax, .65, .78, .13, .34, "XGBoost", fs=7.5)
    box(ax, .65, .24, .13, .34, "ResNet18", fs=7.5)
    box(ax, .82, .5, .12, .3, "Mean of\nprobabilities", fs=7)
    box(ax, .955, .5, .08, .5, "5-class\ndiagnosis", fc="#eef6ea", ec="#4a8f3b", fs=6.5)
    for a in [(.13, .5, .175, .5), (.305, .55, .355, .72), (.305, .45, .355, .30), (.525, .78, .585, .78),
              (.525, .24, .585, .24), (.715, .72, .76, .56), (.715, .30, .76, .44), (.88, .5, .915, .5)]:
        arrow(ax, *a)
    ax.text(.65, .97, "SHAP", fontsize=6.5, ha="center", color="#8f3b4a")
    ax.text(.65, .03, "Grad-CAM", fontsize=6.5, ha="center", color="#8f3b4a")
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=220); plt.close(fig)


if __name__ == "__main__":
    flow(); pipeline(); print("saved to", OUT)
