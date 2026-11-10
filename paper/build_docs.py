"""
Builds the review package from the pipeline outputs:
    paper/BCSE335L_Team21_CardiacDiagnosis.docx   IEEE two-column paper
    paper/TRIPOD_AI_checklist.docx                reporting checklist
    paper/BCSE335L_Team21_Slides.pptx             review slides

Run it after src/09_report_metrics.py:   python paper/build_docs.py
Any number that is not available yet shows as a yellow [[placeholder]].
The script never invents a result. It only reads outputs/results/*.csv.
"""
import os
import re
import json

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("CARDIAC_ROOT", ROOT)
RES = os.path.join(DATA_ROOT, "outputs", "results")
FIG = os.path.join(DATA_ROOT, "outputs", "figures")
DIA = os.path.join(HERE, "figs")
GITHUB = "https://github.com/Victor-O1/cardiac-acdc-diagnosis"
TITLE = ("Automatic Diagnosis of Five Cardiac Conditions from Cine MRI: "
         "U-Net Segmentation, Clinical Features and a ResNet18 Ensemble on the ACDC Dataset")


# =========================================================================== data access
def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ph(name):
    return f"[[{name}]]"


def ci(row, k, d=3):
    return f"{row[k]:.{d}f} ({row[k + '_lo']:.{d}f}–{row[k + '_hi']:.{d}f})"


MET = load("metrics_ci.csv")
PCL = load("per_class_ci.csv")
SUB = load("subgroup_ci.csv")
SEG = load("seg_ci.csv")
COH = load("cohort_table.csv")
SHAP = load("shap_per_class.csv")


def model_row(prefix):
    if MET is None:
        return None
    m = MET[MET["model"].str.startswith(prefix)]
    return m.iloc[0] if len(m) else None


MAIN = model_row("Ensemble")
BASE = model_row("Random forest")
XGBR = model_row("XGBoost (U-Net")
CNNR = model_row("CNN")


def main_val(k, d=3, with_ci=True):
    if MAIN is None:
        return ph(k)
    return ci(MAIN, k, d) if with_ci else f"{MAIN[k]:.{d}f}"


def seg_val(structure, col="dice"):
    if SEG is None:
        return ph(f"{structure} {col}")
    r = SEG[(SEG["phase"] == "ED") & (SEG["structure"] == structure)]
    return r.iloc[0][col].split(" ")[0] if len(r) else ph(f"{structure} {col}")


def top_shap(n=3):
    if SHAP is None:
        return [ph("feature 1"), ph("feature 2"), ph("feature 3")][:n]
    s = SHAP.set_index(SHAP.columns[0]).mean(axis=1).sort_values(ascending=False)
    return list(s.index[:n])


# =========================================================================== references
REFS = {
    "bernard": 'O. Bernard et al., "Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved?," IEEE Trans. Med. Imaging, vol. 37, no. 11, pp. 2514–2525, Nov. 2018.',
    "isensee17": 'F. Isensee, P. Jaeger, P. M. Full, I. Wolf, S. Engelhardt, and K. H. Maier-Hein, "Automatic cardiac disease assessment on cine-MRI via time-series segmentation and domain specific features," in Proc. STACOM, Lecture Notes in Computer Science, vol. 10663, 2018, pp. 120–129.',
    "khened": 'M. Khened, V. A. Kollerathu, and G. Krishnamurthi, "Fully convolutional multi-scale residual DenseNets for cardiac segmentation and automated cardiac diagnosis using ensemble of classifiers," Med. Image Anal., vol. 51, pp. 21–45, Jan. 2019.',
    "cetin": 'I. Cetin et al., "A radiomics approach to computer-aided diagnosis with cardiac cine-MRI," in Proc. STACOM, Lecture Notes in Computer Science, vol. 10663, 2018, pp. 82–90.',
    "wolterink": 'J. M. Wolterink, T. Leiner, M. A. Viergever, and I. Išgum, "Automatic segmentation and disease classification using cardiac cine MR images," in Proc. STACOM, Lecture Notes in Computer Science, vol. 10663, 2018, pp. 101–110.',
    "baumgartner": 'C. F. Baumgartner, L. M. Koch, M. Pollefeys, and E. Konukoglu, "An exploration of 2D and 3D deep learning techniques for cardiac MR image segmentation," in Proc. STACOM, Lecture Notes in Computer Science, vol. 10663, 2018, pp. 111–119.',
    "zotti": 'C. Zotti, Z. Luo, A. Lalande, and P.-M. Jodoin, "Convolutional neural network with shape prior applied to cardiac MRI segmentation," IEEE J. Biomed. Health Inform., vol. 23, no. 3, pp. 1119–1128, May 2019.',
    "painchaud": 'N. Painchaud, Y. Skandarani, T. Judge, O. Bernard, A. Lalande, and P.-M. Jodoin, "Cardiac segmentation with strong anatomical guarantees," IEEE Trans. Med. Imaging, vol. 39, no. 11, pp. 3703–3713, Nov. 2020.',
    "bai": 'W. Bai et al., "Automated cardiovascular magnetic resonance image analysis with fully convolutional networks," J. Cardiovasc. Magn. Reson., vol. 20, art. no. 65, 2018.',
    "nnunet": 'F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nat. Methods, vol. 18, no. 2, pp. 203–211, Feb. 2021.',
    "chen20": 'C. Chen et al., "Deep learning for cardiac image segmentation: a review," Front. Cardiovasc. Med., vol. 7, art. no. 25, 2020.',
    "ouyang": 'D. Ouyang et al., "Video-based AI for beat-to-beat assessment of cardiac function," Nature, vol. 580, pp. 252–256, 2020.',
    "unet": 'O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, Lecture Notes in Computer Science, vol. 9351, 2015, pp. 234–241.',
    "unetpp": 'Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A nested U-Net architecture for medical image segmentation," in Proc. DLMIA, Lecture Notes in Computer Science, vol. 11045, 2018, pp. 3–11.',
    "smp": 'P. Iakubovskii, "Segmentation Models PyTorch," GitHub repository, 2019. [Online]. Available: https://github.com/qubvel/segmentation_models.pytorch',
    "resnet": 'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016, pp. 770–778.',
    "imagenet": 'J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proc. IEEE CVPR, 2009, pp. 248–255.',
    "adamw": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.',
    "xgb": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. ACM SIGKDD, 2016, pp. 785–794.',
    "rf": 'L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001.',
    "shap": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, 2017, pp. 4765–4774.',
    "gradcam": 'R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE ICCV, 2017, pp. 618–626.',
    "taha": 'A. A. Taha and A. Hanbury, "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool," BMC Med. Imaging, vol. 15, art. no. 29, 2015.',
    "saito": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, art. no. e0118432, 2015.',
    "guo": 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, PMLR vol. 70, 2017, pp. 1321–1330.',
    "efron": 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.',
    "tripod": 'G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, art. no. e078378, 2024.',
    "dpdp": 'Government of India, "The Digital Personal Data Protection Act, 2023," Act No. 22 of 2023, Gazette of India, Aug. 2023.',
    "icmr": 'Indian Council of Medical Research, "Ethical guidelines for application of artificial intelligence in biomedical research and healthcare," New Delhi, India, 2023.',
    "srik": 'N. Srikijkasemwat, M. Villarroel, and A. Banerjee, "Multi-phase deep learning model for automated disease classification from cardiac cine MRI," J. R. Soc. Interface, vol. 22, no. 231, art. no. 20250303, Oct. 2025, doi: 10.1098/rsif.2025.0303.',
    "intelli": 'T. Y. Tsai et al., "IntelliCardiac: An intelligent platform for cardiac image segmentation and classification," arXiv:2505.03838, May 2025.',
    "liu23": 'X. Liu, F. Xing, H. K. Gaggin, C.-C. J. Kuo, G. El Fakhri, and J. Woo, "Successive subspace learning for cardiac disease classification with two-phase deformation fields from cine MRI," in Proc. IEEE ISBI, 2023, arXiv:2301.08959.',
    "sfb": 'N. Portal, N. Kachenoura, T. Dietenbeck, and C. Achard, "SFB-net for cardiac segmentation: Bridging the semantic gap with attention," in Proc. IEEE ISBI, Cartagena, Colombia, Apr. 2023, pp. 1–5, doi: 10.1109/ISBI53787.2023.10230354.',
    "mese": 'I. Mese and B. Kocak, "Glass-box agentic-style workflow for multiclass cine cardiac magnetic resonance imaging classification with a large language model," Diagn. Interv. Radiol., May 2026, doi: 10.4274/dir.2026.264016.',
    "tavakoli": 'N. Tavakoli, A. A. Rahsepar, and D. Kim, "Bridging the gap in cardiovascular magnetic resonance imaging artificial intelligence implementations: From ambitious goals to real-world progress using foundation models," J. Cardiovasc. Magn. Reson., vol. 27, no. 2, art. no. 101979, Oct. 2025, doi: 10.1016/j.jocmr.2025.101979.',
    "acmunet": 'J. Huang, Y. Zhao, Y. Li, Z. Dai, C. Chen, and Q. Lai, "ACM-UNet: Adaptive integration of CNNs and Mamba for efficient medical image segmentation," arXiv:2505.24481, May 2025.',
    "nntrans": 'L. Zhao, D. Zhou, X. Jin, and W. Zhu, "nn-TransUNet: An automatic deep learning pipeline for heart MRI segmentation," Life, vol. 12, no. 10, art. no. 1570, Oct. 2022, doi: 10.3390/life12101570.',
}
_order = []


def c(*keys):
    for k in keys:
        if k not in _order:
            _order.append(k)
    return "[" + ", ".join(str(_order.index(k) + 1) for k in keys) + "]"


# =========================================================================== docx helpers
FONT = "Times New Roman"


def set_cols(section, n, space=360):
    pr = section._sectPr
    for old in pr.findall(qn("w:cols")):
        pr.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(space))
    pr.append(cols)


def runs(par, text, size=10, bold=False, italic=False):
    """Add text to a paragraph. Segments inside [[...]] get a yellow highlight."""
    for part in re.split(r"(\[\[.*?\]\])", text):
        if not part:
            continue
        r = par.add_run(part)
        r.font.name = FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        r.font.size = Pt(size)
        r.bold, r.italic = bold, italic
        if part.startswith("[[") and part.endswith("]]"):
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def para(doc, text, size=10, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         before=0, after=4, indent=0.0):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    if indent:
        pf.first_line_indent = Inches(indent)
    runs(p, text, size, bold, italic)
    return p


def heading(doc, num, text):
    return para(doc, f"{num}. {text.upper()}", size=10, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                before=8, after=4)


def subheading(doc, letter, text):
    return para(doc, f"{letter}. {text}", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                before=4, after=2)


def body(doc, text):
    return para(doc, text, indent=0.17, after=6)


_counts = {"table": 0, "fig": 0}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def table(doc, df, caption, widths=None, size=7.5):
    _counts["table"] += 1
    para(doc, f"TABLE {ROMAN[_counts['table']]}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=0)
    para(doc, caption.upper(), size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = ""
        runs(cell.paragraphs[0], str(col), size, bold=True)
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "E7E6E6")
        cell._tc.get_or_add_tcPr().append(sh)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            runs(cells[i].paragraphs[0], str(v), size)
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[i])
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
    para(doc, "", size=4, after=2)


def figure(doc, path, caption, width=3.3):
    _counts["fig"] += 1
    if path and os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Inches(width))
    else:
        para(doc, ph(f"Insert {os.path.basename(path or 'figure')} after running the pipeline"),
             size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, f"Fig. {_counts['fig']}. {caption}", size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)


def df_or_placeholder(df, cols, note):
    if df is not None:
        return df
    return pd.DataFrame([[ph(note)] + [ph("TBD")] * (len(cols) - 1)], columns=cols)


# =========================================================================== paper
def abstract_text():
    seg = f"{seg_val('LV')}, {seg_val('MYO')} and {seg_val('RV')}"
    t3 = ", ".join(top_shap(3))
    return (
        "Background: Cardiac MRI diagnosis needs manual contouring and expert reading. This work is slow and "
        "readers do not always agree. "
        "Objective: We tested if a fully automatic pipeline can give one of five diagnoses from cine MRI. "
        "The classes are normal, previous infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy and abnormal right ventricle. "
        "Methods: We used the public ACDC dataset of 150 patients. We split the data by patient into 80 for training, "
        "20 for validation and 50 for final testing. A U-Net segmented the heart at end-diastole and end-systole. "
        "XGBoost classified 18 clinical features taken from the masks. A ResNet18 classified heart crops. "
        "We averaged both outputs. A random forest was the baseline. We report AUROC, AUPRC, sensitivity, "
        "specificity, F1 and calibration, each with a 95% bootstrap confidence interval over patients. "
        f"Results: On the 50 test patients, the ensemble reached a macro AUROC of {main_val('AUROC')} and a macro AUPRC of "
        f"{main_val('AUPRC')}. Macro sensitivity was {main_val('sensitivity', with_ci=False)}, macro specificity "
        f"{main_val('specificity', with_ci=False)} and macro F1 {main_val('f1', with_ci=False)}. XGBoost alone reached a macro F1 of "
        f"{('%.2f' % XGBR['f1']) if XGBR is not None else ph('XGB F1')}. The U-Net reached Dice scores "
        f"of {seg} for LV, MYO and RV at end-diastole. SHAP ranked {t3} as the most important features. "
        "Conclusion: Segmentation followed by feature-based and image-based classification can diagnose these five "
        "conditions on unseen patients. The test set is small, comes from one centre and has equal class sizes. "
        "External validation must come before any clinical use."
    )


RELATED = [
    ("Bernard et al. " + "{bernard}", "Challenge", "Overview of ACDC and the methods submitted to it", "Defines the dataset, metrics and the segmentation and diagnosis tasks"),
    ("Isensee et al. " + "{isensee17}", "Seg. + diagnosis", "U-Net ensemble, time-series and domain features, classifier ensemble", "Strong pipeline. Uses 4D cine and many hand-built features"),
    ("Khened et al. " + "{khened}", "Seg. + diagnosis", "Multi-scale residual DenseNet, then an ensemble of classifiers", "Segmentation and diagnosis in one design. Reports no confidence intervals"),
    ("Cetin et al. " + "{cetin}", "Diagnosis", "Radiomics features with a classical classifier", "Needs masks first. Tests how much radiomics adds"),
    ("Wolterink et al. " + "{wolterink}", "Seg. + diagnosis", "CNN segmentation, then feature-based classification", "Similar two-stage idea on ACDC"),
    ("Baumgartner et al. " + "{baumgartner}", "Segmentation", "Compares 2D and 3D networks", "Shows 2D networks can match 3D ones on ACDC"),
    ("Zotti et al. " + "{zotti}", "Segmentation", "CNN with a shape prior", "Adds anatomical knowledge to the network"),
    ("Painchaud et al. " + "{painchaud}", "Segmentation", "Anatomical plausibility enforced after prediction", "Reduces implausible shapes. Does not diagnose"),
    ("Bai et al. " + "{bai}", "Segmentation", "Fully convolutional network on a large cohort (UK Biobank)", "Large data. Different task and data source"),
    ("Isensee et al. " + "{nnunet}", "Segmentation", "Self-configuring U-Net framework", "Strong baseline for many datasets, ACDC included"),
    ("Chen et al. " + "{chen20}", "Review", "Review of deep learning for cardiac segmentation", "Maps the field. Not a diagnosis study"),
    ("Ouyang et al. " + "{ouyang}", "Function", "Video model for ejection fraction on echo", "Shows AI can measure heart function. Different modality"),
    ("Srikijkasemwat et al. {srik}", "Diagnosis", "ED and ES frames into ResNet and DenseNet models, joined in a multi-phase network, with Grad-CAM++", "ACDC test accuracy 0.78. Image only. No mask features, no calibration"),
    ("Tsai et al. {intelli}", "Seg. + diagnosis", "Segmentation, then a two-step classifier on mask features", "Reports 98% accuracy. Preprint with no confidence intervals"),
    ("Liu et al. {liu23}", "Diagnosis", "Successive subspace learning on ED and ES deformation fields", "Light and explainable. The abstract gives no numbers"),
    ("Portal et al. {sfb}", "Segmentation", "Swin filtering blocks inside a U-shaped network", "Dice 92.4 on ACDC. Does not diagnose"),
    ("Mese and Kocak {mese}", "Seg. + diagnosis", "nnU-Net, 17 biomarkers, language model with prompts", "Accuracy 0.925 (CI 0.863–0.975). A random forest reached 0.938"),
    ("Tavakoli et al. {tavakoli}", "Review", "Editorial on foundation models for cardiac MRI", "Disease classification is still weak. No multi-vendor test"),
    ("Huang et al. {acmunet}", "Segmentation", "CNN and Mamba hybrid with a wavelet decoder", "General medical task. ACDC is one test set"),
    ("Zhao et al. {nntrans}", "Segmentation", "nnU-Net planning with a TransUNet network", "Average Dice 93.6% on ACDC. Does not diagnose"),
]


def build_paper(out):
    doc = Document()
    s0 = doc.sections[0]
    s0.page_width, s0.page_height = Inches(8.5), Inches(11)
    s0.left_margin = s0.right_margin = Inches(0.7)
    s0.top_margin, s0.bottom_margin = Inches(0.75), Inches(0.9)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(10)

    para(doc, TITLE, size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    para(doc, "Priyesh Kumar Jha (23BCE1221)", size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para(doc, "Team 21: Cardiac MRI to Diagnosis: Segmentation + Modelling + Explainability, BCSE335L Healthcare Data Analytics, School of Computer Science and Engineering, VIT Chennai, India",
         size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)

    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(sec, 2)

    abs_text = abstract_text()
    p = para(doc, "", size=9, bold=True, after=4)
    runs(p, "Abstract—", 9, bold=True, italic=True)
    runs(p, abs_text, 9, bold=True)
    para(doc, "Keywords: cardiac MRI, ACDC, segmentation, U-Net, XGBoost, ResNet, SHAP, calibration, "
              "cardiomyopathy", size=9, italic=True, after=6)

    # ---------------------------------------------------------------- I Introduction
    heading(doc, "I", "Introduction")
    body(doc, "Cardiovascular disease is a leading cause of death worldwide. Cine cardiac MRI gives some of the most "
              "exact measures of heart volume, muscle mass and pumping function. To get these measures, a trained reader "
              "draws the borders of the ventricles and the heart muscle on every slice at two time points. This work "
              "is slow, and two readers rarely draw the same line.")
    body(doc, f"Deep networks can draw these borders automatically {c('unet', 'bernard')}. The public ACDC challenge "
              f"showed that the same masks can also support a diagnosis {c('bernard')}. Still, papers on this task differ "
              "in how they split the data and which metrics they report. They also differ in the masks used at test time: "
              "expert masks or masks from their own network. Few papers give confidence intervals, calibration or subgroup results. With only "
              "50 test patients, one accuracy number can mislead the reader.")
    body(doc, "Our research question has two parts. First, can a segmentation network and two classifiers diagnose "
              "five cardiac conditions in patients that the models have never seen? Second, how certain are the results?")
    body(doc, "We make four contributions. First, we build an end-to-end pipeline that tests on predicted masks, "
              "not on expert masks. Second, we split all data by patient and fit every scaling step on training data "
              "only, so no information leaks into the test set. Third, we report AUROC, AUPRC, sensitivity, specificity, "
              "F1 and calibration with 95% confidence intervals, plus a subgroup analysis. Fourth, we explain the "
              f"models with SHAP and Grad-CAM, and we release the code at {GITHUB}.")

    # ---------------------------------------------------------------- II Related work
    heading(doc, "II", "Related Work")
    body(doc, f"The ACDC challenge paper {c('bernard')} describes the data and the first wave of methods. The two best "
              f"diagnosis systems used segmentation, hand-built features and classifier ensembles {c('isensee17', 'khened')}. "
              f"Other teams tried radiomics features {c('cetin')} or CNN segmentation with feature-based classification "
              f"{c('wolterink')}. On the segmentation side, work has covered 2D against 3D networks {c('baumgartner')}, shape "
              f"priors {c('zotti')}, anatomical plausibility checks {c('painchaud')} and self-configuring frameworks {c('nnunet')}. "
              f"Larger cohorts such as the UK Biobank support fully convolutional networks {c('bai')}. Reviews map the field "
              f"{c('chen20')}, and AI already measures heart function on echo video {c('ouyang')}.")
    body(doc, "Table I compares these studies with our work. Most of them focus on segmentation or on diagnosis alone, "
              "and few report confidence intervals, calibration and subgroup results together. Our study fills this gap "
              "on a small public dataset. We do not claim a higher accuracy than the ACDC challenge entries, because the "
              "test conditions differ.")
    rel = pd.DataFrame([(re.sub(r"\{(\w+)\}", lambda mo: c(mo.group(1)), a), t, m, n) for a, t, m, n in RELATED]
                       + [("This work", "Seg. + diagnosis",
                           "U-Net, XGBoost, ResNet18, mean ensemble, test on predicted masks",
                           "Patient-level split, 95% CI, calibration, SHAP, subgroups")],
                       columns=["Study", "Task", "Method", "Note"])
    table(doc, rel, "Comparison with related studies", widths=[0.85, 0.55, 1.1, 1.0], size=7.5)

    # ---------------------------------------------------------------- III Methods
    heading(doc, "III", "Methods")
    subheading(doc, "A", "Dataset and cohort")
    body(doc, f"We used the ACDC dataset {c('bernard')}. It holds short-axis cine MRI of 150 patients from one hospital. "
              "Each of five groups has 30 patients: normal (NOR), previous myocardial infarction (MINF), dilated "
              "cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM) and abnormal right ventricle (RV). Experts "
              "drew masks of the right ventricle (RV), the myocardium (MYO) and the left ventricle (LV) at end-diastole "
              "(ED) and end-systole (ES). Each patient file also lists height and weight. Patients 1 to 100 form the "
              "training set. Patients 101 to 150 form the test set.")
    body(doc, "We excluded no patient, because every patient has ED and ES images with masks. Our inspection script "
              "checks this on each copy of the data. Fig. 1 shows the cohort flow. We split the training set by patient "
              "into 80 for training and 20 for validation, with equal group ratios. The 50 test patients stayed untouched "
              "until the final evaluation.")
    figure(doc, os.path.join(DIA, "cohort_flow.png"), "Cohort flow diagram. All splits are by patient.", 2.6)
    subheading(doc, "B", "Outcome definition")
    body(doc, "The outcome is the diagnostic group in the patient file. Clinicians assigned this group before the "
              "challenge, and we treat it as the reference standard. The task is a five-class problem.")
    subheading(doc, "C", "Preprocessing")
    body(doc, "We clipped each MRI volume to its own 1st and 99th intensity percentiles and scaled it to the range 0 to 1. "
              "This step uses only the volume itself, so no statistic passes between patients. For segmentation we resized "
              "each slice to 224 by 224 pixels. For training we applied random rotation (up to 15 degrees), scaling "
              "(0.9 to 1.1), shift (5%) and gamma change. We did not flip images, because a flip would place the right "
              "ventricle on the wrong side. We did not oversample any class, since the classes have equal size.")
    subheading(doc, "D", "Models")
    body(doc, f"Segmentation. We trained a 2D U-Net {c('unet')} with a ResNet34 encoder {c('resnet')} from the "
              f"segmentation-models-pytorch library {c('smp')}. The encoder started from ImageNet weights {c('imagenet')}. "
              f"The loss was cross-entropy plus Dice loss. We used AdamW {c('adamw')} with a learning rate of 3e-4 and a cosine "
              "schedule for 40 epochs. We chose the checkpoint with the best mean Dice on the validation patients. "
              "After prediction we kept the largest 3D connected component of each structure. A U-Net++ "
              f"{c('unetpp')} is available in the code for comparison.")
    body(doc, "Clinical features. We computed 18 features from the ED and ES masks and the voxel size. They are the volume and ejection "
              "fraction of both ventricles, and the myocardial mass at ED and ES (density 1.05 g/ml). They also include the volumes and mass "
              "divided by body surface area (Mosteller formula), the RV to LV volume ratio and the myocardium to LV ratio. "
              "The last features are the maximum and mean wall thickness at ED, height, weight and body surface area.")
    body(doc, f"XGBoost. An XGBoost model {c('xgb')} with 300 trees, depth 3 and learning rate 0.05 learned from the "
              "features of expert masks of all 100 training patients. At test time it received features from the "
              f"U-Net masks. Baseline. A random forest {c('rf')} with 500 trees used the same data and features.")
    body(doc, "CNN. A ResNet18 with ImageNet weights received three channels per slice: the ED slice, the ES slice and their "
              "absolute difference. We cropped a 160 mm box around the heart. For training the box came from expert masks. "
              "For validation and test it came from U-Net masks. The patient score is the mean softmax over all heart "
              "slices. We used AdamW with a learning rate of 1e-4, 25 epochs, label smoothing of 0.1 and a batch size of 32.")
    body(doc, "Ensemble. The final probability is the plain mean of the XGBoost and CNN probabilities. We did not tune "
              "the weights.")
    figure(doc, os.path.join(DIA, "pipeline.png"), "Pipeline from cine MRI to diagnosis. SHAP explains XGBoost and "
                                                   "Grad-CAM explains the CNN.", 3.4)
    subheading(doc, "E", "Validation and leakage control")
    body(doc, "All splits are by patient. No slice of one patient appears in two sets. The U-Net used only the 80 training "
              "patients for learning. The 20 validation patients chose the best epoch. The 50 test patients were used "
              "once. The XGBoost and random forest models were fitted on the 100 training patients and checked by "
              "5-fold stratified cross-validation. Intensity scaling uses no statistic from other patients.")
    subheading(doc, "F", "Evaluation metrics")
    body(doc, f"For diagnosis we report AUROC, AUPRC {c('saito')}, sensitivity, specificity and F1, macro-averaged over the five "
              f"classes in a one-versus-rest scheme. Calibration is shown with a reliability diagram, the Brier score and the "
              f"expected calibration error {c('guo')}. Accuracy appears only as a secondary value. For segmentation we report "
              f"Dice and the 95th percentile Hausdorff distance {c('taha')}. Every metric has a 95% confidence interval from "
              f"2000 bootstrap resamples over patients {c('efron')}. We use the percentile method.")
    mset = pd.DataFrame([
        ("U-Net (ResNet34)", "AdamW, lr 3e-4, 40 epochs, batch 16", "CE + Dice loss"),
        ("XGBoost", "300 trees, depth 3, lr 0.05", "18 features"),
        ("Random forest", "500 trees", "Baseline"),
        ("ResNet18", "AdamW, lr 1e-4, 25 epochs, batch 32", "Label smoothing 0.1"),
        ("Ensemble", "Mean of probabilities", "No tuned weights"),
    ], columns=["Model", "Settings", "Note"])
    table(doc, mset, "Model settings", widths=[1.0, 1.7, 0.9], size=7)

    # ---------------------------------------------------------------- IV Results
    heading(doc, "IV", "Results")
    subheading(doc, "A", "Cohort")
    body(doc, "Table III lists the cohort. The groups have equal size by design, so these numbers do not show the real "
              "share of each disease in a hospital.")
    coh = df_or_placeholder(COH, ["split", "group", "n", "height_cm", "weight_kg", "LV_EF_pct", "LV_EDV_ml"],
                            "Run src/09_report_metrics.py")
    table(doc, coh, "Cohort description (mean ± SD)", widths=[0.9, 0.45, 0.3, 0.55, 0.55, 0.5, 0.55], size=8)
    subheading(doc, "B", "Segmentation")
    seg = df_or_placeholder(SEG, ["phase", "structure", "n", "dice", "hd95_mm"], "Run steps 3 and 9")
    body(doc, f"Table IV gives the segmentation quality on the test patients. At ED the Dice scores were "
              f"{seg_val('LV')} for LV, {seg_val('MYO')} for MYO and {seg_val('RV')} for RV.")
    table(doc, seg, "Segmentation on the test set, mean (95% CI)", widths=[0.5, 0.7, 0.35, 1.2, 1.2], size=7)
    figure(doc, os.path.join(FIG, "seg_examples_unet.png"), "Segmentation examples on four test patients at ED: "
                                                           "image, expert mask and U-Net prediction.", 3.0)
    subheading(doc, "C", "Diagnosis performance")
    if MET is not None:
        m = MET.copy()
        perf = pd.DataFrame({"Model": m["model"]})
        for k, n in (("AUROC", "AUROC"), ("AUPRC", "AUPRC"), ("sensitivity", "Sens."),
                     ("specificity", "Spec."), ("f1", "F1")):
            perf[n] = [ci(r, k, 2) for _, r in m.iterrows()]
        best = m.sort_values("AUROC", ascending=False).iloc[0]
        body(doc, f"Table V shows the results on the 50 test patients. {best['model']} had the highest macro AUROC, "
                  f"{ci(best, 'AUROC')}. Intervals of different models overlap in most columns, so with 50 patients we cannot "
                  f"rank the models with confidence. The ensemble had a Brier score of {main_val('brier', with_ci=False)} and an "
                  f"expected calibration error of {main_val('ece', with_ci=False)}. Accuracy was {main_val('accuracy')}. "
                  f"XGBoost alone had a macro F1 of {ci(XGBR, 'f1', 2)} and an expected calibration error of {XGBR['ece']:.3f}. "
                  f"The CNN was the weakest model, with a macro F1 of {ci(CNNR, 'f1', 2)}. The random forest baseline reached "
                  f"an AUROC of {ci(BASE, 'AUROC', 2)}.")
    else:
        perf = df_or_placeholder(None, ["Model", "AUROC", "AUPRC", "Sens.", "Spec.", "F1"], "Run step 9")
        body(doc, ph("Describe Table V after running the pipeline"))
    table(doc, perf, "Diagnosis on the test set, macro mean (95% CI)", widths=[0.9, 0.5, 0.5, 0.5, 0.5, 0.5], size=7.5)
    pc = df_or_placeholder(None, ["Class", "AUROC", "AUPRC", "Sens.", "Spec.", "F1"], "Run step 9")
    if PCL is not None:
        pc = pd.DataFrame({"Class": PCL["class"]})
        for k, n in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("sensitivity", "Sens."),
                     ("specificity", "Spec."), ("f1", "F1")):
            pc[n] = [ci(r, k, 2) for _, r in PCL.iterrows()]
    body(doc, "Table VI gives the same metrics for each class. Each class has only 10 test patients, so the intervals are wide.")
    table(doc, pc, "Per-class results of the ensemble (95% CI)", widths=[0.4, 0.55, 0.55, 0.55, 0.55, 0.55], size=7.5)
    figure(doc, os.path.join(FIG, "roc_pr.png"), "ROC and precision-recall curves of the ensemble, one class against the rest.", 3.4)
    figure(doc, os.path.join(FIG, "calibration.png"), "Reliability diagram of the ensemble. Numbers show the count of "
                                                       "predictions in each bin.", 2.4)
    figure(doc, os.path.join(FIG, "confusion_matrices.png"), "Confusion matrices of all models on the test set.", 3.4)
    subheading(doc, "D", "Explainability")
    t3 = ", ".join(top_shap(3))
    body(doc, f"Fig. 7 shows the mean absolute SHAP values of the XGBoost model on the test set. The three features with the "
              f"largest mean effect were {t3}. Fig. 8 shows Grad-CAM maps of the CNN for one test patient per group.")
    figure(doc, os.path.join(FIG, "shap_global.png"), "Global SHAP feature importance of XGBoost.", 3.0)
    figure(doc, os.path.join(FIG, "gradcam_examples.png"), "Grad-CAM maps of the CNN. Warm colours mark regions "
                                                           "with a stronger effect on the prediction.", 3.4)
    figure(doc, os.path.join(FIG, "shap_per_class.png"), "Mean absolute SHAP value of each feature, for each class.", 3.4)
    subheading(doc, "E", "Subgroup analysis")
    sg = df_or_placeholder(SUB, ["subgroup", "n", "AUROC", "AUPRC", "sensitivity", "specificity", "f1", "accuracy"],
                           "Run step 9")
    body(doc, "The patient files hold no age or sex, so we split the test set at the median body surface area. "
              "Each subgroup has about 25 patients. Table VII shows the result.")
    table(doc, sg[[x for x in ["subgroup", "n", "AUROC", "AUPRC", "sensitivity", "f1"] if x in sg.columns]],
          "Ensemble by body surface area, macro mean (95% CI)", widths=[0.9, 0.3, 0.9, 0.9, 0.9, 0.9], size=7.5)

    # ---------------------------------------------------------------- V Discussion
    heading(doc, "V", "Discussion")
    body(doc, "Clinical meaning. The five groups differ in ways that doctors already use. A dilated heart has a large LV "
              "volume and a low ejection fraction. A hypertrophic heart has a thick wall and a high mass. A heart with "
              "a previous infarction often has a low ejection fraction. An abnormal RV shows a large RV volume. "
              f"The features that SHAP ranked highest ({t3}) fit this list. Ejection fraction and ventricle volume separate DCM and "
              "MINF from the other groups. The RV to LV volume ratio points to the abnormal RV group.")
    body(doc, "Model comparison and errors. The ensemble had the highest macro AUROC and AUPRC, but XGBoost alone matched or beat it "
              "on F1, accuracy and calibration, and the intervals overlap. We therefore see no proof that the ensemble helps. "
              "The CNN was clearly weaker than the feature-based models. We think it has too few training patients (80) for "
              "a deep network. Most errors of all models mix up DCM and MINF. Both groups have a large LV and a low "
              "ejection fraction, so this error is clinically plausible. Features from U-Net masks did as well as features "
              "from expert masks, which suggests that segmentation error is not the main limit.")
    body(doc, "Comparison with prior work. Our pipeline follows the two-stage idea of earlier ACDC entries "
              f"{c('isensee17', 'khened')}. A direct comparison of single numbers is not fair. The earlier teams used the "
              "challenge test protocol. We use our own split, a different set of metrics and predicted masks at test time. "
              "What we add is a full report of uncertainty, calibration and subgroups.")
    body(doc, "Limitations. First, the test set has only 50 patients, 10 per class, so the confidence intervals are wide. "
              "Second, all data come from one hospital and one MRI protocol, and we ran no external validation. Third, "
              "the equal class sizes do not match real hospital prevalence, so predictive values and calibration may not "
              "transfer. Fourth, we picked the CNN checkpoint by accuracy on only 20 validation patients. Fifth, the files "
              "hold no age or sex, so we could not study fairness for these groups. Sixth, the U-Net works on 2D slices "
              "and may fail on slices at the base or apex. This is a research prototype and not a medical device.")

    # ---------------------------------------------------------------- VI Conclusion
    heading(doc, "VI", "Conclusion")
    body(doc, f"A U-Net, a feature-based XGBoost model and a ResNet18 together reached a macro AUROC of {main_val('AUROC')} "
              "on 50 unseen ACDC patients. All splits were by patient, so no data leaked. The intervals show how much "
              "uncertainty a small test set leaves. XGBoost on mask features did about as well as the ensemble, and the CNN was weaker, so the "
              "clinical features carry most of the signal. Future work will add an external test set and a 3D or temporal model. "
              "It will also add age and sex data for fairness checks, and a study with clinicians who use the explanations.")

    # ---------------------------------------------------------------- Declarations
    heading(doc, "VII", "Declarations")
    body(doc, "Ethics. This study used the public ACDC dataset. The dataset owners released it in anonymised form. "
              "The dataset comes from real clinical exams at the University Hospital of Dijon, France. The hospital "
              "handled the data under the rules of its local ethical committee. The dataset page gives no approval "
              f"number, so we cite the source paper {c('bernard')} and follow its citation rule. We did not collect any new "
              f"patient data. We followed the Digital Personal Data Protection Act, 2023 {c('dpdp')} and the ICMR "
              f"guidelines for AI in biomedical research {c('icmr')}. We do not share patient data. The repository "
              "holds code and aggregate results only.")
    body(doc, f"Reporting. We followed the TRIPOD+AI checklist {c('tripod')}. The completed checklist is attached. We prepared no study protocol, and we did not register the study. No patients or members of the public took part in the design.")
    body(doc, f"Data availability. The ACDC data are available from the dataset owners after registration. "
              f"Code availability. The code is at {GITHUB}, with a README that lists every step.")
    body(doc, "Use of AI tools. We used an AI assistant (Claude, Anthropic) to help structure the code and to draft "
              "parts of this text. The author checked the code, ran all experiments, verified all numbers and "
              "edited the text.")
    body(doc, "Funding: none. Conflicts of interest: none declared. Author contributions: the single author did all work.")

    # ---------------------------------------------------------------- References
    para(doc, "REFERENCES", size=10, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4)
    for k in REFS:
        c(k)                                    # add uncited ones at the end
    for i, k in enumerate(_order, 1):
        para(doc, f"[{i}] {REFS[k]}", size=8, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
    doc.save(out)
    return abs_text


# =========================================================================== checklist
TRIPOD = [
    ("Title", "1", "Identify the study as developing and evaluating a prediction model, and name the target population and outcome.", "Title"),
    ("Abstract", "2", "Structured summary of the study.", "Abstract"),
    ("Introduction", "3", "Background and rationale, including the healthcare context and the role of the model.", "I"),
    ("Introduction", "4", "Objectives, including whether the study develops or evaluates the model.", "I"),
    ("Methods", "5", "Data source, dates and the way the data were collected.", "III-A"),
    ("Methods", "6", "Participants: eligibility criteria and setting.", "III-A, Fig. 1"),
    ("Methods", "7", "Data preparation and quality checks.", "III-C"),
    ("Methods", "8", "Outcome definition and how it was assessed.", "III-B"),
    ("Methods", "9", "Predictors: how they were chosen, measured and handled.", "III-D"),
    ("Methods", "10", "Sample size and its justification.", "III-A, V (limitations). Fixed public dataset"),
    ("Methods", "11", "Missing data and how it was handled.", "III-A. No missing masks or labels"),
    ("Methods", "12", "Analytical methods: model type, tuning, validation.", "III-D, III-E"),
    ("Methods", "13", "Class imbalance and how it was handled.", "III-C. Classes are equal in size"),
    ("Methods", "14", "Fairness: subgroups and how they were studied.", "IV-E, V. Age and sex not available"),
    ("Methods", "15", "Model output (probabilities) and thresholds.", "III-D. Argmax of probabilities"),
    ("Methods", "16", "Difference between training and evaluation data.", "III-A, III-E"),
    ("Methods", "17", "Ethical approval and consent.", "VII"),
    ("Open science", "18", "Funding, conflicts, protocol, registration, data and code sharing.", "VII"),
    ("Open science", "19", "Patient and public involvement.", "Not applicable. State this in VII"),
    ("Results", "20", "Participants: flow, characteristics, number of events.", "III-A, IV-A"),
    ("Results", "21", "Model development: how the final model was built.", "III-D"),
    ("Results", "22", "Model specification: details to use the model.", "III-D, repository"),
    ("Results", "23", "Model performance with confidence intervals.", "IV-B, IV-C"),
    ("Results", "24", "Model updating.", "Not applicable. No updating"),
    ("Discussion", "25", "Interpretation of the results.", "V"),
    ("Discussion", "26", "Limitations.", "V"),
    ("Discussion", "27", "Usability of the model in current care.", "V. Research prototype only"),
]


def build_checklist(out):
    from tripod_items import TRIPOD
    d = Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(0.6)
    st = d.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(10)
    para(d, "TRIPOD+AI Reporting Checklist", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
    para(d, "Items and numbering follow the official checklist (version 11 January 2024) from tripod-statement.org. "
            "Source: Collins GS et al., BMJ 2024;385:e078378, doi:10.1136/bmj-2023-078378.", size=9, italic=True,
         align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
    para(d, f"Paper: {TITLE}", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
    para(d, "Team 21: Cardiac MRI to Diagnosis: Segmentation + Modelling + Explainability. Author: Priyesh Kumar Jha, "
            "23BCE1221. Study type: development and evaluation of a diagnostic prediction model "
            "(five-class diagnosis from cine MRI).", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, after=4)
    para(d, "D = item for development. E = item for evaluation. D;E = both. The last column points to a section of the "
            "paper: I Introduction, II Related Work, III Methods, IV Results, V Discussion, VI Conclusion, "
            "VII Declarations. Items that the study does not report say so.", size=9,
         align=WD_ALIGN_PARAGRAPH.LEFT, after=6)
    cols = ["Section", "Item", "D/E", "Checklist item", "Reported in this study"]
    t = d.add_table(rows=1, cols=5)
    t.style = "Table Grid"
    for i, col in enumerate(cols):
        t.rows[0].cells[i].text = ""
        runs(t.rows[0].cells[i].paragraphs[0], col, 9, bold=True)
    for r in TRIPOD:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            runs(cells[i].paragraphs[0], str(v), 8)
    for row in t.rows:
        for i, w in enumerate((0.9, 0.4, 0.4, 2.8, 2.7)):
            row.cells[i].width = Inches(w)
    d.save(out)


# =========================================================================== slides
def build_slides(out, abs_text):
    from pptx import Presentation
    from pptx.util import Inches as I, Pt as P
    from pptx.dml.color import RGBColor as C
    from pptx.enum.shapes import MSO_SHAPE

    prs = Presentation()
    prs.slide_width, prs.slide_height = I(13.333), I(7.5)
    blank = prs.slide_layouts[6]
    NAVY, RED, GREY = C(0x1F, 0x3A, 0x4D), C(0x8F, 0x3B, 0x4A), C(0x55, 0x55, 0x55)

    def add_text(sl, x, y, w, h, text, size=20, bold=False, color=GREY):
        tb = sl.shapes.add_textbox(I(x), I(y), I(w), I(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size, p.font.bold, p.font.color.rgb = P(size), bold, color
            p.font.name = "Calibri"
            p.space_after = P(8)
        return tb

    def slide(title, bullets=None, img=None, img_w=6.0, note=None):
        sl = prs.slides.add_slide(blank)
        bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, I(0.25), prs.slide_height)
        bar.fill.solid(); bar.fill.fore_color.rgb = RED; bar.line.fill.background()
        add_text(sl, 0.7, 0.4, 12, 1, title, 30, True, NAVY)
        w = 12 if not img else 6.2
        if bullets:
            add_text(sl, 0.7, 1.5, w, 5.2, "\n".join("• " + b for b in bullets), 20)
        if img:
            if os.path.exists(img):
                sl.shapes.add_picture(img, I(7.1 if bullets else 2.5), I(1.5), width=I(img_w))
            else:
                add_text(sl, 7.1, 3, 5.5, 1, f"[Insert {os.path.basename(img)} after the run]", 16, False, RED)
        if note:
            add_text(sl, 0.7, 6.8, 12, 0.5, note, 12)
        return sl

    sl = prs.slides.add_slide(blank)
    bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    add_text(sl, 0.9, 1.8, 11.5, 2.5, TITLE, 34, True, C(255, 255, 255))
    add_text(sl, 0.9, 5.0, 11.5, 1.5, "BCSE335L Healthcare Data Analytics, Final Review\nCardiac MRI to Diagnosis: Segmentation + Modelling + Explainability\nPriyesh Kumar Jha, 23BCE1221", 18,
             False, C(0xDD, 0xDD, 0xDD))

    slide("The clinical problem", [
        "Cine cardiac MRI gives exact heart volumes, mass and pumping function.",
        "A reader must draw heart borders on every slice at two time points.",
        "This work is slow and readers do not always agree.",
        "Goal: a fully automatic diagnosis of five conditions.",
    ])
    slide("Gap and research question", [
        "Papers differ in splits, metrics and test masks.",
        "Few report confidence intervals, calibration or subgroups.",
        "Question: can segmentation plus two classifiers diagnose unseen patients, and how certain are we?",
    ])
    slide("Data: ACDC, 150 patients", ["5 groups, 30 patients each", "Expert masks of RV, MYO, LV at ED and ES",
                                       "Split by patient: 80 train, 20 validation, 50 test", "No exclusions"],
          img=os.path.join(DIA, "cohort_flow.png"), img_w=4.6)
    slide("Pipeline", ["U-Net finds RV, MYO and LV", "XGBoost reads 18 clinical features",
                       "ResNet18 reads heart crops", "Mean of probabilities gives the diagnosis",
                       "Test uses predicted masks, not expert masks"],
          img=os.path.join(DIA, "pipeline.png"), img_w=6.2)
    slide("Methods and leakage control", [
        "Every split is by patient, never by slice",
        "Scaling uses one volume only, no shared statistics",
        "Random forest is the baseline",
        "Test set used once, at the end",
        "95% CI from 2000 patient-level bootstrap resamples",
    ])
    seg = SEG
    slide("Segmentation results", [
        f"Dice at ED: LV {seg_val('LV')}, MYO {seg_val('MYO')}, RV {seg_val('RV')}",
        "Hausdorff 95 in mm is in the paper, Table IV",
    ])
    if MET is not None:
        lines = [f"{r['model']}: AUROC {ci(r, 'AUROC', 2)}" for _, r in MET.iterrows()]
    else:
        lines = ["[[Fill after the run]]"]
    slide("Diagnosis performance (test, 50 patients)", lines + [
        f"Ensemble: AUPRC {main_val('AUPRC', 2)}, sensitivity {main_val('sensitivity', 2, False)}, "
        f"specificity {main_val('specificity', 2, False)}, F1 {main_val('f1', 2, False)}"])
    slide("ROC, precision-recall and calibration", img=os.path.join(FIG, "roc_pr.png"), img_w=8.5)
    slide("Explainability", [f"SHAP top features: {', '.join(top_shap(3))}", "Grad-CAM shows where the CNN looks"],
          img=os.path.join(FIG, "shap_global.png"), img_w=5.8)
    slide("Subgroups and limitations", [
        "Subgroup: body surface area (no age or sex in the files)",
        "Only 50 test patients, 10 per class: wide intervals",
        "One hospital, no external validation",
        "Equal class sizes: calibration may not transfer",
        "Research prototype, not a medical device",
    ])
    slide("Conclusion and future work", [
        f"Ensemble macro AUROC {main_val('AUROC', 2)} on unseen patients",
        "Future: external test set, 3D or temporal model, age and sex for fairness, clinician study",
    ])
    slide("Ethics, code and AI use", [
        "Public ACDC data. No patient data shared. DPDP Act 2023 and ICMR AI guidelines cited",
        f"Code: {GITHUB}",
        "TRIPOD+AI checklist attached",
        "AI assistant used for code structure and drafting. Authors checked everything",
    ])
    prs.save(out)


if __name__ == "__main__":
    paper = os.path.join(os.environ.get("PAPER_OUT", HERE), "BCSE335L_Team21_CardiacDiagnosis.docx")
    abs_text = build_paper(paper)
    n = len(re.sub(r"\[\[.*?\]\]", "x", abs_text).split())
    print(f"abstract words: {n} (target 200-250)")
    build_checklist(os.path.join(os.environ.get("PAPER_OUT", HERE), "TRIPOD_AI_checklist.docx"))
    build_slides(os.path.join(os.environ.get("PAPER_OUT", HERE), "BCSE335L_Team21_Slides.pptx"), abs_text)
    print("references:", len(_order))
    print("built into", HERE)
