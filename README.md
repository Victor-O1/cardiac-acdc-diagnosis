# Cardiac MRI diagnosis on ACDC

Course project for BCSE335L Healthcare Data Analytics. The pipeline reads cine cardiac MRI and gives one of five diagnoses: NOR, MINF, DCM, HCM, RV.

**Research prototype. Not a medical device.**

## What the pipeline does

1. A U-Net (ResNet34 encoder) segments RV, myocardium and LV at end-diastole and end-systole.
2. 18 clinical features come from the masks. XGBoost classifies them. A random forest is the baseline.
3. A ResNet18 classifies heart crops made of the ED slice, the ES slice and their difference.
4. The mean of both probability outputs is the final diagnosis.
5. SHAP explains XGBoost. Grad-CAM explains the CNN.

## Data

Use the public ACDC dataset (150 patients, 5 groups). Download it from the ACDC challenge site after registration. Put `training/` and `testing/` inside `data/`. This repository holds no patient data. Please read the dataset licence before you use it.

## Run it

Use a GPU. The easy way is `run_in_colab.ipynb`. On your own machine:

```bash
pip install -r requirements.txt
python src/00_inspect_dataset.py      # must say ALL CHECKS PASSED
python src/01_preprocess.py           # caches volumes, makes the patient-level split
python src/02_train_unet.py --arch unet --epochs 40
python src/03_eval_segmentation.py --arch unet
python src/04_extract_features.py
python src/05_train_xgboost.py
python src/06_train_cnn.py --epochs 25
python src/07_compare_models.py
python src/08_explainability.py
python src/09_report_metrics.py       # AUROC, AUPRC, sens, spec, F1, calibration, 95% CI
python paper/make_diagrams.py
python paper/build_docs.py            # paper, TRIPOD+AI checklist, slides
```

## Leakage controls

- All splits are by patient (80 train, 20 validation, 50 test). The test set is used once.
- Intensity scaling uses one volume only. No statistic passes between patients.
- Classes have equal size, so no oversampling is used.
- Confidence intervals come from a bootstrap over patients (2000 resamples).

## Outputs

`outputs/results/` holds tables. `outputs/figures/` holds plots. `paper/` holds the generated paper, checklist and slides. Numbers that do not exist yet show as yellow `[[placeholders]]`.

## Ethics

Public, anonymised data. No new patient data. See the paper for the DPDP Act 2023 and ICMR AI guideline statements.
