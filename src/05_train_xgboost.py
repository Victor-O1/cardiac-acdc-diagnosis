"""
STEP 5 - Diagnosis from clinical features (XGBoost, with RandomForest as a baseline).
  * 5-fold stratified CV on the 100 training patients
  * final XGBoost fit on all 100 -> evaluated on test (expert masks AND U-Net masks)
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from common import CLASSES, RESULTS_DIR, MODEL_DIR, SEED
from features import FEATURE_COLS


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="multi:softprob",
                         eval_metric="mlogloss", random_state=SEED)


def load(name):
    df = pd.read_csv(os.path.join(RESULTS_DIR, name))
    return df, df[FEATURE_COLS].values, df["group"].map(CLASSES.index).values


def save_probs(df, probs, name):
    out = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    out.insert(0, "true", df["group"].values)
    out.insert(0, "pid", df["pid"].values)
    out["pred"] = [CLASSES[i] for i in probs.argmax(1)]
    out.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def main():
    tr, X, y = load("features_train_gt.csv")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report = {}
    for name, model in (("xgboost", make_xgb()),
                        ("random_forest", RandomForestClassifier(500, random_state=SEED))):
        pred = cross_val_predict(model, X, y, cv=cv)
        report[f"cv_{name}"] = {"accuracy": accuracy_score(y, pred),
                                "macro_f1": f1_score(y, pred, average="macro")}
        print(f"5-fold CV {name}: acc={report[f'cv_{name}']['accuracy']:.3f} "
              f"macroF1={report[f'cv_{name}']['macro_f1']:.3f}")

    xgb = make_xgb().fit(X, y)
    xgb.save_model(os.path.join(MODEL_DIR, "xgb.json"))
    rf = RandomForestClassifier(500, random_state=SEED).fit(X, y)      # baseline model

    for tag in ("gt", "pred"):
        path = os.path.join(RESULTS_DIR, f"features_test_{tag}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        te, Xt, yt = load(f"features_test_{tag}.csv")
        probs = xgb.predict_proba(Xt)
        pred = probs.argmax(1)
        save_probs(te, probs, f"xgb_test_probs_{tag}.csv")
        if tag == "pred":                                  # baseline probabilities for step 9
            save_probs(te, rf.predict_proba(Xt), "rf_test_probs_pred.csv")
            report["test_pred_masks_random_forest"] = {
                "accuracy": accuracy_score(yt, rf.predict(Xt)),
                "macro_f1": f1_score(yt, rf.predict(Xt), average="macro")}
        report[f"test_{tag}_masks"] = {
            "accuracy": accuracy_score(yt, pred), "macro_f1": f1_score(yt, pred, average="macro"),
            "confusion_matrix": confusion_matrix(yt, pred, labels=range(5)).tolist()}
        print(f"\nTEST ({tag} masks) accuracy={report[f'test_{tag}_masks']['accuracy']:.3f}")
        print(classification_report(yt, pred, labels=range(5), target_names=CLASSES, zero_division=0))

    with open(os.path.join(RESULTS_DIR, "xgb_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
