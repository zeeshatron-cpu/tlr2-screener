"""
STEP 5: Leave-one-scaffold-out (LOSO) cross-validation.

The single 80/20 scaffold split in step 3 gives ONE train/test partition,
so its AUC rides on which scaffolds happened to land in the test fold. With
sparse, congeneric data that number has a wide confidence interval.

LOSO removes that luck: every Bemis-Murcko scaffold group is held out in turn,
the model is trained on all the others, and the held-out compounds are scored.
Pooling those out-of-fold predictions over the whole dataset gives ONE honest
AUC computed on predictions the model never saw during its own training --
the defensible single number for a paper.

We also run GroupKFold(k=5) grouped by scaffold to report a fold-to-fold
spread (mean +/- std AUC), which quantifies how much the estimate wobbles.

This script only REPORTS. It does not touch the served model (step 3/4 own
that). Reuses step 3's exact featurization so the numbers are comparable.

Run: python 5_loso_cv.py
Needs: tlr2_with_decoys.csv (or tlr2_clean.csv) from step 2/2b
Output: prints metrics, writes loso_metrics.json
"""

import importlib.util
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import GroupKFold
import xgboost as xgb

# Reuse step 3's featurization / scaffold logic verbatim (the module name
# starts with a digit, so import it by path rather than a plain import).
_step3_path = os.path.join(os.path.dirname(__file__), "3_train_scaffold_split.py")
_spec = importlib.util.spec_from_file_location("train_scaffold_split", _step3_path)
_step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step3)

featurize = _step3.featurize
scaffold_of = _step3.scaffold_of
RANDOM_STATE = _step3.RANDOM_STATE


def make_model(spw):
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=RANDOM_STATE,
        use_label_encoder=False,
    )


def spw_for(y_train):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def main():
    csv_path = "tlr2_with_decoys.csv" if os.path.exists("tlr2_with_decoys.csv") else "tlr2_clean.csv"
    print(f"Loading from {csv_path}")
    df = pd.read_csv(csv_path).reset_index(drop=True)

    X, y, scaffolds = [], [], []
    for _, row in df.iterrows():
        f = featurize(row["smiles"])
        if f is None:
            continue
        sc = scaffold_of(row["smiles"])
        X.append(f)
        y.append(int(row["active"]))
        # unparseable scaffold -> unique singleton group so it can't merge
        scaffolds.append(sc if sc else f"_none_{len(X)}")
    X = np.vstack(X).astype(np.float32)
    y = np.array(y)
    scaffolds = np.array(scaffolds)

    groups = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        groups[sc].append(i)
    n_scaffolds = len(groups)
    n_singletons = sum(1 for v in groups.values() if len(v) == 1)
    print(f"featurized {len(y)} compounds | {int(y.sum())} active / {int((y==0).sum())} inactive")
    print(f"unique scaffolds: {n_scaffolds} ({n_singletons} singletons)")

    # ---- True LOSO: pooled out-of-fold predictions ----
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for sc, idx in groups.items():
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[idx] = True
        train_mask = ~test_mask
        # a fold whose training side lost an entire class can't train sensibly
        if len(set(y[train_mask])) < 2:
            continue
        model = make_model(spw_for(y[train_mask]))
        model.fit(X[train_mask], y[train_mask])
        oof[idx] = model.predict_proba(X[test_mask])[:, 1]

    scored = ~np.isnan(oof)
    n_scored = int(scored.sum())
    y_s = y[scored]
    p_s = oof[scored]
    pred_s = (p_s >= 0.5).astype(int)

    loso = {
        "n_folds": n_scaffolds,
        "n_scored": n_scored,
        "n_skipped_single_class_train": len(y) - n_scored,
    }
    if len(set(y_s)) == 2:
        loso.update({
            "pooled_roc_auc": round(float(roc_auc_score(y_s, p_s)), 4),
            "pooled_precision": round(float(precision_score(y_s, pred_s, zero_division=0)), 4),
            "pooled_recall": round(float(recall_score(y_s, pred_s, zero_division=0)), 4),
            "pooled_f1": round(float(f1_score(y_s, pred_s, zero_division=0)), 4),
            "pooled_confusion_matrix": confusion_matrix(y_s, pred_s).tolist(),
        })
    print("\n--- LEAVE-ONE-SCAFFOLD-OUT (pooled out-of-fold) ---")
    print(f"folds (scaffolds)     : {n_scaffolds}")
    print(f"compounds scored      : {n_scored} / {len(y)}")
    if "pooled_roc_auc" in loso:
        print(f"pooled ROC AUC        : {loso['pooled_roc_auc']:.3f}")
        print(f"pooled Precision      : {loso['pooled_precision']:.3f}")
        print(f"pooled Recall         : {loso['pooled_recall']:.3f}")
        print(f"pooled F1             : {loso['pooled_f1']:.3f}")
        print(f"pooled Confusion      : {loso['pooled_confusion_matrix']}")

    # ---- GroupKFold(5) grouped by scaffold: fold-to-fold spread ----
    fold_aucs = []
    n_splits = min(5, n_scaffolds)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups=scaffolds):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            model = make_model(spw_for(y[tr]))
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[te], proba)))

    grouped = {"n_splits": n_splits, "fold_aucs": [round(a, 4) for a in fold_aucs]}
    if fold_aucs:
        grouped["mean_auc"] = round(float(np.mean(fold_aucs)), 4)
        grouped["std_auc"] = round(float(np.std(fold_aucs)), 4)
        print(f"\n--- GroupKFold(k={n_splits}) by scaffold ---")
        print(f"fold AUCs             : {grouped['fold_aucs']}")
        print(f"mean +/- std          : {grouped['mean_auc']:.3f} +/- {grouped['std_auc']:.3f}")

    out = {
        "method": "leave-one-scaffold-out cross-validation",
        "note": "Pooled out-of-fold AUC is the headline; GroupKFold spread shows estimate variance. Reporting only -- served model comes from step 3/4.",
        "dataset": csv_path,
        "n_compounds": len(y),
        "loso": loso,
        "grouped_kfold": grouped,
    }
    with open("loso_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote loso_metrics.json")


if __name__ == "__main__":
    main()
