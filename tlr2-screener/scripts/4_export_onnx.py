"""
STEP 4: Convert tlr2_model.json (XGBoost native) to ONNX for Vercel serving.

onnxruntime (57MB) replaces xgboost (228MB) + scikit-learn (50MB) in the
Vercel bundle, keeping the deployment well under the 250MB limit.

Run: python 4_export_onnx.py
Needs: tlr2_model.json + feature_meta.json from step 3
Output: ml/model/tlr2_clf.onnx + ml/model/clf_meta.json
"""

import json
import os
import numpy as np
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

meta_path = "feature_meta.json"
model_path = "tlr2_model.json"
out_dir = os.path.join("ml", "model")
os.makedirs(out_dir, exist_ok=True)

with open(meta_path) as f:
    meta = json.load(f)

n_features = meta["total_features"]
model = xgb.XGBClassifier()
model.load_model(model_path)

initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

onnx_out = os.path.join(out_dir, "tlr2_clf.onnx")
with open(onnx_out, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"wrote {onnx_out} ({os.path.getsize(onnx_out)//1024}KB)")

clf_meta = {
    "best_model": "XGBoost",
    "task": "binary_classification",
    "data_source": "ChEMBL (CHEMBL4163, UniProt O60603, human TLR2, EC50 only)",
    "model_task": "TLR2 agonist classifier (functional EC50; decoy-augmented negatives)",
    "split_method": "Bemis-Murcko scaffold split (80/20)",
    "threshold_label": "pchembl_value >= 5.0 (EC50 <= 10uM = active agonist)",
    "feature_order": meta.get("feature_order", "morgan_fp_first_then_descriptors"),
    "fp_bits": meta["fp_bits"],
    "fp_radius": meta["fp_radius"],
    "n_descriptors": meta["n_descriptors"],
    "total_features": n_features,
    "n_train": meta.get("n_train"),
    "n_test": meta.get("n_test"),
    "n_total": meta.get("n_total"),
    "roc_auc": meta.get("roc_auc"),
    "cv_roc_auc": meta.get("roc_auc"),  # single split; no CV in this pipeline
    "precision": meta.get("precision"),
    "recall": meta.get("recall"),
    "f1": meta.get("f1"),
    "confusion_matrix": meta.get("confusion_matrix"),
}

meta_out = os.path.join(out_dir, "clf_meta.json")
with open(meta_out, "w") as f:
    json.dump(clf_meta, f, indent=2)
print(f"wrote {meta_out}")
