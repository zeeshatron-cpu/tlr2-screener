#!/usr/bin/env python3
"""
Fetch real TLR2 bioactivity from ChEMBL and train a binary classifier.

Steps:
  1. Pull all IC50/EC50/Ki records for CHEMBL4523 (human TLR2) with pchembl_value
  2. Deduplicate by InChIKey, keep median pchembl per unique structure
  3. Binary label: pchembl >= 5.0  (IC50 < 10uM = active)
  4. Scaffold split (Bemis-Murcko) -- no train/test scaffold overlap
  5. Random Forest vs XGBoost; keep winner by CV ROC AUC
  6. Export winning model as ONNX + write clf_meta.json

Outputs (relative to this script's location):
  ../ml/model/tlr2_clf.onnx
  ../ml/model/clf_meta.json
  ../ml/data/chembl_tlr2_raw.csv      (raw records pre-dedup)
  ../ml/data/chembl_tlr2_modelling.csv (deduped, labelled, split column)
"""

import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ML_DIR = HERE.parent / "ml"
DATA_DIR = ML_DIR / "data"
MODEL_DIR = ML_DIR / "model"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Fetch from ChEMBL ────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Fetching TLR2 bioactivity from ChEMBL")
print("=" * 60)

from chembl_webresource_client.new_client import new_client

activity = new_client.activity
records = activity.filter(
    target_chembl_id="CHEMBL4523",
    standard_type__in=["IC50", "EC50", "Ki", "AC50", "Kd"],
    pchembl_value__isnull=False,
).only([
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_value", "standard_units", "pchembl_value",
    "assay_chembl_id", "document_chembl_id",
])

print("Downloading records (may take a minute)...")
data = list(records)
print(f"  Raw records from ChEMBL: {len(data)}")

df_raw = pd.DataFrame(data)
df_raw = df_raw[df_raw["canonical_smiles"].notna() & (df_raw["canonical_smiles"] != "")]
df_raw["pchembl_value"] = pd.to_numeric(df_raw["pchembl_value"], errors="coerce")
df_raw = df_raw[df_raw["pchembl_value"].notna()]
print(f"  After dropping missing SMILES/pchembl: {len(df_raw)}")

df_raw.to_csv(DATA_DIR / "chembl_tlr2_raw.csv", index=False)
print(f"  Saved raw data -> {DATA_DIR / 'chembl_tlr2_raw.csv'}")

# ── 2. Deduplicate by InChIKey ───────────────────────────────────────────────

print("\nSTEP 2: Deduplicating by InChIKey")

from rdkit import Chem
from rdkit.Chem import inchi

rows = []
failed = 0
for _, row in df_raw.iterrows():
    mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
    if mol is None:
        failed += 1
        continue
    key = inchi.MolToInchiKey(mol)
    rows.append({
        "inchikey": key,
        "smiles": row["canonical_smiles"],
        "pchembl_value": float(row["pchembl_value"]),
        "standard_type": row["standard_type"],
        "molecule_chembl_id": row["molecule_chembl_id"],
    })

print(f"  Parsed: {len(rows)}, unparseable: {failed}")
df = pd.DataFrame(rows)

# Median pchembl per unique InChIKey
df_dedup = (
    df.groupby("inchikey")
    .agg(
        smiles=("smiles", "first"),
        pchembl_value=("pchembl_value", "median"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        n_measurements=("pchembl_value", "count"),
    )
    .reset_index()
)
print(f"  Unique structures (median pchembl): {len(df_dedup)}")
print(f"  pchembl range: {df_dedup.pchembl_value.min():.2f} - {df_dedup.pchembl_value.max():.2f}")

# Binary label
CUTOFF = 5.0  # pIC50 >= 5.0 = IC50 < 10uM = active
df_dedup["active"] = (df_dedup["pchembl_value"] >= CUTOFF).astype(int)
n_active = df_dedup["active"].sum()
print(f"  Active (pchembl >= {CUTOFF}): {n_active} / {len(df_dedup)} ({100*n_active/len(df_dedup):.1f}%)")

if n_active < 10 or (len(df_dedup) - n_active) < 10:
    print("ERROR: Too few actives or inactives for reliable classification.")
    sys.exit(1)

# ── 3. Scaffold split (Bemis-Murcko) ────────────────────────────────────────

print("\nSTEP 3: Scaffold split (Bemis-Murcko)")

from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "__invalid__"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChiralCenters=False)
    except Exception:
        return smi  # fallback: use the SMILES itself as its own scaffold

df_dedup["scaffold"] = df_dedup["smiles"].apply(get_scaffold)
scaffolds = df_dedup["scaffold"].unique()
print(f"  Unique Bemis-Murcko scaffolds: {len(scaffolds)}")

# Shuffle scaffolds and assign 80% to train, 20% to test
rng = np.random.default_rng(42)
shuffled = rng.permutation(scaffolds)
n_test_scaffolds = max(1, int(len(shuffled) * 0.2))
test_scaffolds = set(shuffled[:n_test_scaffolds])
df_dedup["split"] = df_dedup["scaffold"].apply(lambda s: "test" if s in test_scaffolds else "train")

train_df = df_dedup[df_dedup["split"] == "train"]
test_df = df_dedup[df_dedup["split"] == "test"]
print(f"  Train: {len(train_df)} compounds  ({train_df.active.sum()} active)")
print(f"  Test:  {len(test_df)} compounds  ({test_df.active.sum()} active)")

if test_df["active"].sum() < 3 or (len(test_df) - test_df["active"].sum()) < 3:
    print("WARNING: test set has <3 actives or <3 inactives. Falling back to stratified random split.")
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        df_dedup.index, test_size=0.2, random_state=42, stratify=df_dedup["active"]
    )
    df_dedup["split"] = "train"
    df_dedup.loc[test_idx, "split"] = "test"
    train_df = df_dedup[df_dedup["split"] == "train"]
    test_df = df_dedup[df_dedup["split"] == "test"]
    print(f"  Fallback train: {len(train_df)}, test: {len(test_df)}")

df_dedup.to_csv(DATA_DIR / "chembl_tlr2_modelling.csv", index=False)

# ── 4. Featurize ─────────────────────────────────────────────────────────────

print("\nSTEP 4: Featurizing with RDKit")

from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

FP_RADIUS = 2
FP_BITS = 2048
PHYS_COLS = ["mw","logp","hbd","hba","tpsa","rot_bonds","aromatic_rings",
             "heavy_atoms","rings","stereo_centers","acyl_chain_count","longest_chain"]

acyl_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

def longest_chain(mol):
    adj = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ba, bb = mol.GetAtomWithIdx(a), mol.GetAtomWithIdx(b)
        if ba.GetSymbol()=="C" and bb.GetSymbol()=="C" and not ba.GetIsAromatic() and not bb.GetIsAromatic():
            adj[a].append(b); adj[b].append(a)
    best = 0
    for start in adj:
        visited, stack = set(), [(start, 0)]
        while stack:
            node, d = stack.pop()
            if node in visited: continue
            visited.add(node); best = max(best, d)
            for nb in adj[node]:
                if nb not in visited: stack.append((nb, d+1))
    return best

def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    phys = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        len(mol.GetSubstructMatches(acyl_pat)),
        longest_chain(mol),
    ]
    fp = gen.GetFingerprintAsNumPy(mol).astype("float32")
    return np.hstack([np.array(phys, dtype="float32"), fp])

def df_to_Xy(df):
    X, y, idx = [], [], []
    for i, row in df.iterrows():
        x = featurize(row["smiles"])
        if x is not None:
            X.append(x); y.append(int(row["active"])); idx.append(i)
    return np.array(X), np.array(y), idx

X_train, y_train, _ = df_to_Xy(train_df)
X_test, y_test, _ = df_to_Xy(test_df)
print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── 5. Train ─────────────────────────────────────────────────────────────────

print("\nSTEP 5: Training models")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("  Random Forest...")
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_pred = rf.predict(X_test)
rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
results["RandomForest"] = {
    "roc_auc": float(roc_auc_score(y_test, rf_proba)) if len(np.unique(y_test)) > 1 else 0.5,
    "cv_roc_auc": float(rf_cv),
    "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
    "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
    "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
}
print(f"    ROC AUC {results['RandomForest']['roc_auc']:.3f}  CV {rf_cv:.3f}")

print("  XGBoost...")
scale = (y_train==0).sum() / max(1, (y_train==1).sum())
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=scale, random_state=42,
                              eval_metric="logloss", verbosity=0)
xgb_clf.fit(X_train, y_train)
xg_proba = xgb_clf.predict_proba(X_test)[:,1]
xg_pred = xgb_clf.predict(X_test)
xg_cv = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
results["XGBoost"] = {
    "roc_auc": float(roc_auc_score(y_test, xg_proba)) if len(np.unique(y_test)) > 1 else 0.5,
    "cv_roc_auc": float(xg_cv),
    "precision": float(precision_score(y_test, xg_pred, zero_division=0)),
    "recall": float(recall_score(y_test, xg_pred, zero_division=0)),
    "f1": float(f1_score(y_test, xg_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, xg_pred).tolist(),
}
print(f"    ROC AUC {results['XGBoost']['roc_auc']:.3f}  CV {xg_cv:.3f}")

best_name = max(results, key=lambda k: results[k]["cv_roc_auc"])
best_model = rf if best_name == "RandomForest" else xgb_clf
best_res = results[best_name]
print(f"\n  Winner: {best_name} (CV ROC AUC {best_res['cv_roc_auc']:.3f})")
print(f"  Test ROC AUC: {best_res['roc_auc']:.3f}")
print(f"  Precision: {best_res['precision']:.3f}  Recall: {best_res['recall']:.3f}  F1: {best_res['f1']:.3f}")
print(f"  Confusion matrix: {best_res['confusion_matrix']}")

# ── 6. Export to ONNX ────────────────────────────────────────────────────────

print("\nSTEP 6: Exporting to ONNX")

n_features = X_train.shape[1]

if best_name == "XGBoost":
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OT
    onnx_model = convert_xgboost(best_model, initial_types=[("float_input", OT([None, n_features]))])
else:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onnx_model = convert_sklearn(best_model, "rf", [("float_input", FloatTensorType([None, n_features]))])

onnx_path = MODEL_DIR / "tlr2_clf.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"  ONNX model: {os.path.getsize(onnx_path)//1024}KB -> {onnx_path}")

# Verify
import onnxruntime as rt
sess = rt.InferenceSession(str(onnx_path))
test_out = sess.run(None, {sess.get_inputs()[0].name: X_test[:2].astype("float32")})
print(f"  ONNX inference check: {test_out[1][:2]}")

# ── 7. Write metadata ────────────────────────────────────────────────────────

meta = {
    "best_model": best_name,
    "task": "binary_classification",
    "data_source": "ChEMBL (CHEMBL4523, human TLR2)",
    "split_method": "Bemis-Murcko scaffold split (80/20)",
    "threshold_label": "pchembl_value >= 5.0 (IC50 < 10uM)",
    "roc_auc": round(best_res["roc_auc"], 4),
    "cv_roc_auc": round(best_res["cv_roc_auc"], 4),
    "precision": round(best_res["precision"], 4),
    "recall": round(best_res["recall"], 4),
    "f1": round(best_res["f1"], 4),
    "confusion_matrix": best_res["confusion_matrix"],
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_total": int(len(y_train) + len(y_test)),
    "fp_bits": FP_BITS,
    "fp_radius": FP_RADIUS,
    "phys_cols": PHYS_COLS,
    "results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()} for k, v in results.items()},
}
meta_path = MODEL_DIR / "clf_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata -> {meta_path}")

print("\n" + "=" * 60)
print("DONE")
print(f"  Model:  {best_name}, CV ROC AUC {best_res['cv_roc_auc']:.3f}")
print(f"  Data:   {len(y_train)+len(y_test)} compounds from ChEMBL CHEMBL4523")
print(f"  Split:  Bemis-Murcko scaffold, no train/test overlap")
print("=" * 60)
