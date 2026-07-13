"""
STEP 3: Featurize, scaffold-split, train, report honest metrics.

Fixes the "random split on a congeneric series" leak by using a
Bemis-Murcko scaffold split: whole scaffold groups go entirely to
train OR test, so near-identical lipopeptides can never straddle both.

Expect the test AUC to DROP versus a random split. That drop is the point.
A ~0.65-0.80 scaffold-split AUC on real deduped data is the honest number
and is publishable. A high number from a random split on congeneric data
is an artifact.

Run: python 3_train_scaffold_split.py
Needs: tlr2_clean.csv from step 2
Output: prints metrics, writes tlr2_model.json (XGBoost) + feature_meta.json
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

# Silence RDKit's per-molecule deprecation/parse warnings; they flood CI logs
# and bury the metrics we actually need to read back.
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

RANDOM_STATE = 42
TEST_FRACTION = 0.2
MORGAN_BITS = 2048
MORGAN_RADIUS = 2


# ---- featurization ----
# Feature order: Morgan fingerprint (2048) then 12 physicochemical descriptors.
# IMPORTANT: predict_tlr2.py must use this exact same ordering.
def descriptors(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Lipinski.NumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        Lipinski.RingCount(mol),
        len(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)),
        sum(1 for b in mol.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE),
        _longest_carbon_chain(mol),
    ]


def _longest_carbon_chain(mol):
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbons:
        return 0
    best = 0
    cset = set(carbons)
    adj = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in cset and j in cset:
            adj[i].append(j)
            adj[j].append(i)

    def dfs(node, seen):
        m = 1
        for nb in adj[node]:
            if nb not in seen:
                m = max(m, 1 + dfs(nb, seen | {nb}))
        return m

    for c in carbons:
        best = max(best, dfs(c, {c}))
    return best


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp_arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    # fp first, then descriptors — must match predict_tlr2.py exactly
    return np.concatenate([fp_arr, np.array(descriptors(mol), dtype=np.float32)])


# ---- scaffold split ----
def scaffold_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def scaffold_split(df, test_frac, seed):
    groups = defaultdict(list)
    for idx, smi in zip(df.index, df["smiles"]):
        sc = scaffold_of(smi)
        groups[sc if sc else f"_none_{idx}"].append(idx)
    ordered = sorted(groups.values(), key=len, reverse=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(ordered)
    n_test_target = int(len(df) * test_frac)
    test_idx, train_idx = [], []
    for grp in ordered:
        if len(test_idx) < n_test_target:
            test_idx.extend(grp)
        else:
            train_idx.extend(grp)
    return train_idx, test_idx


def main():
    import os
    csv_path = "tlr2_with_decoys.csv" if os.path.exists("tlr2_with_decoys.csv") else "tlr2_clean.csv"
    print(f"Loading from {csv_path}")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    n_act = int(df["active"].sum()) if "active" in df.columns else "?"
    print(f"loaded {len(df)} compounds ({n_act} active, {len(df)-n_act} inactive)")

    X, y, keep = [], [], []
    for i, row in df.iterrows():
        f = featurize(row["smiles"])
        if f is not None:
            X.append(f)
            y.append(int(row["active"]))
            keep.append(i)
    X = np.vstack(X).astype(np.float32)
    y = np.array(y)
    df = df.loc[keep].reset_index(drop=True)
    print(f"featurized {len(df)} | features per compound: {X.shape[1]}")

    train_idx, test_idx = scaffold_split(df, TEST_FRACTION, RANDOM_STATE)
    pos = {orig: k for k, orig in enumerate(df.index)}
    tr = [pos[i] for i in train_idx if i in pos]
    te = [pos[i] for i in test_idx if i in pos]
    n_train_active = int(y[tr].sum())
    n_test_active = int(y[te].sum())
    print(f"scaffold split -> train {len(tr)} ({n_train_active} active) | test {len(te)} ({n_test_active} active)")
    print(f"unique scaffolds total: {len(set(scaffold_of(s) for s in df['smiles']))}")

    if len(te) < 5 or len(set(y[te])) < 2:
        print("\nWARNING: test set too small or single-class after scaffold split.")
        print("With sparse data, report LEAVE-ONE-SCAFFOLD-OUT cross-val instead of")
        print("a single split. Say this plainly in the paper.")

    # scale_pos_weight compensates for class imbalance
    n_neg = int((y[tr] == 0).sum())
    n_pos = int((y[tr] == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nclass imbalance in train: {n_pos} active / {n_neg} inactive | scale_pos_weight={spw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=RANDOM_STATE,
        use_label_encoder=False,
    )
    model.fit(X[tr], y[tr])

    metrics = {}
    if len(te) >= 5 and len(set(y[te])) == 2:
        proba = model.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        roc = roc_auc_score(y[te], proba)
        prec = precision_score(y[te], pred, zero_division=0)
        rec = recall_score(y[te], pred, zero_division=0)
        f1 = f1_score(y[te], pred, zero_division=0)
        cm = confusion_matrix(y[te], pred).tolist()
        print("\n--- HONEST scaffold-split test metrics ---")
        print(f"ROC AUC   : {roc:.3f}")
        print(f"Precision : {prec:.3f}")
        print(f"Recall    : {rec:.3f}")
        print(f"F1        : {f1:.3f}")
        print(f"Confusion : {cm}")
        metrics = {
            "roc_auc": round(roc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
        }
    else:
        print("\nWARNING: Could not compute test metrics (single-class or too small test set).")
        print("Consider leave-one-scaffold-out cross-validation instead.")

    model.save_model("tlr2_model.json")

    meta = {
        "fp_bits": MORGAN_BITS,
        "fp_radius": MORGAN_RADIUS,
        "n_descriptors": 12,
        "total_features": int(X.shape[1]),
        "feature_order": "morgan_fp_first_then_descriptors",
        "n_train": len(tr),
        "n_test": len(te),
        "n_total": len(df),
        **metrics,
    }
    with open("feature_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nwrote tlr2_model.json + feature_meta.json")


if __name__ == "__main__":
    main()
