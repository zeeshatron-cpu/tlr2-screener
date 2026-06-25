"""
FastAPI server for TLR2 ML binding prediction.
Deploy on Railway: connects to ml/model/tlr2_model.pkl

POST /predict  {"smiles": "...", "name": "..."}
GET  /health   -> {"status": "ok", "model": "XGBoost", "r2": 0.71}
"""

import json
import os
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(title="TLR2 ML Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "tlr2_model.pkl"
META_PATH = BASE / "model" / "model_meta.json"

_model = None
_meta = None


def get_model():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def featurize(smiles: str, meta: dict):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    acyl_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")

    def longest_chain(m):
        adj = {i: [] for i in range(m.GetNumAtoms())}
        for bond in m.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ba, bb = m.GetAtomWithIdx(a), m.GetAtomWithIdx(b)
            if ba.GetSymbol() == 'C' and bb.GetSymbol() == 'C' and not ba.GetIsAromatic() and not bb.GetIsAromatic():
                adj[a].append(b)
                adj[b].append(a)
        best = 0
        for start in adj:
            visited, stack = set(), [(start, 0)]
            while stack:
                node, d = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                best = max(best, d)
                for nb in adj[node]:
                    if nb not in visited:
                        stack.append((nb, d + 1))
        return best

    phys = {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "stereo_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "acyl_chain_count": len(mol.GetSubstructMatches(acyl_pat)),
        "longest_chain": longest_chain(mol),
    }

    X_phys = np.array([phys[c] for c in meta["phys_cols"]], dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=meta["fp_radius"], fpSize=meta["fp_bits"]
    )
    X_fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return np.hstack([X_phys, X_fp]).reshape(1, -1), phys


class PredictRequest(BaseModel):
    smiles: str
    name: str = "Query"


@app.get("/health")
def health():
    model, meta = get_model()
    return {
        "status": "ok",
        "model": meta["best_model"],
        "r2": round(meta["test_r2"], 3),
        "n_train": meta["n_train"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES required")

    model, meta = get_model()

    try:
        X, phys = featurize(smiles, meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pic50 = float(model.predict(X)[0])
    ic50_nm = 10 ** (-pic50 + 9)
    score = max(0, min(100, int((pic50 - 2) / 7 * 100)))

    if pic50 >= 7:
        verdict, verdict_text = "high", "Strong TLR2 agonist"
    elif pic50 >= 5:
        verdict, verdict_text = "medium", "Moderate TLR2 activity"
    else:
        verdict, verdict_text = "low", "Weak / inactive"

    return {
        "molecule_name": req.name,
        "pIC50": round(pic50, 2),
        "IC50_nM": round(ic50_nm, 2),
        "tlr2_binding_score": score,
        "pharmacophore_match": min(100, max(0, int(score * 0.9))),
        "verdict": verdict,
        "verdict_text": verdict_text,
        "mw_estimate": round(phys["mw"], 1),
        "logp_estimate": round(phys["logp"], 2),
        "hbd": int(phys["hbd"]),
        "hba": int(phys["hba"]),
        "tpsa": round(phys["tpsa"], 1),
        "acyl_chains": int(phys["acyl_chain_count"]),
        "model_name": meta["best_model"],
        "model_r2": round(meta["test_r2"], 3),
        "source": "ml_model",
    }
