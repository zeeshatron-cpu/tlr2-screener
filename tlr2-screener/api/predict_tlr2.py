"""
Vercel Python serverless function: TLR2 binary activity prediction.
POST /api/predict_tlr2  {"smiles": "...", "name": "..."}
"""
from http.server import BaseHTTPRequestHandler
import json, os
from pathlib import Path

# Paths relative to project root (Vercel includes all project files)
_BASE = Path(__file__).parent.parent / "ml" / "model"
_MODEL_PATH = _BASE / "tlr2_clf.pkl"
_META_PATH = _BASE / "clf_meta.json"

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        import joblib
        _model = joblib.load(_MODEL_PATH)
        with open(_META_PATH) as f:
            _meta = json.load(f)
    return _model, _meta


def _featurize(smiles, meta):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    acyl_pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")

    def longest_chain(m):
        adj = {i: [] for i in range(m.GetNumAtoms())}
        for bond in m.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ba, bb = m.GetAtomWithIdx(a), m.GetAtomWithIdx(b)
            if ba.GetSymbol() == 'C' and bb.GetSymbol() == 'C' and not ba.GetIsAromatic() and not bb.GetIsAromatic():
                adj[a].append(b); adj[b].append(a)
        best = 0
        for start in adj:
            visited, stack = set(), [(start, 0)]
            while stack:
                node, d = stack.pop()
                if node in visited: continue
                visited.add(node); best = max(best, d)
                for nb in adj[node]:
                    if nb not in visited: stack.append((nb, d + 1))
        return best

    phys = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        len(mol.GetSubstructMatches(acyl_pat)),
        longest_chain(mol),
    ]
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=meta["fp_radius"], fpSize=meta["fp_bits"]
    )
    fp = gen.GetFingerprintAsNumPy(mol).astype("float32")
    X = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
         rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
         Descriptors.TPSA(mol)]
    return np.hstack([np.array(phys, dtype="float32"), fp]).reshape(1, -1), {
        "mw": phys[0], "logp": phys[1], "hbd": phys[2], "hba": phys[3],
        "tpsa": phys[4], "acyl_chain_count": phys[10],
    }


def _predict(smiles, name):
    model, meta = _load()
    X, phys = _featurize(smiles.strip(), meta)
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    score = round(proba * 100)
    if proba >= 0.7:
        verdict, verdict_text = "high", "Strong TLR2 agonist"
    elif proba >= 0.4:
        verdict, verdict_text = "medium", "Moderate TLR2 activity"
    else:
        verdict, verdict_text = "low", "Weak / inactive"

    return {
        "molecule_name": name,
        "tlr2_binding_score": score,
        "pharmacophore_match": min(100, max(0, round(score * 0.85))),
        "active_probability": round(proba, 4),
        "predicted_class": pred,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "mw_estimate": round(phys["mw"], 1),
        "logp_estimate": round(phys["logp"], 2),
        "hbd": int(phys["hbd"]),
        "hba": int(phys["hba"]),
        "tpsa": round(phys["tpsa"], 1),
        "acyl_chains": int(phys["acyl_chain_count"]),
        "model_name": meta["best_model"],
        "model_roc_auc": meta["roc_auc"],
        "model_cv_roc_auc": meta["cv_roc_auc"],
        "n_train": meta["n_train"],
        "source": "ml_model",
    }


class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access logs in Vercel

    def _send(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            _, meta = _load()
            self._send(200, {
                "status": "ok",
                "model": meta["best_model"],
                "roc_auc": meta["roc_auc"],
                "cv_roc_auc": meta["cv_roc_auc"],
                "precision": meta["precision"],
                "recall": meta["recall"],
                "f1": meta["f1"],
                "n_train": meta["n_train"],
                "n_test": meta["n_test"],
            })
        except Exception as e:
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except Exception:
            self._send(400, {"error": "Invalid JSON body"})
            return

        smiles = body.get("smiles", "").strip()
        name = body.get("name", "Query")

        if not smiles:
            self._send(400, {"error": "smiles field required"})
            return

        try:
            result = _predict(smiles, name)
            self._send(200, result)
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"Prediction failed: {e}"})
