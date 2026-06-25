"""
/api/predict - TLR2 binding ML prediction endpoint.
Vercel Python serverless function (BaseHTTPRequestHandler).

POST {"smiles": "...", "name": "..."}
Returns {"pIC50": float, "IC50_nM": float, "tlr2_binding_score": int, ...}
"""

import json
import os
import numpy as np
from http.server import BaseHTTPRequestHandler


def _load_model():
    import joblib
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base, "ml", "model", "tlr2_model.pkl")
    meta_path = os.path.join(base, "ml", "model", "model_meta.json")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


_MODEL = None
_META = None


def _get_model():
    global _MODEL, _META
    if _MODEL is None:
        _MODEL, _META = _load_model()
    return _MODEL, _META


def _featurize(smiles, meta):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    phys_cols = meta["phys_cols"]
    fp_bits = meta["fp_bits"]
    fp_radius = meta["fp_radius"]

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

    X_phys = np.array([phys[c] for c in phys_cols], dtype=np.float32)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    X_fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)

    return np.hstack([X_phys, X_fp]).reshape(1, -1), phys


def _predict(smiles, name):
    model, meta = _get_model()
    X, phys = _featurize(smiles, meta)
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
        "molecule_name": name,
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


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._respond(400, {"error": "Invalid request body"})
            return

        smiles = (body.get("smiles") or "").strip()
        name = body.get("name") or "Query"

        if not smiles:
            self._respond(400, {"error": "SMILES required"})
            return

        try:
            result = _predict(smiles, name)
            self._respond(200, result)
        except Exception as e:
            self._respond(500, {"error": str(e), "source": "ml_model"})

    def do_GET(self):
        self._respond(200, {"status": "ok", "endpoint": "/api/predict", "method": "POST"})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
