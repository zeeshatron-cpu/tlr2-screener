"""
Validate the trained model against Pam3Cys-SNFKK.
Reports pIC50 prediction, converts to IC50, compares to known agonists.
"""

import json
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

MODEL = "model/tlr2_model.pkl"
META = "model/model_meta.json"

COMPOUNDS = {
    "Pam3Cys-SNFKK": "CCCCCCCCCCCCCCCC(=O)NCCSC[C@@H](NC(=O)CCCCCCCCCCCCCCC)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)O",
    "Pam3CSK4": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC",
    "FSL-1": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)NC(=O)[C@@H](N)CC(C)C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def featurize(smiles, meta):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    phys_cols = meta["phys_cols"]
    fp_bits = meta["fp_bits"]
    fp_radius = meta["fp_radius"]

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
        "acyl_chain_count": _count_acyl(mol),
        "longest_chain": _longest_chain(mol),
    }
    X_phys = np.array([phys[c] for c in phys_cols], dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
    X_fp = np.array(fp, dtype=np.float32)

    return np.hstack([X_phys, X_fp]).reshape(1, -1)


def _count_acyl(mol):
    pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")
    return len(mol.GetSubstructMatches(pat))


def _longest_chain(mol):
    adj = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ba, bb = mol.GetAtomWithIdx(a), mol.GetAtomWithIdx(b)
        if ba.GetSymbol() == 'C' and bb.GetSymbol() == 'C' and not ba.GetIsAromatic() and not bb.GetIsAromatic():
            adj[a].append(b)
            adj[b].append(a)
    longest = 0
    for start in adj:
        visited, stack = set(), [(start, 0)]
        while stack:
            node, d = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            longest = max(longest, d)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append((nb, d + 1))
    return longest


def main():
    model = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)

    print(f"Model: {meta['best_model']} | Test R2: {meta['test_r2']:.4f} | Trained on {meta['n_total']} compounds\n")
    print(f"{'Compound':<25} {'pIC50':>8} {'IC50 (nM)':>12} {'Class':>8}")
    print("-" * 58)

    for name, smiles in COMPOUNDS.items():
        try:
            X = featurize(smiles, meta)
            pic50 = float(model.predict(X)[0])
            ic50_nm = 10 ** (-pic50 + 9)
            verdict = "HIGH" if pic50 >= 7 else "MED" if pic50 >= 5 else "LOW"
            print(f"{name:<25} {pic50:>8.2f} {ic50_nm:>12.1f} {verdict:>8}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")


if __name__ == "__main__":
    main()
