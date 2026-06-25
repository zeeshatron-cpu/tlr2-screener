"""
Compute molecular features for TLR2 binding model.

Reads either curated (has pIC50 col) or ChEMBL raw format.
Outputs feature matrix CSV with Morgan fingerprints + physicochemical descriptors.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

RAW = "data/chembl_tlr2_raw.csv"
OUTPUT = "data/chembl_tlr2_features.csv"

FP_RADIUS = 2
FP_BITS = 2048


def compute_physchem(mol):
    return {
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
    }


def count_acyl_chains(mol):
    pat = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")
    return len(mol.GetSubstructMatches(pat))


def longest_carbon_chain(mol):
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


_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

def morgan_fp(mol):
    return np.array(_morgan_gen.GetFingerprintAsNumPy(mol))


def to_pic50(row):
    # Curated dataset already has pIC50
    if "pIC50" in row and not pd.isna(row.get("pIC50")):
        return float(row["pIC50"])
    # ChEMBL format
    if "pchembl_value" in row and not pd.isna(row.get("pchembl_value")):
        return float(row["pchembl_value"])
    val = row.get("standard_value")
    units = str(row.get("standard_units", "")).lower()
    if val is None or pd.isna(val):
        return None
    try:
        val = float(val)
    except (ValueError, TypeError):
        return None
    if "nm" in units:
        val_m = val * 1e-9
    elif "um" in units or "µm" in units:
        val_m = val * 1e-6
    elif "mm" in units:
        val_m = val * 1e-3
    else:
        val_m = val * 1e-9
    if val_m <= 0:
        return None
    return -np.log10(val_m)


def prepare(df=None):
    if df is None:
        df = pd.read_csv(RAW)

    print(f"Input: {len(df)} records")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        smiles = row.get("smiles")
        if not smiles or pd.isna(smiles):
            skipped += 1
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            skipped += 1
            continue

        pic50 = to_pic50(row)
        if pic50 is None or pic50 < 0 or pic50 > 15:
            skipped += 1
            continue

        phys = compute_physchem(mol)
        fp = morgan_fp(mol)

        rec = {
            "name": row.get("name", row.get("chembl_id", "")),
            "smiles": smiles,
            "pIC50": pic50,
            "compound_class": row.get("class", ""),
            **phys,
            "acyl_chain_count": count_acyl_chains(mol),
            "longest_chain": longest_carbon_chain(mol),
        }
        for i, bit in enumerate(fp):
            rec[f"fp_{i}"] = int(bit)

        records.append(rec)

    print(f"Skipped: {skipped} (invalid SMILES or missing pIC50)")
    out = pd.DataFrame(records)
    print(f"Valid compounds: {len(out)}")

    # Deduplicate by SMILES - keep the first occurrence (curated data is already unique)
    out = out.drop_duplicates(subset="smiles", keep="first")
    print(f"After dedup: {len(out)} unique SMILES")

    out.to_csv(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")
    return out


if __name__ == "__main__":
    prepare()
