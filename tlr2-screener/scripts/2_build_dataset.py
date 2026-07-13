"""
STEP 2: Pull real TLR2 activities, standardize, deduplicate.

Fixes the four leaks you diagnosed:
  - duplicate SMILES with conflicting labels -> collapsed to ONE row by median pchembl
  - interpolated/constructed values -> not present, this only takes measured ChEMBL rows
  - wrong-target "inactives" -> only rows measured against the TLR2 target id are used
  - unit chaos -> standardized via ChEMBL's precomputed pchembl_value (-log10 molar)

Edit the CONFIG block after you run step 1 and see what actually exists.

Run: python 2_build_dataset.py
Output: tlr2_clean.csv  with columns [chembl_id, smiles, pchembl, active]
"""

import requests
import pandas as pd
from rdkit import Chem

BASE = "https://www.ebi.ac.uk/chembl/api/data"

# ---- CONFIG ----
# VERIFIED by scripts/verify_target.py: human TLR2 (UniProt O60603) is the
# SINGLE PROTEIN target CHEMBL4163. (CHEMBL4523, used by the old pipeline, is
# actually pim-2 kinase — the wrong-target bug this pins down.)
# Pinned explicitly for reproducibility; leave [] to auto-discover from O60603.
TARGET_IDS = ["CHEMBL4163"]
UNIPROT = "O60603"
STANDARD_TYPES = ["IC50", "EC50"]  # keep only comparable assay readouts
ACTIVE_PCHEMBL_CUTOFF = 5.0        # pchembl >= 5.0  == IC50/EC50 <= 10 uM == active
# ----------------------------------------------


def get_json(url, params=None):
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def discover_targets():
    url = f"{BASE}/target.json"
    params = {"target_components__accession": UNIPROT, "limit": 100}
    data = get_json(url, params)
    ids = [t["target_chembl_id"] for t in data.get("targets", [])
           if t.get("target_type") == "SINGLE PROTEIN"]
    print(f"Discovered single-protein targets for {UNIPROT}: {ids}")
    return ids


def pull_activities(target_id):
    url = f"{BASE}/activity.json"
    params = {
        "target_chembl_id": target_id,
        "pchembl_value__isnull": "false",  # only rows with a usable standardized value
        "limit": 1000,
    }
    rows = []
    next_url = None
    while True:
        data = get_json(next_url if next_url else url, None if next_url else params)
        for a in data.get("activities", []):
            if a.get("standard_type") not in STANDARD_TYPES:
                continue
            smi = a.get("canonical_smiles")
            pch = a.get("pchembl_value")
            if smi is None or pch is None:
                continue
            rows.append({
                "chembl_id": a.get("molecule_chembl_id"),
                "smiles": smi,
                "pchembl": float(pch),
                "assay_type": a.get("assay_type"),
                "standard_type": a.get("standard_type"),
            })
        meta = data.get("page_meta", {})
        nxt = meta.get("next")
        if not nxt:
            break
        next_url = "https://www.ebi.ac.uk" + nxt
    return rows


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def main():
    target_ids = TARGET_IDS if TARGET_IDS else discover_targets()
    if not target_ids:
        raise RuntimeError("No single-protein TLR2 targets found for UniProt O60603")

    all_rows = []
    for tid in target_ids:
        print(f"Pulling activities for {tid} ...")
        all_rows.extend(pull_activities(tid))
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows pulled. Recheck TARGET_IDS / STANDARD_TYPES from step 1.")
        return
    print(f"\nraw usable rows: {len(df)}")

    df["smiles"] = df["smiles"].apply(canonical)
    df = df.dropna(subset=["smiles"])
    print(f"rows after RDKit parse: {len(df)}")

    # DEDUP: one row per unique structure, activity = median of measurements
    grp = df.groupby("smiles").agg(
        chembl_id=("chembl_id", "first"),
        pchembl=("pchembl", "median"),
        n_measurements=("pchembl", "size"),
        pchembl_spread=("pchembl", lambda x: round(x.max() - x.min(), 2)),
    ).reset_index()
    print(f"unique structures after dedup: {len(grp)}")

    contradictory = grp[grp["pchembl_spread"] >= 2.0]
    if len(contradictory):
        print(f"\nWARNING: {len(contradictory)} structures have >=2 log unit spread")
        print("across measurements. Inspect these, they may be assay-format mismatches.")

    grp["active"] = (grp["pchembl"] >= ACTIVE_PCHEMBL_CUTOFF).astype(int)
    n_act = int(grp["active"].sum())
    print(f"\nfinal dataset: {len(grp)} compounds | {n_act} active | {len(grp)-n_act} inactive")
    print(f"class balance: {n_act/len(grp):.1%} active")

    out = grp[["chembl_id", "smiles", "pchembl", "n_measurements", "pchembl_spread", "active"]]
    out.to_csv("tlr2_clean.csv", index=False)
    print("\nwrote tlr2_clean.csv")
    if len(grp) < 40 or n_act / len(grp) > 0.95:
        print("WARNING: dataset is very small or severely imbalanced.")
        print("Say so plainly in any paper. Do not backfill with constructed values.")


if __name__ == "__main__":
    main()
