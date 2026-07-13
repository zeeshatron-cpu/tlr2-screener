"""
STEP 2b: Generate property-matched decoys from ChEMBL.

TLR2 ChEMBL data is ~99% active — not because TLR2 is promiscuous, but
because negative results aren't systematically deposited. To train a model
that can distinguish active from inactive we need a negative class.

Strategy: property-matched ChEMBL decoys
  - Pull compounds measured against metabolic enzyme targets (CYP3A4, DHFR)
    that have NO recorded TLR1/2/4/6 activity in ChEMBL
  - Filter to MW and LogP ranges that overlap the TLR2 active set
  - Randomly sample to 1:1 active:decoy ratio (or 1:2 if data allows)
  - Label all decoys class 0 and disclose this clearly

Disclosure: these are computational decoys, not experimentally confirmed
TLR2 inactives. Say so explicitly in any paper. The model learns to
separate TLR2-active chemistry from drug-like chemistry in general, which
is the correct first-pass filter for a virtual screen.

Run: python 2b_generate_decoys.py
Needs: tlr2_clean.csv (from step 2)
Output: tlr2_with_decoys.csv
"""

import requests
import random
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Silence RDKit's per-molecule warnings so they don't flood CI logs.
RDLogger.DisableLog('rdApp.*')

BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Targets used as decoy sources — well-studied, structurally diverse,
# no overlap with TLR signalling pathway
DECOY_SOURCE_TARGETS = [
    "CHEMBL340",   # CYP3A4 (metabolic enzyme)
    "CHEMBL205",   # DHFR (folate pathway)
    "CHEMBL2971",  # Thymidylate synthase
    "CHEMBL1827",  # Acetylcholinesterase
]

# ChEMBL IDs for TLR2 — used to exclude any TLR2-active compound from decoys.
# VERIFIED against UniProt O60603 by scripts/verify_target.py:
#   CHEMBL4163     = Toll-like receptor 2 (SINGLE PROTEIN)  <- the real one
#   CHEMBL3301399  = TLR2/TLR6 (PROTEIN COMPLEX)
#   CHEMBL3885643  = Toll-like receptor 1/2 (PROTEIN COMPLEX)
# NOTE: CHEMBL4523 was previously (wrongly) listed here as "TLR2"; it is
# actually Serine/threonine-protein kinase pim-2. Do not re-add it.
TLR_TARGETS = {
    "CHEMBL4163",     # TLR2 (single protein)
    "CHEMBL3301399",  # TLR2/TLR6 complex
    "CHEMBL3885643",  # TLR1/2 complex
}

RANDOM_SEED = 42


def get_json(url, params=None):
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def pull_smiles_for_target(target_id, limit=500):
    url = f"{BASE}/activity.json"
    params = {"target_chembl_id": target_id, "pchembl_value__isnull": "false", "limit": 1000}
    smiles_set = set()
    next_url = None
    while len(smiles_set) < limit:
        data = get_json(next_url if next_url else url, None if next_url else params)
        for a in data.get("activities", []):
            smi = a.get("canonical_smiles")
            if smi:
                smiles_set.add(smi)
        meta = data.get("page_meta", {})
        nxt = meta.get("next")
        if not nxt:
            break
        next_url = "https://www.ebi.ac.uk" + nxt
    return smiles_set


def get_tlr_compound_ids():
    """Return set of molecule_chembl_ids with any TLR activity."""
    ids = set()
    for tid in TLR_TARGETS:
        url = f"{BASE}/activity.json"
        params = {"target_chembl_id": tid, "limit": 1000}
        next_url = None
        while True:
            data = get_json(next_url if next_url else url, None if next_url else params)
            for a in data.get("activities", []):
                mol_id = a.get("molecule_chembl_id")
                if mol_id:
                    ids.add(mol_id)
            meta = data.get("page_meta", {})
            nxt = meta.get("next")
            if not nxt:
                break
            next_url = "https://www.ebi.ac.uk" + nxt
        print(f"  TLR exclusion: {tid} -> {len(ids)} total excluded compounds so far")
    return ids


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def phys_props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    return Descriptors.MolWt(m), Descriptors.MolLogP(m)


def main():
    actives = pd.read_csv("tlr2_clean.csv")
    n_active = int(actives["active"].sum())
    n_target_decoys = n_active  # 1:1 ratio

    # Property bounds from the active set (5th-95th percentile)
    active_mols = actives[actives["active"] == 1]
    mw_vals = []
    lp_vals = []
    for smi in active_mols["smiles"]:
        mw, lp = phys_props(smi)
        if mw is not None:
            mw_vals.append(mw)
            lp_vals.append(lp)
    mw_lo, mw_hi = sorted(mw_vals)[len(mw_vals)//20], sorted(mw_vals)[len(mw_vals)*19//20]
    lp_lo, lp_hi = sorted(lp_vals)[len(lp_vals)//20], sorted(lp_vals)[len(lp_vals)*19//20]
    print(f"Active property range: MW {mw_lo:.0f}-{mw_hi:.0f}, LogP {lp_lo:.1f}-{lp_hi:.1f}")

    print("\nBuilding TLR exclusion list...")
    tlr_ids = get_tlr_compound_ids()
    print(f"Excluding {len(tlr_ids)} compounds with any TLR activity")

    active_smiles = set(actives["smiles"].tolist())

    print("\nPulling decoy candidates from ChEMBL...")
    candidates = []
    for tid in DECOY_SOURCE_TARGETS:
        print(f"  Pulling from {tid} ...")
        smiles_set = pull_smiles_for_target(tid, limit=2000)
        candidates.extend(smiles_set)
    print(f"Raw candidates: {len(candidates)}")

    # Canonicalize, deduplicate, filter
    seen = set()
    filtered = []
    for smi in candidates:
        csmi = canonical(smi)
        if csmi is None or csmi in seen or csmi in active_smiles:
            continue
        seen.add(csmi)
        mw, lp = phys_props(csmi)
        if mw is None:
            continue
        if mw_lo <= mw <= mw_hi and lp_lo <= lp <= lp_hi:
            filtered.append(csmi)

    print(f"Property-matched candidates after filtering: {len(filtered)}")

    if len(filtered) < 50:
        print("WARNING: very few decoys found. Widening property window by 20%.")
        margin_mw = (mw_hi - mw_lo) * 0.1
        margin_lp = (lp_hi - lp_lo) * 0.1
        filtered = []
        for smi in seen:
            mw, lp = phys_props(smi)
            if mw is None:
                continue
            if (mw_lo - margin_mw) <= mw <= (mw_hi + margin_mw) and \
               (lp_lo - margin_lp) <= lp <= (lp_hi + margin_lp):
                filtered.append(smi)
        print(f"After widening: {len(filtered)}")

    random.seed(RANDOM_SEED)
    n_decoys = min(n_target_decoys, len(filtered))
    chosen = random.sample(filtered, n_decoys)
    print(f"Selected {n_decoys} decoys (target: {n_target_decoys})")

    decoy_rows = []
    for smi in chosen:
        mw, lp = phys_props(smi)
        decoy_rows.append({
            "chembl_id": "DECOY",
            "smiles": smi,
            "pchembl": 0.0,
            "n_measurements": 0,
            "pchembl_spread": 0.0,
            "active": 0,
        })

    decoys_df = pd.DataFrame(decoy_rows)
    combined = pd.concat([actives, decoys_df], ignore_index=True)
    combined.to_csv("tlr2_with_decoys.csv", index=False)

    n_act = int(combined["active"].sum())
    n_inact = len(combined) - n_act
    print(f"\nFinal dataset: {len(combined)} compounds | {n_act} active | {n_inact} inactive (decoys)")
    print(f"Class balance: {n_act/len(combined):.1%} active")
    print("\nDisclosure: inactive class = property-matched ChEMBL decoys with no TLR activity.")
    print("State this in any publication. This is NOT a set of experimentally confirmed TLR2 inactives.")
    print("\nwrote tlr2_with_decoys.csv")


if __name__ == "__main__":
    main()
