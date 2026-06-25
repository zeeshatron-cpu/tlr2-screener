"""
Fetch TLR2 binding data from ChEMBL.
Target: CHEMBL2364 (TLR2, human)
Pulls IC50/Ki from binding assays, saves raw CSV.
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_CHEMBL_ID = "CHEMBL2364"
OUTPUT = "data/chembl_tlr2_raw.csv"


def fetch():
    activity = new_client.activity
    print(f"Fetching binding data for {TARGET_CHEMBL_ID}...")

    res = activity.filter(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type__in=["IC50", "Ki", "Kd", "EC50"],
        assay_type="B",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "pchembl_value",
        "assay_chembl_id",
        "target_chembl_id",
    ])

    rows = []
    for r in res:
        smiles = r.get("canonical_smiles")
        val = r.get("standard_value")
        pval = r.get("pchembl_value")
        if not smiles or (val is None and pval is None):
            continue
        rows.append({
            "chembl_id": r.get("molecule_chembl_id"),
            "smiles": smiles,
            "standard_type": r.get("standard_type"),
            "standard_value": val,
            "standard_units": r.get("standard_units"),
            "standard_relation": r.get("standard_relation"),
            "pchembl_value": pval,
            "assay_chembl_id": r.get("assay_chembl_id"),
        })

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} activity records")
    df.to_csv(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")
    return df


if __name__ == "__main__":
    fetch()
