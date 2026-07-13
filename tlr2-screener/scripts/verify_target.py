"""
Target verification: prove the ChEMBL id we train on is actually human TLR2.

This is the gate. Everything downstream is invalid if the pref_name that
comes back is not "Toll-like receptor 2" carrying UniProt accession O60603.

Runs three checks and prints clean, greppable output:
  1. What does CHEMBL4523 claim to be? (pref_name, type, components)
  2. What id(s) does human TLR2 (UniProt O60603) actually resolve to?
  3. For the resolved id, what is the assay-type / standard-type breakdown?

Run on GitHub Actions (unrestricted internet). The agent proxy blocks
www.ebi.ac.uk locally, so this cannot run in the dev container.
"""

import requests
from collections import Counter

BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "O60603"
SUSPECT_ID = "CHEMBL4523"


def get_json(url, params=None):
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def check_suspect_id():
    print("=" * 70)
    print(f"CHECK 1: what does {SUSPECT_ID} claim to be?")
    print("=" * 70)
    data = get_json(f"{BASE}/target/{SUSPECT_ID}.json")
    print(f"  RESULT_PREF_NAME: {data.get('pref_name')}")
    print(f"  RESULT_TYPE: {data.get('target_type')}")
    print(f"  RESULT_ORGANISM: {data.get('organism')}")
    comps = [(c.get("accession"), c.get("component_description"))
             for c in data.get("target_components", [])]
    print(f"  RESULT_COMPONENTS: {comps}")
    accessions = [c.get("accession") for c in data.get("target_components", [])]
    print(f"  RESULT_CARRIES_O60603: {UNIPROT in accessions}")
    return data.get("pref_name"), (UNIPROT in accessions)


def resolve_uniprot():
    print("\n" + "=" * 70)
    print(f"CHECK 2: what id(s) does UniProt {UNIPROT} resolve to?")
    print("=" * 70)
    data = get_json(f"{BASE}/target.json",
                    {"target_components__accession": UNIPROT, "limit": 100})
    ids = []
    for t in data.get("targets", []):
        tid = t["target_chembl_id"]
        ids.append((tid, t.get("target_type")))
        print(f"  RESOLVED: {tid} | {t.get('pref_name')} | {t.get('target_type')} | {t.get('organism')}")
    single = [tid for tid, ttype in ids if ttype == "SINGLE PROTEIN"]
    print(f"  RESULT_SINGLE_PROTEIN_IDS: {single}")
    return single


def assay_breakdown(target_id):
    print("\n" + "=" * 70)
    print(f"CHECK 3: assay-type breakdown for {target_id}")
    print("=" * 70)
    url = f"{BASE}/activity.json"
    params = {"target_chembl_id": target_id, "limit": 1000}
    types = Counter()
    assay_types = Counter()
    units = Counter()
    with_pchembl = 0
    total = 0
    next_url = None
    while True:
        data = get_json(next_url if next_url else url, None if next_url else params)
        for a in data.get("activities", []):
            total += 1
            types[a.get("standard_type")] += 1
            assay_types[a.get("assay_type")] += 1
            units[a.get("standard_units")] += 1
            if a.get("pchembl_value") is not None:
                with_pchembl += 1
        nxt = data.get("page_meta", {}).get("next")
        if not nxt:
            break
        next_url = "https://www.ebi.ac.uk" + nxt
    print(f"  RESULT_TOTAL_ROWS: {total}")
    print(f"  RESULT_ROWS_WITH_PCHEMBL: {with_pchembl}")
    print("  RESULT_STANDARD_TYPE:")
    for k, v in types.most_common():
        print(f"    {str(k):15s} {v}")
    print("  RESULT_ASSAY_TYPE (B=binding, F=functional, A=ADMET):")
    for k, v in assay_types.most_common():
        print(f"    {str(k):15s} {v}")
    print("  RESULT_UNITS:")
    for k, v in units.most_common(10):
        print(f"    {str(k):15s} {v}")


def main():
    pref_name, carries = check_suspect_id()
    single_ids = resolve_uniprot()

    for tid in single_ids:
        assay_breakdown(tid)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    is_tlr2 = pref_name and "toll-like receptor 2" in pref_name.lower()
    print(f"  {SUSPECT_ID} pref_name is TLR2: {is_tlr2}")
    print(f"  {SUSPECT_ID} carries {UNIPROT}: {carries}")
    print(f"  {UNIPROT} single-protein ids: {single_ids}")
    if is_tlr2 and carries:
        print("  GATE: PASS -- suspect id is genuinely human TLR2")
    else:
        print("  GATE: FAIL -- retrain from the resolved single-protein id above")


if __name__ == "__main__":
    main()
