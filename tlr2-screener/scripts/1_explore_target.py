"""
STEP 1: Find the human TLR2 target in ChEMBL and inventory its assays.

Human TLR2 UniProt accession = O60603 (this is the robust anchor,
NOT a hardcoded CHEMBL id, because target ids can be ambiguous).

Run: python 1_explore_target.py
Output: prints target ids, then an assay-type breakdown so YOU decide
        which assays are comparable before pulling activities.

Exact REST queries are printed so you can paste them into a browser.
"""

import requests
from collections import Counter

BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT = "O60603"  # human TLR2


def get_json(url, params=None):
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def find_targets():
    # Query: targets whose protein component has UniProt accession O60603
    url = f"{BASE}/target.json"
    params = {"target_components__accession": UNIPROT, "limit": 100}
    print("QUERY:", requests.Request("GET", url, params=params).prepare().url)
    data = get_json(url, params)
    targets = data.get("targets", [])
    print(f"\nFound {len(targets)} target record(s) for UniProt {UNIPROT}:\n")
    ids = []
    for t in targets:
        tid = t["target_chembl_id"]
        ids.append(tid)
        print(f"  {tid}  | {t['target_type']:20s} | {t['organism']} | {t['pref_name']}")
    return ids


def inventory_activities(target_ids):
    """For each target id, page through activities and summarize what exists."""
    for tid in target_ids:
        print("\n" + "=" * 70)
        print(f"ACTIVITY INVENTORY for {tid}")
        url = f"{BASE}/activity.json"
        params = {"target_chembl_id": tid, "limit": 1000}
        print("QUERY:", requests.Request("GET", url, params=params).prepare().url)

        types = Counter()
        units = Counter()
        assay_types = Counter()
        with_pchembl = 0
        total = 0
        next_url = None
        while True:
            data = get_json(next_url if next_url else url, None if next_url else params)
            acts = data.get("activities", [])
            for a in acts:
                total += 1
                types[a.get("standard_type")] += 1
                units[a.get("standard_units")] += 1
                assay_types[a.get("assay_type")] += 1
                if a.get("pchembl_value") is not None:
                    with_pchembl += 1
            meta = data.get("page_meta", {})
            nxt = meta.get("next")
            if not nxt:
                break
            next_url = "https://www.ebi.ac.uk" + nxt

        print(f"\n  total activity rows: {total}")
        print(f"  rows with pchembl_value: {with_pchembl}  <-- these are the usable ones")
        print("\n  standard_type breakdown:")
        for k, v in types.most_common():
            print(f"    {str(k):15s} {v}")
        print("\n  assay_type breakdown (B=binding, F=functional, A=ADMET):")
        for k, v in assay_types.most_common():
            print(f"    {str(k):15s} {v}")
        print("\n  standard_units breakdown:")
        for k, v in units.most_common(10):
            print(f"    {str(k):15s} {v}")


if __name__ == "__main__":
    ids = find_targets()
    if not ids:
        print("No targets found. Check the UniProt accession.")
    else:
        inventory_activities(ids)
    print("\nDONE. Decide which target id(s), standard_type(s), and assay_type(s)")
    print("you trust, then set them in 2_build_dataset.py.")
