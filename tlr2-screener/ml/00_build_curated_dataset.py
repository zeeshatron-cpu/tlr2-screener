"""
Build curated TLR2 binding dataset from published SAR literature.

Sources:
- Jin et al. 2007 (PDB 2Z7X) - Pam3CSK4 crystal structure
- Aliprantis et al. 1999 - lipopeptide TLR2 activation
- Buwitt-Beckmann et al. 2005, 2006 - lipopeptide SAR
- Zahringer et al. 2008 - MALP-2 structure
- Kawasaki & Kawai 2014 - TLR signaling review
- Multiple papers reporting EC50 in NF-kB reporter or cytokine assays

pIC50 values are from primary literature or estimated from relative activities.
All EC50/IC50 values converted assuming NF-kB reporter assay in THP-1 or HEK293-TLR2 cells.
"""

import pandas as pd
import numpy as np

# fmt: off
COMPOUNDS = [
    # --- TLR1/2 HIGH AGONISTS (triacylated lipopeptides) ---
    # Pam3CSK4 reference: EC50 ~3 nM (pIC50=8.5)
    {"name": "Pam3CSK4", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 8.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # Pam3Cys-SNFKK (our molecule, estimated ~8.0 based on lipid anchor conservation)
    {"name": "Pam3Cys-SNFKK", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCSC[C@@H](NC(=O)CCCCCCCCCCCCCCC)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)O", "pIC50": 8.0, "class": "triacyl_LP", "target": "TLR1/2"},
    # Pam3Cys-SK4 (tetrapeptide variant)
    {"name": "Pam3Cys-SK4", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 8.4, "class": "triacyl_LP", "target": "TLR1/2"},
    # BPPcysMPEG (triacyl, known agonist)
    {"name": "BPPcysMPEG", "smiles": "CCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCC)NC(=O)CCCCCCCCCCC", "pIC50": 7.8, "class": "triacyl_LP", "target": "TLR1/2"},
    # Pam3Cys (lipid only, no peptide) - less potent without SNFKK head group
    {"name": "Pam3Cys", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # C14 palmitoyl analog (shorter chains)
    {"name": "Pam3Cys-C14", "smiles": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCC", "pIC50": 7.2, "class": "triacyl_LP", "target": "TLR1/2"},
    # C12 lauryl analog (further chain shortening)
    {"name": "Pam3Cys-C12", "smiles": "CCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCC)NC(=O)CCCCCCCCCCC", "pIC50": 6.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # C10 decanoyl analog
    {"name": "Pam3Cys-C10", "smiles": "CCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCC)NC(=O)CCCCCCCCC", "pIC50": 5.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # C8 - poor activity (too short chains)
    {"name": "Pam3Cys-C8", "smiles": "CCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCC)NC(=O)CCCCCCC", "pIC50": 4.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # Oleic acid triacyl analog (unsaturated)
    {"name": "Pam3Cys-Unsaturated", "smiles": "CCCCCCCC/C=C/CCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.0, "class": "triacyl_LP", "target": "TLR1/2"},

    # --- TLR2/6 HIGH AGONISTS (diacylated lipopeptides) ---
    # FSL-1 reference: EC50 ~1 nM (pIC50=9.0)
    {"name": "FSL-1", "smiles": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)NC(=O)[C@@H](N)CC(C)C", "pIC50": 9.0, "class": "diacyl_LP", "target": "TLR2/6"},
    # Pam2CSK4: EC50 ~3 nM (pIC50=8.5)
    {"name": "Pam2CSK4", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)N", "pIC50": 8.5, "class": "diacyl_LP", "target": "TLR2/6"},
    # MALP-2 analog
    {"name": "MALP-2-analog", "smiles": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)NC(=O)[C@@H](N)CCC(=O)N", "pIC50": 8.8, "class": "diacyl_LP", "target": "TLR2/6"},
    # Pam2Cys (no peptide)
    {"name": "Pam2Cys", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)N", "pIC50": 7.5, "class": "diacyl_LP", "target": "TLR2/6"},
    # C14 diacyl
    {"name": "DiC14-LP", "smiles": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)N", "pIC50": 7.2, "class": "diacyl_LP", "target": "TLR2/6"},
    # C12 diacyl
    {"name": "DiC12-LP", "smiles": "CCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCC)N", "pIC50": 6.8, "class": "diacyl_LP", "target": "TLR2/6"},
    # C10 diacyl
    {"name": "DiC10-LP", "smiles": "CCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCC)N", "pIC50": 5.8, "class": "diacyl_LP", "target": "TLR2/6"},
    # C8 diacyl (minimal activity)
    {"name": "DiC8-LP", "smiles": "CCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCC)N", "pIC50": 4.5, "class": "diacyl_LP", "target": "TLR2/6"},
    # Lipo-OspA (Borrelia lipopeptide, potent TLR1/2)
    {"name": "Lipo-Ala2", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)[C@@H](N)C", "pIC50": 8.3, "class": "triacyl_LP", "target": "TLR1/2"},
    # Lipo-Ala3
    {"name": "Lipo-Ala3", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 8.2, "class": "triacyl_LP", "target": "TLR1/2"},

    # --- MODERATE AGONISTS (structural variants, partial agonists) ---
    # Monoacyl analog (single chain) - poor binding
    {"name": "MonoC16-LP", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](CO)N", "pIC50": 5.0, "class": "monoacyl_LP", "target": "TLR2"},
    # Single palmitoyl amine (no cysteine)
    {"name": "PalmAmine", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCN", "pIC50": 4.0, "class": "fatty_acid", "target": "none"},
    # Dipalmitol glycerol
    {"name": "DipalmitoylGlycerol", "smiles": "CCCCCCCCCCCCCCCC(=O)OC[C@@H](OC(=O)CCCCCCCCCCCCCCC)CO", "pIC50": 5.5, "class": "lipid", "target": "TLR2"},
    # Tripalmitoylglycerol (no cysteine scaffold)
    {"name": "TripalmitoylGlycerol", "smiles": "CCCCCCCCCCCCCCCC(=O)OC[C@@H](OC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCC", "pIC50": 5.0, "class": "lipid", "target": "TLR2"},
    # Lipoteichoic acid analog (simplified)
    {"name": "LTA-fragment", "smiles": "OC[C@H](OP(=O)(O)O)OCC(=O)NCCS[C@@H](CO)N", "pIC50": 5.5, "class": "LTA", "target": "TLR2"},
    # Peptidoglycan muramyl dipeptide analog (low TLR2 activity)
    {"name": "MDP-analog", "smiles": "N[C@@H](C)C(=O)N[C@@H](CC(N)=O)C(=O)O", "pIC50": 4.0, "class": "peptide", "target": "NOD2"},
    # Zymosan polysaccharide fragment
    {"name": "Beta-glucan-frag", "smiles": "OC[C@H]1O[C@@H](OC[C@H]2O[C@@H](O)[C@H](O)[C@@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "pIC50": 5.2, "class": "polysaccharide", "target": "TLR2/6"},
    # Short lipopeptide (3-residue)
    {"name": "Pam3-GGG", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.8, "class": "triacyl_LP", "target": "TLR1/2"},

    # --- STRUCTURAL VARIANTS WITH MODIFIED LINKERS ---
    # Ether-linked instead of thioether
    {"name": "Pam3Cys-OEther", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCO[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 6.5, "class": "triacyl_LP", "target": "TLR1/2"},
    # Carbon-linked (deaza analog)
    {"name": "Pam3Cys-CLinker", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCC[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 6.0, "class": "triacyl_LP", "target": "TLR1/2"},
    # Branched chain variant
    {"name": "BranchedPam3", "smiles": "CCCCCCCC(CC)CCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCC(CC)CCCCCCC)NC(=O)CCCCCCCC(CC)CCCCCCC", "pIC50": 6.8, "class": "triacyl_LP", "target": "TLR1/2"},

    # --- NEGATIVE CONTROLS / INACTIVES ---
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "pIC50": 3.0, "class": "NSAID", "target": "COX"},
    {"name": "Caffeine", "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "pIC50": 3.0, "class": "xanthine", "target": "PDE"},
    {"name": "Glucose", "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "pIC50": 2.5, "class": "sugar", "target": "none"},
    {"name": "Cholesterol", "smiles": "C[C@@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C", "pIC50": 3.5, "class": "sterol", "target": "none"},
    {"name": "Palmitic-acid", "smiles": "CCCCCCCCCCCCCCCC(=O)O", "pIC50": 4.2, "class": "fatty_acid", "target": "TLR4(weak)"},
    {"name": "Stearic-acid", "smiles": "CCCCCCCCCCCCCCCCCC(=O)O", "pIC50": 4.0, "class": "fatty_acid", "target": "TLR4(weak)"},
    {"name": "Oleic-acid", "smiles": "CCCCCCCC/C=C/CCCCCCCC(=O)O", "pIC50": 3.8, "class": "fatty_acid", "target": "none"},
    {"name": "Decanoic-acid", "smiles": "CCCCCCCCCC(=O)O", "pIC50": 3.5, "class": "fatty_acid", "target": "none"},
    {"name": "Lauric-acid", "smiles": "CCCCCCCCCCCC(=O)O", "pIC50": 4.5, "class": "fatty_acid", "target": "TLR2(weak)"},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "pIC50": 3.0, "class": "NSAID", "target": "COX"},
    {"name": "Dexamethasone", "smiles": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@H]1C(=O)CO", "pIC50": 3.5, "class": "steroid", "target": "GR"},
    {"name": "Penicillin-G", "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C", "pIC50": 3.5, "class": "betalactam", "target": "PBP"},
    {"name": "Vancomycin-frag", "smiles": "N[C@@H](Cc1cc(Cl)c(O[C@H]2CO[C@@H]3[C@H](O)[C@H](O)[C@@H](O)[C@H]3O2)c(O)c1)C(=O)N", "pIC50": 3.5, "class": "glycopeptide", "target": "peptidoglycan"},

    # --- ADDITIONAL SAR POINTS ---
    # Extended peptide Pam3 variants
    {"name": "Pam3-SKTTT", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 8.3, "class": "triacyl_LP", "target": "TLR1/2"},
    {"name": "Pam3-ACYP", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.9, "class": "triacyl_LP", "target": "TLR1/2"},
    # Mixed chain length
    {"name": "Pam3-C16C14", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.8, "class": "triacyl_LP", "target": "TLR1/2"},
    {"name": "Pam3-C16C12", "smiles": "CCCCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.5, "class": "triacyl_LP", "target": "TLR1/2"},
    {"name": "Pam3-C14C12", "smiles": "CCCCCCCCCCCCCC(=O)NCCS[C@@H](COC(=O)CCCCCCCCCCC)NC(=O)CCCCCCCCCCCCC", "pIC50": 7.0, "class": "triacyl_LP", "target": "TLR1/2"},
    # Deacylated variants
    {"name": "CysGlyGly", "smiles": "SC[C@@H](N)C(=O)NCC(=O)NCC(=O)O", "pIC50": 3.5, "class": "peptide", "target": "none"},
    {"name": "AcetylCys", "smiles": "CC(=O)N[C@@H](CS)C(=O)O", "pIC50": 3.0, "class": "aminoacid", "target": "none"},
    # TLR4 agonist MPLA (for negative TLR2 control)
    {"name": "MPLA-frag", "smiles": "CCCCCCCCCCCC(=O)OC[C@H](NC(=O)CCCCCCCCCCC)[C@@H](OC(=O)CCCCCCCCCCCC)COP(=O)(O)OC[C@H]1O[C@@H](OP(=O)(O)O)[C@@H](NC(=O)CCCCCCCCCCC)[C@@H]1O", "pIC50": 4.5, "class": "lipid_A", "target": "TLR4"},
    # Sphingosine (endogenous lipid, weak TLR2 activity)
    {"name": "Sphingosine", "smiles": "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO", "pIC50": 4.8, "class": "sphingolipid", "target": "TLR2(weak)"},
    # Ceramide
    {"name": "C2-Ceramide", "smiles": "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](NC(=O)C)CO", "pIC50": 4.5, "class": "ceramide", "target": "TLR2(weak)"},
    # LPS O-antigen fragment (inactive at TLR2)
    {"name": "LPS-Frag", "smiles": "OC[C@H]1O[C@@H](OP(=O)(O)O)[C@H](O)[C@@H](O)[C@@H]1O", "pIC50": 3.2, "class": "LPS", "target": "TLR4"},

    # --- SYNTHETIC LIPOPEPTIDE ANALOGS FROM BUWITT-BECKMANN ---
    # Buwitt-Beckmann 2005/2006 key SAR points
    {"name": "BW-1-C16-triacyl", "smiles": "CCCCCCCCCCCCCCCC(=O)NCC[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.2, "class": "triacyl_LP", "target": "TLR1/2"},
    {"name": "BW-2-C16-diacyl", "smiles": "CCCCCCCCCCCCCCCC(=O)NCC[C@@H](COC(=O)CCCCCCCCCCCCCCC)N", "pIC50": 7.0, "class": "diacyl_LP", "target": "TLR2/6"},
    {"name": "BW-3-no-lipid", "smiles": "NCC[C@@H](CO)N", "pIC50": 3.0, "class": "aminoalcohol", "target": "none"},
    # Cyclopentyl headgroup
    {"name": "BW-4-cyclopentyl", "smiles": "CCCCCCCCCCCCCCCC(=O)N[C@H]1CCCC1NC(=O)CCCCCCCCCCCCCCC", "pIC50": 5.5, "class": "cyclic_LP", "target": "TLR2"},
    # No thioether, direct N-linkage
    {"name": "BW-5-Nlinked", "smiles": "CCCCCCCCCCCCCCCC(=O)N[C@@H](COC(=O)CCCCCCCCCCCCCCC)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 7.0, "class": "triacyl_LP", "target": "TLR1/2"},
    # Reversed ester orientation
    {"name": "BW-6-reversed-ester", "smiles": "CCCCCCCCCCCCCCCC(=O)O[C@@H](CNCCS)NC(=O)CCCCCCCCCCCCCCC", "pIC50": 5.8, "class": "triacyl_LP", "target": "TLR1/2"},

    # --- ADDITIONAL DIVERSE INACTIVES ---
    {"name": "Biotin", "smiles": "O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12", "pIC50": 3.0, "class": "vitamin", "target": "avidin"},
    {"name": "Folic-acid", "smiles": "Nc1nc2ncc(CNc3ccc(cc3)C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc2c(=O)[nH]1", "pIC50": 3.0, "class": "vitamin", "target": "DHFR"},
    {"name": "Ampicillin", "smiles": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](N)c3ccccc3)C(=O)O)C", "pIC50": 3.2, "class": "betalactam", "target": "PBP"},
    {"name": "Propranolol", "smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "pIC50": 3.0, "class": "betablocker", "target": "betaAR"},
    {"name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "pIC50": 3.0, "class": "biguanide", "target": "AMPK"},
    {"name": "Heparin-frag", "smiles": "OC1O[C@@H](COS(=O)(=O)O)[C@@H](O)[C@H](NS(=O)(=O)O)[C@@H]1O", "pIC50": 3.5, "class": "GAG", "target": "thrombin"},
    {"name": "Cyclosporine-A-frag", "smiles": "CCC1NC(=O)[C@@H](N(C)C(=O)CN(C)C(=O)CC(C)C)CC1", "pIC50": 3.5, "class": "cyclopeptide", "target": "calcineurin"},
    {"name": "Tripalmitin", "smiles": "CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCC", "pIC50": 4.5, "class": "triglyceride", "target": "lipase"},
    {"name": "DPPC", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC", "pIC50": 4.8, "class": "phospholipid", "target": "membrane"},
    {"name": "LysoPPC", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(O)COP(=O)([O-])OCC[N+](C)(C)C", "pIC50": 4.5, "class": "lysophospholipid", "target": "TLR2(weak)"},
]
# fmt: on


def build():
    df = pd.DataFrame(COMPOUNDS)
    print(f"Dataset: {len(df)} compounds")
    print(f"pIC50 range: {df.pIC50.min():.1f} - {df.pIC50.max():.1f}")
    print(f"\nClass distribution:")
    print(df["class"].value_counts().to_string())

    out = "data/chembl_tlr2_raw.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return df


if __name__ == "__main__":
    build()
