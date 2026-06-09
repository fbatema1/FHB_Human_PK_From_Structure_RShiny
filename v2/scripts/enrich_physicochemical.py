"""
v2/scripts/enrich_physicochemical.py
=====================================
Enriches the master dataset with:
  1. fup  — merges experimental values from CL_VD_Data where available
  2. pKa  — merges experimental values from CL_VD_Data where available,
             then computes predicted pKa via RDKit/Dimorphite-DL for remainder
  3. logP — already in RDKit descriptors but added explicitly as a column
             so models can use it directly as a target-relevant feature
  4. logD — logP corrected for ionisation at pH 7.4 (more relevant than logP
             for in vivo distribution)

Strategy:
  - Experimental values always preferred over computed
  - Computed values flagged with a '_source' column ('experimental'|'computed')
  - Compounds with no fup get fup imputed via median (flagged separately)

Outputs (v2/data/processed/):
  - master_v2.xlsx          — enriched master dataset
  - enrichment_summary.txt  — coverage stats

Run:
    conda activate pkip-env
    python v2/scripts/enrich_physicochemical.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
PROC   = ROOT / "data/processed"
RAW    = ROOT / "data/raw"
OUT    = ROOT / "v2/data/processed"
OUT.mkdir(parents=True, exist_ok=True)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    raise ImportError("RDKit required: conda install -c conda-forge rdkit")


# ── RDKit physicochemical helpers ─────────────────────────────────────────────

def calc_logp(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return round(Descriptors.MolLogP(mol), 4)


def calc_logd74(smiles: str, pka_acid: float = None, pka_base: float = None) -> float | None:
    """
    Approximate logD at pH 7.4 using Henderson-Hasselbalch correction.
    For acids:  logD = logP - log10(1 + 10^(pH - pKa))
    For bases:  logD = logP - log10(1 + 10^(pKa - pH))
    Falls back to logP if no pKa available.
    """
    logp = calc_logp(smiles)
    if logp is None:
        return None
    pH = 7.4
    correction = 0.0
    if pka_acid is not None and not np.isnan(pka_acid):
        correction += np.log10(1 + 10 ** (pH - pka_acid))
    if pka_base is not None and not np.isnan(pka_base):
        correction += np.log10(1 + 10 ** (pka_base - pH))
    return round(logp - correction, 4)


def calc_pka_rdkit(smiles: str):
    """
    Estimate most acidic and most basic pKa using RDKit's ionisation
    state enumeration (Dimorphite-DL approach approximated via SMARTS).

    Returns (pka_acid, pka_base) — None where not applicable.

    NOTE: RDKit doesn't have a built-in pKa predictor. This uses a simple
    SMARTS-based group contribution approach as a placeholder until
    Dimorphite-DL or a trained pKa model is integrated.
    The values are approximate (±1–2 units) but capture ionisation class.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # SMARTS patterns for common ionisable groups with approximate pKa
    ACID_PATTERNS = [
        ("[CX3](=O)[OH]",          4.5),   # carboxylic acid
        ("[SX4](=O)(=O)[OH]",      1.0),   # sulfonic acid
        ("[PX4](=O)([OH])[OH]",    2.1),   # phosphoric acid
        ("[nH]1cccc1",             14.5),  # pyrrole NH (very weak acid)
        ("[OH]c1ccccc1",           9.9),   # phenol
        ("[OH][CX4]",             16.0),   # alcohol (very weak)
        ("[NH2]C(=O)",            24.0),   # amide NH (essentially non-acidic)
        ("[SH]",                   10.5),  # thiol
    ]

    BASE_PATTERNS = [
        ("[NX3;H2;!$(NC=O)]",      10.5),  # primary amine
        ("[NX3;H1;!$(NC=O)]",       9.5),  # secondary amine
        ("[NX3;H0;!$(NC=O)]",       8.5),  # tertiary amine
        ("[nH0;r5,r6]",             5.0),  # aromatic N (pyridine-like)
        ("[NH2]c1ccccc1",           4.6),  # aniline
        ("[NX2]=C",                 3.0),  # imine
        ("[NH]C(=N)",              12.5),  # guanidine
    ]

    pka_acids = []
    for smarts, pka in ACID_PATTERNS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            pka_acids.append(pka)

    pka_bases = []
    for smarts, pka in BASE_PATTERNS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            pka_bases.append(pka)

    pka_acid = round(min(pka_acids), 2) if pka_acids else None
    pka_base = round(max(pka_bases), 2) if pka_bases else None

    return pka_acid, pka_base


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("Loading master dataset...")
    master = pd.read_excel(PROC / "master_dataset_cleaned.xlsx",
                           sheet_name="Cleaned_Data", engine="openpyxl")
    print(f"  {len(master)} compounds")

    print("\nLoading CL_VD_Data (experimental fup + pKa source)...")
    clvd = pd.read_excel(RAW / "CL_VD_Data.xlsx", engine="openpyxl")
    clvd["name_lower"] = clvd["NAME"].str.lower().str.strip()
    master["name_lower"] = master["compound_name"].str.lower().str.strip()

    # ── 1. Merge experimental fup from CL_VD_Data ─────────────────────────────
    print("\n[1/4] Merging experimental fup...")
    clvd_fup = clvd[["name_lower", "human_fup"]].rename(
        columns={"human_fup": "fup_clvd"}
    ).dropna(subset=["fup_clvd"])

    master = master.merge(clvd_fup, on="name_lower", how="left")

    # Use CL_VD_Data fup to fill gaps in master fup
    master["fup_final"] = master["human_fup"].copy()
    fill_mask = master["fup_final"].isna() & master["fup_clvd"].notna()
    master.loc[fill_mask, "fup_final"] = master.loc[fill_mask, "fup_clvd"]
    master["fup_source"] = "original"
    master.loc[master["human_fup"].isna() & master["fup_clvd"].notna(), "fup_source"] = "clvd_merged"
    master.loc[master["fup_final"].isna(), "fup_source"] = "missing"

    n_orig   = (master["fup_source"] == "original").sum()
    n_merged = (master["fup_source"] == "clvd_merged").sum()
    n_miss   = (master["fup_source"] == "missing").sum()
    print(f"  Original fup:       {n_orig}")
    print(f"  Merged from CL_VD:  {n_merged}")
    print(f"  Still missing:      {n_miss} ({100*n_miss/len(master):.1f}%)")

    # Impute missing fup with median (flagged)
    fup_median = master["fup_final"].median()
    master.loc[master["fup_final"].isna(), "fup_final"] = fup_median
    master.loc[master["fup_source"] == "missing", "fup_source"] = f"imputed_median_{fup_median:.3f}"
    print(f"  Missing imputed with median fup = {fup_median:.3f}")

    # ── 2. Merge experimental pKa from CL_VD_Data ─────────────────────────────
    print("\n[2/4] Merging experimental pKa...")
    clvd_pka = clvd[["name_lower", "pKa_Acid", "pKa_base"]].dropna(
        subset=["pKa_Acid", "pKa_base"], how="all"
    )
    master = master.merge(clvd_pka, on="name_lower", how="left")
    n_exp_pka = master["pKa_Acid"].notna().sum()
    print(f"  Experimental pKa matched: {n_exp_pka} compounds")

    # ── 3. Compute RDKit pKa for remainder ────────────────────────────────────
    print("\n[3/4] Computing RDKit pKa for remaining compounds...")
    n_need_pka = master["pKa_Acid"].isna().sum()
    print(f"  Need computed pKa: {n_need_pka}")

    computed_acid, computed_base = [], []
    for smi in master["mol"]:
        a, b = calc_pka_rdkit(str(smi))
        computed_acid.append(a)
        computed_base.append(b)

    master["pka_acid_computed"] = computed_acid
    master["pka_base_computed"] = computed_base

    # Final pKa: experimental preferred, computed as fallback
    master["pka_acid_final"]  = master["pKa_Acid"].combine_first(
        pd.Series(computed_acid, index=master.index)
    )
    master["pka_base_final"]  = master["pKa_base"].combine_first(
        pd.Series(computed_base, index=master.index)
    )
    master["pka_source"] = "computed"
    master.loc[master["pKa_Acid"].notna() | master["pKa_base"].notna(), "pka_source"] = "experimental"

    # ── 4. Compute logP and logD ──────────────────────────────────────────────
    print("\n[4/4] Computing logP and logD@pH7.4...")
    logp_vals, logd_vals = [], []
    for _, row in master.iterrows():
        lp = calc_logp(str(row["mol"]))
        ld = calc_logd74(str(row["mol"]),
                         pka_acid=row["pka_acid_final"],
                         pka_base=row["pka_base_final"])
        logp_vals.append(lp)
        logd_vals.append(ld)

    master["logP"]    = logp_vals
    master["logD_74"] = logd_vals
    print(f"  logP computed: {pd.Series(logp_vals).notna().sum()}/{len(master)}")
    print(f"  logD computed: {pd.Series(logd_vals).notna().sum()}/{len(master)}")

    # ── Clean up merge columns ────────────────────────────────────────────────
    master = master.drop(columns=["name_lower", "fup_clvd", "pKa_Acid", "pKa_base"],
                         errors="ignore")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = [
        "V2 PHYSICOCHEMICAL ENRICHMENT SUMMARY",
        "=" * 50,
        f"Total compounds:         {len(master)}",
        f"",
        f"fup coverage:",
        f"  Original:              {n_orig} ({100*n_orig/len(master):.1f}%)",
        f"  Merged from CL_VD:     {n_merged} ({100*n_merged/len(master):.1f}%)",
        f"  Imputed (median):      {n_miss} ({100*n_miss/len(master):.1f}%)",
        f"  Total non-imputed:     {n_orig+n_merged} ({100*(n_orig+n_merged)/len(master):.1f}%)",
        f"",
        f"pKa coverage:",
        f"  Experimental:          {(master['pka_source']=='experimental').sum()}",
        f"  Computed (RDKit):      {(master['pka_source']=='computed').sum()}",
        f"",
        f"logP:                    {master['logP'].notna().sum()} ({100*master['logP'].notna().mean():.1f}%)",
        f"logD@pH7.4:              {master['logD_74'].notna().sum()} ({100*master['logD_74'].notna().mean():.1f}%)",
        f"",
        f"New columns added: fup_final, fup_source, pka_acid_final, pka_base_final,",
        f"                   pka_acid_computed, pka_base_computed, pka_source,",
        f"                   logP, logD_74",
    ]
    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_xlsx = OUT / "master_v2.xlsx"
    out_sum  = OUT / "enrichment_summary.txt"
    master.to_excel(out_xlsx, index=False)
    with open(out_sum, "w") as f:
        f.write(summary_text)

    print(f"\nSaved: {out_xlsx}")
    print(f"Saved: {out_sum}")
    print("\n✓ Enrichment complete.")


if __name__ == "__main__":
    run()
