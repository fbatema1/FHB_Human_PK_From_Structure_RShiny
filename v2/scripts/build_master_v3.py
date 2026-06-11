"""
v2/scripts/build_master_v3.py
=============================
Builds master_v3.xlsx for Track B by enriching master_v2_preclinical with the
ChEMBL Caco-2 permeability dataset (v2/data/permeability_data.csv).

Caco-2 handling is deliberately conservative — the two sources are kept in
SEPARATE columns plus a unified "best" column with a provenance flag, so we
never silently average values that may be on different scales:
  - caco2_clvd        : existing Caco-2 from CL_VD_Data (master_v2_preclinical)
  - caco2_chembl      : median apical→basolateral Papp (×10⁻⁶ cm/s) from ChEMBL
  - caco2_final       : caco2_clvd if present, else caco2_chembl
  - caco2_source      : 'clvd' | 'chembl' | 'missing'

ChEMBL cleaning:
  - filter to apical→basolateral (absorptive Papp) records only
  - harmonise units to ×10⁻⁶ cm/s
  - median per RDKit-canonical compound

Outputs (v2/data/processed/):
  - master_v3.xlsx
  - master_v3_summary.txt

Run:
    python v2/scripts/build_master_v3.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT   = Path(__file__).resolve().parents[2]
V2DATA = ROOT / "v2/data"
PROC   = V2DATA / "processed"


def canon(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


# Unit → multiplier to express Papp in ×10⁻⁶ cm/s
UNIT_FACTOR = {
    "ucm/s":       1.0,    # µcm/s == ×10⁻⁶ cm/s
    "10'-6 cm/s":  1.0,
    "10'-6cm/s":   1.0,
    "10'6cm/s":    1.0,    # dropped-minus typo for 10⁻⁶
    "10'-5cm/s":   10.0,   # 10⁻⁵ = 10 × 10⁻⁶
}


def clean_chembl_caco2(perm: pd.DataFrame):
    """Filter to A→B, harmonise units to ×10⁻⁶ cm/s, median per canonical compound."""
    desc = perm["description"].fillna("").str.lower()
    ab_mask = desc.str.contains("apical to basolateral")
    ba_mask = desc.str.contains("basolateral to apical")
    n_ab, n_ba, n_other = ab_mask.sum(), ba_mask.sum(), (~ab_mask & ~ba_mask).sum()

    ab = perm[ab_mask].copy()

    # Harmonise units
    ab["unit_factor"] = ab["standard_units"].map(UNIT_FACTOR)
    n_unmapped = ab["unit_factor"].isna().sum()
    ab = ab.dropna(subset=["unit_factor", "standard_value"])
    ab["papp_e6"] = ab["standard_value"].astype(float) * ab["unit_factor"]

    # Sanity: drop non-physical values
    ab = ab[(ab["papp_e6"] > 0) & (ab["papp_e6"] < 1000)]

    ab["canon"] = ab["canonical_smiles"].map(canon)
    ab = ab.dropna(subset=["canon"])

    per_cmpd = ab.groupby("canon")["papp_e6"].median()

    return per_cmpd, dict(n_ab=int(n_ab), n_ba=int(n_ba), n_other=int(n_other),
                          n_unmapped=int(n_unmapped), n_compounds=int(per_cmpd.nunique()))


def run():
    print("Loading master_v2_preclinical...")
    master = pd.read_excel(PROC / "master_v2_preclinical.xlsx", engine="openpyxl")
    master["canon"] = master["mol"].map(canon)
    print(f"  {len(master)} compounds")

    print("Loading + cleaning ChEMBL permeability...")
    perm = pd.read_csv(V2DATA / "permeability_data.csv", low_memory=False)
    chembl_caco2, stats = clean_chembl_caco2(perm)
    print(f"  records: A→B={stats['n_ab']}, B→A={stats['n_ba']}, other/ambiguous={stats['n_other']}")
    print(f"  unmapped units dropped: {stats['n_unmapped']}")
    print(f"  unique A→B compounds (cleaned): {stats['n_compounds']}")

    # ── Build columns — two DISTINCT Caco-2 features (different scales!) ─────────
    # caco2_flag : binary high/low permeability class (CL_VD_Data, 0/1)
    # caco2_papp : continuous apical→basolateral Papp in ×10⁻⁶ cm/s (ChEMBL)
    # These are NOT the same quantity and are intentionally never merged.
    master["caco2_flag"] = master["caco2"] if "caco2" in master.columns else np.nan
    master["caco2_papp"] = master["canon"].map(chembl_caco2)

    # ── Distribution report ──────────────────────────────────────────────────────
    def describe(s):
        s = s.dropna()
        if len(s) == 0:
            return "  (none)"
        q = s.quantile([0, .25, .5, .75, 1.0]).round(2).tolist()
        return f"  n={len(s)}  min/Q1/med/Q3/max = {q}"

    n = len(master)
    n_flag = master["caco2_flag"].notna().sum()
    n_papp = master["caco2_papp"].notna().sum()
    both   = (master["caco2_flag"].notna() & master["caco2_papp"].notna()).sum()

    summary = "\n".join([
        "MASTER_V3 BUILD SUMMARY",
        "=" * 55,
        f"Total compounds:            {n}",
        "",
        "TWO DISTINCT Caco-2 features (different scales — never merged):",
        f"  caco2_flag  — binary permeability class (CL_VD_Data, 0/1):",
        describe(master["caco2_flag"]),
        f"  caco2_papp  — continuous Papp ×10⁻⁶ cm/s (ChEMBL, A→B):",
        describe(master["caco2_papp"]),
        "",
        f"Coverage:",
        f"  caco2_flag (binary):       {n_flag}  ({100*n_flag/n:.1f}%)",
        f"  caco2_papp (continuous):   {n_papp}  ({100*n_papp/n:.1f}%)",
        f"  have BOTH:                 {both}",
        "",
        "INTERPRETATION:",
        "  The legacy 'caco2' (from CL_VD_Data) is a BINARY high/low flag, not a",
        "  permeability value — kept as caco2_flag. The ChEMBL data is the real",
        "  continuous Papp (caco2_papp) but only ~6% coverage. For Track B, prefer",
        "  caco2_papp as a continuous feature; caco2_flag is categorical and only",
        "  loosely related. Do not average or combine the two.",
    ])
    print("\n" + summary)

    # Drop the now-redundant legacy 'caco2' alias to avoid confusion
    master = master.drop(columns=["caco2"], errors="ignore")

    # ── Save ─────────────────────────────────────────────────────────────────────
    master = master.drop(columns=["canon"])
    out_xlsx = PROC / "master_v3.xlsx"
    master.to_excel(out_xlsx, index=False)
    with open(PROC / "master_v3_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved: {out_xlsx}")
    print(f"Saved: {PROC / 'master_v3_summary.txt'}")


if __name__ == "__main__":
    run()
