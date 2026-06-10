"""
v2/scripts/enrich_preclinical.py
=================================
Merges preclinical PK and in vitro ADME data from CL_VD_Data.xlsx
into master_v2.xlsx, producing master_v2_preclinical.xlsx.

New columns added:
  - rat_cl          (rat CL, mL/min/kg)
  - rat_vd          (rat Vd,ss, L/kg)
  - rat_fup         (rat fraction unbound)
  - monkey_cl       (monkey CL, mL/min/kg)
  - dog_cl          (dog CL, mL/min/kg)
  - caco2           (Caco-2 permeability)
  - water_sol       (water solubility)

All matched by compound name (lowercase). Missing values left as NaN
for now — imputation happens at featurization time.

Run:
    conda activate pkip-env
    python v2/scripts/enrich_preclinical.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
V2PROC = ROOT / "v2/data/processed"
RAW    = ROOT / "data/raw"


def run():
    print("Loading master_v2...")
    master = pd.read_excel(V2PROC / "master_v2.xlsx", engine="openpyxl")
    print(f"  {len(master)} compounds")

    print("Loading CL_VD_Data...")
    clvd = pd.read_excel(RAW / "CL_VD_Data.xlsx", engine="openpyxl")
    print(f"  {len(clvd)} compounds in CL_VD_Data")

    # Name-match key
    clvd["name_lower"]   = clvd["NAME"].str.lower().str.strip()
    master["name_lower"] = master["compound_name"].str.lower().str.strip()

    # Select and rename preclinical columns
    preclinical = clvd[[
        "name_lower",
        "rat_CL_mL_min_kg",
        "rat_VDss_L_kg",
        "rat_fup",
        "monkey_CL_mL_min_kg",
        "monkey_VDss_L_kg",
        "dog_CL_mL_min_kg",
        "dog_VDss_L_kg",
        "Caco_2",
        "water_solubility",
    ]].rename(columns={
        "rat_CL_mL_min_kg":    "rat_cl",
        "rat_VDss_L_kg":       "rat_vd",
        "rat_fup":             "rat_fup",
        "monkey_CL_mL_min_kg": "monkey_cl",
        "monkey_VDss_L_kg":    "monkey_vd",
        "dog_CL_mL_min_kg":    "dog_cl",
        "dog_VDss_L_kg":       "dog_vd",
        "Caco_2":              "caco2",
        "water_solubility":    "water_sol",
    })

    master = master.merge(preclinical, on="name_lower", how="left")
    master = master.drop(columns=["name_lower"])

    n = len(master)
    print("\nCoverage after merge:")
    for col in ["rat_cl","rat_vd","rat_fup","monkey_cl","dog_cl","caco2","water_sol"]:
        k = master[col].notna().sum()
        print(f"  {col:<12} {k}/{n} ({100*k/n:.1f}%)")

    out = V2PROC / "master_v2_preclinical.xlsx"
    master.to_excel(out, index=False)
    print(f"\nSaved: {out}")
    print("✓ Preclinical enrichment complete.")


if __name__ == "__main__":
    run()
