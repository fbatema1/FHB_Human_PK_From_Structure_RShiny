"""
v2/scripts/enrich_rat_pk.py
============================
Queries ChEMBL REST API for rat in vivo PK data (clearance, volume of
distribution) and Caco-2 permeability for compounds in master_v2 that
are NOT already covered by the CL_VD_Data.xlsx name-match.

Strategy:
  1. Load master_v2 — identify compounds missing rat CL/Vd
  2. For each compound, search ChEMBL by compound name → get ChEMBL ID
  3. Query assay results for:
       - rat CL   (assay_type=ADME, standard_type IN ['CL','Clearance'])
       - rat Vd   (assay_type=ADME, standard_type IN ['VDss','Vd'])
       - Caco-2   (assay_type=ADME, standard_type IN ['Papp','Permeability'])
  4. Filter to rat organism (standard_organism LIKE '%rattus%')
  5. Take median where multiple values exist, flag source as 'chembl'
  6. Merge into master_v2 → save as master_v2_rat.xlsx

Outputs (v2/data/processed/):
  - master_v2_rat.xlsx          — enriched with rat PK + Caco-2
  - rat_pk_fetch_log.txt        — per-compound fetch log
  - rat_pk_coverage_summary.txt — coverage stats

Run:
    conda activate pkip-env
    python v2/scripts/enrich_rat_pk.py

Note: ChEMBL REST API has no auth requirement but rate-limits to ~1 req/s.
      ~1000 new compounds → ~20-30 min runtime. Run on Longleaf or overnight.
"""

import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
V2PROC = ROOT / "v2/data/processed"
RAW    = ROOT / "data/raw"
V2PROC.mkdir(parents=True, exist_ok=True)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
SLEEP_S    = 0.5    # seconds between requests (polite rate limiting)

# ── ChEMBL assay type mappings ─────────────────────────────────────────────────
CL_TYPES   = {"cl", "clearance", "total clearance", "clp", "clint"}
VD_TYPES   = {"vdss", "vd", "volume of distribution", "vss"}
CACO_TYPES = {"papp", "papp a-b", "permeability", "papp ab"}
RAT_ORGS   = {"rattus norvegicus", "rat"}


# ── API helpers ────────────────────────────────────────────────────────────────

def search_chembl_id(name: str) -> str | None:
    """Search ChEMBL for a compound by name, return first ChEMBL ID or None."""
    url = f"{CHEMBL_API}/molecule/search.json"
    params = {"q": name, "limit": 5}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        mols = r.json().get("molecules", [])
        if mols:
            return mols[0]["molecule_chembl_id"]
    except Exception:
        pass
    return None


def get_adme_data(chembl_id: str) -> list[dict]:
    """Fetch all ADME activity records for a ChEMBL molecule ID."""
    url = f"{CHEMBL_API}/activity.json"
    params = {
        "molecule_chembl_id": chembl_id,
        "assay_type":         "A",    # ADME
        "limit":              1000,
        "format":             "json",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("activities", [])
    except Exception:
        return []


def parse_rat_pk(activities: list[dict]) -> dict:
    """
    Extract rat CL, Vd, and Caco-2 from ChEMBL activity records.
    Returns dict with keys: rat_cl, rat_vd, caco2_papp (median values, or None).
    Units: CL in mL/min/kg, Vd in L/kg, Caco-2 in 10^-6 cm/s.
    """
    rat_cl_vals, rat_vd_vals, caco2_vals = [], [], []

    for act in activities:
        stype = (act.get("standard_type") or "").lower().strip()
        org   = (act.get("assay_organism") or "").lower().strip()
        val   = act.get("standard_value")
        units = (act.get("standard_units") or "").lower().strip()

        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        is_rat = any(r in org for r in RAT_ORGS)

        if is_rat and stype in CL_TYPES:
            # Convert to mL/min/kg if needed
            if "ml/min/kg" in units or units == "":
                rat_cl_vals.append(val)
            elif "ul/min/mg" in units:
                pass  # intrinsic, skip
            elif "ml/min" in units and "kg" not in units:
                pass  # total body CL, skip without BW

        elif is_rat and stype in VD_TYPES:
            if "l/kg" in units or units == "":
                rat_vd_vals.append(val)

        elif stype in CACO_TYPES:
            # Caco-2 is in vitro, no organism filter needed
            if "10-6" in units or "cm/s" in units or units == "":
                caco2_vals.append(val)

    return {
        "rat_cl_chembl":    float(np.median(rat_cl_vals)) if rat_cl_vals else None,
        "rat_vd_chembl":    float(np.median(rat_vd_vals)) if rat_vd_vals else None,
        "caco2_chembl":     float(np.median(caco2_vals))  if caco2_vals  else None,
        "rat_cl_n":         len(rat_cl_vals),
        "rat_vd_n":         len(rat_vd_vals),
        "caco2_n":          len(caco2_vals),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("RAT PK + CACO-2 ENRICHMENT (ChEMBL)")
    print("=" * 60)

    # ── Load master_v2 ────────────────────────────────────────────────────────
    master = pd.read_excel(V2PROC / "master_v2.xlsx", engine="openpyxl")
    print(f"  master_v2: {len(master)} compounds")

    # ── Load existing rat PK from CL_VD_Data ─────────────────────────────────
    clvd = pd.read_excel(RAW / "CL_VD_Data.xlsx", engine="openpyxl")
    clvd["name_lower"] = clvd["NAME"].str.lower().str.strip()
    master["name_lower"] = master["compound_name"].str.lower().str.strip()

    # Columns to pull from CL_VD_Data (adjust names to match your file)
    rat_cols = []
    for col in ["rat_CL", "rat_cl", "Rat_CL", "rat_VDss", "rat_vd", "Rat_VDss",
                "Caco2", "caco2", "Caco_2"]:
        if col in clvd.columns:
            rat_cols.append(col)

    if rat_cols:
        print(f"  Existing rat/Caco-2 columns in CL_VD_Data: {rat_cols}")
        clvd_rat = clvd[["name_lower"] + rat_cols]
        master = master.merge(clvd_rat, on="name_lower", how="left")
    else:
        print("  No rat PK columns found in CL_VD_Data — all to be fetched from ChEMBL")
        master["rat_CL"]   = np.nan
        master["rat_VDss"] = np.nan
        master["caco2"]    = np.nan

    # Standardise column names
    for old, new in [("rat_CL","rat_cl_existing"), ("Rat_CL","rat_cl_existing"),
                     ("rat_VDss","rat_vd_existing"), ("Rat_VDss","rat_vd_existing"),
                     ("Caco2","caco2_existing"), ("caco2","caco2_existing"),
                     ("Caco_2","caco2_existing")]:
        if old in master.columns and "rat_cl_existing" not in master.columns:
            master = master.rename(columns={old: new})

    if "rat_cl_existing" not in master.columns:
        master["rat_cl_existing"] = np.nan
    if "rat_vd_existing" not in master.columns:
        master["rat_vd_existing"] = np.nan
    if "caco2_existing" not in master.columns:
        master["caco2_existing"] = np.nan

    # ── Identify compounds needing ChEMBL fetch ───────────────────────────────
    need_fetch = master[
        master["rat_cl_existing"].isna() |
        master["rat_vd_existing"].isna() |
        master["caco2_existing"].isna()
    ].copy()
    print(f"\n  Compounds needing ChEMBL fetch: {len(need_fetch)}")
    print(f"  Already have rat CL: {master['rat_cl_existing'].notna().sum()}")
    print(f"  Already have rat Vd: {master['rat_vd_existing'].notna().sum()}")
    print(f"  Already have Caco-2: {master['caco2_existing'].notna().sum()}")

    # ── ChEMBL fetch loop ─────────────────────────────────────────────────────
    print(f"\nFetching ChEMBL data for {len(need_fetch)} compounds...")
    print("(This will take ~20-30 minutes — each compound = 2 API calls)\n")

    log_rows = []
    chembl_results = {}

    for i, (idx, row) in enumerate(need_fetch.iterrows()):
        name = row["compound_name"]
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(need_fetch)}] {name}")

        # Step 1: get ChEMBL ID
        time.sleep(SLEEP_S)
        chembl_id = search_chembl_id(name)
        if chembl_id is None:
            log_rows.append({"compound": name, "chembl_id": None,
                             "status": "not_found", "rat_cl": None,
                             "rat_vd": None, "caco2": None})
            continue

        # Step 2: fetch ADME activities
        time.sleep(SLEEP_S)
        activities = get_adme_data(chembl_id)
        pk = parse_rat_pk(activities)

        chembl_results[idx] = pk
        log_rows.append({
            "compound":  name,
            "chembl_id": chembl_id,
            "status":    "found",
            "rat_cl":    pk["rat_cl_chembl"],
            "rat_vd":    pk["rat_vd_chembl"],
            "caco2":     pk["caco2_chembl"],
            "rat_cl_n":  pk["rat_cl_n"],
            "rat_vd_n":  pk["rat_vd_n"],
            "caco2_n":   pk["caco2_n"],
        })

    print(f"\n  Fetch complete. {len(chembl_results)} compounds with any data.")

    # ── Merge ChEMBL results back ─────────────────────────────────────────────
    master["rat_cl_chembl"] = np.nan
    master["rat_vd_chembl"] = np.nan
    master["caco2_chembl"]  = np.nan

    for idx, pk in chembl_results.items():
        master.at[idx, "rat_cl_chembl"] = pk["rat_cl_chembl"]
        master.at[idx, "rat_vd_chembl"] = pk["rat_vd_chembl"]
        master.at[idx, "caco2_chembl"]  = pk["caco2_chembl"]

    # ── Final combined columns ────────────────────────────────────────────────
    master["rat_cl_final"]  = master["rat_cl_existing"].combine_first(
                                  master["rat_cl_chembl"])
    master["rat_vd_final"]  = master["rat_vd_existing"].combine_first(
                                  master["rat_vd_chembl"])
    master["caco2_final"]   = master["caco2_existing"].combine_first(
                                  master["caco2_chembl"])

    master["rat_cl_source"] = "missing"
    master.loc[master["rat_cl_existing"].notna(), "rat_cl_source"] = "clvd_file"
    master.loc[(master["rat_cl_existing"].isna()) &
               (master["rat_cl_chembl"].notna()), "rat_cl_source"] = "chembl"

    master["rat_vd_source"] = "missing"
    master.loc[master["rat_vd_existing"].notna(), "rat_vd_source"] = "clvd_file"
    master.loc[(master["rat_vd_existing"].isna()) &
               (master["rat_vd_chembl"].notna()), "rat_vd_source"] = "chembl"

    # ── Coverage summary ──────────────────────────────────────────────────────
    n = len(master)
    n_rat_cl   = master["rat_cl_final"].notna().sum()
    n_rat_vd   = master["rat_vd_final"].notna().sum()
    n_caco2    = master["caco2_final"].notna().sum()
    n_all_rat  = (master["rat_cl_final"].notna() &
                  master["rat_vd_final"].notna()).sum()

    summary = "\n".join([
        "RAT PK + CACO-2 ENRICHMENT SUMMARY",
        "=" * 50,
        f"Total compounds:        {n}",
        f"",
        f"Rat CL coverage:        {n_rat_cl} ({100*n_rat_cl/n:.1f}%)",
        f"  from CL_VD_Data:      {(master['rat_cl_source']=='clvd_file').sum()}",
        f"  from ChEMBL:          {(master['rat_cl_source']=='chembl').sum()}",
        f"  missing:              {(master['rat_cl_source']=='missing').sum()}",
        f"",
        f"Rat Vd coverage:        {n_rat_vd} ({100*n_rat_vd/n:.1f}%)",
        f"  from CL_VD_Data:      {(master['rat_vd_source']=='clvd_file').sum()}",
        f"  from ChEMBL:          {(master['rat_vd_source']=='chembl').sum()}",
        f"  missing:              {(master['rat_vd_source']=='missing').sum()}",
        f"",
        f"Caco-2 coverage:        {n_caco2} ({100*n_caco2/n:.1f}%)",
        f"Both rat CL + Vd:       {n_all_rat} ({100*n_all_rat/n:.1f}%)",
    ])
    print("\n" + summary)

    # ── Save ──────────────────────────────────────────────────────────────────
    master = master.drop(columns=["name_lower"], errors="ignore")
    master.to_excel(V2PROC / "master_v2_rat.xlsx", index=False)
    pd.DataFrame(log_rows).to_csv(V2PROC / "rat_pk_fetch_log.csv", index=False)
    with open(V2PROC / "rat_pk_coverage_summary.txt", "w") as f:
        f.write(summary)

    print(f"\nSaved: master_v2_rat.xlsx")
    print(f"Saved: rat_pk_fetch_log.csv")
    print(f"Saved: rat_pk_coverage_summary.txt")
    print("\n✓ Rat PK enrichment complete.")


if __name__ == "__main__":
    run()
