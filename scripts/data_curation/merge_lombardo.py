"""
merge_lombardo.py
=================
Merges the full Lombardo dataset into the master dataset.

Actions:
  1. Adds 654 new compounds from Lombardo not already in master
  2. Fills missing CL, Vd, and fup for 713 overlapping compounds
     where Lombardo has values and master does not
  3. Adds data_source column to track provenance
  4. Never overwrites existing master values

Overlap detection uses InChIKeys (structure-based, not name-based)
to avoid duplicate entries from naming differences.

Run:
    conda activate pkip-env
    python scripts/data_curation/merge_lombardo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MASTER_IN  = ROOT / "data/raw/master_dataset_FHB_06052026_v3.xlsx"
LOMBARDO   = ROOT / "data/raw/lombardo_full_dataset.xlsx"
MASTER_OUT = ROOT / "data/raw/master_dataset_with_lombardo.xlsx"


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_inchikey(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return MolToInchiKey(mol)
    except:
        pass
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')


# ── Main ──────────────────────────────────────────────────────────────────────
def run():

    # ── Load master ───────────────────────────────────────────────────────────
    print(f"Loading master: {MASTER_IN.name}")
    xl = pd.ExcelFile(MASTER_IN, engine='openpyxl')
    master = pd.read_excel(xl, sheet_name='All_Compounds', engine='openpyxl')
    other_sheets = {
        s: pd.read_excel(xl, sheet_name=s, engine='openpyxl')
        for s in xl.sheet_names if s != 'All_Compounds'
    }

    print(f"  Master compounds:            {len(master)}")
    print(f"  Complete (CL+Vd+fup):        {(master.human_CL_mL_min_kg.notna() & master.human_VDss_L_kg.notna() & master.human_fup.notna()).sum()}")

    # ── Load Lombardo ─────────────────────────────────────────────────────────
    print(f"\nLoading Lombardo: {LOMBARDO.name}")
    lomb = pd.read_excel(LOMBARDO, sheet_name='Data_sheet', header=8, engine='openpyxl')
    lomb.columns = [str(c).strip() for c in lomb.columns]
    lomb = lomb.rename(columns={
        'Name':                             'compound_name',
        'SMILES':                           'mol',
        'human VDss (L/kg)':               'human_VDss_L_kg',
        'human CL (mL/min/kg)':            'human_CL_mL_min_kg',
        'fraction unbound \nin plasma (fu)':'human_fup',
        'terminal  t1/2 (h)':              'terminal_t12_h',
        'MRT (h)':                         'MRT_h',
    })

    # Keep only columns we need
    keep_cols = ['compound_name', 'mol', 'human_CL_mL_min_kg',
                 'human_VDss_L_kg', 'human_fup', 'terminal_t12_h', 'MRT_h']
    lomb = lomb[[c for c in keep_cols if c in lomb.columns]].copy()

    # Clean up
    lomb = lomb[lomb['mol'].notna() & (lomb['mol'].astype(str) != 'nan')]
    lomb['human_CL_mL_min_kg'] = to_numeric_safe(lomb['human_CL_mL_min_kg'])
    lomb['human_VDss_L_kg']    = to_numeric_safe(lomb['human_VDss_L_kg'])
    lomb['human_fup']          = to_numeric_safe(lomb['human_fup'])

    print(f"  Lombardo compounds:          {len(lomb)}")
    print(f"  With CL:                     {lomb.human_CL_mL_min_kg.notna().sum()}")
    print(f"  With Vd:                     {lomb.human_VDss_L_kg.notna().sum()}")
    print(f"  With fup:                    {lomb.human_fup.notna().sum()}")

    # ── Generate InChIKeys ────────────────────────────────────────────────────
    print("\nGenerating InChIKeys for structure-based matching...")
    master['inchikey'] = master['mol'].apply(safe_inchikey)
    lomb['inchikey']   = lomb['mol'].apply(safe_inchikey)

    master_keys = set(master['inchikey'].dropna())
    lomb_keys   = set(lomb['inchikey'].dropna())
    overlap     = master_keys & lomb_keys
    new_only    = lomb_keys - master_keys

    print(f"  Overlapping compounds:       {len(overlap)}")
    print(f"  New from Lombardo:           {len(new_only)}")

    # ── Step 1: Fill gaps in overlapping compounds ────────────────────────────
    print("\nStep 1: Filling gaps in overlapping compounds...")

    # Build lookup from Lombardo by inchikey (take first match if duplicates)
    lomb_lookup = lomb.dropna(subset=['inchikey']).drop_duplicates('inchikey').set_index('inchikey')

    filled_cl = filled_vd = filled_fup = 0
    for idx, row in master.iterrows():
        ik = row['inchikey']
        if ik not in overlap or ik not in lomb_lookup.index:
            continue
        lrow = lomb_lookup.loc[ik]
        if pd.isna(row['human_CL_mL_min_kg']) and pd.notna(lrow['human_CL_mL_min_kg']):
            master.at[idx, 'human_CL_mL_min_kg'] = lrow['human_CL_mL_min_kg']
            filled_cl += 1
        if pd.isna(row['human_VDss_L_kg']) and pd.notna(lrow['human_VDss_L_kg']):
            master.at[idx, 'human_VDss_L_kg'] = lrow['human_VDss_L_kg']
            filled_vd += 1
        if pd.isna(row['human_fup']) and pd.notna(lrow['human_fup']):
            master.at[idx, 'human_fup'] = lrow['human_fup']
            filled_fup += 1

    print(f"  CL gaps filled:   {filled_cl}")
    print(f"  Vd gaps filled:   {filled_vd}")
    print(f"  fup gaps filled:  {filled_fup}")

    # ── Step 2: Add new Lombardo compounds ────────────────────────────────────
    print("\nStep 2: Adding new Lombardo compounds...")
    new_lomb = lomb[lomb['inchikey'].isin(new_only)].copy()

    # Only add compounds that have at least CL + Vd
    new_lomb_valid = new_lomb[
        new_lomb['human_CL_mL_min_kg'].notna() &
        new_lomb['human_VDss_L_kg'].notna()
    ].copy()

    print(f"  New compounds with CL+Vd:    {len(new_lomb_valid)}")
    print(f"  New compounds with CL+Vd+fup:{(new_lomb_valid.human_CL_mL_min_kg.notna() & new_lomb_valid.human_VDss_L_kg.notna() & new_lomb_valid.human_fup.notna()).sum()}")

    # Add data_source column to track provenance
    if 'data_source' not in master.columns:
        master['data_source'] = 'master_original'
    new_lomb_valid = new_lomb_valid.copy()
    new_lomb_valid['data_source'] = 'lombardo_2018'

    # Align columns — add any missing columns as NaN
    for col in master.columns:
        if col not in new_lomb_valid.columns:
            new_lomb_valid[col] = np.nan
    new_lomb_valid = new_lomb_valid[master.columns]

    # Concatenate
    master_merged = pd.concat([master, new_lomb_valid], ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"MERGE RESULTS")
    print(f"{'='*55}")
    print(f"  Original master compounds:   {len(master)}")
    print(f"  New Lombardo compounds added:{len(new_lomb_valid)}")
    print(f"  Final dataset size:          {len(master_merged)}")
    print(f"\n  Before merge:")
    print(f"    Complete (CL+Vd+fup):      {(master.human_CL_mL_min_kg.notna() & master.human_VDss_L_kg.notna() & master.human_fup.notna()).sum()}")
    print(f"    With CL+Vd:                {(master.human_CL_mL_min_kg.notna() & master.human_VDss_L_kg.notna()).sum()}")
    print(f"\n  After merge:")
    complete_after = (master_merged.human_CL_mL_min_kg.notna() & master_merged.human_VDss_L_kg.notna() & master_merged.human_fup.notna()).sum()
    cl_vd_after    = (master_merged.human_CL_mL_min_kg.notna() & master_merged.human_VDss_L_kg.notna()).sum()
    print(f"    Complete (CL+Vd+fup):      {complete_after}")
    print(f"    With CL+Vd:                {cl_vd_after}")
    print(f"    Missing CL:                {master_merged.human_CL_mL_min_kg.isna().sum()}")
    print(f"    Missing Vd:                {master_merged.human_VDss_L_kg.isna().sum()}")
    print(f"    Missing fup:               {master_merged.human_fup.isna().sum()}")
    print(f"{'='*55}")

    # Drop inchikey column before saving (internal use only)
    master_merged = master_merged.drop(columns=['inchikey'], errors='ignore')

    # ── Save ──────────────────────────────────────────────────────────────────
    with pd.ExcelWriter(MASTER_OUT, engine='openpyxl') as writer:
        master_merged.to_excel(writer, sheet_name='All_Compounds', index=False)
        for sname, sdf in other_sheets.items():
            sdf.to_excel(writer, sheet_name=sname, index=False)

    print(f"\nSaved → {MASTER_OUT}")


if __name__ == '__main__':
    run()
