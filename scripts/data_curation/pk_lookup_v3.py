"""
pk_lookup_v3.py
===============
Looks up missing human PK data (CL, Vd, fup) for compounds in the
All_Compounds sheet of the master dataset via the ChEMBL API.

Handles three cases:
  1. Missing ALL PK values (CL, Vd, fup all null)
  2. Missing fup only (CL and Vd already present)
  3. Missing any individual value

Results are merged back into the master dataset and saved in-place.

Run:
    conda activate pkip-env
    python scripts/data_curation/pk_lookup_v3.py

Dependencies:
    pip install pandas chembl-webresource-client openpyxl
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings

Settings.Instance().TIMEOUT = 30

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MASTER_IN  = ROOT / "data/raw/master_dataset_FHB_06052026_v2.xlsx"
MASTER_OUT = ROOT / "data/raw/master_dataset_FHB_06052026_v3.xlsx"

BATCH_SIZE     = 50   # name → ChEMBL ID lookup batch size
ACT_BATCH_SIZE = 20   # activity fetch batch size (keep small to avoid ChEMBL 500s)
BODY_WT        = 70.0 # kg, for unit conversions

# ── Unit conversions ──────────────────────────────────────────────────────────
def convert_cl(value, unit):
    """Convert CL to mL/min/kg."""
    if value is None:
        return None
    v, u = float(value), str(unit).lower().strip()
    if 'ul/min/mg' in u:  return None          # microsomal — not in vivo
    if 'ml/min/kg' in u:  return v
    if 'ml/kg/min' in u:  return v
    if u == 'ml/min':     return v / BODY_WT
    if 'l/h/kg'  in u:    return v * 1000 / 60
    if 'l/hr/kg' in u:    return v * 1000 / 60
    if u == 'l/h':        return v * 1000 / 60 / BODY_WT
    if 'ml/h/kg' in u:    return v / 60
    return None

def convert_vd(value, unit):
    """Convert Vd to L/kg."""
    if value is None:
        return None
    v, u = float(value), str(unit).lower().strip()
    if 'l/kg'   in u:     return v
    if 'l kg-1' in u:     return v
    if 'ml/kg'  in u:     return v / 1000
    if u == 'l':          return v / BODY_WT
    if u == 'ml':         return v / 1000 / BODY_WT
    return None

def convert_fup(value, unit):
    """Convert fup to fraction (0–1)."""
    if value is None:
        return None
    v, u = float(value), str(unit).lower().strip()
    if '%' in u:          return v / 100
    if 0 <= v <= 1:       return v
    if 0 < v <= 100:      return v / 100
    return None

# ── ChEMBL standard type sets ─────────────────────────────────────────────────
CL_TYPES = {
    'clearance', 'cl', 'cltot', 'clsys', 'clp', 'clb',
    'total clearance', 'systemic clearance', 'plasma clearance'
}
VD_TYPES = {
    'vdss', 'vd', 'volume of distribution', 'vss',
    'apparent volume of distribution',
    'volume of distribution at steady state'
}
FUP_TYPES = {
    'fu', 'fup', 'fraction unbound', 'f unbound', 'fu,p',
    'fraction unbound in plasma', 'free fraction'
}

HUMAN_KEYWORDS = ('human', 'homo sapien', 'in vivo', 'iv ', 'oral', 'po ')


# ── Core lookup functions ─────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    """Strip common salt/hydrate suffixes for better ChEMBL name matching."""
    suffixes = [
        ' hydrochloride', ' hcl', ' sodium', ' potassium',
        ' sulfate', ' mesylate', ' maleate', ' tartrate',
        ' acetate', ' phosphate', ' fumarate', ' citrate',
        ' dihydrochloride', ' monohydrate', ' hydrate'
    ]
    n = name.lower()
    for s in suffixes:
        if n.endswith(s):
            return name[:len(name) - len(s)].strip()
    return name


def lookup_chembl_ids(names: list, molecule) -> dict:
    """
    Batch-resolve compound names to ChEMBL IDs.
    Returns dict: {compound_name: chembl_id}
    """
    name_to_chembl = {}
    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]

    for b_idx, batch in enumerate(batches):
        print(f"  Name batch {b_idx + 1}/{len(batches)}...", end=' ', flush=True)
        found = 0

        # Try exact preferred name match
        try:
            hits = molecule.filter(
                pref_name__in=[n.upper() for n in batch]
            ).only(['molecule_chembl_id', 'pref_name'])
            for h in hits:
                pname = (h.get('pref_name') or '').title()
                cid = h.get('molecule_chembl_id')
                if pname and cid:
                    name_to_chembl[pname] = cid
                    found += 1
        except Exception as e:
            print(f"  [name batch error: {e}]", end=' ')

        # Retry unmatched with cleaned names (salt stripped)
        unmatched = [n for n in batch if n not in name_to_chembl]
        cleaned_pairs = [(n, clean_name(n)) for n in unmatched if clean_name(n) != n]
        if cleaned_pairs:
            try:
                clean_map = {c[1].upper(): c[0] for c in cleaned_pairs}
                hits2 = molecule.filter(
                    pref_name__in=list(clean_map.keys())
                ).only(['molecule_chembl_id', 'pref_name'])
                for h in hits2:
                    pname_upper = (h.get('pref_name') or '').upper()
                    cid = h.get('molecule_chembl_id')
                    if pname_upper in clean_map and cid:
                        original_name = clean_map[pname_upper]
                        name_to_chembl[original_name] = cid
                        found += 1
            except Exception as e:
                print(f"  [cleaned name batch error: {e}]", end=' ')

        print(f"{found} found (running total: {len(name_to_chembl)})")
        time.sleep(0.4)

    return name_to_chembl


def fetch_pk_activities(chembl_ids: list, activity) -> dict:
    """
    Batch-fetch PK activity data for a list of ChEMBL IDs.
    Returns dict: {chembl_id: {'cl': [values], 'vd': [values], 'fup': [values]}}
    """
    pk = {cid: {'cl': [], 'vd': [], 'fup': []} for cid in chembl_ids}
    id_batches = [chembl_ids[i:i + ACT_BATCH_SIZE]
                  for i in range(0, len(chembl_ids), ACT_BATCH_SIZE)]

    for b_idx, batch in enumerate(id_batches):
        print(f"  Activity batch {b_idx + 1}/{len(id_batches)}...", end=' ', flush=True)

        for attempt in range(3):
            try:
                acts = activity.filter(
                    molecule_chembl_id__in=list(batch),
                    assay_type='A',
                ).only(['molecule_chembl_id', 'standard_type', 'standard_value',
                        'standard_units', 'assay_description'])

                batch_hits = 0
                for a in acts:
                    cid   = a.get('molecule_chembl_id')
                    stype = str(a.get('standard_type') or '').lower().strip()
                    sval  = a.get('standard_value')
                    sunit = str(a.get('standard_units') or '')
                    adesc = str(a.get('assay_description') or '').lower()

                    if cid not in pk or sval is None:
                        continue
                    # Filter to human in vivo assays
                    if not any(w in adesc for w in HUMAN_KEYWORDS):
                        continue

                    if stype in CL_TYPES:
                        v = convert_cl(sval, sunit)
                        if v and 0.001 < v < 500:
                            pk[cid]['cl'].append(v)
                            batch_hits += 1
                    elif stype in VD_TYPES:
                        v = convert_vd(sval, sunit)
                        if v and 0.001 < v < 500:
                            pk[cid]['vd'].append(v)
                            batch_hits += 1
                    elif stype in FUP_TYPES:
                        v = convert_fup(sval, sunit)
                        if v and 0 < v <= 1:
                            pk[cid]['fup'].append(v)
                            batch_hits += 1

                print(f"{batch_hits} PK values retrieved")
                break

            except Exception as e:
                print(f"  [attempt {attempt + 1} failed: {e}]", end=' ')
                time.sleep(2)

        time.sleep(0.8)

    return pk


def median_or_nan(lst: list):
    return float(np.median(lst)) if lst else np.nan


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    molecule = new_client.molecule
    activity = new_client.activity

    print(f"Loading: {MASTER_IN}")
    xl = pd.ExcelFile(MASTER_IN, engine='openpyxl')
    df = pd.read_excel(xl, sheet_name='All_Compounds', engine='openpyxl')

    # ── Identify compounds needing lookup ─────────────────────────────────────
    missing_any = df[
        df['human_CL_mL_min_kg'].isna() |
        df['human_VDss_L_kg'].isna()    |
        df['human_fup'].isna()
    ].copy()

    missing_all  = missing_any[missing_any['human_CL_mL_min_kg'].isna() &
                                missing_any['human_VDss_L_kg'].isna()].shape[0]
    missing_fup  = missing_any[missing_any['human_fup'].isna()].shape[0]

    print(f"\nDataset summary:")
    print(f"  Total compounds:              {len(df)}")
    print(f"  Complete (CL + Vd + fup):     {(df['human_CL_mL_min_kg'].notna() & df['human_VDss_L_kg'].notna() & df['human_fup'].notna()).sum()}")
    print(f"  Missing CL or Vd:             {missing_all}")
    print(f"  Missing fup:                  {missing_fup}")
    print(f"  Total needing lookup:         {len(missing_any)}")

    # ── Step 1: Name → ChEMBL ID ──────────────────────────────────────────────
    names = missing_any['compound_name'].dropna().unique().tolist()
    print(f"\nStep 1: Resolving {len(names)} compound names to ChEMBL IDs...")
    name_to_chembl = lookup_chembl_ids(names, molecule)
    print(f"  Resolved: {len(name_to_chembl)}/{len(names)}")

    missing_any['chembl_id'] = missing_any['compound_name'].map(name_to_chembl)

    # ── Step 2: Fetch PK activities ───────────────────────────────────────────
    chembl_ids = missing_any['chembl_id'].dropna().unique().tolist()
    print(f"\nStep 2: Fetching PK activities for {len(chembl_ids)} ChEMBL IDs...")
    all_pk = fetch_pk_activities(chembl_ids, activity)

    # ── Step 3: Fill in values (only overwrite if currently null) ─────────────
    def fill_if_missing(row, field, pk_key):
        if pd.notna(row[field]):
            return row[field]  # already has a value — don't overwrite
        cid = row.get('chembl_id')
        if pd.isna(cid):
            return np.nan
        return median_or_nan(all_pk.get(cid, {}).get(pk_key, []))

    missing_any['human_CL_mL_min_kg'] = missing_any.apply(
        lambda r: fill_if_missing(r, 'human_CL_mL_min_kg', 'cl'), axis=1)
    missing_any['human_VDss_L_kg'] = missing_any.apply(
        lambda r: fill_if_missing(r, 'human_VDss_L_kg', 'vd'), axis=1)
    missing_any['human_fup'] = missing_any.apply(
        lambda r: fill_if_missing(r, 'human_fup', 'fup'), axis=1)

    # ── Results summary ───────────────────────────────────────────────────────
    got_cl  = missing_any['human_CL_mL_min_kg'].notna().sum()
    got_vd  = missing_any['human_VDss_L_kg'].notna().sum()
    got_fup = missing_any['human_fup'].notna().sum()

    print(f"\n{'='*55}")
    print(f"LOOKUP RESULTS")
    print(f"  ChEMBL IDs resolved:     {len(name_to_chembl)}/{len(names)}")
    print(f"  CL values retrieved:     {got_cl}")
    print(f"  Vd values retrieved:     {got_vd}")
    print(f"  fup values retrieved:    {got_fup}")
    print(f"{'='*55}")

    # ── Merge back into full dataframe ────────────────────────────────────────
    df.update(missing_any[['human_CL_mL_min_kg', 'human_VDss_L_kg', 'human_fup']])

    complete_after = (
        df['human_CL_mL_min_kg'].notna() &
        df['human_VDss_L_kg'].notna() &
        df['human_fup'].notna()
    ).sum()
    print(f"\nCompounds with complete PK after update: {complete_after}/{len(df)}")

    # ── Save output ───────────────────────────────────────────────────────────
    other_sheets = {
        s: pd.read_excel(xl, sheet_name=s, engine='openpyxl')
        for s in xl.sheet_names if s != 'All_Compounds'
    }
    with pd.ExcelWriter(MASTER_OUT, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Compounds', index=False)
        for sheet_name, sheet_df in other_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved → {MASTER_OUT}")


if __name__ == '__main__':
    run()
