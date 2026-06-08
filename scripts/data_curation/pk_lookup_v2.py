"""
pk_lookup_v2.py
===============
Looks up human CL, Vd, and fup for compounds missing PK data.
Searches ChEMBL by compound name (much faster + higher hit rate than SMILES).

Run:
    conda activate pkip-env
    python pk_lookup_v2.py
"""

import pandas as pd
import numpy as np
import time
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings
Settings.Instance().TIMEOUT = 30

MASTER_PATH = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/master_dataset_FHB_06052026_v2.xlsx"
OUT_PATH    = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/master_dataset_FHB_06052026_v2.xlsx"
BATCH_SIZE  = 50
BODY_WT     = 70.0

# ── Unit conversions ─────────────────────────────────────────────────────────
def convert_cl(value, unit):
    if value is None: return None
    v = float(value); u = str(unit).lower().strip()
    if 'ul/min/mg' in u:   return None
    if 'ml/min/kg' in u:   return v
    if 'ml/kg/min' in u:   return v
    if u == 'ml/min':      return v / BODY_WT
    if 'l/h/kg'  in u:     return v * 1000 / 60
    if 'l/hr/kg' in u:     return v * 1000 / 60
    if u == 'l/h':         return v * 1000 / 60 / BODY_WT
    if 'ml/h/kg' in u:     return v / 60
    return None

def convert_vd(value, unit):
    if value is None: return None
    v = float(value); u = str(unit).lower().strip()
    if 'l/kg'   in u:      return v
    if 'l kg-1' in u:      return v
    if 'ml/kg'  in u:      return v / 1000
    if u == 'l':           return v / BODY_WT
    if u == 'ml':          return v / 1000 / BODY_WT
    return None

def convert_fup(value, unit):
    if value is None: return None
    v = float(value); u = str(unit).lower().strip()
    if '%' in u:           return v / 100
    if 0 <= v <= 1:        return v
    if 0 < v <= 100:       return v / 100
    return None

CL_TYPES  = {'clearance','cl','cltot','clsys','clp','clb',
             'total clearance','systemic clearance','plasma clearance'}
VD_TYPES  = {'vdss','vd','volume of distribution','vss',
             'apparent volume of distribution',
             'volume of distribution at steady state'}
FUP_TYPES = {'fu','fup','fraction unbound','f unbound','fu,p',
             'fraction unbound in plasma','free fraction'}

def get_pk_for_chembl_ids(chembl_ids, activity):
    """Batch fetch PK activities for a list of ChEMBL IDs."""
    pk = {cid: {'cl': [], 'vd': [], 'fup': []} for cid in chembl_ids}
    try:
        acts = activity.filter(
            molecule_chembl_id__in=list(chembl_ids),
            assay_type='A',
        ).only(['molecule_chembl_id','standard_type','standard_value',
                'standard_units','assay_description'])
        for a in acts:
            cid   = a.get('molecule_chembl_id')
            stype = str(a.get('standard_type') or '').lower().strip()
            sval  = a.get('standard_value')
            sunit = str(a.get('standard_units') or '')
            adesc = str(a.get('assay_description') or '').lower()
            if cid not in pk or sval is None: continue
            # Human filter — flexible
            if not any(w in adesc for w in ('human','homo sapien','in vivo','iv ','oral','po ')):
                continue
            if stype in CL_TYPES:
                v = convert_cl(sval, sunit)
                if v and 0.001 < v < 500: pk[cid]['cl'].append(v)
            elif stype in VD_TYPES:
                v = convert_vd(sval, sunit)
                if v and 0.001 < v < 500: pk[cid]['vd'].append(v)
            elif stype in FUP_TYPES:
                v = convert_fup(sval, sunit)
                if v and 0 < v <= 1: pk[cid]['fup'].append(v)
    except Exception as e:
        print(f"  Activity batch error: {e}")
    return pk

def median_or_nan(lst):
    return float(np.median(lst)) if lst else np.nan

def run():
    molecule = new_client.molecule
    activity = new_client.activity

    xl = pd.ExcelFile(MASTER_PATH, engine='calamine')
    df = pd.read_excel(xl, sheet_name='All_Compounds', engine='calamine')

    # Compounds missing PK
    missing = df[df['human_CL_mL_min_kg'].isna() & df['human_VDss_L_kg'].isna()].copy()
    print(f"Compounds missing PK: {len(missing)}")

    # ── Step 1: Name → ChEMBL ID (batch by preferred name) ──────────────────
    print(f"\nStep 1: Looking up ChEMBL IDs by name (batches of {BATCH_SIZE})...")
    names = missing['compound_name'].dropna().unique().tolist()

    # Strip salt suffixes for better matching
    def clean_name(n):
        for suffix in [' hydrochloride',' hcl',' sodium',' potassium',
                       ' sulfate',' mesylate',' maleate',' tartrate',
                       ' acetate',' phosphate',' fumarate',' citrate',
                       ' dihydrochloride',' monohydrate']:
            if n.lower().endswith(suffix):
                return n[:len(n)-len(suffix)].strip()
        return n

    name_to_chembl = {}
    batches = [names[i:i+BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        print(f"  Name batch {b_idx+1}/{len(batches)}...", end=' ', flush=True)
        found = 0
        try:
            # Try exact preferred name match
            hits = molecule.filter(
                pref_name__in=[n.upper() for n in batch]
            ).only(['molecule_chembl_id','pref_name'])
            for h in hits:
                pname = (h.get('pref_name') or '').title()
                cid   = h.get('molecule_chembl_id')
                if pname and cid:
                    name_to_chembl[pname] = cid
                    found += 1
        except Exception as e:
            print(f"error: {e}")

        # For unmatched, try cleaned names
        unmatched = [n for n in batch if n not in name_to_chembl]
        cleaned   = [(n, clean_name(n)) for n in unmatched if clean_name(n) != n]
        if cleaned:
            try:
                clean_names = [c[1] for c in cleaned]
                hits2 = molecule.filter(
                    pref_name__in=[n.upper() for n in clean_names]
                ).only(['molecule_chembl_id','pref_name'])
                lookup = {h['pref_name'].title(): h['molecule_chembl_id'] for h in hits2 if h.get('pref_name')}
                for orig, clean in cleaned:
                    if clean.title() in lookup:
                        name_to_chembl[orig] = lookup[clean.title()]
                        found += 1
            except:
                pass

        print(f"{found} found (total: {len(name_to_chembl)})")
        time.sleep(0.4)

    print(f"\nChEMBL IDs found: {len(name_to_chembl)}/{len(names)}")
    missing['chembl_id'] = missing['compound_name'].map(name_to_chembl)

    # ── Step 2: Batch PK activity lookup ────────────────────────────────────
    chembl_ids = missing['chembl_id'].dropna().unique().tolist()
    ACT_BATCH  = 20   # smaller batches to avoid ChEMBL 500 errors
    print(f"\nStep 2: Fetching PK for {len(chembl_ids)} compounds (batches of {ACT_BATCH})...")
    all_pk = {}
    id_batches = [chembl_ids[i:i+ACT_BATCH] for i in range(0, len(chembl_ids), ACT_BATCH)]
    for b_idx, batch in enumerate(id_batches):
        print(f"  Activity batch {b_idx+1}/{len(id_batches)}...", end=' ', flush=True)
        # Retry up to 3 times on failure
        for attempt in range(3):
            pk = get_pk_for_chembl_ids(batch, activity)
            if pk:
                break
            print(f"retry {attempt+1}...", end=' ', flush=True)
            time.sleep(2)
        all_pk.update(pk)
        with_pk = sum(1 for v in pk.values() if v['cl'] or v['vd'] or v['fup'])
        print(f"{with_pk} with PK data")
        time.sleep(0.8)

    # ── Fill in values ───────────────────────────────────────────────────────
    missing['human_CL_mL_min_kg'] = missing['chembl_id'].apply(
        lambda c: median_or_nan(all_pk.get(c, {}).get('cl', [])))
    missing['human_VDss_L_kg'] = missing['chembl_id'].apply(
        lambda c: median_or_nan(all_pk.get(c, {}).get('vd', [])))
    missing['human_fup'] = missing['chembl_id'].apply(
        lambda c: median_or_nan(all_pk.get(c, {}).get('fup', [])))

    got_cl  = missing['human_CL_mL_min_kg'].notna().sum()
    got_vd  = missing['human_VDss_L_kg'].notna().sum()
    got_fup = missing['human_fup'].notna().sum()
    got_any = (missing['human_CL_mL_min_kg'].notna() | missing['human_VDss_L_kg'].notna()).sum()

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"  ChEMBL ID found:      {len(name_to_chembl)}/{len(names)}")
    print(f"  CL retrieved:         {got_cl}")
    print(f"  Vd retrieved:         {got_vd}")
    print(f"  fup retrieved:        {got_fup}")
    print(f"  Compounds with any PK:{got_any}")
    print(f"{'='*50}")

    # ── Merge back into full dataset ─────────────────────────────────────────
    df.update(missing[['human_CL_mL_min_kg','human_VDss_L_kg','human_fup']])

    has_any = (df['human_CL_mL_min_kg'].notna() | df['human_VDss_L_kg'].notna()).sum()
    print(f"\nUpdated All_Compounds — compounds with PK: {has_any}/{len(df)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    other_sheets = {s: pd.read_excel(xl, sheet_name=s, engine='calamine')
                    for s in xl.sheet_names if s != 'All_Compounds'}
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Compounds', index=False)
        for sheet, sdf in other_sheets.items():
            sdf.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved → {OUT_PATH}")

if __name__ == '__main__':
    run()
