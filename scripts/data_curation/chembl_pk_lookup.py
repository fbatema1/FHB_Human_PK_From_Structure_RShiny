"""
ChEMBL Bulk PK Lookup — Batched
================================
Looks up human CL, Vd, and fup for compounds in the Enamine_New_Pending_PK
sheet of PKIP_master_dataset_FHB_06052026.xlsx.

Uses batch API requests (50 compounds per call) — ~5 min for 900 compounds.

Run:
    python chembl_pk_lookup.py
"""

import pandas as pd
import numpy as np
import time
import warnings
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
MASTER_PATH  = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026.xlsx"
OUT_PATH     = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026_PKfilled.xlsx"
BATCH_SIZE   = 50
BODY_WEIGHT_KG = 70.0

# ── UNIT CONVERSIONS ─────────────────────────────────────────────────────────
def convert_cl(value, unit):
    if value is None: return None
    v = float(value); u = str(unit).lower().strip()
    if 'ul/min/mg' in u:   return None   # microsomal — skip
    if 'ml/min/kg' in u:   return v
    if 'ml/kg/min' in u:   return v
    if u == 'ml/min':      return v / BODY_WEIGHT_KG
    if 'l/h/kg' in u:      return v * 1000 / 60
    if 'l/hr/kg' in u:     return v * 1000 / 60
    if u == 'l/h':         return v * 1000 / 60 / BODY_WEIGHT_KG
    if 'ml/h/kg' in u:     return v / 60
    return None

def convert_vd(value, unit):
    if value is None: return None
    v = float(value); u = str(unit).lower().strip()
    if 'l/kg' in u:        return v
    if 'l kg-1' in u:      return v
    if 'ml/kg' in u:       return v / 1000
    if u == 'l':           return v / BODY_WEIGHT_KG
    if u == 'ml':          return v / 1000 / BODY_WEIGHT_KG
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

# ── STEP 1: Generate InChIKeys locally ───────────────────────────────────────
def get_inchikey(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return MolToInchiKey(mol) if mol else None
    except:
        return None

# ── STEP 2: Batch ChEMBL ID lookup by InChIKey ───────────────────────────────
def batch_get_chembl_ids(inchikeys, molecule):
    """Send up to BATCH_SIZE InChIKeys at once, return {inchikey: chembl_id}."""
    valid = [k for k in inchikeys if k]
    if not valid:
        return {}
    results = {}
    try:
        hits = molecule.filter(
            molecule_structures__standard_inchi_key__in=valid
        ).only(['molecule_chembl_id', 'molecule_structures'])
        for h in hits:
            ik = (h.get('molecule_structures') or {}).get('standard_inchi_key')
            if ik:
                results[ik] = h['molecule_chembl_id']
    except Exception as e:
        print(f"  Batch ID lookup error: {e}")
    return results

# ── STEP 3: Batch activity lookup by ChEMBL ID ───────────────────────────────
def batch_get_activities(chembl_ids, activity):
    """Pull ADME activities for a batch of ChEMBL IDs."""
    if not chembl_ids:
        return {}
    pk_data = {cid: {'cl': [], 'vd': [], 'fup': []} for cid in chembl_ids}
    try:
        acts = activity.filter(
            molecule_chembl_id__in=list(chembl_ids),
            assay_type='A',
        ).only(['molecule_chembl_id', 'standard_type', 'standard_value',
                'standard_units', 'assay_description'])
        for a in acts:
            cid   = a.get('molecule_chembl_id')
            stype = str(a.get('standard_type') or '').lower().strip()
            sval  = a.get('standard_value')
            sunit = str(a.get('standard_units') or '')
            adesc = str(a.get('assay_description') or '').lower()
            if cid not in pk_data or sval is None:
                continue
            if not any(w in adesc for w in ('human', 'homo sapiens')):
                continue
            if stype in CL_TYPES:
                v = convert_cl(sval, sunit)
                if v and 0.001 < v < 500:
                    pk_data[cid]['cl'].append(v)
            elif stype in VD_TYPES:
                v = convert_vd(sval, sunit)
                if v and 0.001 < v < 500:
                    pk_data[cid]['vd'].append(v)
            elif stype in FUP_TYPES:
                v = convert_fup(sval, sunit)
                if v and 0 < v <= 1:
                    pk_data[cid]['fup'].append(v)
    except Exception as e:
        print(f"  Batch activity error: {e}")
    return pk_data

# ── MAIN ─────────────────────────────────────────────────────────────────────
def run_lookup():
    from chembl_webresource_client.new_client import new_client
    from chembl_webresource_client.settings import Settings
    Settings.Instance().TIMEOUT = 30

    molecule = new_client.molecule
    activity = new_client.activity

    xl = pd.ExcelFile(MASTER_PATH, engine='calamine')
    pending = pd.read_excel(xl, sheet_name='Enamine_New_Pending_PK')
    n = len(pending)
    print(f"Loaded {n} compounds")

    # Generate InChIKeys locally (instant)
    print("Generating InChIKeys locally...")
    pending['inchikey'] = pending['mol'].apply(get_inchikey)
    valid_ik = pending['inchikey'].notna().sum()
    print(f"Valid InChIKeys: {valid_ik}/{n}")

    # Batch ChEMBL ID lookup
    print(f"\nStep 1/2: Looking up ChEMBL IDs in batches of {BATCH_SIZE}...")
    ik_to_chembl = {}
    inchikeys = pending['inchikey'].tolist()
    batches = [inchikeys[i:i+BATCH_SIZE] for i in range(0, len(inchikeys), BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        print(f"  ID batch {b_idx+1}/{len(batches)}...", end=' ', flush=True)
        hits = batch_get_chembl_ids(batch, molecule)
        ik_to_chembl.update(hits)
        print(f"{len(hits)} hits")
        time.sleep(0.5)

    pending['chembl_id'] = pending['inchikey'].map(ik_to_chembl)
    found_ids = pending['chembl_id'].notna().sum()
    print(f"ChEMBL IDs found: {found_ids}/{n}")

    # Batch activity lookup
    print(f"\nStep 2/2: Fetching PK activities in batches of {BATCH_SIZE}...")
    chembl_ids = pending['chembl_id'].dropna().unique().tolist()
    all_pk = {}
    id_batches = [chembl_ids[i:i+BATCH_SIZE] for i in range(0, len(chembl_ids), BATCH_SIZE)]
    for b_idx, batch in enumerate(id_batches):
        print(f"  Activity batch {b_idx+1}/{len(id_batches)}...", end=' ', flush=True)
        pk = batch_get_activities(batch, activity)
        all_pk.update(pk)
        found_in_batch = sum(1 for v in pk.values() if v['cl'] or v['vd'] or v['fup'])
        print(f"{found_in_batch} with PK data")
        time.sleep(0.5)

    # Assemble results
    def median_or_nan(lst): return float(np.median(lst)) if lst else np.nan

    pending['human_CL_mL_min_kg'] = pending['chembl_id'].apply(
        lambda cid: median_or_nan(all_pk.get(cid, {}).get('cl', [])))
    pending['human_VDss_L_kg'] = pending['chembl_id'].apply(
        lambda cid: median_or_nan(all_pk.get(cid, {}).get('vd', [])))
    pending['human_fup'] = pending['chembl_id'].apply(
        lambda cid: median_or_nan(all_pk.get(cid, {}).get('fup', [])))

    found_cl  = pending['human_CL_mL_min_kg'].notna().sum()
    found_vd  = pending['human_VDss_L_kg'].notna().sum()
    found_fup = pending['human_fup'].notna().sum()
    has_pk    = pending[pending['human_CL_mL_min_kg'].notna() | pending['human_VDss_L_kg'].notna()]
    no_pk     = pending[pending['human_CL_mL_min_kg'].isna() & pending['human_VDss_L_kg'].isna()]

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"  ChEMBL ID found:        {found_ids}/{n}")
    print(f"  CL retrieved:           {found_cl}")
    print(f"  Vd retrieved:           {found_vd}")
    print(f"  fup retrieved:          {found_fup}")
    print(f"  Compounds with ≥1 PK:   {len(has_pk)}")
    print(f"{'='*50}")

    # Rebuild All_Compounds
    all_orig = pd.read_excel(xl, sheet_name='All_Compounds')
    pk_cols = ['NAME', 'mol', 'human_CL_mL_min_kg', 'human_VDss_L_kg', 'human_fup']
    new_with_pk = has_pk[pk_cols].copy()
    new_with_pk['source'] = 'Enamine_FDA_ChEMBL'
    all_updated = pd.concat([all_orig, new_with_pk], ignore_index=True)
    print(f"  Updated All_Compounds: {len(all_updated)} total compounds")

    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
        all_updated.to_excel(writer, sheet_name='All_Compounds', index=False)
        all_orig.to_excel(writer, sheet_name='Previous_824', index=False)
        has_pk.to_excel(writer, sheet_name='Enamine_PK_Found', index=False)
        no_pk[['NAME','mol','Catalog_ID','MW','ClogP']].to_excel(
            writer, sheet_name='Enamine_Still_Pending', index=False)
        pd.DataFrame({
            'Metric': ['Previous master','Searched','ChEMBL matched',
                       'CL retrieved','Vd retrieved','fup retrieved',
                       'New compounds added','Updated total'],
            'Value':  [len(all_orig), n, found_ids, found_cl, found_vd,
                       found_fup, len(new_with_pk), len(all_updated)]
        }).to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nSaved → {OUT_PATH}")

if __name__ == '__main__':
    run_lookup()
