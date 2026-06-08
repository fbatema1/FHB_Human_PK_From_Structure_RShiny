"""
chembl_sqlite_pull.py
=====================
Downloads ChEMBL 35 SQLite database (~7GB, one-time) and extracts
ALL human in vivo CL, Vd, and fup data.

Then merges with master dataset to fill in missing PK values.

Run:
    pip install chembl-downloader
    conda activate pkip-env
    python chembl_sqlite_pull.py

First run takes 20-40 min to download. Subsequent runs are instant
(file is cached locally by chembl-downloader).
"""

import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

MASTER_PATH = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/master_dataset_FHB_06052026_v2.xlsx"
OUT_PATH    = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/master_dataset_FHB_06052026_v2.xlsx"
BODY_WT     = 70.0

# ── Unit conversions ─────────────────────────────────────────────────────────
def convert_cl(value, unit):
    if value is None: return None
    try: v = float(value)
    except: return None
    u = str(unit).lower().strip()
    if 'ul/min/mg' in u:   return None   # microsomal — skip
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
    try: v = float(value)
    except: return None
    u = str(unit).lower().strip()
    if 'l/kg'   in u:      return v
    if 'l kg-1' in u:      return v
    if 'ml/kg'  in u:      return v / 1000
    if u == 'l':           return v / BODY_WT
    if u == 'ml':          return v / 1000 / BODY_WT
    return None

def convert_fup(value, unit):
    if value is None: return None
    try: v = float(value)
    except: return None
    u = str(unit).lower().strip()
    if '%' in u:           return v / 100
    if 0 <= v <= 1:        return v
    if 0 < v <= 100:       return v / 100
    return None

def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None

# ── Step 1: Download / locate SQLite ─────────────────────────────────────────
print("Step 1: Getting ChEMBL SQLite database...")
print("  (First run downloads ~7GB — takes 20-40 min on typical connection)")
print("  (Subsequent runs use cached file — instant)")

import chembl_downloader
db_path = chembl_downloader.download_extract_sqlite()
print(f"  Database path: {db_path}")

# ── Step 2: Query all human ADME PK data ─────────────────────────────────────
print("\nStep 2: Querying for human CL, Vd, fup...")

query = """
SELECT
    md.chembl_id,
    md.pref_name,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_value,
    act.standard_units,
    ass.description        AS assay_description,
    ass.assay_organism
FROM activities act
JOIN assays           ass ON act.assay_id  = ass.assay_id
JOIN molecule_dictionary md ON act.molregno = md.molregno
LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE ass.assay_type = 'A'
  AND upper(act.standard_type) IN (
      'CLEARANCE','CL','CLTOT','CLSYS','CLP','CLB',
      'TOTAL CLEARANCE','SYSTEMIC CLEARANCE','PLASMA CLEARANCE',
      'VDSS','VD','VOLUME OF DISTRIBUTION','VSS',
      'APPARENT VOLUME OF DISTRIBUTION',
      'VOLUME OF DISTRIBUTION AT STEADY STATE',
      'FU','FUP','FRACTION UNBOUND','F UNBOUND','FU,P',
      'FRACTION UNBOUND IN PLASMA','FREE FRACTION'
  )
  AND act.standard_value IS NOT NULL
  AND (
      lower(ass.assay_organism) LIKE '%homo sapiens%'
      OR lower(ass.description) LIKE '%human%'
      OR lower(ass.description) LIKE '%in vivo%'
  )
"""

conn = sqlite3.connect(db_path)
print("  Running query (may take 1-2 min)...")
raw = pd.read_sql_query(query, conn)
conn.close()

print(f"  Raw activity records: {len(raw)}")
print(f"  Unique compounds:     {raw['chembl_id'].nunique()}")
print(f"  standard_type values: {raw['standard_type'].value_counts().head(10).to_dict()}")

# ── Step 3: Convert units and aggregate per compound ─────────────────────────
print("\nStep 3: Converting units and aggregating...")

CL_TYPES  = {'clearance','cl','cltot','clsys','clp','clb',
             'total clearance','systemic clearance','plasma clearance'}
VD_TYPES  = {'vdss','vd','volume of distribution','vss',
             'apparent volume of distribution',
             'volume of distribution at steady state'}
FUP_TYPES = {'fu','fup','fraction unbound','f unbound','fu,p',
             'fraction unbound in plasma','free fraction'}

records = {}  # chembl_id → {cl:[], vd:[], fup:[], smiles, name}

for _, row in raw.iterrows():
    cid   = row['chembl_id']
    stype = str(row['standard_type'] or '').lower().strip()
    sval  = row['standard_value']
    sunit = str(row['standard_units'] or '')

    if cid not in records:
        records[cid] = {
            'chembl_id': cid,
            'pref_name': row['pref_name'],
            'smiles':    row['canonical_smiles'],
            'cl': [], 'vd': [], 'fup': []
        }

    if stype in CL_TYPES:
        v = convert_cl(sval, sunit)
        if v and 0.001 < v < 500:
            records[cid]['cl'].append(v)
    elif stype in VD_TYPES:
        v = convert_vd(sval, sunit)
        if v and 0.001 < v < 500:
            records[cid]['vd'].append(v)
    elif stype in FUP_TYPES:
        v = convert_fup(sval, sunit)
        if v and 0 < v <= 1:
            records[cid]['fup'].append(v)

def med(lst): return float(np.median(lst)) if lst else np.nan

chembl_pk = pd.DataFrame([{
    'chembl_id':          r['chembl_id'],
    'pref_name':          r['pref_name'],
    'canonical_smiles':   r['smiles'],
    'human_CL_mL_min_kg': med(r['cl']),
    'human_VDss_L_kg':    med(r['vd']),
    'human_fup':          med(r['fup']),
} for r in records.values()])

has_cl  = chembl_pk['human_CL_mL_min_kg'].notna().sum()
has_vd  = chembl_pk['human_VDss_L_kg'].notna().sum()
has_fup = chembl_pk['human_fup'].notna().sum()
has_any = (chembl_pk['human_CL_mL_min_kg'].notna() | chembl_pk['human_VDss_L_kg'].notna()).sum()

print(f"  Compounds with CL:  {has_cl}")
print(f"  Compounds with Vd:  {has_vd}")
print(f"  Compounds with fup: {has_fup}")
print(f"  Compounds with any: {has_any}")

# Canonicalize SMILES for matching
print("  Canonicalizing SMILES...")
chembl_pk['canon_smi'] = chembl_pk['canonical_smiles'].apply(canonical_smiles)

# ── Step 4: Match against master and fill missing PK ─────────────────────────
print("\nStep 4: Merging with master dataset...")

xl      = pd.ExcelFile(MASTER_PATH, engine='calamine')
master  = pd.read_excel(xl, sheet_name='All_Compounds', engine='calamine')
master['canon_smi'] = master['mol'].apply(canonical_smiles)

# Build SMILES lookup from ChEMBL PK data (only compounds with any PK)
chembl_has_pk = chembl_pk[chembl_pk['human_CL_mL_min_kg'].notna() | chembl_pk['human_VDss_L_kg'].notna()]
smi_lookup = chembl_has_pk.set_index('canon_smi')[['human_CL_mL_min_kg','human_VDss_L_kg','human_fup']].to_dict('index')

# Also build name lookup (lowercase)
name_lookup = chembl_has_pk.copy()
name_lookup['name_lower'] = name_lookup['pref_name'].str.lower().str.strip()
name_lookup = name_lookup.dropna(subset=['name_lower']).set_index('name_lower')[['human_CL_mL_min_kg','human_VDss_L_kg','human_fup']].to_dict('index')

filled_cl = filled_vd = filled_fup = 0
missing_mask = master['human_CL_mL_min_kg'].isna() & master['human_VDss_L_kg'].isna()

for idx in master[missing_mask].index:
    smi   = master.at[idx, 'canon_smi']
    cname = str(master.at[idx, 'compound_name'] or '').lower().strip()

    pk = smi_lookup.get(smi) or name_lookup.get(cname)
    if not pk:
        continue

    if pd.notna(pk.get('human_CL_mL_min_kg')) and pd.isna(master.at[idx, 'human_CL_mL_min_kg']):
        master.at[idx, 'human_CL_mL_min_kg'] = pk['human_CL_mL_min_kg']
        filled_cl += 1
    if pd.notna(pk.get('human_VDss_L_kg')) and pd.isna(master.at[idx, 'human_VDss_L_kg']):
        master.at[idx, 'human_VDss_L_kg'] = pk['human_VDss_L_kg']
        filled_vd += 1
    if pd.notna(pk.get('human_fup')) and pd.isna(master.at[idx, 'human_fup']):
        master.at[idx, 'human_fup'] = pk['human_fup']
        filled_fup += 1

master = master.drop(columns=['canon_smi'], errors='ignore')

has_any_after = (master['human_CL_mL_min_kg'].notna() | master['human_VDss_L_kg'].notna()).sum()

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"  CL filled:   {filled_cl}")
print(f"  Vd filled:   {filled_vd}")
print(f"  fup filled:  {filled_fup}")
print(f"  Compounds with PK before: 787")
print(f"  Compounds with PK after:  {has_any_after}")
print(f"{'='*50}")

# ── Step 5: Save ──────────────────────────────────────────────────────────────
other_sheets = {s: pd.read_excel(xl, sheet_name=s, engine='calamine')
                for s in xl.sheet_names if s != 'All_Compounds'}

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    master.to_excel(writer, sheet_name='All_Compounds', index=False)
    for sheet, df in other_sheets.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nSaved → {OUT_PATH}")
