"""
add_names.py
Adds compound_name column to master sheet, resolves ChEMBL IDs to real names.
Run: python add_names.py
"""

import pandas as pd
import numpy as np
import time
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings
Settings.Instance().TIMEOUT = 30

IN_PATH  = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026_named.xlsx"
OUT_PATH = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026_named.xlsx"

molecule = new_client.molecule

# ── Step 1: Collect all unique ChEMBL IDs across the file ───────────────────
xl = pd.ExcelFile(IN_PATH, engine='calamine')
all_sheets = {s: pd.read_excel(xl, sheet_name=s, engine='calamine') for s in xl.sheet_names}

chembl_ids = set()
for df in all_sheets.values():
    if 'NAME' in df.columns:
        ids = df['NAME'].dropna()
        ids = ids[ids.str.match(r'^CHEMBL\d+$', na=False)]
        chembl_ids.update(ids.tolist())

print(f"Unique ChEMBL IDs to resolve: {len(chembl_ids)}")

# ── Step 2: Batch lookup preferred names ────────────────────────────────────
id_to_name = {}
id_list = list(chembl_ids)
batches = [id_list[i:i+50] for i in range(0, len(id_list), 50)]

for i, batch in enumerate(batches):
    print(f"  Batch {i+1}/{len(batches)}...", end=' ', flush=True)
    try:
        hits = molecule.filter(
            molecule_chembl_id__in=batch
        ).only(['molecule_chembl_id', 'pref_name'])
        n = 0
        for h in hits:
            cid  = h.get('molecule_chembl_id')
            name = h.get('pref_name')
            if cid and name:
                id_to_name[cid] = name.title()
                n += 1
        print(f"{n} resolved")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(0.3)

print(f"\nTotal resolved: {len(id_to_name)}/{len(chembl_ids)}")

# ── Step 3: Add compound_name to each sheet ─────────────────────────────────
def build_compound_name(row):
    name = str(row.get('NAME', ''))
    # Already have a real compound_name that isn't a ChEMBL ID
    existing = row.get('compound_name')
    if pd.notna(existing) and not str(existing).startswith('CHEMBL'):
        return existing
    # ChEMBL ID → look up
    if name.startswith('CHEMBL'):
        return id_to_name.get(name, np.nan)
    # Real name already in NAME
    return name if name and name != 'nan' else np.nan

updated = {}
for sheet, df in all_sheets.items():
    if 'NAME' not in df.columns:
        updated[sheet] = df
        continue
    df['compound_name'] = df.apply(build_compound_name, axis=1)
    resolved = df['compound_name'].notna().sum()
    print(f"{sheet}: {resolved}/{len(df)} have names")
    updated[sheet] = df

# ── Step 4: Sort — named first, unnamed/unresolved last ─────────────────────
def sort_named_first(df):
    if 'compound_name' not in df.columns:
        return df
    has_name = df['compound_name'].notna() & ~df['compound_name'].str.match(r'^CHEMBL\d+$', na=False)
    return pd.concat([df[has_name], df[~has_name]], ignore_index=True)

# ── Step 5: Reorder columns so compound_name is first ───────────────────────
def reorder_cols(df):
    if 'compound_name' not in df.columns:
        return df
    return df[['compound_name'] + [c for c in df.columns if c != 'compound_name']]

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    for sheet, df in updated.items():
        df = sort_named_first(df)
        df = reorder_cols(df)
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nSaved → {OUT_PATH}")
