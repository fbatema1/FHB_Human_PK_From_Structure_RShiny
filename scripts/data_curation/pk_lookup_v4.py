"""
pk_lookup_v4.py
===============
Multi-source PK lookup for compounds missing CL, Vd, or fup.

Sources attempted in order:
  1. ChEMBL   — relaxed human filter, all assay types
  2. DrugBank  — structured PK fields (requires free academic API key)
  3. PubChem   — bioassay + pharmacology section
  4. OpenFDA   — drug label PK section

Results from all sources are aggregated per compound and medians taken.
Only null fields are filled — existing values are never overwritten.

Setup:
    pip install pandas chembl-webresource-client openpyxl requests

DrugBank API key (free for academic use):
    Register at https://go.drugbank.com/releases/latest#academic
    Set env variable: export DRUGBANK_API_KEY="your_key_here"
    Or paste directly into DRUGBANK_API_KEY variable below.

Run:
    conda activate pkip-env
    python scripts/data_curation/pk_lookup_v4.py
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings

Settings.Instance().TIMEOUT = 30

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MASTER_IN  = ROOT / "data/raw/master_dataset_FHB_06052026_v2.xlsx"
MASTER_OUT = ROOT / "data/raw/master_dataset_FHB_06052026_v4.xlsx"

BODY_WT = 70.0

# ── API Keys ──────────────────────────────────────────────────────────────────
DRUGBANK_API_KEY = os.environ.get("DRUGBANK_API_KEY", "")  # set env var or paste here


# ══════════════════════════════════════════════════════════════════════════════
# Unit conversions
# ══════════════════════════════════════════════════════════════════════════════
def convert_cl(value, unit):
    if value is None: return None
    v, u = float(value), str(unit).lower().strip()
    if 'ul/min/mg' in u:  return None
    if 'ml/min/kg' in u:  return v
    if 'ml/kg/min' in u:  return v
    if u == 'ml/min':     return v / BODY_WT
    if 'l/h/kg'  in u:    return v * 1000 / 60
    if 'l/hr/kg' in u:    return v * 1000 / 60
    if u == 'l/h':        return v * 1000 / 60 / BODY_WT
    if 'ml/h/kg' in u:    return v / 60
    return None

def convert_vd(value, unit):
    if value is None: return None
    v, u = float(value), str(unit).lower().strip()
    if 'l/kg'   in u:     return v
    if 'l kg-1' in u:     return v
    if 'ml/kg'  in u:     return v / 1000
    if u == 'l':          return v / BODY_WT
    if u == 'ml':         return v / 1000 / BODY_WT
    return None

def convert_fup(value, unit):
    if value is None: return None
    v, u = float(value), str(unit).lower().strip()
    if '%' in u:          return v / 100
    if 0 <= v <= 1:       return v
    if 0 < v <= 100:      return v / 100
    return None

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

def median_or_nan(lst):
    return float(np.median(lst)) if lst else np.nan

def merge_pk(base: dict, new: dict):
    """Merge new PK lists into base dict."""
    for k in ('cl', 'vd', 'fup'):
        base[k].extend(new.get(k, []))
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Source 1: ChEMBL (relaxed human filter)
# ══════════════════════════════════════════════════════════════════════════════
def clean_name(name: str) -> str:
    suffixes = [
        ' hydrochloride', ' hcl', ' sodium', ' potassium', ' sulfate',
        ' mesylate', ' maleate', ' tartrate', ' acetate', ' phosphate',
        ' fumarate', ' citrate', ' dihydrochloride', ' monohydrate', ' hydrate'
    ]
    n = name.lower()
    for s in suffixes:
        if n.endswith(s):
            return name[:len(name) - len(s)].strip()
    return name


def chembl_name_to_id(names: list, batch_size=50) -> dict:
    """Resolve compound names to ChEMBL IDs."""
    molecule = new_client.molecule
    name_to_chembl = {}
    batches = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"  [ChEMBL] Name batch {b_idx + 1}/{len(batches)}...", end=' ', flush=True)
        found = 0
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
            print(f"[err: {e}]", end=' ')

        # Retry with salt-stripped names
        unmatched = [n for n in batch if n not in name_to_chembl]
        cleaned_pairs = [(n, clean_name(n)) for n in unmatched if clean_name(n) != n]
        if cleaned_pairs:
            try:
                clean_map = {c[1].upper(): c[0] for c in cleaned_pairs}
                hits2 = molecule.filter(
                    pref_name__in=list(clean_map.keys())
                ).only(['molecule_chembl_id', 'pref_name'])
                for h in hits2:
                    pu = (h.get('pref_name') or '').upper()
                    cid = h.get('molecule_chembl_id')
                    if pu in clean_map and cid:
                        name_to_chembl[clean_map[pu]] = cid
                        found += 1
            except:
                pass

        print(f"{found} found")
        time.sleep(0.4)

    return name_to_chembl


def chembl_fetch_pk(chembl_ids: list, batch_size=20) -> dict:
    """
    Fetch PK activities from ChEMBL.
    Relaxed: no human keyword filter (too many false negatives).
    Excludes microsomal/in vitro assays by unit.
    """
    activity = new_client.activity
    pk = {cid: {'cl': [], 'vd': [], 'fup': []} for cid in chembl_ids}
    batches = [chembl_ids[i:i + batch_size] for i in range(0, len(chembl_ids), batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"  [ChEMBL] Activity batch {b_idx + 1}/{len(batches)}...", end=' ', flush=True)
        hits = 0
        for attempt in range(3):
            try:
                acts = activity.filter(
                    molecule_chembl_id__in=list(batch),
                    assay_type__in=['A', 'P'],   # ADMET + PK assay types
                ).only(['molecule_chembl_id', 'standard_type', 'standard_value',
                        'standard_units', 'assay_description', 'assay_type'])

                for a in acts:
                    cid   = a.get('molecule_chembl_id')
                    stype = str(a.get('standard_type') or '').lower().strip()
                    sval  = a.get('standard_value')
                    sunit = str(a.get('standard_units') or '')
                    adesc = str(a.get('assay_description') or '').lower()

                    if cid not in pk or sval is None:
                        continue
                    # Exclude clearly in vitro microsomal assays
                    if any(w in adesc for w in ('microsom', 'hepatocyte', 's9 ', 'caco', 'in vitro')):
                        continue
                    # Exclude rat/mouse/dog unless no human qualifier at all
                    if any(w in adesc for w in ('rat ', 'mouse', 'murine', 'canine', 'dog ')):
                        continue

                    if stype in CL_TYPES:
                        v = convert_cl(sval, sunit)
                        if v and 0.001 < v < 500:
                            pk[cid]['cl'].append(v); hits += 1
                    elif stype in VD_TYPES:
                        v = convert_vd(sval, sunit)
                        if v and 0.001 < v < 500:
                            pk[cid]['vd'].append(v); hits += 1
                    elif stype in FUP_TYPES:
                        v = convert_fup(sval, sunit)
                        if v and 0 < v <= 1:
                            pk[cid]['fup'].append(v); hits += 1
                break
            except Exception as e:
                print(f"[retry {attempt+1}: {e}]", end=' ')
                time.sleep(2)

        print(f"{hits} values")
        time.sleep(0.8)

    return pk


# ══════════════════════════════════════════════════════════════════════════════
# Source 2: DrugBank API
# ══════════════════════════════════════════════════════════════════════════════
def drugbank_fetch_pk(names: list) -> dict:
    """
    Look up PK data from DrugBank API.
    Returns dict: {compound_name: {'cl': [], 'vd': [], 'fup': []}}

    Requires DRUGBANK_API_KEY. Register free at:
    https://go.drugbank.com/releases/latest#academic
    """
    if not DRUGBANK_API_KEY:
        print("  [DrugBank] No API key set — skipping. See script header to register.")
        return {}

    headers = {
        'Authorization': f'Bearer {DRUGBANK_API_KEY}',
        'Content-Type': 'application/json'
    }
    base_url = "https://api.drugbank.com/v1"
    pk_results = {}

    print(f"  [DrugBank] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 50 == 0:
            print(f"    {i}/{len(names)}...", flush=True)
        try:
            # Search by name
            r = requests.get(
                f"{base_url}/drugs",
                headers=headers,
                params={'q': name, 'per_page': 1},
                timeout=10
            )
            if r.status_code != 200:
                continue
            results = r.json()
            if not results:
                continue

            drug = results[0]
            drug_id = drug.get('drugbank_id')
            if not drug_id:
                continue

            # Fetch full drug record for PK fields
            r2 = requests.get(
                f"{base_url}/drugs/{drug_id}",
                headers=headers,
                timeout=10
            )
            if r2.status_code != 200:
                continue

            data = r2.json()
            props = {p['kind']: p['value']
                     for p in data.get('pharmacokinetics', [])
                     if p.get('value')}

            entry = {'cl': [], 'vd': [], 'fup': []}

            # Volume of distribution
            vd_raw = props.get('volume-of-distribution', '')
            if vd_raw:
                # Extract numeric value from strings like "0.6 L/kg"
                import re
                nums = re.findall(r'[\d.]+', str(vd_raw))
                units = str(vd_raw).lower()
                for num in nums:
                    v = convert_vd(float(num), 'l/kg' if 'l/kg' in units else 'l')
                    if v and 0.001 < v < 500:
                        entry['vd'].append(v)

            # Protein binding → fup
            pb_raw = props.get('protein-binding', '')
            if pb_raw:
                import re
                nums = re.findall(r'[\d.]+', str(pb_raw))
                for num in nums:
                    try:
                        pb = float(num)
                        if pb > 1:   pb /= 100   # convert % to fraction
                        fup = 1 - pb
                        if 0 < fup <= 1:
                            entry['fup'].append(fup)
                            break
                    except:
                        pass

            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.2)

        except Exception as e:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [DrugBank] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Source 3: PubChem
# ══════════════════════════════════════════════════════════════════════════════
def pubchem_fetch_pk(names: list) -> dict:
    """
    Look up PK data from PubChem pharmacology section.
    No API key required.
    """
    import re
    pk_results = {}
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    print(f"  [PubChem] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 100 == 0:
            print(f"    {i}/{len(names)}...", flush=True)
        try:
            # Get CID by name
            r = requests.get(
                f"{base}/compound/name/{requests.utils.quote(name)}/property/MolecularWeight/JSON",
                timeout=10
            )
            if r.status_code != 200:
                continue
            cid_data = r.json()
            cids = [p['CID'] for p in cid_data.get('PropertyTable', {}).get('Properties', [])]
            if not cids:
                continue
            cid = cids[0]

            # Fetch pharmacology/PK section
            r2 = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
                f"?heading=Pharmacokinetics",
                timeout=15
            )
            if r2.status_code != 200:
                continue

            text = r2.text.lower()
            entry = {'cl': [], 'vd': [], 'fup': []}

            # Vd patterns: "0.6 l/kg", "600 ml/kg"
            for m in re.finditer(r'([\d.]+)\s*(l/kg|ml/kg|l kg)', text):
                v = convert_vd(float(m.group(1)), m.group(2))
                if v and 0.001 < v < 500:
                    entry['vd'].append(v)

            # fup / protein binding patterns
            for m in re.finditer(r'([\d.]+)\s*%?\s*(?:protein binding|bound to plasma)', text):
                try:
                    pb = float(m.group(1))
                    if pb > 1: pb /= 100
                    fup = 1 - pb
                    if 0 < fup <= 1:
                        entry['fup'].append(fup)
                except:
                    pass

            for m in re.finditer(r'(?:fu|fraction unbound)[^\d]*([\d.]+)', text):
                v = convert_fup(float(m.group(1)), '')
                if v and 0 < v <= 1:
                    entry['fup'].append(v)

            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.5)  # PubChem rate limit: ~5 req/sec

        except Exception:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [PubChem] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Source 4: OpenFDA drug labels
# ══════════════════════════════════════════════════════════════════════════════
def openfda_fetch_pk(names: list) -> dict:
    """
    Look up PK data from FDA drug labels via OpenFDA API.
    Parses the clinical_pharmacology / pharmacokinetics section.
    No API key required (rate limited to 240 req/min).
    """
    import re
    pk_results = {}
    base = "https://api.fda.gov/drug/label.json"

    print(f"  [OpenFDA] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 100 == 0:
            print(f"    {i}/{len(names)}...", flush=True)
        try:
            r = requests.get(
                base,
                params={
                    'search': f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"',
                    'limit': 1
                },
                timeout=10
            )
            if r.status_code != 200:
                continue

            results = r.json().get('results', [])
            if not results:
                continue

            label = results[0]
            pk_text = ' '.join(
                label.get('clinical_pharmacology', []) +
                label.get('pharmacokinetics', [])
            ).lower()

            entry = {'cl': [], 'vd': [], 'fup': []}

            # CL patterns: "clearance of 10 ml/min/kg", "cl = 5 l/h"
            for m in re.finditer(r'clearance[^\d]*([\d.]+)\s*(ml/min/kg|ml/min|l/h/kg|l/h)', pk_text):
                v = convert_cl(float(m.group(1)), m.group(2))
                if v and 0.001 < v < 500:
                    entry['cl'].append(v)

            # Vd patterns
            for m in re.finditer(r'(?:volume of distribution|vd|vss)[^\d]*([\d.]+)\s*(l/kg|l kg|ml/kg|liters/kg)', pk_text):
                v = convert_vd(float(m.group(1)), m.group(2))
                if v and 0.001 < v < 500:
                    entry['vd'].append(v)

            # Protein binding / fup
            for m in re.finditer(r'([\d.]+)\s*%\s*(?:bound|protein binding)', pk_text):
                try:
                    pb = float(m.group(1)) / 100
                    fup = 1 - pb
                    if 0 < fup <= 1:
                        entry['fup'].append(fup)
                except:
                    pass

            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.26)  # stay under 240/min

        except Exception:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [OpenFDA] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run():
    print(f"Loading: {MASTER_IN}")
    xl = pd.ExcelFile(MASTER_IN, engine='openpyxl')
    df = pd.read_excel(xl, sheet_name='All_Compounds', engine='openpyxl')

    # Compounds missing at least one PK value
    missing_mask = (
        df['human_CL_mL_min_kg'].isna() |
        df['human_VDss_L_kg'].isna()    |
        df['human_fup'].isna()
    )
    missing = df[missing_mask].copy()
    names = missing['compound_name'].dropna().unique().tolist()

    print(f"\nDataset: {len(df)} total | {missing_mask.sum()} missing at least one PK value")
    print(f"Unique compound names to look up: {len(names)}\n")

    # Accumulate PK from all sources
    # Structure: {compound_name: {'cl': [], 'vd': [], 'fup': []}}
    all_pk_by_name = {n: {'cl': [], 'vd': [], 'fup': []} for n in names}

    # ── Source 1: ChEMBL ──────────────────────────────────────────────────────
    print("=" * 55)
    print("SOURCE 1: ChEMBL")
    print("=" * 55)
    name_to_chembl = chembl_name_to_id(names)
    print(f"  ChEMBL IDs resolved: {len(name_to_chembl)}/{len(names)}")

    chembl_ids = list(set(name_to_chembl.values()))
    chembl_pk_by_id = chembl_fetch_pk(chembl_ids)

    # Map back to names
    for name, cid in name_to_chembl.items():
        if name in all_pk_by_name and cid in chembl_pk_by_id:
            merge_pk(all_pk_by_name[name], chembl_pk_by_id[cid])

    # ── Source 2: DrugBank ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 2: DrugBank")
    print("=" * 55)
    drugbank_pk = drugbank_fetch_pk(names)
    for name, pk in drugbank_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Source 3: PubChem ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 3: PubChem")
    print("=" * 55)
    pubchem_pk = pubchem_fetch_pk(names)
    for name, pk in pubchem_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Source 4: OpenFDA ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 4: OpenFDA")
    print("=" * 55)
    fda_pk = openfda_fetch_pk(names)
    for name, pk in fda_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Fill values (never overwrite existing) ────────────────────────────────
    print("\n" + "=" * 55)
    print("MERGING RESULTS")
    print("=" * 55)

    def fill_if_missing(row, field, pk_key):
        if pd.notna(row[field]):
            return row[field]
        pk = all_pk_by_name.get(row['compound_name'], {})
        return median_or_nan(pk.get(pk_key, []))

    missing['human_CL_mL_min_kg'] = missing.apply(
        lambda r: fill_if_missing(r, 'human_CL_mL_min_kg', 'cl'), axis=1)
    missing['human_VDss_L_kg'] = missing.apply(
        lambda r: fill_if_missing(r, 'human_VDss_L_kg', 'vd'), axis=1)
    missing['human_fup'] = missing.apply(
        lambda r: fill_if_missing(r, 'human_fup', 'fup'), axis=1)

    got_cl  = missing['human_CL_mL_min_kg'].notna().sum()
    got_vd  = missing['human_VDss_L_kg'].notna().sum()
    got_fup = missing['human_fup'].notna().sum()

    print(f"  CL filled:   {got_cl}")
    print(f"  Vd filled:   {got_vd}")
    print(f"  fup filled:  {got_fup}")

    df.update(missing[['human_CL_mL_min_kg', 'human_VDss_L_kg', 'human_fup']])

    complete = (
        df['human_CL_mL_min_kg'].notna() &
        df['human_VDss_L_kg'].notna() &
        df['human_fup'].notna()
    ).sum()
    print(f"\nComplete compounds (CL + Vd + fup) after update: {complete}/{len(df)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    other_sheets = {
        s: pd.read_excel(xl, sheet_name=s, engine='openpyxl')
        for s in xl.sheet_names if s != 'All_Compounds'
    }
    with pd.ExcelWriter(MASTER_OUT, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Compounds', index=False)
        for sname, sdf in other_sheets.items():
            sdf.to_excel(writer, sheet_name=sname, index=False)

    print(f"Saved → {MASTER_OUT}")


if __name__ == '__main__':
    run()
