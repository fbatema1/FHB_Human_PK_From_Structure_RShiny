"""
pk_lookup_v5.py
===============
Multi-source PK lookup for compounds missing CL, Vd, or fup.

Sources attempted in order:
  1. ChEMBL   — relaxed human filter, assay types A + P
  2. PubChem  — pharmacology/PK section (NLM/NCBI hosted)
  3. OpenFDA  — FDA drug label PK section
  4. DailyMed — NLM full drug label database (more complete than OpenFDA)

Results from all sources are aggregated per compound; medians taken.
Only null fields are filled — existing values are never overwritten.
No API keys required.

Setup:
    pip install pandas chembl-webresource-client openpyxl requests

Run:
    conda activate pkip-env
    python scripts/data_curation/pk_lookup_v5.py
"""

import re
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
MASTER_IN  = ROOT / "data/raw/master_dataset_FHB_06052026_v3.xlsx"
MASTER_OUT = ROOT / "data/raw/master_dataset_FHB_06052026_v5.xlsx"

BODY_WT    = 70.0  # kg

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

def convert_fup(value, unit=''):
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
    for k in ('cl', 'vd', 'fup'):
        base[k].extend(new.get(k, []))
    return base

def empty_pk():
    return {'cl': [], 'vd': [], 'fup': []}


# ══════════════════════════════════════════════════════════════════════════════
# Shared text parser — used by OpenFDA and DailyMed
# ══════════════════════════════════════════════════════════════════════════════
def parse_pk_text(text: str) -> dict:
    """
    Extract CL, Vd, and fup values from free-text PK sections.
    Returns {'cl': [], 'vd': [], 'fup': []}.
    """
    t = text.lower()
    entry = empty_pk()

    # CL: "clearance of 10 ml/min/kg", "cl = 5 l/h", "10 ml/min/kg"
    for m in re.finditer(
        r'(?:clearance|cl\b)[^\d]{0,20}([\d.]+)\s*(ml/min/kg|ml/min|l/h/kg|l/hr/kg|l/h\b|ml/h/kg)', t
    ):
        v = convert_cl(m.group(1), m.group(2))
        if v and 0.001 < v < 500:
            entry['cl'].append(v)

    # Vd: "volume of distribution 0.6 l/kg", "vdss of 12 l"
    for m in re.finditer(
        r'(?:volume of distribution|vd\b|vdss|vss)[^\d]{0,20}([\d.]+)\s*(l/kg|l kg|ml/kg|liters/kg|l\b)', t
    ):
        v = convert_vd(m.group(1), m.group(2))
        if v and 0.001 < v < 500:
            entry['vd'].append(v)

    # Protein binding → fup: "97% protein bound", "plasma protein binding: 94%"
    for m in re.finditer(
        r'([\d.]+)\s*%\s*(?:protein[\s-]?bound|bound to plasma protein|plasma protein binding)', t
    ):
        try:
            pb = float(m.group(1)) / 100
            fup = round(1 - pb, 4)
            if 0 < fup <= 1:
                entry['fup'].append(fup)
        except:
            pass

    # Direct fup: "fraction unbound 0.03", "fu = 5%"
    for m in re.finditer(
        r'(?:fraction unbound|fu\b|fup\b)[^\d]{0,10}([\d.]+)\s*(%?)', t
    ):
        v = convert_fup(m.group(1), m.group(2))
        if v and 0 < v <= 1:
            entry['fup'].append(v)

    return entry


# ══════════════════════════════════════════════════════════════════════════════
# Source 1: ChEMBL
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
    molecule = new_client.molecule
    name_to_chembl = {}
    batches = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    for b_idx, batch in enumerate(batches):
        print(f"  [ChEMBL] Name batch {b_idx+1}/{len(batches)}...", end=' ', flush=True)
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
    activity = new_client.activity
    pk = {cid: empty_pk() for cid in chembl_ids}
    batches = [chembl_ids[i:i + batch_size] for i in range(0, len(chembl_ids), batch_size)]
    for b_idx, batch in enumerate(batches):
        print(f"  [ChEMBL] Activity batch {b_idx+1}/{len(batches)}...", end=' ', flush=True)
        hits = 0
        for attempt in range(3):
            try:
                acts = activity.filter(
                    molecule_chembl_id__in=list(batch),
                    assay_type__in=['A', 'P'],
                ).only(['molecule_chembl_id', 'standard_type', 'standard_value',
                        'standard_units', 'assay_description'])
                for a in acts:
                    cid   = a.get('molecule_chembl_id')
                    stype = str(a.get('standard_type') or '').lower().strip()
                    sval  = a.get('standard_value')
                    sunit = str(a.get('standard_units') or '')
                    adesc = str(a.get('assay_description') or '').lower()
                    if cid not in pk or sval is None: continue
                    if any(w in adesc for w in ('microsom', 'hepatocyte', 's9 ', 'caco', 'in vitro')): continue
                    if any(w in adesc for w in ('rat ', 'mouse', 'murine', 'canine', 'dog ')): continue
                    if stype in CL_TYPES:
                        v = convert_cl(sval, sunit)
                        if v and 0.001 < v < 500: pk[cid]['cl'].append(v); hits += 1
                    elif stype in VD_TYPES:
                        v = convert_vd(sval, sunit)
                        if v and 0.001 < v < 500: pk[cid]['vd'].append(v); hits += 1
                    elif stype in FUP_TYPES:
                        v = convert_fup(sval, sunit)
                        if v and 0 < v <= 1: pk[cid]['fup'].append(v); hits += 1
                break
            except Exception as e:
                print(f"[retry {attempt+1}: {e}]", end=' ')
                time.sleep(2)
        print(f"{hits} values")
        time.sleep(0.8)
    return pk


# ══════════════════════════════════════════════════════════════════════════════
# Source 2: PubChem (NLM/NCBI)
# ══════════════════════════════════════════════════════════════════════════════
def pubchem_fetch_pk(names: list) -> dict:
    """
    Fetches PK data from PubChem pharmacology sections.
    Uses PUG-View API — no key required.
    """
    pk_results = {}
    base_pug = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    base_view = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

    print(f"  [PubChem] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(names)} processed...", flush=True)
        try:
            # Name → CID
            r = requests.get(
                f"{base_pug}/compound/name/{requests.utils.quote(name)}/cids/JSON",
                timeout=10
            )
            if r.status_code != 200:
                continue
            cids = r.json().get('IdentifierList', {}).get('CID', [])
            if not cids:
                continue
            cid = cids[0]

            # CID → PK section via PUG-View
            r2 = requests.get(
                f"{base_view}/data/compound/{cid}/JSON",
                params={'heading': 'Pharmacokinetics'},
                timeout=15
            )
            if r2.status_code != 200:
                continue

            # Extract all string values from nested JSON
            def extract_strings(obj):
                if isinstance(obj, str):
                    return [obj]
                if isinstance(obj, dict):
                    return [s for v in obj.values() for s in extract_strings(v)]
                if isinstance(obj, list):
                    return [s for item in obj for s in extract_strings(item)]
                return []

            all_text = ' '.join(extract_strings(r2.json()))
            entry = parse_pk_text(all_text)

            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.5)

        except Exception:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [PubChem] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Source 3: OpenFDA
# ══════════════════════════════════════════════════════════════════════════════
def openfda_fetch_pk(names: list) -> dict:
    """
    Fetches PK data from FDA drug labels via OpenFDA API.
    No key required (rate limit: 240 req/min).
    """
    pk_results = {}
    base = "https://api.fda.gov/drug/label.json"

    print(f"  [OpenFDA] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(names)} processed...", flush=True)
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
            )

            entry = parse_pk_text(pk_text)
            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.26)

        except Exception:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [OpenFDA] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Source 4: DailyMed (NLM)
# ══════════════════════════════════════════════════════════════════════════════
def dailymed_fetch_pk(names: list) -> dict:
    """
    Fetches PK data from DailyMed (NLM full FDA label database).
    More comprehensive than OpenFDA — includes older and generic labels.
    No API key required.
    API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
    """
    pk_results = {}
    search_url  = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    section_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/sections.json"

    # DailyMed section codes for clinical pharmacology / PK
    PK_SECTION_CODES = {
        '43685-7',  # Clinical Pharmacology
        '43680-8',  # Pharmacokinetics
    }

    print(f"  [DailyMed] Looking up {len(names)} compounds...")
    for i, name in enumerate(names):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(names)} processed...", flush=True)
        try:
            # Search by drug name
            r = requests.get(
                search_url,
                params={'drug_name': name, 'pagesize': 1},
                timeout=10
            )
            if r.status_code != 200:
                continue

            data = r.json()
            spls = data.get('data', [])
            if not spls:
                # Try cleaned name (strip salt)
                cleaned = clean_name(name)
                if cleaned != name:
                    r2 = requests.get(
                        search_url,
                        params={'drug_name': cleaned, 'pagesize': 1},
                        timeout=10
                    )
                    if r2.status_code == 200:
                        spls = r2.json().get('data', [])
            if not spls:
                continue

            setid = spls[0].get('setid')
            if not setid:
                continue

            # Fetch sections for this label
            r3 = requests.get(
                section_url.format(setid=setid),
                timeout=10
            )
            if r3.status_code != 200:
                continue

            sections = r3.json().get('data', [])
            pk_text = ''
            for section in sections:
                code = section.get('loinc_code', '')
                if code in PK_SECTION_CODES:
                    pk_text += ' ' + section.get('text', '')

            if not pk_text.strip():
                continue

            entry = parse_pk_text(pk_text)
            if any(entry[k] for k in entry):
                pk_results[name] = entry

            time.sleep(0.3)

        except Exception:
            continue

    found = sum(1 for v in pk_results.values() if any(v[k] for k in v))
    print(f"  [DailyMed] Retrieved PK for {found} compounds")
    return pk_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run():
    print(f"Loading: {MASTER_IN}")
    xl = pd.ExcelFile(MASTER_IN, engine='openpyxl')
    df = pd.read_excel(xl, sheet_name='All_Compounds', engine='openpyxl')

    missing_mask = (
        df['human_CL_mL_min_kg'].isna() |
        df['human_VDss_L_kg'].isna()    |
        df['human_fup'].isna()
    )
    missing = df[missing_mask].copy()
    names = missing['compound_name'].dropna().unique().tolist()

    print(f"\nDataset: {len(df)} total compounds")
    print(f"  Complete (CL + Vd + fup): {(df.human_CL_mL_min_kg.notna() & df.human_VDss_L_kg.notna() & df.human_fup.notna()).sum()}")
    print(f"  Missing at least one PK:  {missing_mask.sum()}")
    print(f"  Unique names to look up:  {len(names)}\n")

    # Accumulate PK across all sources
    all_pk_by_name = {n: empty_pk() for n in names}

    # ── Source 1: ChEMBL ──────────────────────────────────────────────────────
    print("=" * 55)
    print("SOURCE 1: ChEMBL")
    print("=" * 55)
    name_to_chembl = chembl_name_to_id(names)
    print(f"  Resolved: {len(name_to_chembl)}/{len(names)}")
    chembl_ids = list(set(name_to_chembl.values()))
    chembl_pk_by_id = chembl_fetch_pk(chembl_ids)
    for name, cid in name_to_chembl.items():
        if name in all_pk_by_name and cid in chembl_pk_by_id:
            merge_pk(all_pk_by_name[name], chembl_pk_by_id[cid])

    # ── Source 2: PubChem ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 2: PubChem (NLM/NCBI)")
    print("=" * 55)
    pubchem_pk = pubchem_fetch_pk(names)
    for name, pk in pubchem_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Source 3: OpenFDA ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 3: OpenFDA")
    print("=" * 55)
    fda_pk = openfda_fetch_pk(names)
    for name, pk in fda_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Source 4: DailyMed ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SOURCE 4: DailyMed (NLM)")
    print("=" * 55)
    dailymed_pk = dailymed_fetch_pk(names)
    for name, pk in dailymed_pk.items():
        if name in all_pk_by_name:
            merge_pk(all_pk_by_name[name], pk)

    # ── Fill missing values ───────────────────────────────────────────────────
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
    cl_vd = (df['human_CL_mL_min_kg'].notna() & df['human_VDss_L_kg'].notna()).sum()
    print(f"\n  Complete (CL + Vd + fup) after update: {complete}/{len(df)}")
    print(f"  Complete (CL + Vd only)  after update: {cl_vd}/{len(df)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    other_sheets = {
        s: pd.read_excel(xl, sheet_name=s, engine='openpyxl')
        for s in xl.sheet_names if s != 'All_Compounds'
    }
    with pd.ExcelWriter(MASTER_OUT, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Compounds', index=False)
        for sname, sdf in other_sheets.items():
            sdf.to_excel(writer, sheet_name=sname, index=False)

    print(f"\nSaved → {MASTER_OUT}")


if __name__ == '__main__':
    run()
