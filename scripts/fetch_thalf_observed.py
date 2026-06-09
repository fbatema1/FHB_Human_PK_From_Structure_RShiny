"""
scripts/fetch_thalf_observed.py
================================
Fetch published human half-life (t½) values for test set compounds.

Sources queried (in order):
  1. PubChem — compound pharmacology / clinical data sections
  2. ChEMBL  — clinical pharmacokinetics activities table
  3. openFDA — drug labels (pharmacokinetics section, text-mined)

Output:
  data/processed/thalf_observed.csv
    compound_name, smiles, cid, thalf_h, thalf_source, thalf_notes

Usage:
    python scripts/fetch_thalf_observed.py
"""

import re
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
OUT_PATH   = ROOT / "data" / "processed" / "thalf_observed.csv"
TEST_PATH  = ROOT / "data" / "processed" / "test.xlsx"
SLEEP      = 0.25    # seconds between API calls (be polite)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url, params=None, timeout=15):
    """GET with retries."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


# ── Half-life parsing ─────────────────────────────────────────────────────────
# Patterns: "3.5 hours", "t1/2 = 2h", "half-life of 12 h", "t½ ~6 hours"
def parse_thalf_from_text(text: str):
    """
    Extract a numeric t½ in hours from free text.
    Returns (value_h, raw_match) or (None, None).
    For ranges, returns the midpoint.

    Handles patterns like:
      half-life is 2 hr
      half-life of about 2 hours
      t1/2 = 3.5 h
      t½ ~6 hours
      half-life of 1.2-2 hours
      elimination half-life of approximately 12 hours
    """
    if not text:
        return None, None

    # Step 1: find sentences/phrases mentioning half-life
    hl_pattern = re.compile(
        r'(?:half[-\s]?life|t\s*[½1][/½2]*)',
        re.IGNORECASE
    )

    # Step 2: after finding a half-life mention, look for a number + hour unit nearby
    number_pattern = re.compile(
        r'(\d+(?:\.\d+)?)'           # first number
        r'(?:\s*[-–to]\s*'           # optional range separator
        r'(\d+(?:\.\d+)?))?'         # second number (range end)
        r'\s*'
        r'(h(?:ou?r?s?)?|hr?s?)\b',  # hour unit
        re.IGNORECASE
    )

    for hl_match in hl_pattern.finditer(text):
        # Search for number+unit within 80 chars after the half-life mention
        search_start = hl_match.start()
        search_end   = min(len(text), hl_match.end() + 120)
        window       = text[search_start:search_end]

        for num_match in number_pattern.finditer(window):
            lo_str = num_match.group(1)
            hi_str = num_match.group(2)
            try:
                lo = float(lo_str)
                if lo <= 0 or lo > 1000:   # sanity check
                    continue
                if hi_str:
                    hi  = float(hi_str)
                    val = round((lo + hi) / 2, 2)
                else:
                    val = round(lo, 2)
                raw = text[search_start : search_start + num_match.end()].strip()
                return val, raw[:120]
            except ValueError:
                continue

    return None, None


# ── Source 1: PubChem ─────────────────────────────────────────────────────────

def pubchem_cid_from_name(name: str):
    """Look up PubChem CID from compound name."""
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON".format(
        requests.utils.quote(name)
    )
    r = get(url)
    if r is None:
        return None
    try:
        return r.json()["IdentifierList"]["CID"][0]
    except (KeyError, IndexError):
        return None


def pubchem_cid_from_smiles(smiles: str):
    """Look up PubChem CID from SMILES."""
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON".format(
        requests.utils.quote(smiles)
    )
    r = get(url)
    if r is None:
        return None
    try:
        return r.json()["IdentifierList"]["CID"][0]
    except (KeyError, IndexError):
        return None


def pubchem_thalf(cid: int):
    """
    Fetch t½ from PubChem pharmacology + clinical data sections.
    Returns (thalf_h, notes) or (None, None).
    """
    # Section headings that may contain t½
    headings = [
        "Absorption, Distribution and Excretion",
        "Pharmacokinetics",
        "Human Metabolite",
    ]

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    r = get(url)
    if r is None:
        return None, None

    try:
        data = r.json()
    except Exception:
        return None, None

    # Walk all sections recursively, collect strings
    texts = []

    def walk(node):
        if isinstance(node, dict):
            if "StringWithMarkup" in node:
                for s in node["StringWithMarkup"]:
                    texts.append(s.get("String", ""))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)

    for text in texts:
        val, raw = parse_thalf_from_text(text)
        if val is not None:
            return val, f"PubChem: {raw.strip()}"

    return None, None


# ── Source 2: ChEMBL ──────────────────────────────────────────────────────────

def chembl_thalf(name: str):
    """
    Query ChEMBL for half-life activity values (standard_type = 'T1/2').
    Returns (thalf_h, notes) or (None, None).
    """
    # Find ChEMBL molecule ID
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    r = get(url, params={"pref_name__iexact": name, "limit": 1})
    if r is None:
        return None, None
    try:
        mols = r.json()["molecules"]
        if not mols:
            return None, None
        chembl_id = mols[0]["molecule_chembl_id"]
    except (KeyError, IndexError):
        return None, None

    time.sleep(SLEEP)

    # Fetch T1/2 activities
    url2 = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    r2 = get(url2, params={
        "molecule_chembl_id": chembl_id,
        "standard_type":      "T1/2",
        "assay_type":         "PK",
        "limit":              10,
    })
    if r2 is None:
        return None, None

    try:
        acts = r2.json()["activities"]
    except KeyError:
        return None, None

    # Filter to human, IV, hours
    human_vals = []
    for a in acts:
        desc = (a.get("assay_description") or "").lower()
        units = (a.get("standard_units") or "").lower()
        val   = a.get("standard_value")
        if val is None:
            continue

        # Convert to hours
        try:
            val = float(val)
        except ValueError:
            continue

        if "min" in units:
            val = val / 60
        elif "day" in units:
            val = val * 24
        elif "h" not in units:
            continue   # unknown units

        # Prefer human data
        if "human" in desc or "clinical" in desc:
            human_vals.append(val)
        elif not human_vals:
            human_vals.append(val)   # keep non-human as fallback

    if human_vals:
        med = float(np.median(human_vals))
        return round(med, 2), f"ChEMBL: {len(human_vals)} value(s), median={med:.2f}h"

    return None, None


# ── Source 3: openFDA drug labels ─────────────────────────────────────────────

def fda_thalf(name: str):
    """
    Search openFDA drug labels for t½ in the pharmacokinetics section.
    Returns (thalf_h, notes) or (None, None).
    """
    url = "https://api.fda.gov/drug/label.json"
    r = get(url, params={
        "search": f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"',
        "limit":  3,
    })
    if r is None:
        return None, None

    try:
        results = r.json().get("results", [])
    except Exception:
        return None, None

    for label in results:
        # Check pharmacokinetics section first
        for section in ["pharmacokinetics", "clinical_pharmacology",
                        "description", "clinical_studies"]:
            text_list = label.get(section, [])
            if isinstance(text_list, str):
                text_list = [text_list]
            for text in text_list:
                val, raw = parse_thalf_from_text(text)
                if val is not None:
                    return val, f"FDA label: {raw.strip()}"

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fetching observed t½ for test set compounds")
    print("=" * 60)

    df = pd.read_excel(TEST_PATH)
    df.columns = [c.lower() for c in df.columns]
    names  = df["compound_name"].tolist()
    smiles = df["mol"].tolist()
    n      = len(names)
    print(f"Test set: {n} compounds\n")

    records = []

    for i, (name, smi) in enumerate(zip(names, smiles)):
        print(f"[{i+1:3d}/{n}] {name}")
        rec = {
            "compound_name": name,
            "smiles":        smi,
            "cid":           None,
            "thalf_h":       None,
            "thalf_source":  None,
            "thalf_notes":   None,
        }

        # ── PubChem CID lookup ────────────────────────────────────────────
        cid = pubchem_cid_from_name(name)
        if cid is None:
            cid = pubchem_cid_from_smiles(smi)
        rec["cid"] = cid
        time.sleep(SLEEP)

        # ── PubChem t½ ────────────────────────────────────────────────────
        if cid is not None:
            val, notes = pubchem_thalf(cid)
            time.sleep(SLEEP)
            if val is not None:
                rec["thalf_h"]      = val
                rec["thalf_source"] = "PubChem"
                rec["thalf_notes"]  = notes
                print(f"       → PubChem: {val} h")
                records.append(rec)
                continue

        # ── ChEMBL t½ ─────────────────────────────────────────────────────
        val, notes = chembl_thalf(name)
        time.sleep(SLEEP)
        if val is not None:
            rec["thalf_h"]      = val
            rec["thalf_source"] = "ChEMBL"
            rec["thalf_notes"]  = notes
            print(f"       → ChEMBL: {val} h")
            records.append(rec)
            continue

        # ── openFDA label ─────────────────────────────────────────────────
        val, notes = fda_thalf(name)
        time.sleep(SLEEP)
        if val is not None:
            rec["thalf_h"]      = val
            rec["thalf_source"] = "FDA"
            rec["thalf_notes"]  = notes
            print(f"       → FDA: {val} h")
            records.append(rec)
            continue

        print(f"       → not found")
        records.append(rec)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)

    found = out["thalf_h"].notna().sum()
    print(f"\n{'='*60}")
    print(f"Found t½ for {found}/{n} compounds ({found/n*100:.1f}%)")
    print(f"Source breakdown:\n{out['thalf_source'].value_counts().to_string()}")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
