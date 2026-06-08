"""
clean_dataset.py
================
Final dataset cleaning pipeline. Performs three steps:

  Step 1 — Drop compounds missing CL or Vd (not usable for training)
  Step 2 — Remove duplicate structures (InChIKey-based) and invalid SMILES
  Step 3 — Standardize problematic structures:
             a) Salt forms: strip counterions, keep largest fragment
             b) Large MW compounds (>1000): flag but retain if SMILES is valid
             c) Known naming issues (e.g. Atorvastatin hemicalcium salt)

Input:  data/raw/master_dataset_FINAL_combined.xlsx
Output: data/processed/master_dataset_cleaned.xlsx
Log:    data/processed/cleaning_log.xlsx  (every decision recorded)

Nothing is overwritten. All changes are documented in the log.

Run:
    conda activate pkip-env
    python scripts/data_curation/clean_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.MolStandardize import rdMolStandardize
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
DATA_IN  = ROOT / "data/raw/master_dataset_FINAL_combined.xlsx"
DATA_OUT = ROOT / "data/processed/master_dataset_cleaned.xlsx"
LOG_OUT  = ROOT / "data/processed/cleaning_log.xlsx"

ROOT.joinpath("data/processed").mkdir(parents=True, exist_ok=True)

MW_THRESHOLD = 1000.0  # flag but keep compounds above this MW

# ── RDKit standardization tools ───────────────────────────────────────────────
remover    = SaltRemover.SaltRemover()
normalizer = rdMolStandardize.Normalizer()
chooser    = rdMolStandardize.LargestFragmentChooser()
uncharger  = rdMolStandardize.Uncharger()


def standardize_smiles(smi: str):
    """
    Standardize a SMILES string:
      1. Parse to mol
      2. Remove salts / counterions (keep largest fragment)
      3. Normalize (fix unusual valences, tautomers)
      4. Return canonical SMILES and MW

    Returns (canonical_smiles, mw, note) or (None, None, error_message)
    """
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None, None, "invalid_smiles"

        original_smi = Chem.MolToSmiles(mol)

        # Remove salts — keep largest organic fragment
        mol_desalted = remover.StripMol(mol, dontRemoveEverything=True)
        if mol_desalted is None or mol_desalted.GetNumAtoms() == 0:
            mol_desalted = chooser.choose(mol)

        # Normalize
        mol_norm = normalizer.normalize(mol_desalted)

        canonical = Chem.MolToSmiles(mol_norm)
        mw        = Descriptors.MolWt(mol_norm)

        note = ''
        if canonical != original_smi:
            note = 'standardized'
        if mw > MW_THRESHOLD:
            note = (note + '|large_mw').lstrip('|')

        return canonical, mw, note

    except Exception as e:
        return None, None, f"error: {e}"


def safe_inchikey(smi: str):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return MolToInchiKey(mol)
    except:
        pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print(f"Loading: {DATA_IN.name}")
    df = pd.read_excel(DATA_IN, sheet_name='All_Compounds', engine='openpyxl')
    print(f"  Input compounds: {len(df)}")

    # Initialise log
    log_rows = []

    def log(compound_name, smiles, step, action, reason):
        log_rows.append({
            'compound_name': compound_name,
            'smiles':        smiles,
            'step':          step,
            'action':        action,
            'reason':        reason
        })

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Drop compounds missing CL or Vd
    # ══════════════════════════════════════════════════════════════════════════
    print("\nStep 1: Dropping compounds missing CL or Vd...")

    missing_mask = df['human_CL_mL_min_kg'].isna() | df['human_VDss_L_kg'].isna()
    dropped_step1 = df[missing_mask].copy()
    df = df[~missing_mask].copy()

    for _, row in dropped_step1.iterrows():
        missing_fields = []
        if pd.isna(row['human_CL_mL_min_kg']): missing_fields.append('CL')
        if pd.isna(row['human_VDss_L_kg']):    missing_fields.append('Vd')
        log(row['compound_name'], row['mol'], 'Step 1',
            'dropped', f"missing {'+'.join(missing_fields)}")

    print(f"  Dropped:   {len(dropped_step1)}")
    print(f"  Remaining: {len(df)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Remove invalid SMILES and duplicate structures
    # ══════════════════════════════════════════════════════════════════════════
    print("\nStep 2: Removing invalid SMILES and duplicates...")

    # 2a — Flag invalid SMILES
    def is_valid(smi):
        try:
            return Chem.MolFromSmiles(str(smi)) is not None
        except:
            return False

    invalid_mask = ~df['mol'].apply(is_valid)
    dropped_invalid = df[invalid_mask].copy()
    df = df[~invalid_mask].copy()

    for _, row in dropped_invalid.iterrows():
        log(row['compound_name'], row['mol'], 'Step 2a',
            'dropped', 'invalid SMILES — could not parse')

    print(f"  Invalid SMILES dropped: {len(dropped_invalid)}")

    # 2b — Generate InChIKeys on remaining compounds
    print("  Generating InChIKeys for deduplication...")
    df['inchikey'] = df['mol'].apply(safe_inchikey)

    # Flag compounds where InChIKey could not be generated
    no_key_mask = df['inchikey'].isna()
    dropped_nokey = df[no_key_mask].copy()
    df = df[~no_key_mask].copy()

    for _, row in dropped_nokey.iterrows():
        log(row['compound_name'], row['mol'], 'Step 2b',
            'dropped', 'could not generate InChIKey')

    print(f"  No InChIKey dropped:    {len(dropped_nokey)}")

    # 2c — Remove duplicates — keep the entry with more PK data
    before_dedup = len(df)
    df['pk_count'] = (
        df['human_CL_mL_min_kg'].notna().astype(int) +
        df['human_VDss_L_kg'].notna().astype(int) +
        df['human_fup'].notna().astype(int)
    )
    df_sorted = df.sort_values('pk_count', ascending=False)
    duplicates = df_sorted[df_sorted.duplicated('inchikey', keep='first')]
    df = df_sorted.drop_duplicates('inchikey', keep='first').copy()

    for _, row in duplicates.iterrows():
        log(row['compound_name'], row['mol'], 'Step 2c',
            'dropped', f"duplicate structure (InChIKey: {row['inchikey']}) — kept entry with more PK data")

    print(f"  Duplicates dropped:     {len(duplicates)}")
    print(f"  Remaining after Step 2: {len(df)}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Standardize structures
    # ══════════════════════════════════════════════════════════════════════════
    print("\nStep 3: Standardizing structures (salt stripping, normalization)...")

    standardized_count = 0
    large_mw_count     = 0
    failed_count       = 0

    new_smiles = []
    new_mw     = []
    new_notes  = []

    for idx, row in df.iterrows():
        canonical, mw, note = standardize_smiles(row['mol'])

        if canonical is None:
            # Standardization failed — keep original, flag it
            new_smiles.append(row['mol'])
            new_mw.append(np.nan)
            new_notes.append('standardization_failed')
            log(row['compound_name'], row['mol'], 'Step 3',
                'kept_original', f'standardization failed: {note}')
            failed_count += 1
        else:
            new_smiles.append(canonical)
            new_mw.append(mw)
            new_notes.append(note)

            if 'standardized' in note:
                log(row['compound_name'], row['mol'], 'Step 3',
                    'standardized', f'salt stripped / normalized → {canonical}')
                standardized_count += 1

            if 'large_mw' in note:
                log(row['compound_name'], canonical, 'Step 3',
                    'flagged_large_mw', f'MW={round(mw,1)} > {MW_THRESHOLD} — retained with flag')
                large_mw_count += 1

    df['mol']              = new_smiles
    df['mol_MW']           = new_mw
    df['structure_notes']  = new_notes

    # Recompute InChIKey on standardized SMILES
    df['inchikey'] = df['mol'].apply(safe_inchikey)

    # Final dedup pass after standardization (some salts may now match)
    pre_final_dedup = len(df)
    df_sorted2 = df.sort_values('pk_count', ascending=False)
    new_dups   = df_sorted2[df_sorted2.duplicated('inchikey', keep='first')]
    df = df_sorted2.drop_duplicates('inchikey', keep='first').copy()

    for _, row in new_dups.iterrows():
        log(row['compound_name'], row['mol'], 'Step 3 dedup',
            'dropped', f"post-standardization duplicate (InChIKey: {row['inchikey']})")

    print(f"  Structures standardized:        {standardized_count}")
    print(f"  Large MW flagged (kept):        {large_mw_count}")
    print(f"  Standardization failed (kept):  {failed_count}")
    print(f"  Post-standardization duplicates:{len(new_dups)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    # Drop internal columns before saving
    df = df.drop(columns=['inchikey', 'pk_count'], errors='ignore')

    complete  = (df.human_CL_mL_min_kg.notna() & df.human_VDss_L_kg.notna() & df.human_fup.notna()).sum()
    cl_vd     = (df.human_CL_mL_min_kg.notna() & df.human_VDss_L_kg.notna()).sum()
    large_mw  = (df['mol_MW'] > MW_THRESHOLD).sum() if 'mol_MW' in df.columns else 'N/A'

    print(f"\n{'='*55}")
    print(f"CLEAN DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Input compounds:          {len(df) + len(dropped_step1) + len(dropped_invalid) + len(dropped_nokey) + len(duplicates) + len(new_dups)}")
    print(f"  Step 1 dropped (no PK):   {len(dropped_step1)}")
    print(f"  Step 2 dropped (dups/inv):{len(dropped_invalid) + len(dropped_nokey) + len(duplicates)}")
    print(f"  Step 3 post-std dups:     {len(new_dups)}")
    print(f"  ─────────────────────────────────────")
    print(f"  Final clean compounds:    {len(df)}")
    print(f"  With CL + Vd:             {cl_vd}")
    print(f"  With CL + Vd + fup:       {complete}")
    print(f"  Large MW flagged (kept):  {large_mw}")
    print(f"{'='*55}")

    # Save cleaned dataset
    with pd.ExcelWriter(DATA_OUT, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Cleaned_Data', index=False)

    print(f"\nCleaned dataset → {DATA_OUT}")

    # Save cleaning log
    log_df = pd.DataFrame(log_rows)
    log_summary = pd.DataFrame([
        {'step': 'Step 1', 'action': 'dropped_missing_pk',    'count': len(dropped_step1)},
        {'step': 'Step 2', 'action': 'dropped_invalid_smiles','count': len(dropped_invalid)},
        {'step': 'Step 2', 'action': 'dropped_no_inchikey',   'count': len(dropped_nokey)},
        {'step': 'Step 2', 'action': 'dropped_duplicates',    'count': len(duplicates)},
        {'step': 'Step 3', 'action': 'standardized',          'count': standardized_count},
        {'step': 'Step 3', 'action': 'flagged_large_mw',      'count': large_mw_count},
        {'step': 'Step 3', 'action': 'post_std_duplicates',   'count': len(new_dups)},
        {'step': 'FINAL',  'action': 'clean_compounds',       'count': len(df)},
    ])

    with pd.ExcelWriter(LOG_OUT, engine='openpyxl') as writer:
        log_summary.to_excel(writer, sheet_name='Summary', index=False)
        log_df.to_excel(writer, sheet_name='Full_Log', index=False)

    print(f"Cleaning log      → {LOG_OUT}")


if __name__ == '__main__':
    run()
