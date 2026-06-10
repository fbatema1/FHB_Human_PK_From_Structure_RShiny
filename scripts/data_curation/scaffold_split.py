"""
scaffold_split.py
=================
Splits the cleaned dataset into train (80%) / test (20%) using a
Bemis-Murcko scaffold split.

Why scaffold split?
  A random split leaks structural information — compounds with nearly
  identical scaffolds appear in both train and test, inflating metrics.
  A scaffold split places all compounds sharing a carbon skeleton in the
  same partition, so the test set contains genuinely novel scaffolds.
  This is the standard rigorous benchmark for ML in drug discovery
  (Wu et al., MoleculeNet, 2018; Bemis & Murcko, J. Med. Chem., 1996).

Algorithm:
  1. Compute Bemis-Murcko scaffold for every compound (via RDKit).
  2. Group compounds by scaffold.
  3. Sort scaffold groups by size (largest first) for stable splitting.
  4. Assign groups to train/test greedily to hit ~80/20 split.

Outputs (all in data/processed/):
  - train.xlsx            (replaces old random-split train.xlsx)
  - test.xlsx             (replaces old random-split test.xlsx)
  - scaffold_split_summary.txt

Usage:
    conda activate pkip-env
    python scripts/data_curation/scaffold_split.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit required: conda install -c conda-forge rdkit")

ROOT      = Path(__file__).resolve().parents[2]
DATA_IN   = ROOT / "data/processed/master_dataset_cleaned.xlsx"
TRAIN_OUT = ROOT / "data/processed/scaffold_train.xlsx"
TEST_OUT  = ROOT / "data/processed/scaffold_test.xlsx"
LOG_OUT   = ROOT / "data/processed/scaffold_split_summary.txt"

TEST_FRAC    = 0.20
RANDOM_STATE = 42   # used only for tie-breaking within same-size scaffold groups


# ── Scaffold helper ────────────────────────────────────────────────────────────

def get_scaffold(smiles: str) -> str:
    """Return the generic Bemis-Murcko scaffold SMILES, or '' on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        # Generic scaffold: replace heteroatoms with C (carbon skeleton only)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(generic, isomericSmiles=False)
    except Exception:
        return ""


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print(f"Loading: {DATA_IN.name}")
    df = pd.read_excel(DATA_IN, sheet_name="Cleaned_Data", engine="openpyxl")
    print(f"  Total compounds: {len(df)}")

    smiles_col = None
    for candidate in ["mol", "smiles", "smi", "structure", "canonical_smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")
    print(f"  Using SMILES column: '{smiles_col}'")

    # Compute scaffolds
    print("  Computing Bemis-Murcko scaffolds…")
    df["_scaffold"] = df[smiles_col].astype(str).apply(get_scaffold)

    n_fail = (df["_scaffold"] == "").sum()
    if n_fail:
        print(f"  ⚠  {n_fail} compound(s) failed scaffold parsing → assigned unique scaffold")
        # Give each failed compound a unique scaffold so they go to train
        mask = df["_scaffold"] == ""
        df.loc[mask, "_scaffold"] = [f"__invalid_{i}__" for i in df.index[mask]]

    n_scaffolds = df["_scaffold"].nunique()
    print(f"  Unique scaffolds: {n_scaffolds} across {len(df)} compounds")

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold_to_indices[row["_scaffold"]].append(idx)

    # Sort scaffold groups: largest first, then shuffle within same size for reproducibility
    rng = np.random.default_rng(RANDOM_STATE)
    groups = list(scaffold_to_indices.values())
    groups.sort(key=lambda g: (-len(g), rng.random()))  # stable: large first

    target_test_n = int(round(len(df) * TEST_FRAC))
    test_indices  = []
    train_indices = []

    # Greedy assignment: fill test up to ~20%, rest goes to train
    for group in groups:
        if len(test_indices) < target_test_n:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train = df.loc[train_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    test  = df.loc[test_indices].drop(columns=["_scaffold"]).reset_index(drop=True)

    actual_test_pct = 100 * len(test) / len(df)
    print(f"  Train: {len(train)} ({100-actual_test_pct:.1f}%)")
    print(f"  Test:  {len(test)} ({actual_test_pct:.1f}%)")

    # ── Summary stats ──────────────────────────────────────────────────────────
    def pk_stats(d, label):
        lines = [f"\n{label} (n={len(d)})"]
        for col, name in [("human_CL_mL_min_kg", "CL"), ("human_VDss_L_kg", "Vd")]:
            if col not in d.columns:
                continue
            vals = d[col].dropna().astype(float)
            lines.append(
                f"  {name}: median={vals.median():.2f}  "
                f"mean={vals.mean():.2f}  "
                f"min={vals.min():.3f}  "
                f"max={vals.max():.1f}  "
                f"log10_std={np.log10(vals).std():.3f}"
            )
        return "\n".join(lines)

    # Scaffold overlap check (should be zero)
    train_scaffolds = set(df.loc[train_indices, "_scaffold"] if "_scaffold" in df.columns
                          else [get_scaffold(s) for s in train[smiles_col]])
    # Re-derive for overlap check
    test_df_scaf  = df.loc[test_indices, "_scaffold"] if "_scaffold" in df.columns else \
                    pd.Series([get_scaffold(s) for s in test[smiles_col]])
    # We dropped _scaffold above, recompute quickly
    train_scafs = set(scaffold_to_indices.keys()) - \
                  set(df.loc[test_indices, "_scaffold"] if "_scaffold" in df.columns
                      else [])
    # Simple: recompute overlap using original grouped dict
    test_scafs_set  = {df.loc[i, "_scaffold"] for i in test_indices
                       if "_scaffold" in df.columns}
    train_scafs_set = {df.loc[i, "_scaffold"] for i in train_indices
                       if "_scaffold" in df.columns}
    overlap = len(test_scafs_set & train_scafs_set)

    summary_lines = [
        "SCAFFOLD SPLIT SUMMARY",
        "=" * 50,
        f"Method:           Bemis-Murcko generic scaffold split",
        f"Random seed:      {RANDOM_STATE}",
        f"Target test frac: {TEST_FRAC:.0%}",
        f"Total:            {len(df)}",
        f"Train:            {len(train)} ({100-actual_test_pct:.1f}%)",
        f"Test:             {len(test)} ({actual_test_pct:.1f}%)",
        f"Unique scaffolds: {n_scaffolds}",
        f"Scaffold overlap (must be 0): {overlap}",
        pk_stats(train, "TRAIN"),
        pk_stats(test,  "TEST"),
    ]

    if "data_source" in train.columns:
        summary_lines.append("\nData source breakdown (train):")
        for src, cnt in train["data_source"].value_counts().items():
            summary_lines.append(f"  {src}: {cnt}")
        summary_lines.append("\nData source breakdown (test):")
        for src, cnt in test["data_source"].value_counts().items():
            summary_lines.append(f"  {src}: {cnt}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # ── Save ──────────────────────────────────────────────────────────────────
    train.to_excel(TRAIN_OUT, index=False)
    test.to_excel(TEST_OUT,  index=False)
    with open(LOG_OUT, "w") as fh:
        fh.write(summary_text)

    print(f"\nTrain → {TRAIN_OUT}")
    print(f"Test  → {TEST_OUT}")
    print(f"Log   → {LOG_OUT}")
    print("\n✓ Scaffold split complete.")


if __name__ == "__main__":
    run()
