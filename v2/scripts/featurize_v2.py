"""
v2/scripts/featurize_v2.py
===========================
Featurizes the enriched master_v2 dataset for v2 model training.

Extends the scaffold featurization pipeline by appending four new
physicochemical features to the RDKit descriptor+fingerprint matrix:
  - fup_final      (fraction unbound in plasma)
  - logP           (Crippen logP)
  - logD_74        (logD at pH 7.4)
  - pka_acid_final (most acidic pKa; NaN-imputed with median)
  - pka_base_final (most basic pKa; NaN-imputed with median)

Strategy:
  - New features appended AFTER existing descriptors and fingerprints
    so feat_idx from v1 models still indexes correctly into the v1 block
  - pKa NaNs imputed with training-set median (same approach as fup)
  - Scaffold split applied to master_v2 using the same scaffold assignments
    as the v1 split (compound_name match) to ensure identical train/test
    partitioning

Outputs (v2/data/processed/):
  - v2_scaffold_train.xlsx
  - v2_scaffold_test.xlsx
  - v2_X_train.npy          shape (n_train, n_v1_features + 5)
  - v2_X_test.npy           shape (n_test,  n_v1_features + 5)
  - v2_y_train.npy          shape (n_train, 2)  [log10_CL, log10_Vd]
  - v2_y_test.npy           shape (n_test,  2)
  - v2_featurizer.pkl       RDKitFeaturizer fitted on v2 train SMILES
  - v2_train_graphs.pt      PyG graphs for GNN
  - v2_test_graphs.pt
  - v2_feature_names.txt    names for all columns including new features
  - v2_featurization_summary.txt

Run:
    conda activate pkip-env
    python v2/scripts/featurize_v2.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import sys

ROOT   = Path(__file__).resolve().parents[2]
V2PROC = ROOT / "v2/data/processed"
PROC   = ROOT / "data/processed"
V2PROC.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from features.rdkit_features import RDKitFeaturizer

try:
    from features.graph_builder import MolGraphBuilder
    import torch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("⚠️  torch_geometric not available — skipping GNN graph building.")
    print("   Run on Longleaf to generate GNN graphs.\n")

# ── New feature columns from master_v2 ────────────────────────────────────────
NEW_FEATURES = ["fup_final", "logP", "logD_74", "pka_acid_final", "pka_base_final"]


def load_master_v2():
    path = V2PROC / "master_v2.xlsx"
    print(f"Loading {path}...")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"  {len(df)} compounds, {len(df.columns)} columns")
    return df


def apply_scaffold_split(df):
    """
    Re-use the v1 scaffold split assignments by matching on compound_name.
    Compounds not present in v1 splits (shouldn't happen) go to train.
    """
    train_v1 = pd.read_excel(PROC / "scaffold_train.xlsx", engine="openpyxl")
    test_v1  = pd.read_excel(PROC / "scaffold_test.xlsx",  engine="openpyxl")

    train_names = set(train_v1["compound_name"].str.lower().str.strip())
    test_names  = set(test_v1["compound_name"].str.lower().str.strip())

    df["_name_lower"] = df["compound_name"].str.lower().str.strip()
    test_mask  = df["_name_lower"].isin(test_names)
    train_mask = ~test_mask   # everything not in test → train

    train_df = df[train_mask].copy().drop(columns=["_name_lower"])
    test_df  = df[test_mask].copy().drop(columns=["_name_lower"])

    print(f"  Scaffold train: {len(train_df)}  |  test: {len(test_df)}")
    unmatched = (~df["_name_lower"].isin(train_names | test_names)).sum()
    if unmatched:
        print(f"  ⚠️  {unmatched} compounds not in v1 splits → assigned to train")

    return train_df, test_df


def impute_pka(train_df, test_df):
    """Impute NaN pKa values with training-set medians."""
    for col in ["pka_acid_final", "pka_base_final"]:
        median_val = train_df[col].median()
        n_train = train_df[col].isna().sum()
        n_test  = test_df[col].isna().sum()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)
        print(f"  {col}: imputed {n_train} train / {n_test} test NaNs "
              f"with median={median_val:.2f}")
    return train_df, test_df


def build_new_feature_matrix(df, col_medians=None):
    """
    Extract the 5 new physicochemical features as a (n, 5) float array.
    col_medians: if provided, use these for imputation (for test set).
    """
    arr = df[NEW_FEATURES].values.astype(np.float32)
    # Should have no NaNs after imputation, but guard anyway
    if np.isnan(arr).any():
        if col_medians is None:
            col_medians = np.nanmedian(arr, axis=0)
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return arr


def run():
    print("=" * 60)
    print("V2 FEATURIZATION PIPELINE")
    print("=" * 60)

    # ── Load and split ────────────────────────────────────────────────────────
    df = load_master_v2()
    print("\nApplying scaffold split (re-using v1 assignments)...")
    train_df, test_df = apply_scaffold_split(df)

    # ── Impute pKa NaNs ───────────────────────────────────────────────────────
    print("\nImputing missing pKa values...")
    train_df, test_df = impute_pka(train_df, test_df)

    # ── Save split xlsx ───────────────────────────────────────────────────────
    train_df.to_excel(V2PROC / "v2_scaffold_train.xlsx", index=False)
    test_df.to_excel( V2PROC / "v2_scaffold_test.xlsx",  index=False)
    print(f"\n  Saved v2_scaffold_train.xlsx / v2_scaffold_test.xlsx")

    # ── Targets ───────────────────────────────────────────────────────────────
    print("\nPreparing targets (log10 scale)...")
    y_train = np.column_stack([
        np.log10(train_df["human_CL_mL_min_kg"].astype(np.float32).values),
        np.log10(train_df["human_VDss_L_kg"].astype(np.float32).values),
    ])
    y_test = np.column_stack([
        np.log10(test_df["human_CL_mL_min_kg"].astype(np.float32).values),
        np.log10(test_df["human_VDss_L_kg"].astype(np.float32).values),
    ])
    print(f"  y_train: {y_train.shape}  y_test: {y_test.shape}")

    # ── RDKit descriptors + fingerprints ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("RDKit Descriptors + Morgan Fingerprints")
    print("─" * 60)
    train_smi = train_df["mol"].tolist()
    test_smi  = test_df["mol"].tolist()

    feat = RDKitFeaturizer()
    feat.fit(train_smi, verbose=True)

    print("\nTransforming train set...")
    X_train_rdkit = feat.transform(train_smi)
    print(f"  X_train_rdkit: {X_train_rdkit.shape}")

    print("Transforming test set...")
    X_test_rdkit = feat.transform(test_smi)
    print(f"  X_test_rdkit:  {X_test_rdkit.shape}")

    # ── New physicochemical features ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Appending new physicochemical features")
    print("─" * 60)
    train_new = build_new_feature_matrix(train_df)
    col_medians = np.median(train_new, axis=0)
    test_new  = build_new_feature_matrix(test_df, col_medians=col_medians)
    print(f"  New features: {NEW_FEATURES}")
    print(f"  train_new: {train_new.shape}  test_new: {test_new.shape}")

    # Stats on new features
    for i, name in enumerate(NEW_FEATURES):
        print(f"    {name}: train mean={train_new[:,i].mean():.3f}  "
              f"std={train_new[:,i].std():.3f}  "
              f"min={train_new[:,i].min():.3f}  max={train_new[:,i].max():.3f}")

    # ── Concatenate ───────────────────────────────────────────────────────────
    X_train = np.hstack([X_train_rdkit, train_new]).astype(np.float32)
    X_test  = np.hstack([X_test_rdkit,  test_new]).astype(np.float32)
    print(f"\n  Final X_train: {X_train.shape}  "
          f"(+{X_train.shape[1] - X_train_rdkit.shape[1]} new features)")
    print(f"  Final X_test:  {X_test.shape}")

    # ── Feature names ─────────────────────────────────────────────────────────
    rdkit_names = feat.get_feature_names() if hasattr(feat, "get_feature_names") else \
                  [f"feat_{i}" for i in range(X_train_rdkit.shape[1])]
    all_feature_names = rdkit_names + NEW_FEATURES
    assert len(all_feature_names) == X_train.shape[1], \
        f"Feature name count mismatch: {len(all_feature_names)} vs {X_train.shape[1]}"

    # ── GNN graphs (only if torch_geometric available) ────────────────────────
    train_graphs, test_graphs = [], []
    if HAS_TORCH_GEOMETRIC:
        print("\n" + "─" * 60)
        print("Building GNN graphs")
        print("─" * 60)
        builder = MolGraphBuilder()

        print("  Building train graphs...")
        for smi, y in zip(train_smi, y_train):
            g = builder.build(smi)
            if g is not None:
                g.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                train_graphs.append(g)
        print(f"  Train graphs: {len(train_graphs)}/{len(train_smi)}")

        print("  Building test graphs...")
        for smi, y in zip(test_smi, y_test):
            g = builder.build(smi)
            if g is not None:
                g.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                test_graphs.append(g)
        print(f"  Test graphs: {len(test_graphs)}/{len(test_smi)}")
    else:
        print("\n⚠️  Skipping GNN graphs (no torch_geometric). "
              "Run on Longleaf to generate v2_train_graphs.pt / v2_test_graphs.pt")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    np.save(V2PROC / "v2_X_train.npy", X_train)
    np.save(V2PROC / "v2_X_test.npy",  X_test)
    np.save(V2PROC / "v2_y_train.npy", y_train)
    np.save(V2PROC / "v2_y_test.npy",  y_test)
    feat.save(str(V2PROC / "v2_featurizer.pkl"))
    if HAS_TORCH_GEOMETRIC and train_graphs:
        torch.save(train_graphs, V2PROC / "v2_train_graphs.pt")
        torch.save(test_graphs,  V2PROC / "v2_test_graphs.pt")

    with open(V2PROC / "v2_feature_names.txt", "w") as f:
        f.write("\n".join(all_feature_names))

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_lines = [
        "V2 FEATURIZATION SUMMARY",
        "=" * 50,
        f"Total compounds:      {len(df)}",
        f"Train:                {len(train_df)}",
        f"Test:                 {len(test_df)}",
        f"",
        f"Feature dimensions:",
        f"  RDKit desc+FP:      {X_train_rdkit.shape[1]}",
        f"  New physchem:       {len(NEW_FEATURES)}",
        f"  Total:              {X_train.shape[1]}",
        f"",
        f"New features appended: {NEW_FEATURES}",
        f"",
        f"pKa imputation (training median):",
    ]
    for col in ["pka_acid_final", "pka_base_final"]:
        summary_lines.append(f"  {col}: {train_df[col].median():.2f}")
    gnn_str = (f"{len(train_graphs)} train / {len(test_graphs)} test"
               if HAS_TORCH_GEOMETRIC else "skipped (run on Longleaf)")
    summary_lines += [f"", f"GNN graphs: {gnn_str}"]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(V2PROC / "v2_featurization_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n✓ V2 featurization complete.")
    print(f"  Outputs in: {V2PROC}")


if __name__ == "__main__":
    run()
