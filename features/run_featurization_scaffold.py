"""
run_featurization_scaffold.py
=============================
Runs the full featurization pipeline on the SCAFFOLD split train/test sets.

Reads:
  data/processed/scaffold_train.xlsx
  data/processed/scaffold_test.xlsx

Outputs (data/processed/) — all prefixed with scaffold_ so originals are untouched:
  - scaffold_X_train_desc_fp.npy
  - scaffold_X_test_desc_fp.npy
  - scaffold_y_train.npy
  - scaffold_y_test.npy
  - scaffold_featurizer.pkl
  - scaffold_train_graphs.pt
  - scaffold_test_graphs.pt
  - scaffold_feature_names.txt

Run:
    conda activate pkip-env
    python features/run_featurization_scaffold.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
PROC      = ROOT / "data/processed"
TRAIN_IN  = PROC / "scaffold_train.xlsx"
TEST_IN   = PROC / "scaffold_test.xlsx"

import sys
sys.path.insert(0, str(ROOT))

from features.rdkit_features import RDKitFeaturizer
from features.graph_builder  import MolGraphBuilder


def load_split(path):
    df = pd.read_excel(path, engine='openpyxl')
    smiles = df['mol'].tolist()
    cl     = df['human_CL_mL_min_kg'].astype(float).tolist()
    vd     = df['human_VDss_L_kg'].astype(float).tolist()
    return smiles, cl, vd, df


def run():
    print("=" * 55)
    print("FEATURIZATION PIPELINE (SCAFFOLD SPLIT)")
    print("=" * 55)

    print("\nLoading scaffold train/test splits...")
    train_smi, train_cl, train_vd, train_df = load_split(TRAIN_IN)
    test_smi,  test_cl,  test_vd,  test_df  = load_split(TEST_IN)
    print(f"  Train: {len(train_smi)} compounds")
    print(f"  Test:  {len(test_smi)} compounds")

    print("\nPreparing targets (log10 scale)...")
    y_train = np.column_stack([
        np.log10(np.array(train_cl, dtype=np.float32)),
        np.log10(np.array(train_vd, dtype=np.float32))
    ])
    y_test = np.column_stack([
        np.log10(np.array(test_cl, dtype=np.float32)),
        np.log10(np.array(test_vd, dtype=np.float32))
    ])
    print(f"  y_train shape: {y_train.shape}  (cols: log10_CL, log10_Vd)")
    print(f"  y_test shape:  {y_test.shape}")

    print("\n" + "─" * 55)
    print("RDKit Descriptors + Morgan Fingerprints")
    print("─" * 55)
    feat = RDKitFeaturizer()
    feat.fit(train_smi, verbose=True)

    print("\nTransforming train set...")
    X_train = feat.transform(train_smi)
    print(f"  X_train shape: {X_train.shape}")

    print("Transforming test set...")
    X_test = feat.transform(test_smi)
    print(f"  X_test shape:  {X_test.shape}")

    nan_train = np.isnan(X_train).sum()
    nan_test  = np.isnan(X_test).sum()
    if nan_train > 0 or nan_test > 0:
        print(f"  WARNING: NaN values — train: {nan_train}, test: {nan_test}")
    else:
        print("  No NaN values detected")

    print("\n" + "─" * 55)
    print("PyTorch Geometric Graph Builder")
    print("─" * 55)
    builder = MolGraphBuilder()
    print(f"  Atom feature dim: {builder.n_atom_features}")
    print(f"  Bond feature dim: {builder.n_bond_features}")

    print("\nBuilding train graphs...")
    train_graphs = builder.build_dataset(train_smi, train_cl, train_vd, verbose=True)

    print("Building test graphs...")
    test_graphs = builder.build_dataset(test_smi, test_cl, test_vd, verbose=True)

    g = train_graphs[0]
    print(f"\n  Sample graph: {g.num_nodes} atoms, {g.num_edges} directed edges")

    print("\n" + "─" * 55)
    print("Saving outputs (scaffold_ prefix)...")
    PROC.mkdir(parents=True, exist_ok=True)

    np.save(PROC / "scaffold_X_train_desc_fp.npy", X_train)
    np.save(PROC / "scaffold_X_test_desc_fp.npy",  X_test)
    np.save(PROC / "scaffold_y_train.npy",          y_train)
    np.save(PROC / "scaffold_y_test.npy",           y_test)

    feat.save(str(PROC / "scaffold_featurizer.pkl"))

    torch.save(train_graphs, PROC / "scaffold_train_graphs.pt")
    torch.save(test_graphs,  PROC / "scaffold_test_graphs.pt")

    feature_names = feat.get_feature_names()
    with open(PROC / "scaffold_feature_names.txt", 'w') as f:
        f.write('\n'.join(feature_names))

    print(f"\n  scaffold_X_train_desc_fp.npy : {X_train.shape}")
    print(f"  scaffold_X_test_desc_fp.npy  : {X_test.shape}")
    print(f"  scaffold_y_train.npy          : {y_train.shape}")
    print(f"  scaffold_y_test.npy           : {y_test.shape}")
    print(f"  scaffold_train_graphs.pt      : {len(train_graphs)} graphs")
    print(f"  scaffold_test_graphs.pt       : {len(test_graphs)} graphs")
    print(f"  scaffold_feature_names.txt    : {len(feature_names)} features")
    print(f"  scaffold_featurizer.pkl       : saved")

    print("\n" + "=" * 55)
    print("Scaffold featurization complete.")
    print("=" * 55)


if __name__ == '__main__':
    run()
