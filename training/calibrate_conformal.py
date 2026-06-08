"""
training/calibrate_conformal.py
================================
Fit split conformal predictors for each model × parameter combination.

Method:
  Uses the training set with a random 15% holdout as the calibration set.
  The calibration set was never used for model training or hyperparameter
  tuning — it is only used here to compute conformal quantiles.

  This preserves the full test set as a clean held-out evaluation set
  for reporting final metrics and coverage.

Why not use the test set for calibration?
  Using the test set for calibration would invalidate its use as an
  independent evaluation set. The 15% training holdout gives ~170 compounds
  for calibration — sufficient for reliable 95% quantile estimation.

Outputs (models/saved/conformal/):
  - conformal_{model}_{param}.pkl   e.g. conformal_RF_CL.pkl
  - conformal_summary.json          coverage + quantile for all combinations

Run:
    conda activate pkip-env
    python training/calibrate_conformal.py

Note: Run this AFTER all three model training scripts have completed.
"""

import json
import pickle
import numpy as np
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.rdkit_features  import RDKitFeaturizer
from models.random_forest     import PKRandomForest
from models.xgboost_model     import PKXGBoost
from models.gnn_model         import PKAttentiveFP
from evaluation.conformal     import PKConformalPredictor

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC     = ROOT / "data/processed"
RF_DIR   = ROOT / "models/saved/rf"
XGB_DIR  = ROOT / "models/saved/xgb"
GNN_DIR  = ROOT / "models/saved/gnn"
SAVE_DIR = ROOT / "models/saved/conformal"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
PARAMS       = ['CL', 'Vd']
CAL_FRACTION = 0.15    # fraction of training set used for calibration
RANDOM_STATE = 42
COVERAGE     = 0.95

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    print("=" * 55)
    print("CONFORMAL CALIBRATION")
    print("=" * 55)

    # ── Load training data ────────────────────────────────────────────────────
    print("\nLoading training data...")
    X_train      = np.load(PROC / "X_train_desc_fp.npy")
    y_train      = np.load(PROC / "y_train.npy")
    train_graphs = torch.load(str(PROC / "train_graphs.pt"), weights_only=False)
    device       = get_device()
    print(f"  X_train: {X_train.shape}  |  Graphs: {len(train_graphs)}")
    print(f"  Device: {device}")

    # ── Calibration split (from training set only) ────────────────────────────
    n      = len(X_train)
    n_cal  = int(n * CAL_FRACTION)
    rng    = np.random.default_rng(RANDOM_STATE)
    cal_idx = rng.choice(n, size=n_cal, replace=False)

    X_cal        = X_train[cal_idx]
    y_cal        = y_train[cal_idx]
    cal_graphs   = [train_graphs[i] for i in cal_idx]
    print(f"\nCalibration set: {n_cal} compounds ({CAL_FRACTION:.0%} of training set)")

    summary = {}

    # ══════════════════════════════════════════════════════════════════════════
    # RF
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*40}")
    print("Random Forest")
    print(f"{'─'*40}")

    for param_idx, param in enumerate(PARAMS):
        model_path = RF_DIR / f"rf_{param}_best.pkl"
        if not model_path.exists():
            print(f"  [{param}] Model not found: {model_path} — skipping")
            continue

        rf: PKRandomForest = PKRandomForest.load(str(model_path))
        X_cal_sel = X_cal[:, rf.feat_idx]
        y_pred_cal = rf.predict(X_cal_sel)
        y_true_cal = y_cal[:, param_idx]

        cp = PKConformalPredictor(coverage=COVERAGE)
        cp.calibrate(y_true_cal, y_pred_cal)

        save_path = SAVE_DIR / f"conformal_RF_{param}.pkl"
        cp.save(str(save_path))
        print(f"  [{param}] Saved → {save_path}")

        summary[f'RF_{param}'] = {
            'quantile':  cp.quantile_,
            'fold_mult': round(10 ** cp.quantile_, 3),
            'n_cal':     cp.n_cal_,
            'coverage':  COVERAGE,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # XGBoost
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*40}")
    print("XGBoost")
    print(f"{'─'*40}")

    for param_idx, param in enumerate(PARAMS):
        model_path = XGB_DIR / f"xgb_{param}_best.pkl"
        if not model_path.exists():
            print(f"  [{param}] Model not found: {model_path} — skipping")
            continue

        xgb: PKXGBoost = PKXGBoost.load(str(model_path))
        X_cal_sel  = X_cal[:, xgb.feat_idx]
        y_pred_cal = xgb.predict(X_cal_sel)
        y_true_cal = y_cal[:, param_idx]

        cp = PKConformalPredictor(coverage=COVERAGE)
        cp.calibrate(y_true_cal, y_pred_cal)

        save_path = SAVE_DIR / f"conformal_XGB_{param}.pkl"
        cp.save(str(save_path))
        print(f"  [{param}] Saved → {save_path}")

        summary[f'XGB_{param}'] = {
            'quantile':  cp.quantile_,
            'fold_mult': round(10 ** cp.quantile_, 3),
            'n_cal':     cp.n_cal_,
            'coverage':  COVERAGE,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # GNN
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*40}")
    print("GNN")
    print(f"{'─'*40}")

    gnn_model_path = GNN_DIR / "gnn_best.pt"
    if gnn_model_path.exists():
        gnn: PKAttentiveFP = PKAttentiveFP.load(str(gnn_model_path), device=str(device))
        gnn.eval()

        cal_loader = GeoDataLoader(cal_graphs, batch_size=64, shuffle=False)
        y_pred_all = []
        with torch.no_grad():
            for batch in cal_loader:
                batch = batch.to(device)
                out   = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y_pred_all.append(out.cpu().numpy())
        y_pred_gnn = np.vstack(y_pred_all)  # (n_cal, 2)

        for param_idx, param in enumerate(PARAMS):
            y_true_cal = y_cal[:, param_idx]
            y_pred_cal = y_pred_gnn[:, param_idx]

            cp = PKConformalPredictor(coverage=COVERAGE)
            cp.calibrate(y_true_cal, y_pred_cal)

            save_path = SAVE_DIR / f"conformal_GNN_{param}.pkl"
            cp.save(str(save_path))
            print(f"  [{param}] Saved → {save_path}")

            summary[f'GNN_{param}'] = {
                'quantile':  cp.quantile_,
                'fold_mult': round(10 ** cp.quantile_, 3),
                'n_cal':     cp.n_cal_,
                'coverage':  COVERAGE,
            }
    else:
        print(f"  GNN model not found: {gnn_model_path} — skipping")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("CONFORMAL CALIBRATION SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<12} {'Param':<6} {'Quantile':>10} {'Fold ×':>10} {'n_cal':>8}")
    print(f"{'─'*50}")
    for key, vals in summary.items():
        model, param = key.split('_')
        print(f"  {model:<10} {param:<6} {vals['quantile']:>10.4f} {vals['fold_mult']:>10.3f}× {vals['n_cal']:>8}")

    with open(SAVE_DIR / "conformal_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {SAVE_DIR / 'conformal_summary.json'}")


if __name__ == '__main__':
    main()
