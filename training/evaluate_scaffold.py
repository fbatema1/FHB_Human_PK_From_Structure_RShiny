"""
training/evaluate_scaffold.py
==============================
Evaluates all three scaffold-split models (RF, XGB, GNN) on the held-out
scaffold test set and saves results to models/saved/scaffold_results/.

Outputs:
  - scaffold_rf_results.json
  - scaffold_xgb_results.json
  - scaffold_gnn_results.json
  - scaffold_eval_summary.txt   — side-by-side comparison with random-split

Run:
    /nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python training/evaluate_scaffold.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.rdkit_features import RDKitFeaturizer
from models.random_forest    import PKRandomForest
from models.xgboost_model    import PKXGBoost
from models.gnn_model        import PKAttentiveFP

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC    = ROOT / "data/processed"
RF_DIR  = ROOT / "models/saved/scaffold_rf"
XGB_DIR = ROOT / "models/saved/scaffold_xgb"
GNN_DIR = ROOT / "models/saved/scaffold_gnn"
OUT_DIR = ROOT / "models/saved/scaffold_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = ["CL", "Vd"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_metrics(y_obs, y_pred):
    """Compute GMFE, R², RMSE, within-2/3-fold on log10 scale."""
    log_obs  = np.log10(y_obs)
    log_pred = np.log10(y_pred)
    residuals = log_pred - log_obs
    abs_res   = np.abs(residuals)

    gmfe  = 10 ** np.mean(abs_res)
    rmse  = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    r2    = 1 - ss_res / ss_tot

    fe_ratio = np.where(y_pred > y_obs, y_pred / y_obs, y_obs / y_pred)
    w2 = 100 * np.mean(fe_ratio <= 2.0)
    w3 = 100 * np.mean(fe_ratio <= 3.0)

    bias = 10 ** np.mean(residuals)

    return {
        "n":           int(len(y_obs)),
        "gmfe":        round(float(gmfe),  4),
        "r2":          round(float(r2),    4),
        "rmse":        round(float(rmse),  4),
        "within_2fold": round(float(w2),   2),
        "within_3fold": round(float(w3),   2),
        "bias":        round(float(bias),  4),
    }


def load_test_data():
    print("Loading scaffold test data...")
    feat = RDKitFeaturizer()

    # Descriptor + fingerprint features
    X_test = np.load(PROC / "scaffold_X_test_desc_fp.npy")
    print(f"  X_test: {X_test.shape}")

    # Raw targets from scaffold y arrays (stored as log10, convert back)
    y_test = np.load(PROC / "scaffold_y_test.npy")
    y_cl = 10 ** y_test[:, 0]
    y_vd = 10 ** y_test[:, 1]
    print(f"  Test compounds: {len(y_cl)}")

    # GNN graphs
    graphs = torch.load(PROC / "scaffold_test_graphs.pt", map_location=DEVICE)
    print(f"  GNN graphs: {len(graphs)}")

    return X_test, y_cl, y_vd, graphs


def evaluate_rf(X_test, y_cl, y_vd):
    print("\n── Random Forest ─────────────────────────────────")
    results = {}
    for param, y_obs in zip(PARAMS, [y_cl, y_vd]):
        model = PKRandomForest.load(str(RF_DIR / f"rf_{param}_best.pkl"))
        # Apply the same feature selection used during training
        X_sel = X_test[:, model.feat_idx]
        y_pred = model.predict_original_scale(X_sel)
        m = eval_metrics(y_obs, y_pred)
        results[param] = m
        print(f"  {param}: GMFE={m['gmfe']:.3f}  R²={m['r2']:.3f}  "
              f"W2={m['within_2fold']:.1f}%  W3={m['within_3fold']:.1f}%  "
              f"(features used: {len(model.feat_idx)})")
    out = OUT_DIR / "scaffold_rf_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")
    return results


def evaluate_xgb(X_test, y_cl, y_vd):
    print("\n── XGBoost ────────────────────────────────────────")
    results = {}
    for param, y_obs in zip(PARAMS, [y_cl, y_vd]):
        model = PKXGBoost.load(str(XGB_DIR / f"xgb_{param}_best.pkl"))
        # Apply the same feature selection used during training
        X_sel = X_test[:, model.feat_idx]
        y_pred = model.predict_original_scale(X_sel)
        m = eval_metrics(y_obs, y_pred)
        results[param] = m
        print(f"  {param}: GMFE={m['gmfe']:.3f}  R²={m['r2']:.3f}  "
              f"W2={m['within_2fold']:.1f}%  W3={m['within_3fold']:.1f}%  "
              f"(features used: {len(model.feat_idx)})")
    out = OUT_DIR / "scaffold_xgb_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")
    return results


def evaluate_gnn(graphs, y_cl, y_vd):
    print("\n── GNN ────────────────────────────────────────────")
    # GNN is a single dual-head model → one .pt file for both CL and Vd
    model_path = GNN_DIR / "gnn_best.pt"
    model = PKAttentiveFP.load(str(model_path), device=str(DEVICE))
    model.eval()

    loader = GeoDataLoader(graphs, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.append(out.cpu().numpy())

    preds = np.vstack(preds)   # (n, 2) — columns: log10_CL, log10_Vd

    results = {}
    for idx, (param, y_obs) in enumerate(zip(PARAMS, [y_cl, y_vd])):
        y_pred = 10 ** preds[:, idx]
        m = eval_metrics(y_obs, y_pred)
        results[param] = m
        print(f"  {param}: GMFE={m['gmfe']:.3f}  R²={m['r2']:.3f}  "
              f"W2={m['within_2fold']:.1f}%  W3={m['within_3fold']:.1f}%")

    out = OUT_DIR / "scaffold_gnn_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")
    return results


def print_comparison(rf, xgb, gnn):
    # Random-split results (from models/saved/)
    rand_split = {
        "RF":  {"CL": {"gmfe":2.256,"r2":0.433,"within_2fold":60.3,"within_3fold":77.4},
                "Vd": {"gmfe":1.913,"r2":0.653,"within_2fold":66.2,"within_3fold":82.2}},
        "XGB": {"CL": {"gmfe":2.272,"r2":0.417,"within_2fold":56.8,"within_3fold":73.9},
                "Vd": {"gmfe":1.815,"r2":0.694,"within_2fold":71.1,"within_3fold":85.0}},
        "GNN": {"CL": {"gmfe":2.550,"r2":0.312,"within_2fold":48.4,"within_3fold":68.3},
                "Vd": {"gmfe":2.048,"r2":0.588,"within_2fold":56.4,"within_3fold":78.7}},
    }
    scaffold = {"RF": rf, "XGB": xgb, "GNN": gnn}

    lines = [
        "=" * 72,
        "SCAFFOLD vs RANDOM SPLIT — TEST SET COMPARISON",
        "=" * 72,
        f"{'Model':<6} {'Param':<5} {'Split':<10} {'GMFE':>6} {'R²':>6} {'W2%':>7} {'W3%':>7}",
        "─" * 72,
    ]
    for model_name in ["RF", "XGB", "GNN"]:
        for param in PARAMS:
            r = rand_split[model_name][param]
            s = scaffold[model_name][param]
            lines.append(
                f"{model_name:<6} {param:<5} {'random':<10} "
                f"{r['gmfe']:>6.3f} {r['r2']:>6.3f} {r['within_2fold']:>7.1f} {r['within_3fold']:>7.1f}"
            )
            lines.append(
                f"{'':6} {'':5} {'scaffold':<10} "
                f"{s['gmfe']:>6.3f} {s['r2']:>6.3f} {s['within_2fold']:>7.1f} {s['within_3fold']:>7.1f}"
            )
        lines.append("─" * 72)

    summary = "\n".join(lines)
    print("\n" + summary)

    out = OUT_DIR / "scaffold_eval_summary.txt"
    with open(out, "w") as f:
        f.write(summary)
    print(f"\nSaved → {out}")


def run():
    print("=" * 55)
    print("SCAFFOLD TEST SET EVALUATION")
    print("=" * 55)

    X_test, y_cl, y_vd, graphs = load_test_data()
    rf  = evaluate_rf(X_test, y_cl, y_vd)
    xgb = evaluate_xgb(X_test, y_cl, y_vd)
    gnn = evaluate_gnn(graphs, y_cl, y_vd)
    print_comparison(rf, xgb, gnn)
    print("\n✓ Scaffold evaluation complete.")


if __name__ == "__main__":
    run()
