"""
scripts/validate_thalf.py
==========================
Compare Wadhams-predicted t½ against published observed values.

Requires:
  - data/processed/thalf_observed.csv   (from fetch_thalf_observed.py)
  - data/processed/test.xlsx            (test set with SMILES)
  - Hybrid predictor loaded

Outputs:
  - data/processed/thalf_validation.csv  (full comparison table)
  - Prints GMFE, R², within-2fold for t½

Usage:
    python scripts/validate_thalf.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid.predictor import HybridPredictor

OBS_PATH = ROOT / "data" / "processed" / "thalf_observed.csv"
OUT_PATH = ROOT / "data" / "processed" / "thalf_validation.csv"


def gmfe(y_true, y_pred):
    return 10 ** np.mean(np.abs(np.log10(y_pred / y_true)))


def within_fold(y_true, y_pred, fold=2):
    return 100 * np.mean((y_pred / y_true <= fold) & (y_true / y_pred <= fold))


def main():
    print("=" * 60)
    print("t½ Validation — Predicted vs Observed")
    print("=" * 60)

    obs = pd.read_csv(OBS_PATH)
    obs = obs[obs["thalf_h"].notna()].copy()
    print(f"\nCompounds with observed t½: {len(obs)}")
    print(f"Source breakdown:\n{obs['thalf_source'].value_counts().to_string()}\n")

    # Run hybrid predictor
    print("Running hybrid predictor...")
    predictor = HybridPredictor.load()
    results   = predictor.predict(obs["smiles"].tolist(), ci=False)

    pred_thalf = []
    pred_cl    = []
    pred_vd    = []
    for r in results:
        pred_thalf.append(r.get("thalf_pred"))
        pred_cl.append(r.get("CL_pred"))
        pred_vd.append(r.get("Vd_pred"))

    obs["thalf_pred"] = pred_thalf
    obs["CL_pred"]    = pred_cl
    obs["Vd_pred"]    = pred_vd

    # Filter to valid predictions
    valid = obs[
        obs["thalf_pred"].notna() &
        obs["thalf_h"].notna() &
        (obs["thalf_pred"] > 0) &
        (obs["thalf_h"]    > 0)
    ].copy()

    print(f"Valid pairs for evaluation: {len(valid)}")

    y_obs  = valid["thalf_h"].values
    y_pred = valid["thalf_pred"].values

    # Metrics
    gmfe_val   = gmfe(y_obs, y_pred)
    w2f        = within_fold(y_obs, y_pred, 2)
    w3f        = within_fold(y_obs, y_pred, 3)
    log_obs    = np.log10(y_obs)
    log_pred   = np.log10(y_pred)
    ss_res     = np.sum((log_obs - log_pred) ** 2)
    ss_tot     = np.sum((log_obs - log_obs.mean()) ** 2)
    r2         = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse_log   = float(np.sqrt(np.mean((log_obs - log_pred) ** 2)))

    print(f"\n{'='*45}")
    print(f"t½ VALIDATION RESULTS (n={len(valid)})")
    print(f"{'='*45}")
    print(f"  GMFE:           {gmfe_val:.3f}")
    print(f"  R²  (log10):    {r2:.3f}")
    print(f"  RMSE (log10):   {rmse_log:.3f}")
    print(f"  Within 2-fold:  {w2f:.1f}%")
    print(f"  Within 3-fold:  {w3f:.1f}%")

    # Comparison table
    valid["log10_obs"]   = np.log10(y_obs)
    valid["log10_pred"]  = np.log10(y_pred)
    valid["fold_error"]  = y_pred / y_obs
    valid["abs_log_err"] = np.abs(np.log10(valid["fold_error"]))

    valid = valid.sort_values("abs_log_err", ascending=False)
    valid.to_csv(OUT_PATH, index=False)
    print(f"\nFull table saved → {OUT_PATH}")

    print(f"\nTop 10 largest errors:")
    print(valid[["compound_name","thalf_h","thalf_pred","fold_error","thalf_source"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
