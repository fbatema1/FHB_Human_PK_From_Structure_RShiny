"""
evaluation/metrics.py
=====================
Evaluation metrics for PK parameter prediction.

Primary metrics (per parameter):
  - GMFE  : Geometric Mean Fold Error — primary metric, target < 1.5
  - R²    : Coefficient of determination on log10 scale, target > 0.7
  - RMSE  : Root mean squared error on log10 scale
  - within_2fold : % of predictions within 2-fold of measured value
  - within_3fold : % of predictions within 3-fold of measured value

All metrics operate on log10-transformed values internally.
Inputs can be raw (original scale) or log10 — specify with log_scale flag.

Usage:
    from evaluation.metrics import evaluate, gmfe, r2_log

    results = evaluate(y_true, y_pred, param_name='CL', log_scale=False)
    print(results)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Individual metric functions
# ══════════════════════════════════════════════════════════════════════════════

def gmfe(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> float:
    """
    Geometric Mean Fold Error.

    GMFE = 10 ^ ( mean( |log10(pred/true)| ) )
         = 10 ^ ( mean( |log10_pred - log10_true| ) )

    A perfect model has GMFE = 1.0. Target: GMFE < 1.5.

    Args:
        y_true:     true values
        y_pred:     predicted values
        log_scale:  if True, inputs are already log10-transformed
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if not log_scale:
        y_true = np.log10(np.clip(y_true, 1e-10, None))
        y_pred = np.log10(np.clip(y_pred, 1e-10, None))

    return float(10 ** np.mean(np.abs(y_pred - y_true)))


def r2_log(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> float:
    """
    R² (coefficient of determination) computed on log10 scale.
    Target: R² > 0.7.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if not log_scale:
        y_true = np.log10(np.clip(y_true, 1e-10, None))
        y_pred = np.log10(np.clip(y_pred, 1e-10, None))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> float:
    """RMSE on log10 scale."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if not log_scale:
        y_true = np.log10(np.clip(y_true, 1e-10, None))
        y_pred = np.log10(np.clip(y_pred, 1e-10, None))

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def within_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold:   float = 2.0,
    log_scale: bool = True
) -> float:
    """
    Percentage of predictions within N-fold of true value.
    e.g. within_fold(..., fold=2.0) → % within 2-fold error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if not log_scale:
        y_true = np.log10(np.clip(y_true, 1e-10, None))
        y_pred = np.log10(np.clip(y_pred, 1e-10, None))

    log_fold = np.log10(fold)
    within   = np.abs(y_pred - y_true) <= log_fold
    return float(100 * within.mean())


# ══════════════════════════════════════════════════════════════════════════════
# Full evaluation report
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    param_name: str  = '',
    log_scale:  bool = True,
    verbose:    bool = True,
) -> dict:
    """
    Full evaluation of predictions for a single PK parameter.

    Args:
        y_true:     true values
        y_pred:     predicted values
        param_name: name of the parameter (for display)
        log_scale:  if True, inputs are log10-transformed
        verbose:    print results

    Returns:
        dict with keys: gmfe, r2, rmse, within_2fold, within_3fold, n
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Remove any NaN pairs
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    results = {
        'param':        param_name,
        'n':            len(y_true),
        'gmfe':         gmfe(y_true, y_pred, log_scale),
        'r2':           r2_log(y_true, y_pred, log_scale),
        'rmse':         rmse_log(y_true, y_pred, log_scale),
        'within_2fold': within_fold(y_true, y_pred, fold=2.0, log_scale=log_scale),
        'within_3fold': within_fold(y_true, y_pred, fold=3.0, log_scale=log_scale),
    }

    if verbose:
        label = f"[{param_name}]" if param_name else ''
        meets_gmfe = '✅' if results['gmfe'] < 1.5  else '❌'
        meets_r2   = '✅' if results['r2']   > 0.7  else '❌'
        print(f"\n{'─'*45}")
        print(f"  {label} n={results['n']}")
        print(f"  GMFE:         {results['gmfe']:.3f}  {meets_gmfe} (target < 1.5)")
        print(f"  R²:           {results['r2']:.3f}  {meets_r2} (target > 0.7)")
        print(f"  RMSE (log10): {results['rmse']:.3f}")
        print(f"  Within 2-fold:{results['within_2fold']:.1f}%")
        print(f"  Within 3-fold:{results['within_3fold']:.1f}%")
        print(f"{'─'*45}")

    return results


def evaluate_all(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    log_scale: bool = True,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Evaluate predictions for both CL and Vd simultaneously.

    Args:
        y_true:    array of shape (n, 2) — columns: [log10_CL, log10_Vd]
        y_pred:    array of shape (n, 2) — same layout
        log_scale: True if inputs are log10-transformed

    Returns:
        DataFrame with one row per parameter
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    params = ['CL', 'Vd']
    rows   = []

    if verbose:
        print("\n" + "=" * 45)
        print("  EVALUATION RESULTS")
        print("=" * 45)

    for i, param in enumerate(params):
        results = evaluate(
            y_true[:, i], y_pred[:, i],
            param_name=param,
            log_scale=log_scale,
            verbose=verbose
        )
        rows.append(results)

    df = pd.DataFrame(rows).set_index('param')

    if verbose:
        targets_met = (df['gmfe'] < 1.5).all() and (df['r2'] > 0.7).all()
        print(f"\n  All targets met: {'✅ YES' if targets_met else '❌ NO'}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective helper
# ══════════════════════════════════════════════════════════════════════════════

def optuna_objective_score(y_true: np.ndarray, y_pred: np.ndarray, log_scale: bool = True) -> float:
    """
    Single scalar score for Optuna to minimize.
    Uses log10 RMSE (proxy for GMFE, differentiable-friendly).
    Lower is better.
    """
    return rmse_log(y_true, y_pred, log_scale=log_scale)
