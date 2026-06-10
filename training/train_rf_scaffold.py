"""
training/train_rf_scaffold.py
====================
Optuna-driven hyperparameter tuning for Random Forest models.

Pipeline per parameter (CL, Vd):
  Phase 1 — SHAP ranking
    Fit a baseline RF on full training set, compute SHAP values,
    rank descriptors by mean |SHAP|. Ranking saved to featurizer.
    (Morgan FP bits are always included — only descriptor count is tuned.)

  Phase 2 — Optuna tuning (300 trials, MedianPruner)
    Each trial samples:
      - top_n_desc    : int [20, 162]  — how many SHAP-ranked descriptors to use
      - n_estimators  : int [100, 1000]
      - max_depth     : int [3, 30] or None
      - min_samples_leaf : int [1, 20]
      - min_samples_split: int [2, 20]
      - max_features  : 'sqrt' | 'log2' | float [0.1, 0.5]
    Objective: mean 5-fold CV log10-RMSE (lower = better)
    Note: SHAP ranking is computed once in Phase 1, not per trial,
          to keep tuning tractable. Feature *count* is tuned per trial.

  Phase 3 — Final fit
    Refit best model on full training set with best hyperparameters.
    Evaluate on held-out test set. Save model and results.

Outputs (models/saved/rf/):
  - rf_CL_best.pkl        — fitted PKRandomForest for CL
  - rf_Vd_best.pkl        — fitted PKRandomForest for Vd
  - rf_CL_study.pkl       — Optuna study object (all trials)
  - rf_Vd_study.pkl       — Optuna study object
  - rf_results.json       — test set metrics for both parameters

Run:
    conda activate pkip-env
    python training/train_rf_scaffold.py

Cluster (SLURM) — see scripts/slurm/train_rf.sh
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from sklearn.model_selection import KFold

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.rdkit_features import RDKitFeaturizer
from models.random_forest    import PKRandomForest
from evaluation.metrics      import evaluate, optuna_objective_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC     = ROOT / "data/processed"
SAVE_DIR = ROOT / "models/saved/scaffold_rf"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS     = 300
N_CV_FOLDS   = 5
RANDOM_STATE = 42
PARAMS       = ['CL', 'Vd']   # column indices 0 and 1 in y arrays
N_SHAP_SAMPLE = 200            # compounds to subsample for SHAP (speed)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — SHAP ranking
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap_ranking(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    featurizer: RDKitFeaturizer,
    param_name: str,
    n_sample:   int = N_SHAP_SAMPLE,
) -> np.ndarray:
    """
    Fit a baseline RF and compute SHAP-based descriptor ranking.
    Returns indices of descriptors sorted by mean |SHAP| (descending).
    Morgan FP bits are excluded from ranking (always included in full).

    Note: Ranking is computed on training set only — no test data involved.
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — pip install shap")
        raise

    n_desc = len(featurizer.kept_desc_names)
    X_desc = X_train[:, :n_desc]   # descriptor columns only

    # Subsample for speed
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_desc), size=min(n_sample, len(X_desc)), replace=False)
    X_sample = X_desc[idx]
    y_sample = y_train[idx]

    print(f"  [SHAP] Fitting baseline RF on {len(X_desc)} compounds...")
    baseline_rf = PKRandomForest(n_estimators=200, random_state=RANDOM_STATE)
    baseline_rf.fit(X_desc, y_train, param_name=param_name)

    print(f"  [SHAP] Computing SHAP values on {len(X_sample)} compounds...")
    explainer   = shap.TreeExplainer(baseline_rf.model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    ranking     = np.argsort(mean_abs)[::-1]   # descending

    print(f"  [SHAP] Top 10 descriptors: {[featurizer.kept_desc_names[i] for i in ranking[:10]]}")
    return ranking


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def make_objective(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    featurizer: RDKitFeaturizer,
    n_desc:     int,
):
    """
    Returns an Optuna objective function for a single PK parameter.
    Closed over X_train, y_train, featurizer, and n_desc.
    """
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        # ── Hyperparameter suggestions ────────────────────────────────────────
        top_n = trial.suggest_int('top_n_desc', 20, n_desc)

        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)

        use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
        max_depth = trial.suggest_int('max_depth', 3, 30) if use_max_depth else None

        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

        max_features_type = trial.suggest_categorical('max_features_type', ['sqrt', 'log2', 'float'])
        if max_features_type == 'float':
            max_features = trial.suggest_float('max_features_float', 0.1, 0.5)
        else:
            max_features = max_features_type

        # ── Select top-N descriptors using pre-computed SHAP ranking ──────────
        # Morgan FP always appended after descriptors
        desc_idx = featurizer.shap_ranking[:top_n]
        fp_start = n_desc
        fp_idx   = np.arange(fp_start, X_train.shape[1])
        feat_idx = np.concatenate([desc_idx, fp_idx])
        X_sel    = X_train[:, feat_idx]

        # ── 5-fold CV ─────────────────────────────────────────────────────────
        cv_scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sel)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            rf = PKRandomForest(
                n_estimators     = n_estimators,
                max_depth        = max_depth,
                min_samples_leaf = min_samples_leaf,
                min_samples_split= min_samples_split,
                max_features     = max_features,
                random_state     = RANDOM_STATE,
            )
            rf.fit(X_tr, y_tr)
            y_pred    = rf.predict(X_val)
            fold_rmse = optuna_objective_score(y_val, y_pred, log_scale=True)
            cv_scores.append(fold_rmse)

            # Report intermediate value for pruning
            trial.report(np.mean(cv_scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(cv_scores))

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_rf():
    print("=" * 55)
    print("RANDOM FOREST TRAINING")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading features and targets...")
    X_train = np.load(PROC / "scaffold_X_train_desc_fp.npy")
    X_test  = np.load(PROC / "scaffold_X_test_desc_fp.npy")
    y_train = np.load(PROC / "scaffold_y_train.npy")
    y_test  = np.load(PROC / "scaffold_y_test.npy")

    featurizer: RDKitFeaturizer = RDKitFeaturizer.load(str(PROC / "scaffold_featurizer.pkl"))
    n_desc = len(featurizer.kept_desc_names)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Descriptors: {n_desc}  |  Total features: {X_train.shape[1]}")

    all_results = {}

    for param_idx, param_name in enumerate(PARAMS):
        print(f"\n{'='*55}")
        print(f"  PARAMETER: {param_name}  (col {param_idx})")
        print(f"{'='*55}")

        y_tr = y_train[:, param_idx]
        y_te = y_test[:,  param_idx]

        # ── Phase 1: SHAP ranking ─────────────────────────────────────────────
        print(f"\nPhase 1: SHAP descriptor ranking for {param_name}...")
        shap_ranking = compute_shap_ranking(
            X_train[:, :n_desc], y_tr, featurizer, param_name
        )
        featurizer.shap_ranking = shap_ranking
        featurizer.save(str(PROC / f"scaffold_featurizer_{param_name}.pkl"))

        # ── Phase 2: Optuna tuning ────────────────────────────────────────────
        print(f"\nPhase 2: Optuna tuning — {N_TRIALS} trials, {N_CV_FOLDS}-fold CV...")
        sampler = TPESampler(seed=RANDOM_STATE)
        pruner  = MedianPruner(n_startup_trials=20, n_warmup_steps=2)
        study   = optuna.create_study(
            direction  = 'minimize',
            sampler    = sampler,
            pruner     = pruner,
            study_name = f'rf_{param_name}',
        )

        objective = make_objective(X_train, y_tr, featurizer, n_desc)

        study.optimize(
            objective,
            n_trials  = N_TRIALS,
            show_progress_bar = True,
        )

        best = study.best_params
        print(f"\n  Best trial:  #{study.best_trial.number}")
        print(f"  Best CV RMSE: {study.best_value:.4f}")
        print(f"  Best params:  {best}")

        # Save study
        with open(SAVE_DIR / f"rf_{param_name}_study.pkl", 'wb') as f:
            pickle.dump(study, f)

        # ── Phase 3: Final fit on full training set ───────────────────────────
        print(f"\nPhase 3: Final fit on full training set...")

        top_n = best['top_n_desc']
        max_features = (
            best['max_features_float']
            if best['max_features_type'] == 'float'
            else best['max_features_type']
        )
        max_depth = best.get('max_depth') if best.get('use_max_depth') else None

        # Rebuild feature selection with best top_n
        desc_idx = featurizer.shap_ranking[:top_n]
        fp_idx   = np.arange(n_desc, X_train.shape[1])
        feat_idx = np.concatenate([desc_idx, fp_idx])

        X_tr_sel = X_train[:, feat_idx]
        X_te_sel = X_test[:,  feat_idx]

        final_model = PKRandomForest(
            n_estimators     = best['n_estimators'],
            max_depth        = max_depth,
            min_samples_leaf = best['min_samples_leaf'],
            min_samples_split= best['min_samples_split'],
            max_features     = max_features,
            random_state     = RANDOM_STATE,
        )
        final_model.fit(X_tr_sel, y_tr, param_name=param_name)

        # Save best feature indices with model for inference
        final_model.feat_idx  = feat_idx
        final_model.top_n_desc = top_n

        # ── Evaluate on train set (overfitting diagnostic) ────────────────────
        y_pred_tr  = final_model.predict(X_tr_sel)
        res_train  = evaluate(y_tr, y_pred_tr, param_name=f'RF {param_name} [TRAIN]', log_scale=True)
        print(f"\nTrain set evaluation — {param_name}:")
        print(f"  GMFE={res_train['gmfe']:.3f}  R²={res_train['r2']:.3f}  within-2fold={res_train['within_2fold']:.1f}%")

        # ── Evaluate on test set ──────────────────────────────────────────────
        print(f"\nTest set evaluation — {param_name}:")
        y_pred = final_model.predict(X_te_sel)
        results = evaluate(y_te, y_pred, param_name=f'RF {param_name}', log_scale=True)
        results['train_gmfe'] = float(res_train['gmfe'])
        results['train_r2']   = float(res_train['r2'])
        all_results[param_name] = results

        # Overfitting flag
        gmfe_gap = res_train['gmfe'] / results['gmfe']   # <1 means train better than test
        if gmfe_gap < 0.7:
            print(f"  ⚠️  Possible overfit: train GMFE {res_train['gmfe']:.3f} vs test GMFE {results['gmfe']:.3f}")
        else:
            print(f"  ✅ Generalisation gap acceptable (train/test GMFE ratio: {gmfe_gap:.2f})")

        # Save model
        model_path = SAVE_DIR / f"rf_{param_name}_best.pkl"
        final_model.save(str(model_path))
        print(f"  Model saved → {model_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("RANDOM FOREST — FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    for param, res in all_results.items():
        gmfe_flag = '✅' if res['gmfe'] < 2.2 else '❌'
        r2_flag   = '✅' if res['r2']   > 0.45 else '❌'
        print(f"  {param}: GMFE={res['gmfe']:.3f}{gmfe_flag}  R²={res['r2']:.3f}{r2_flag}  within-2fold={res['within_2fold']:.1f}%")

    # Save results JSON
    results_out = {k: {m: float(v) for m, v in r.items() if m != 'param'}
                   for k, r in all_results.items()}
    with open(SAVE_DIR / "rf_results.json", 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  Results → {SAVE_DIR / 'rf_results.json'}")


if __name__ == '__main__':
    train_rf()
