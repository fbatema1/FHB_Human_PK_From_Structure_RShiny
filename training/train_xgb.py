"""
training/train_xgb.py
=====================
Optuna-driven hyperparameter tuning for XGBoost models.

Follows the same 3-phase structure as train_rf.py.
XGBoost-specific additions:
  - Early stopping on each CV fold's validation split (faster, better)
  - Larger hyperparameter space: learning_rate, subsample, colsample_bytree,
    gamma, reg_alpha, reg_lambda (L1/L2 regularization)
  - SHAP rankings reused from RF phase if already computed, otherwise recomputed

Pipeline per parameter (CL, Vd):
  Phase 1 — SHAP ranking (reuses RF ranking if featurizer_{param}.pkl exists)
  Phase 2 — 300-trial Optuna TPE + MedianPruner, 5-fold CV with early stopping
  Phase 3 — Final fit, test set evaluation, save

Outputs (models/saved/xgb/):
  - xgb_CL_best.pkl
  - xgb_Vd_best.pkl
  - xgb_CL_study.pkl
  - xgb_Vd_study.pkl
  - xgb_results.json

Run:
    conda activate pkip-env
    python training/train_xgb.py

Cluster (SLURM) — see scripts/slurm/train_xgb.sh
"""

import json
import pickle
import warnings
import numpy as np
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from sklearn.model_selection import KFold

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.rdkit_features import RDKitFeaturizer
from models.xgboost_model    import PKXGBoost
from evaluation.metrics      import evaluate, optuna_objective_score
from training.train_rf       import compute_shap_ranking

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC     = ROOT / "data/processed"
SAVE_DIR = ROOT / "models/saved/xgb"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS              = 300
N_CV_FOLDS            = 5
RANDOM_STATE          = 42
PARAMS                = ['CL', 'Vd']
EARLY_STOPPING_ROUNDS = 50


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def make_objective(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    featurizer: RDKitFeaturizer,
    n_desc:     int,
):
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        # ── Hyperparameter suggestions ────────────────────────────────────────
        top_n = trial.suggest_int('top_n_desc', 20, n_desc)

        n_estimators     = trial.suggest_int('n_estimators', 100, 1500, step=100)
        learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
        max_depth        = trial.suggest_int('max_depth', 3, 12)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        subsample        = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1.0)
        gamma            = trial.suggest_float('gamma', 0.0, 5.0)
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)

        # ── Feature selection ─────────────────────────────────────────────────
        desc_idx = featurizer.shap_ranking[:top_n]
        fp_idx   = np.arange(n_desc, X_train.shape[1])
        feat_idx = np.concatenate([desc_idx, fp_idx])
        X_sel    = X_train[:, feat_idx]

        # ── 5-fold CV with early stopping ─────────────────────────────────────
        cv_scores = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sel)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = PKXGBoost(
                n_estimators         = n_estimators,
                learning_rate        = learning_rate,
                max_depth            = max_depth,
                min_child_weight     = min_child_weight,
                subsample            = subsample,
                colsample_bytree     = colsample_bytree,
                gamma                = gamma,
                reg_alpha            = reg_alpha,
                reg_lambda           = reg_lambda,
                early_stopping_rounds= EARLY_STOPPING_ROUNDS,
                random_state         = RANDOM_STATE,
            )
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            y_pred    = model.predict(X_val)
            fold_rmse = optuna_objective_score(y_val, y_pred, log_scale=True)
            cv_scores.append(fold_rmse)

            # Report for pruning
            trial.report(np.mean(cv_scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(cv_scores))

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def train_xgb():
    print("=" * 55)
    print("XGBOOST TRAINING")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading features and targets...")
    X_train = np.load(PROC / "X_train_desc_fp.npy")
    X_test  = np.load(PROC / "X_test_desc_fp.npy")
    y_train = np.load(PROC / "y_train.npy")
    y_test  = np.load(PROC / "y_test.npy")

    featurizer: RDKitFeaturizer = RDKitFeaturizer.load(str(PROC / "featurizer.pkl"))
    n_desc = len(featurizer.kept_desc_names)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    all_results = {}

    for param_idx, param_name in enumerate(PARAMS):
        print(f"\n{'='*55}")
        print(f"  PARAMETER: {param_name}  (col {param_idx})")
        print(f"{'='*55}")

        y_tr = y_train[:, param_idx]
        y_te = y_test[:,  param_idx]

        # ── Phase 1: SHAP ranking ─────────────────────────────────────────────
        # Reuse RF ranking if available (same training data, same descriptors)
        rf_feat_path = PROC / f"featurizer_{param_name}.pkl"
        if rf_feat_path.exists():
            print(f"\nPhase 1: Reusing SHAP ranking from RF phase ({rf_feat_path.name})")
            feat_with_shap = RDKitFeaturizer.load(str(rf_feat_path))
            featurizer.shap_ranking = feat_with_shap.shap_ranking
        else:
            print(f"\nPhase 1: Computing SHAP ranking for {param_name}...")
            shap_ranking = compute_shap_ranking(
                X_train[:, :n_desc], y_tr, featurizer, param_name
            )
            featurizer.shap_ranking = shap_ranking

        top5 = [featurizer.kept_desc_names[i] for i in featurizer.shap_ranking[:5]]
        print(f"  Top 5 descriptors: {top5}")

        # ── Phase 2: Optuna tuning ────────────────────────────────────────────
        print(f"\nPhase 2: Optuna tuning — {N_TRIALS} trials, {N_CV_FOLDS}-fold CV...")
        sampler = TPESampler(seed=RANDOM_STATE)
        pruner  = MedianPruner(n_startup_trials=20, n_warmup_steps=2)
        study   = optuna.create_study(
            direction  = 'minimize',
            sampler    = sampler,
            pruner     = pruner,
            study_name = f'xgb_{param_name}',
        )

        objective = make_objective(X_train, y_tr, featurizer, n_desc)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best = study.best_params
        print(f"\n  Best trial:   #{study.best_trial.number}")
        print(f"  Best CV RMSE: {study.best_value:.4f}")
        print(f"  Best params:  {best}")

        # Save study
        with open(SAVE_DIR / f"xgb_{param_name}_study.pkl", 'wb') as f:
            pickle.dump(study, f)

        # ── Phase 3: Final fit ────────────────────────────────────────────────
        print(f"\nPhase 3: Final fit on full training set...")

        top_n    = best['top_n_desc']
        desc_idx = featurizer.shap_ranking[:top_n]
        fp_idx   = np.arange(n_desc, X_train.shape[1])
        feat_idx = np.concatenate([desc_idx, fp_idx])

        X_tr_sel = X_train[:, feat_idx]
        X_te_sel = X_test[:,  feat_idx]

        # Final fit — no early stopping (use full n_estimators on all data)
        final_model = PKXGBoost(
            n_estimators         = best['n_estimators'],
            learning_rate        = best['learning_rate'],
            max_depth            = best['max_depth'],
            min_child_weight     = best['min_child_weight'],
            subsample            = best['subsample'],
            colsample_bytree     = best['colsample_bytree'],
            gamma                = best['gamma'],
            reg_alpha            = best['reg_alpha'],
            reg_lambda           = best['reg_lambda'],
            early_stopping_rounds= None,   # no early stopping on final fit
            random_state         = RANDOM_STATE,
        )
        final_model.fit(X_tr_sel, y_tr, param_name=param_name)
        final_model.feat_idx   = feat_idx
        final_model.top_n_desc = top_n

        # ── Evaluate ──────────────────────────────────────────────────────────
        print(f"\nTest set evaluation — {param_name}:")
        y_pred  = final_model.predict(X_te_sel)
        results = evaluate(y_te, y_pred, param_name=f'XGB {param_name}', log_scale=True)
        all_results[param_name] = results

        model_path = SAVE_DIR / f"xgb_{param_name}_best.pkl"
        final_model.save(str(model_path))
        print(f"  Model saved → {model_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("XGBOOST — FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    for param, res in all_results.items():
        gmfe_flag = '✅' if res['gmfe'] < 1.5 else '❌'
        r2_flag   = '✅' if res['r2']   > 0.7 else '❌'
        print(f"  {param}: GMFE={res['gmfe']:.3f}{gmfe_flag}  R²={res['r2']:.3f}{r2_flag}  within-2fold={res['within_2fold']:.1f}%")

    results_out = {k: {m: float(v) for m, v in r.items() if m != 'param'}
                   for k, r in all_results.items()}
    with open(SAVE_DIR / "xgb_results.json", 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  Results → {SAVE_DIR / 'xgb_results.json'}")


if __name__ == '__main__':
    train_xgb()
