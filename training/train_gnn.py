"""
training/train_gnn.py
=====================
Optuna-driven hyperparameter tuning for the AttentiveFP GNN.

Architecture: AttentiveFP encoder → dual linear heads (CL, Vd)
Targets:      log10(CL) and log10(Vd) — same scale as RF/XGB

Pipeline:
  Phase 1 — Optuna tuning (175 trials, MedianPruner)
    Tunes: hidden_channels, num_layers, num_timesteps, dropout,
           learning_rate, weight_decay, batch_size
    Each trial: 5-fold CV, max 100 epochs per fold, early stopping
    Objective: mean CV log10-RMSE averaged over CL and Vd

  Phase 2 — Final fit
    Train best config on full training set (no CV split)
    Max 200 epochs, early stopping on 10% holdout of train
    Evaluate on held-out test set
    Save model + results

Outputs (models/saved/gnn/):
  - gnn_best.pt          — model weights + hyperparams
  - gnn_study.pkl        — Optuna study object
  - gnn_results.json     — test set metrics for CL and Vd

Data:
  - Reads pre-built PyG graph lists: data/processed/train_graphs.pt
                                      data/processed/test_graphs.pt
  - y values already embedded in graph.y as [log10_CL, log10_Vd]

Run:
    conda activate pkip-env
    python training/train_gnn.py

Cluster (SLURM) — see scripts/slurm/train_gnn.sh
"""

import json
import pickle
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import KFold

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gnn_model    import PKAttentiveFP
from evaluation.metrics  import evaluate, optuna_objective_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC     = ROOT / "data/processed"
SAVE_DIR = ROOT / "models/saved/gnn"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS     = 175
N_CV_FOLDS   = 5
RANDOM_STATE = 42
MAX_EPOCHS_TRIAL = 100    # per fold during Optuna
MAX_EPOCHS_FINAL = 200    # for final fit
PATIENCE     = 15         # early stopping patience (epochs)
PARAMS       = ['CL', 'Vd']

# Device selection: MPS (Apple Silicon) > CUDA > CPU
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y.view(-1, 2))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """One validation epoch. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss  = criterion(out, batch.y.view(-1, 2))
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, device) -> np.ndarray:
    """Run inference on a DataLoader. Returns (N, 2) numpy array."""
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(out.cpu().numpy())
    return np.vstack(preds)   # (N, 2)


def fit_model(
    model:       PKAttentiveFP,
    train_data:  list,
    val_data:    list,
    lr:          float,
    weight_decay: float,
    batch_size:  int,
    max_epochs:  int,
    patience:    int,
    device:      torch.device,
    trial:       optuna.Trial = None,
    fold:        int = 0,
) -> float:
    """
    Train model until early stopping or max_epochs.
    Returns best validation loss achieved.
    Optionally reports to Optuna pruner if trial is provided.
    """
    model = model.to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    no_improve     = 0
    best_state     = None

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        # Report to Optuna pruner every 10 epochs
        if trial is not None and epoch % 10 == 9:
            trial.report(best_val_loss, step=fold * max_epochs + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_improve >= patience:
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_val_loss


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(train_graphs: list, device: torch.device):
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    indices = np.arange(len(train_graphs))

    def objective(trial: optuna.Trial) -> float:
        # ── Hyperparameter suggestions ────────────────────────────────────────
        hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 192, 256])
        num_layers      = trial.suggest_int('num_layers', 2, 6)
        num_timesteps   = trial.suggest_int('num_timesteps', 2, 6)
        dropout         = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
        lr              = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        weight_decay    = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size      = trial.suggest_categorical('batch_size', [32, 64, 128])

        # ── 5-fold CV ─────────────────────────────────────────────────────────
        fold_rmse_cl = []
        fold_rmse_vd = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(indices)):
            tr_data  = [train_graphs[i] for i in tr_idx]
            val_data = [train_graphs[i] for i in val_idx]

            torch.manual_seed(RANDOM_STATE + fold)
            model = PKAttentiveFP(
                hidden_channels = hidden_channels,
                num_layers      = num_layers,
                num_timesteps   = num_timesteps,
                dropout         = dropout,
            )

            fit_model(
                model, tr_data, val_data,
                lr=lr, weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=MAX_EPOCHS_TRIAL,
                patience=PATIENCE,
                device=device,
                trial=trial,
                fold=fold,
            )

            # Predict on val fold
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            y_pred_all = predict_all(model, val_loader, device)   # (N_val, 2)
            y_true_all = np.stack([g.y.numpy() for g in val_data]) # (N_val, 2)

            rmse_cl = optuna_objective_score(y_true_all[:, 0], y_pred_all[:, 0], log_scale=True)
            rmse_vd = optuna_objective_score(y_true_all[:, 1], y_pred_all[:, 1], log_scale=True)
            fold_rmse_cl.append(rmse_cl)
            fold_rmse_vd.append(rmse_vd)

        # Objective: average log10-RMSE across both params and all folds
        mean_rmse = float(np.mean(fold_rmse_cl + fold_rmse_vd))
        return mean_rmse

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn():
    print("=" * 55)
    print("GNN (AttentiveFP) TRAINING")
    print("=" * 55)

    device = get_device()
    print(f"\nDevice: {device}")

    # ── Load graphs ───────────────────────────────────────────────────────────
    print("\nLoading graph datasets...")
    train_graphs: list = torch.load(str(PROC / "train_graphs.pt"), weights_only=False)
    test_graphs:  list = torch.load(str(PROC / "test_graphs.pt"),  weights_only=False)
    print(f"  Train graphs: {len(train_graphs)}  |  Test graphs: {len(test_graphs)}")

    # Sanity check — peek at first graph
    g0 = train_graphs[0]
    print(f"  Node features: {g0.x.shape[1]}  |  Edge features: {g0.edge_attr.shape[1]}  |  y shape: {g0.y.shape}")

    # ── Phase 1: Optuna tuning ────────────────────────────────────────────────
    print(f"\nPhase 1: Optuna tuning — {N_TRIALS} trials, {N_CV_FOLDS}-fold CV...")
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner  = MedianPruner(n_startup_trials=15, n_warmup_steps=N_CV_FOLDS * 20)
    study   = optuna.create_study(
        direction  = 'minimize',
        sampler    = sampler,
        pruner     = pruner,
        study_name = 'gnn_attentivefp',
    )

    objective = make_objective(train_graphs, device)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best trial:   #{study.best_trial.number}")
    print(f"  Best CV RMSE: {study.best_value:.4f}")
    print(f"  Best params:  {best}")

    # Save study
    with open(SAVE_DIR / "gnn_study.pkl", 'wb') as f:
        pickle.dump(study, f)

    # ── Phase 2: Final fit on full training set ───────────────────────────────
    print(f"\nPhase 2: Final fit on full training set ({len(train_graphs)} compounds)...")

    # Use a small validation holdout from train (10%) just for early stopping
    n_train = len(train_graphs)
    n_val   = max(50, int(0.10 * n_train))
    rng     = np.random.default_rng(RANDOM_STATE)
    val_idx = rng.choice(n_train, size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(n_train), val_idx)

    final_train = [train_graphs[i] for i in tr_idx]
    final_val   = [train_graphs[i] for i in val_idx]

    torch.manual_seed(RANDOM_STATE)
    final_model = PKAttentiveFP(
        hidden_channels = best['hidden_channels'],
        num_layers      = best['num_layers'],
        num_timesteps   = best['num_timesteps'],
        dropout         = best['dropout'],
    )

    fit_model(
        final_model, final_train, final_val,
        lr           = best['lr'],
        weight_decay = best['weight_decay'],
        batch_size   = best['batch_size'],
        max_epochs   = MAX_EPOCHS_FINAL,
        patience     = PATIENCE,
        device       = device,
    )
    final_model.eval()

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\nTest set evaluation:")
    test_loader = DataLoader(test_graphs, batch_size=best['batch_size'], shuffle=False)
    y_pred_all  = predict_all(final_model, test_loader, device)   # (N_test, 2)
    y_true_all  = np.stack([g.y.numpy() for g in test_graphs])    # (N_test, 2)

    all_results = {}
    for param_idx, param_name in enumerate(PARAMS):
        results = evaluate(
            y_true_all[:, param_idx],
            y_pred_all[:, param_idx],
            param_name = f'GNN {param_name}',
            log_scale  = True,
        )
        all_results[param_name] = results

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("GNN — FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    for param, res in all_results.items():
        gmfe_flag = '✅' if res['gmfe'] < 2.2 else '❌'
        r2_flag   = '✅' if res['r2']   > 0.45 else '❌'
        print(f"  {param}: GMFE={res['gmfe']:.3f}{gmfe_flag}  R²={res['r2']:.3f}{r2_flag}  within-2fold={res['within_2fold']:.1f}%")

    # Save model
    model_path = SAVE_DIR / "gnn_best.pt"
    final_model.save(str(model_path), hyperparams={
        'hidden_channels': best['hidden_channels'],
        'num_layers':      best['num_layers'],
        'num_timesteps':   best['num_timesteps'],
        'dropout':         best['dropout'],
    })
    print(f"\n  Model saved → {model_path}")

    # Save results JSON
    results_out = {k: {m: float(v) for m, v in r.items() if m != 'param'}
                   for k, r in all_results.items()}
    with open(SAVE_DIR / "gnn_results.json", 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"  Results → {SAVE_DIR / 'gnn_results.json'}")


if __name__ == '__main__':
    train_gnn()
