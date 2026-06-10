"""
scripts/generate_paper_figures.py
===================================
Generates publication-quality figures for the Wadhams PK Predictor paper.

Figure 1: Observed vs. predicted scatter plots (RF + XGB, CL + Vd)
           2x2 panel, scaffold split test set
Figure 2: t½ prediction accuracy by elimination rate stratum (bar chart)
Figure 3: Top 15 SHAP features for RF CL and RF Vd (horizontal bar)

Outputs (figures/):
  - fig1_obs_vs_pred.png   (300 dpi)
  - fig2_thalf_strat.png   (300 dpi)
  - fig3_shap.png          (300 dpi)

Run:
    conda activate pkip-env
    python scripts/generate_paper_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pickle
import json
from pathlib import Path
import sys

ROOT    = Path(__file__).resolve().parents[1]
PROC    = ROOT / "data/processed"
RF_DIR  = ROOT / "models/saved/scaffold_rf"
XGB_DIR = ROOT / "models/saved/scaffold_xgb"
GNN_DIR = ROOT / "models/saved/scaffold_gnn"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from models.random_forest  import PKRandomForest
from models.xgboost_model  import PKXGBoost
from models.gnn_model      import PKAttentiveFP
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Colour palette (JCIM-friendly, colourblind-safe) ──────────────────────────
C_RF    = "#2166AC"   # blue
C_XGB   = "#D6604D"   # red-orange
C_UNITY = "#555555"   # grey unity line
C_2FOLD = "#AAAAAA"   # light grey 2-fold band
ALPHA   = 0.45
MS      = 18          # marker size (scatter)

# ── Load scaffold test data ───────────────────────────────────────────────────
def load_test():
    X_test  = np.load(PROC / "scaffold_X_test_desc_fp.npy")
    y_test  = np.load(PROC / "scaffold_y_test.npy")
    y_cl    = 10 ** y_test[:, 0]
    y_vd    = 10 ** y_test[:, 1]
    graphs  = torch.load(PROC / "scaffold_test_graphs.pt", map_location=DEVICE)
    return X_test, y_cl, y_vd, graphs


def get_preds(X_test, y_cl, y_vd, graphs):
    preds = {}

    # RF
    for param, y_obs in [("CL", y_cl), ("Vd", y_vd)]:
        model = PKRandomForest.load(str(RF_DIR / f"rf_{param}_best.pkl"))
        X_sel = X_test[:, model.feat_idx]
        y_pred = model.predict_original_scale(X_sel)
        fe   = np.where(y_pred > y_obs, y_pred / y_obs, y_obs / y_pred)
        preds[("RF", param)] = {
            "obs":  y_obs, "pred": y_pred,
            "gmfe": 10 ** np.mean(np.abs(np.log10(y_pred / y_obs))),
            "w2":   100 * np.mean(fe <= 2.0),
        }

    # XGB
    for param, y_obs in [("CL", y_cl), ("Vd", y_vd)]:
        model = PKXGBoost.load(str(XGB_DIR / f"xgb_{param}_best.pkl"))
        X_sel = X_test[:, model.feat_idx]
        y_pred = model.predict_original_scale(X_sel)
        fe   = np.where(y_pred > y_obs, y_pred / y_obs, y_obs / y_pred)
        preds[("XGB", param)] = {
            "obs":  y_obs, "pred": y_pred,
            "gmfe": 10 ** np.mean(np.abs(np.log10(y_pred / y_obs))),
            "w2":   100 * np.mean(fe <= 2.0),
        }

    # GNN (dual-head)
    gnn = PKAttentiveFP.load(str(GNN_DIR / "gnn_best.pt"), device=str(DEVICE))
    gnn.eval()
    loader = GeoDataLoader(graphs, batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.append(out.cpu().numpy())
    gnn_preds = np.vstack(all_preds)
    for idx, (param, y_obs) in enumerate(zip(["CL", "Vd"], [y_cl, y_vd])):
        y_pred = 10 ** gnn_preds[:, idx]
        fe   = np.where(y_pred > y_obs, y_pred / y_obs, y_obs / y_pred)
        preds[("GNN", param)] = {
            "obs":  y_obs, "pred": y_pred,
            "gmfe": 10 ** np.mean(np.abs(np.log10(y_pred / y_obs))),
            "w2":   100 * np.mean(fe <= 2.0),
        }

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Observed vs Predicted (2×2 panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_obs_vs_pred(preds):
    C_GNN = "#1B7837"   # green
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.38, wspace=0.35)

    panels = [
        (("RF",  "CL"), axes[0, 0], C_RF,  "RF",  "CL (mL/min/kg)"),
        (("XGB", "CL"), axes[0, 1], C_XGB, "XGB", "CL (mL/min/kg)"),
        (("GNN", "CL"), axes[0, 2], C_GNN, "GNN", "CL (mL/min/kg)"),
        (("RF",  "Vd"), axes[1, 0], C_RF,  "RF",  "Vd,ss (L/kg)"),
        (("XGB", "Vd"), axes[1, 1], C_XGB, "XGB", "Vd,ss (L/kg)"),
        (("GNN", "Vd"), axes[1, 2], C_GNN, "GNN", "Vd,ss (L/kg)"),
    ]

    for key, ax, color, mname, units in panels:
        d    = preds[key]
        obs  = d["obs"]
        pred = d["pred"]
        param = key[1]

        lo = np.floor(np.log10(min(obs.min(), pred.min()))) - 0.2
        hi = np.ceil( np.log10(max(obs.max(), pred.max()))) + 0.2

        # 2-fold band
        x_band = np.logspace(lo, hi, 200)
        ax.fill_between(x_band, x_band / 2, x_band * 2,
                        color=C_2FOLD, alpha=0.25, zorder=0, label="2-fold")

        # Unity line
        ax.plot([10**lo, 10**hi], [10**lo, 10**hi],
                color=C_UNITY, lw=1.2, ls="--", zorder=1)

        # Scatter
        ax.scatter(obs, pred, color=color, alpha=ALPHA, s=MS,
                   edgecolors="none", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10**lo, 10**hi)
        ax.set_ylim(10**lo, 10**hi)

        ax.set_xlabel(f"Observed {units}", fontsize=9)
        ax.set_ylabel(f"Predicted {units}", fontsize=9)
        ax.set_title(f"{mname} — {param}", fontsize=10, fontweight="bold")

        # Annotation
        ax.text(0.05, 0.92,
                f"GMFE = {d['gmfe']:.2f}\n{d['w2']:.0f}% within 2-fold\nn = {len(obs)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

        ax.tick_params(labelsize=8)

    # Legend (shared)
    legend_elements = [
        Line2D([0], [0], ls="--", color=C_UNITY, lw=1.2, label="Unity"),
        plt.Rectangle((0, 0), 1, 1, fc=C_2FOLD, alpha=0.4, label="2-fold range"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Observed vs. Predicted PK Parameters\n(Scaffold Split Test Set, n = 294)",
        fontsize=11, fontweight="bold", y=0.98
    )

    out = FIG_DIR / "fig1_obs_vs_pred.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — t½ stratification
# ─────────────────────────────────────────────────────────────────────────────
def fig2_thalf_strat():
    df = pd.read_csv(PROC / "thalf_validation.csv")

    # Bin by observed t½
    bins   = [0, 6, 24, 72, np.inf]
    labels = ["Fast\n(<6 h)", "Moderate\n(6–24 h)", "Slow\n(24–72 h)", "Very slow\n(≥72 h)"]
    df["stratum"] = pd.cut(df["thalf_h"], bins=bins, labels=labels)

    summary = df.groupby("stratum", observed=True).apply(
        lambda g: pd.Series({
            "n":     len(g),
            "w2":    100 * (np.where(g["thalf_pred"] > g["thalf_h"],
                                     g["thalf_pred"] / g["thalf_h"],
                                     g["thalf_h"]    / g["thalf_pred"]) <= 2.0).mean(),
            "gmfe":  10 ** np.mean(np.abs(np.log10(
                         g["thalf_pred"].clip(lower=0.001) /
                         g["thalf_h"].clip(lower=0.001)))),
        })
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.subplots_adjust(wspace=0.38)

    colors = ["#2166AC", "#4DAC26", "#D6604D", "#8E0152"]
    x = np.arange(len(summary))

    # Panel A — Within 2-fold %
    bars = ax1.bar(x, summary["w2"], color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    ax1.axhline(50, color="0.4", ls="--", lw=1, label="50% reference")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\n(n={int(n)})"
                         for s, n in zip(summary["stratum"], summary["n"])],
                        fontsize=8.5)
    ax1.set_ylabel("Within 2-fold (%)", fontsize=9)
    ax1.set_ylim(0, 80)
    ax1.set_title("A   Half-life: Within 2-fold Accuracy", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, summary["w2"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5)

    # Panel B — GMFE
    bars2 = ax2.bar(x, summary["gmfe"], color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    ax2.axhline(2.0, color="0.4", ls="--", lw=1, label="GMFE = 2.0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n(n={int(n)})"
                         for s, n in zip(summary["stratum"], summary["n"])],
                        fontsize=8.5)
    ax2.set_ylabel("GMFE", fontsize=9)
    ax2.set_ylim(0, summary["gmfe"].max() * 1.25)
    ax2.set_title("B   Half-life: GMFE by Elimination Rate", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, summary["gmfe"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Half-life Prediction Accuracy by Elimination Rate Stratum",
                 fontsize=11, fontweight="bold")

    out = FIG_DIR / "fig2_thalf_strat.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Feature importance (RF MDI, top 15 per endpoint)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_feature_importance():
    feat_names = (PROC / "scaffold_feature_names.txt").read_text().strip().split("\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(wspace=0.55)

    for ax, param, color, label in [
        (axes[0], "CL", C_RF,  "A   RF Feature Importance — CL"),
        (axes[1], "Vd", C_XGB, "B   RF Feature Importance — Vd,ss"),
    ]:
        model = PKRandomForest.load(str(RF_DIR / f"rf_{param}_best.pkl"))
        imp   = model.feature_importances_

        # Map feat_idx back to names
        indexed_imp = [(feat_names[i] if i < len(feat_names) else f"feat_{i}", imp[j])
                       for j, i in enumerate(model.feat_idx)]
        top15 = sorted(indexed_imp, key=lambda x: x[1], reverse=True)[:15]
        names_top, vals_top = zip(*top15)

        # Clean up long names
        clean = [n.replace("rdkit_", "").replace("_", " ") for n in names_top]

        y_pos = np.arange(len(clean))
        ax.barh(y_pos, vals_top, color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Random Forest Feature Importance (Scaffold Split Training Set)",
                 fontsize=11, fontweight="bold")

    out = FIG_DIR / "fig3_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print("Generating paper figures...")
    print("\n[1/3] Loading test data and generating predictions...")
    X_test, y_cl, y_vd, graphs = load_test()
    preds = get_preds(X_test, y_cl, y_vd, graphs)

    print("\n[2/3] Figure 1 — Observed vs Predicted...")
    fig1_obs_vs_pred(preds)

    print("\n[3/3] Figure 2 — t½ stratification...")
    fig2_thalf_strat()

    print("\n[4/4] Figure 3 — Feature importance...")
    fig3_feature_importance()

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    run()
