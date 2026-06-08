"""
evaluation/plots.py
===================
Publication-quality figures for the PK predictor manuscript.

All figures:
  - 300 DPI, exportable as PDF / SVG / PNG
  - Color-blind-friendly palette (Wong 2011)
  - Consistent font sizes and axis styling

Available functions:
  obs_vs_pred()          — observed vs predicted scatter (log10 + original scale)
  residual_plot()        — residuals vs predicted
  model_comparison()     — GMFE / R² bar chart across RF, XGB, GNN
  shap_summary()         — SHAP beeswarm + bar chart
  conformal_coverage()   — empirical vs nominal coverage across alpha levels
  interval_width_dist()  — distribution of PI widths
  all_manuscript_figs()  — generate every figure in one call

Usage:
    from evaluation.plots import all_manuscript_figs
    all_manuscript_figs(results_dict, output_dir='figures/')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for cluster + local)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Color palette (Wong 2011, color-blind safe) ───────────────────────────────
COLORS = {
    'RF':    '#0072B2',   # blue
    'XGB':   '#E69F00',   # orange
    'GNN':   '#009E73',   # green
    'obs':   '#333333',   # dark grey
    'fold2': '#CC79A7',   # pink  — 2-fold boundary
    'fold3': '#D55E00',   # vermillion — 3-fold boundary
    'CI':    '#56B4E9',   # sky blue — conformal intervals
}

PARAM_LABELS = {
    'CL': 'Clearance (CL)',
    'Vd': 'Volume of Distribution (Vd)',
}
PARAM_UNITS = {
    'CL': 'mL/min/kg',
    'Vd': 'L/kg',
}

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path, formats=('pdf', 'png')):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), bbox_inches='tight')
    plt.close(fig)


def _add_fold_lines(ax, lims, folds=(2, 3)):
    """Add 2-fold and 3-fold error lines to an obs-vs-pred plot."""
    x = np.linspace(lims[0], lims[1], 300)
    styles = {2: ('--', COLORS['fold2'], '2-fold'), 3: (':', COLORS['fold3'], '3-fold')}
    for fold, (ls, col, lbl) in styles.items():
        if fold in folds:
            ax.plot(x, x + np.log10(fold),  ls=ls, color=col, lw=1.2, label=f'+{lbl}')
            ax.plot(x, x - np.log10(fold),  ls=ls, color=col, lw=1.2, label=f'-{lbl}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Observed vs Predicted
# ══════════════════════════════════════════════════════════════════════════════

def obs_vs_pred(
    y_true_log:  np.ndarray,
    y_pred_log:  np.ndarray,
    param_name:  str,
    model_name:  str,
    metrics:     dict,
    output_path: Optional[str] = None,
    ci_lower_log: Optional[np.ndarray] = None,
    ci_upper_log: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Observed vs predicted scatter on log10 scale with 2/3-fold error bands.
    Optionally overlays conformal prediction intervals.

    Args:
        y_true_log:   true log10 values (n,)
        y_pred_log:   predicted log10 values (n,)
        param_name:   'CL' or 'Vd'
        model_name:   'RF', 'XGB', or 'GNN'
        metrics:      dict with keys gmfe, r2, within_2fold, within_3fold
        output_path:  if provided, save figure to this path (without extension)
        ci_lower_log: optional lower CI bound, log10 scale (n,)
        ci_upper_log: optional upper CI bound, log10 scale (n,)
    """
    color = COLORS.get(model_name, '#333333')
    units = PARAM_UNITS[param_name]
    label = PARAM_LABELS[param_name]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # ── Conformal intervals (if provided) ────────────────────────────────────
    if ci_lower_log is not None and ci_upper_log is not None:
        yerr = np.stack([y_pred_log - ci_lower_log,
                         ci_upper_log - y_pred_log])
        ax.errorbar(
            y_pred_log, y_true_log,
            xerr=yerr,
            fmt='none', color=COLORS['CI'], alpha=0.35, lw=0.8, zorder=1,
            label='95% PI',
        )

    # ── Scatter ───────────────────────────────────────────────────────────────
    ax.scatter(y_pred_log, y_true_log, color=color, alpha=0.65,
               s=28, linewidths=0, zorder=3, label='Compounds')

    # ── Reference lines ───────────────────────────────────────────────────────
    lims_raw = [min(y_true_log.min(), y_pred_log.min()) - 0.3,
                max(y_true_log.max(), y_pred_log.max()) + 0.3]
    ax.plot(lims_raw, lims_raw, 'k-', lw=1.2, zorder=2, label='Identity')
    _add_fold_lines(ax, lims_raw)

    # ── Metrics box ───────────────────────────────────────────────────────────
    txt = (f"GMFE = {metrics['gmfe']:.3f}\n"
           f"R² = {metrics['r2']:.3f}\n"
           f"Within 2-fold: {metrics['within_2fold']:.1f}%")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8, ec='#cccccc'))

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel(f'Predicted log₁₀({param_name}) [{units}]')
    ax.set_ylabel(f'Observed log₁₀({param_name}) [{units}]')
    ax.set_title(f'{model_name} — {label}')
    ax.set_xlim(lims_raw)
    ax.set_ylim(lims_raw)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.8, fontsize=9)

    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Residual Plot
# ══════════════════════════════════════════════════════════════════════════════

def residual_plot(
    y_true_log:  np.ndarray,
    y_pred_log:  np.ndarray,
    param_name:  str,
    model_name:  str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Residuals (observed - predicted) vs predicted values."""
    color = COLORS.get(model_name, '#333333')
    residuals = y_true_log - y_pred_log

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(y_pred_log, residuals, color=color, alpha=0.6, s=25, linewidths=0)
    ax.axhline(0, color='black', lw=1.2, linestyle='-')
    ax.axhline( np.log10(2), color=COLORS['fold2'], lw=1.0, linestyle='--', label='±2-fold')
    ax.axhline(-np.log10(2), color=COLORS['fold2'], lw=1.0, linestyle='--')
    ax.axhline( np.log10(3), color=COLORS['fold3'], lw=1.0, linestyle=':', label='±3-fold')
    ax.axhline(-np.log10(3), color=COLORS['fold3'], lw=1.0, linestyle=':')

    ax.set_xlabel(f'Predicted log₁₀({param_name})')
    ax.set_ylabel('Residual (Observed − Predicted)')
    ax.set_title(f'{model_name} — {PARAM_LABELS[param_name]} Residuals')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def model_comparison(
    results: Dict[str, Dict[str, Dict]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing GMFE, R², and within-2fold% across models and parameters.

    Args:
        results: nested dict — results[model_name][param_name] = metrics_dict
                 e.g. results['RF']['CL'] = {'gmfe': 1.4, 'r2': 0.75, ...}
    """
    models = list(results.keys())
    params = ['CL', 'Vd']
    metrics_to_plot = [
        ('gmfe',         'GMFE',              1.5,  'lower'),
        ('r2',           'R²',                0.7,  'higher'),
        ('within_2fold', 'Within 2-fold (%)', 70.0, 'higher'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    x = np.arange(len(params))
    width = 0.25

    for ax, (metric_key, metric_label, benchmark, direction) in zip(axes, metrics_to_plot):
        for i, model in enumerate(models):
            vals = [results[model][p][metric_key] for p in params]
            bars = ax.bar(x + i * width, vals, width * 0.9,
                          label=model, color=COLORS.get(model, '#999999'),
                          alpha=0.85, zorder=3)
            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * ax.get_ylim()[1],
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # Benchmark line
        ax.axhline(benchmark, color='red', lw=1.2, linestyle='--',
                   label=f'Benchmark ({benchmark})', zorder=4)

        ax.set_xticks(x + width)
        ax.set_xticklabels(params)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.3, len(params) - 0.1)
        ax.yaxis.grid(True, alpha=0.4, zorder=0)

    fig.suptitle('Model Comparison — Test Set Performance', fontsize=13, y=1.02)
    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHAP Summary
# ══════════════════════════════════════════════════════════════════════════════

def shap_summary(
    shap_values:    np.ndarray,
    feature_names:  List[str],
    param_name:     str,
    model_name:     str,
    top_n:          int = 20,
    output_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N features by mean |SHAP| value.

    Args:
        shap_values:   (n_compounds, n_features) SHAP value matrix
        feature_names: list of feature names
        param_name:    'CL' or 'Vd'
        model_name:    'RF', 'XGB', or 'GNN'
        top_n:         number of top features to show
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    top_vals = mean_abs[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 1.5))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_vals[::-1], color=COLORS.get(model_name, '#0072B2'), alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value| (log₁₀ units)')
    ax.set_title(f'{model_name} — Top {top_n} Features for {PARAM_LABELS[param_name]}')
    ax.xaxis.grid(True, alpha=0.4)
    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Conformal Coverage Plot
# ══════════════════════════════════════════════════════════════════════════════

def conformal_coverage(
    y_true_log:  np.ndarray,
    y_pred_log:  np.ndarray,
    param_name:  str,
    model_name:  str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Empirical coverage vs nominal coverage across alpha levels.
    The conformal guarantee means the curve should sit at or above the diagonal.
    """
    alphas   = np.linspace(0.01, 0.50, 50)
    residuals = np.abs(y_true_log - y_pred_log)
    n = len(residuals)

    empirical = []
    for alpha in alphas:
        level    = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        quantile = np.quantile(residuals, level)
        covered  = np.mean(residuals <= quantile)
        empirical.append(covered)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(1 - alphas, empirical, color=COLORS.get(model_name, '#0072B2'),
            lw=2, label='Empirical coverage')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Ideal (diagonal)')
    ax.fill_between(1 - alphas, empirical, 1 - alphas,
                    where=np.array(empirical) >= 1 - alphas,
                    alpha=0.15, color=COLORS.get(model_name, '#0072B2'),
                    label='Coverage surplus')
    ax.set_xlabel('Nominal Coverage (1 − α)')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'{model_name} — Conformal Coverage ({PARAM_LABELS[param_name]})')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prediction Interval Width Distribution
# ══════════════════════════════════════════════════════════════════════════════

def interval_width_dist(
    y_pred_log:   np.ndarray,
    quantile:     float,
    param_name:   str,
    model_name:   str,
    output_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of 95% PI widths on the original scale.
    Split conformal gives constant-width intervals on log scale,
    but variable width on original scale (wider for larger predicted values).
    """
    lower = 10 ** (y_pred_log - quantile)
    upper = 10 ** (y_pred_log + quantile)
    widths = upper - lower
    units  = PARAM_UNITS[param_name]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(widths, bins=40, color=COLORS.get(model_name, '#0072B2'),
            alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(np.median(widths), color='red', lw=1.5, linestyle='--',
               label=f'Median: {np.median(widths):.2f} {units}')
    ax.set_xlabel(f'95% PI Width [{units}]')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_name} — PI Width Distribution ({PARAM_LABELS[param_name]})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if output_path:
        _save(fig, Path(output_path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. All Manuscript Figures
# ══════════════════════════════════════════════════════════════════════════════

def all_manuscript_figs(
    predictions:  Dict[str, Dict[str, np.ndarray]],
    y_true:       Dict[str, np.ndarray],
    metrics:      Dict[str, Dict[str, dict]],
    output_dir:   str = 'figures/',
    conformal:    Optional[Dict[str, Dict]] = None,
    shap_data:    Optional[Dict[str, Dict]] = None,
):
    """
    Generate all manuscript figures in one call.

    Args:
        predictions:  predictions[model][param] = y_pred_log array
        y_true:       y_true[param] = y_true_log array
        metrics:      metrics[model][param] = metrics dict
        output_dir:   directory to save all figures
        conformal:    optional — conformal[model][param] = {'lower': ..., 'upper': ..., 'quantile': ...}
        shap_data:    optional — shap_data[model][param] = {'values': ..., 'feature_names': ...}

    Example:
        all_manuscript_figs(
            predictions = {'RF': {'CL': rf_cl_pred, 'Vd': rf_vd_pred}, ...},
            y_true      = {'CL': y_te_cl, 'Vd': y_te_vd},
            metrics     = {'RF': {'CL': rf_cl_metrics, ...}, ...},
            output_dir  = 'figures/',
        )
    """
    out = Path(output_dir)
    models = list(predictions.keys())
    params = ['CL', 'Vd']

    print(f"Generating manuscript figures → {out}/")

    for model in models:
        for param in params:
            y_t = y_true[param]
            y_p = predictions[model][param]
            m   = metrics[model][param]
            tag = f'{model}_{param}'

            # Obs vs pred
            ci_lo = ci_hi = None
            if conformal and model in conformal and param in conformal[model]:
                ci_lo = conformal[model][param].get('lower')
                ci_hi = conformal[model][param].get('upper')

            obs_vs_pred(y_t, y_p, param, model, m,
                        output_path=str(out / f'obs_vs_pred_{tag}'),
                        ci_lower_log=ci_lo, ci_upper_log=ci_hi)
            print(f"  ✓ obs_vs_pred_{tag}")

            # Residuals
            residual_plot(y_t, y_p, param, model,
                          output_path=str(out / f'residuals_{tag}'))
            print(f"  ✓ residuals_{tag}")

            # Conformal coverage
            conformal_coverage(y_t, y_p, param, model,
                               output_path=str(out / f'coverage_{tag}'))
            print(f"  ✓ coverage_{tag}")

            # PI width distribution
            if conformal and model in conformal and param in conformal[model]:
                q = conformal[model][param].get('quantile')
                if q is not None:
                    interval_width_dist(y_p, q, param, model,
                                        output_path=str(out / f'pi_width_{tag}'))
                    print(f"  ✓ pi_width_{tag}")

            # SHAP
            if shap_data and model in shap_data and param in shap_data[model]:
                sd = shap_data[model][param]
                shap_summary(sd['values'], sd['feature_names'], param, model,
                             output_path=str(out / f'shap_{tag}'))
                print(f"  ✓ shap_{tag}")

    # Model comparison (all models together)
    model_comparison(metrics, output_path=str(out / 'model_comparison'))
    print(f"  ✓ model_comparison")
    print(f"\nDone — {len(list(out.glob('*.pdf')))} PDF figures saved to {out}/")
