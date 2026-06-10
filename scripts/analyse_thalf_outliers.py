"""
analyse_thalf_outliers.py
=========================
Quantifies which compounds fall outside the 2-fold t½ agreement window
and characterises their disposition profiles.

Outputs:
  - prints summary stats by disposition class
  - data/processed/thalf_outlier_analysis.csv  — full per-compound table
  - data/processed/thalf_disposition_summary.csv — group-level summary

Run:
    conda activate pkip-env
    python scripts/analyse_thalf_outliers.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
VAL_IN  = ROOT / "data/processed/thalf_validation.csv"
OUT_CSV = ROOT / "data/processed/thalf_outlier_analysis.csv"
SUM_CSV = ROOT / "data/processed/thalf_disposition_summary.csv"

# ── Known multi-compartment / atypical disposition drugs ──────────────────────
# Manually curated based on published PK literature.
# Categories:
#   multi_compartment  — deep tissue distribution, terminal t½ >> 0.693*Vd/CL
#   renal_tubular      — active secretion/reabsorption distorts apparent t½
#   plasma_binding     — very high fup variability
#   prodrug            — observed t½ is metabolite, not parent
#   enterohepatic      — recirculation extends apparent t½

DISPOSITION_FLAGS = {
    # Multi-compartment / deep tissue
    "Diatrizoate":        "multi_compartment",
    "Iohexol":            "multi_compartment",
    "Iodixanol":          "multi_compartment",
    "Iopamidol":          "multi_compartment",
    "Iopromide":          "multi_compartment",
    "Alendronate":        "multi_compartment",  # bisphosphonate — bone binding
    "Risedronate":        "multi_compartment",
    "Zoledronic acid":    "multi_compartment",
    "Pamidronate":        "multi_compartment",
    "Ibandronate":        "multi_compartment",
    "Etidronate":         "multi_compartment",
    "Chloroquine":        "multi_compartment",  # extreme tissue distribution
    "Hydroxychloroquine": "multi_compartment",
    "Amiodarone":         "multi_compartment",  # weeks-long terminal t½
    "Digoxin":            "multi_compartment",
    "Methotrexate":       "multi_compartment",
    "Bleomycin":          "multi_compartment",
    "Vancomycin":         "multi_compartment",
    # Renal tubular secretion / reabsorption
    "Metformin":          "renal_tubular",
    "Ciprofloxacin":      "renal_tubular",
    "Levofloxacin":       "renal_tubular",
    "Cephalexin":         "renal_tubular",
    "Lithium":            "renal_tubular",
    # Enterohepatic recirculation
    "Mycophenolic acid":  "enterohepatic",
    "Estradiol":          "enterohepatic",
    "Morphine":           "enterohepatic",
    "Indomethacin":       "enterohepatic",
    # Prodrug (observed t½ is metabolite)
    "Enalapril":          "prodrug",
    "Codeine":            "prodrug",
    "Oseltamivir":        "prodrug",
    "Clopidogrel":        "prodrug",
    "Valacyclovir":       "prodrug",
}


def run():
    df = pd.read_csv(VAL_IN)
    print(f"Loaded {len(df)} compounds from validation set\n")

    # ── Classify each compound ────────────────────────────────────────────────
    df["disposition"] = df["compound_name"].map(DISPOSITION_FLAGS).fillna("standard")
    df["within_2fold"] = df["fold_error"] >= 0.5   # fold_error = pred/obs, within 2× means 0.5–2.0

    # Re-derive a cleaner fold error column (ratio, always ≥1 direction)
    df["fold_error_ratio"] = df.apply(
        lambda r: max(r["thalf_pred"], r["thalf_h"]) / min(r["thalf_pred"], r["thalf_h"])
        if min(r["thalf_pred"], r["thalf_h"]) > 0 else np.nan, axis=1
    )
    df["within_2fold_clean"] = df["fold_error_ratio"] <= 2.0
    df["outlier"]            = ~df["within_2fold_clean"]

    # ── Overall stats ──────────────────────────────────────────────────────────
    n_total   = len(df)
    n_within2 = df["within_2fold_clean"].sum()
    gmfe      = 10 ** df["abs_log_err"].mean()

    print("=" * 55)
    print("OVERALL")
    print("=" * 55)
    print(f"  N:                {n_total}")
    print(f"  Within 2-fold:    {n_within2} ({100*n_within2/n_total:.1f}%)")
    print(f"  GMFE:             {gmfe:.3f}")
    print(f"  Outliers (>2×):   {n_total - n_within2} ({100*(n_total-n_within2)/n_total:.1f}%)")

    # ── Stats by disposition class ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("BY DISPOSITION CLASS")
    print("=" * 55)

    summary_rows = []
    for grp, gdf in df.groupby("disposition"):
        n       = len(gdf)
        w2      = gdf["within_2fold_clean"].sum()
        g       = 10 ** gdf["abs_log_err"].mean()
        med_obs = gdf["thalf_h"].median()
        med_pred= gdf["thalf_pred"].median()
        summary_rows.append({
            "disposition":    grp,
            "n":              n,
            "within_2fold":   w2,
            "within_2fold_%": round(100 * w2 / n, 1),
            "GMFE":           round(g, 3),
            "median_obs_h":   round(med_obs, 1),
            "median_pred_h":  round(med_pred, 1),
        })
        print(f"\n  {grp.upper()} (n={n})")
        print(f"    Within 2-fold: {w2}/{n} ({100*w2/n:.1f}%)")
        print(f"    GMFE:          {g:.3f}")
        print(f"    Median obs t½: {med_obs:.1f} h")
        print(f"    Median pred t½:{med_pred:.1f} h")

    # ── Worst outliers ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TOP 15 WORST OUTLIERS (highest fold error)")
    print("=" * 55)
    worst = df.nlargest(15, "fold_error_ratio")[
        ["compound_name", "thalf_h", "thalf_pred", "fold_error_ratio", "disposition"]
    ]
    print(worst.to_string(index=False))

    # ── Stats excluding multi-compartment ─────────────────────────────────────
    df_std = df[df["disposition"] == "standard"]
    n_std  = len(df_std)
    w2_std = df_std["within_2fold_clean"].sum()
    g_std  = 10 ** df_std["abs_log_err"].mean()

    print("\n" + "=" * 55)
    print("EXCLUDING KNOWN ATYPICAL DISPOSITION (standard only)")
    print("=" * 55)
    print(f"  N:                {n_std}")
    print(f"  Within 2-fold:    {w2_std} ({100*w2_std/n_std:.1f}%)")
    print(f"  GMFE:             {g_std:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(SUM_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.name}")
    print(f"Saved: {SUM_CSV.name}")


if __name__ == "__main__":
    run()
