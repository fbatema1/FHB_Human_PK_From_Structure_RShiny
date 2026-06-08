"""
split_dataset.py
================
Splits the cleaned dataset into train (80%) and test (20%) sets.

Stratification:
  - CL and Vd are log-transformed and binned into 5 quantiles
  - Stratification is performed on combined CL+Vd bins to ensure
    the test set covers the full PK distribution

Outputs (all in data/processed/):
  - train.xlsx
  - test.xlsx
  - split_summary.txt  — statistics confirming distribution match

Run:
    conda activate pkip-env
    python scripts/data_curation/split_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT     = Path(__file__).resolve().parents[2]
DATA_IN  = ROOT / "data/processed/master_dataset_cleaned.xlsx"
TRAIN_OUT = ROOT / "data/processed/train.xlsx"
TEST_OUT  = ROOT / "data/processed/test.xlsx"
LOG_OUT   = ROOT / "data/processed/split_summary.txt"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

def run():
    print(f"Loading: {DATA_IN.name}")
    df = pd.read_excel(DATA_IN, sheet_name='Cleaned_Data', engine='openpyxl')
    print(f"  Total compounds: {len(df)}")

    # Log-transform CL and Vd for stratification binning
    df['log_CL'] = np.log10(df['human_CL_mL_min_kg'].astype(float))
    df['log_Vd'] = np.log10(df['human_VDss_L_kg'].astype(float))

    # Bin into 5 quantiles each, combine into a stratification label
    df['cl_bin'] = pd.qcut(df['log_CL'], q=5, labels=False, duplicates='drop')
    df['vd_bin'] = pd.qcut(df['log_Vd'], q=5, labels=False, duplicates='drop')
    df['strat_label'] = df['cl_bin'].astype(str) + '_' + df['vd_bin'].astype(str)

    # Stratified split
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['strat_label']
    )

    # Drop internal columns before saving
    drop_cols = ['log_CL', 'log_Vd', 'cl_bin', 'vd_bin', 'strat_label']
    train = train.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)
    test  = test.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)

    # Summary stats
    def pk_stats(d, label):
        lines = [f"\n{label} (n={len(d)})"]
        for col, name in [('human_CL_mL_min_kg','CL'), ('human_VDss_L_kg','Vd')]:
            vals = d[col].dropna()
            lines.append(
                f"  {name}: median={vals.median():.2f}  "
                f"mean={vals.mean():.2f}  "
                f"min={vals.min():.3f}  "
                f"max={vals.max():.1f}  "
                f"log10_std={np.log10(vals).std():.3f}"
            )
        return '\n'.join(lines)

    summary = [
        "TRAIN/TEST SPLIT SUMMARY",
        "=" * 50,
        f"Random seed:  {RANDOM_STATE}",
        f"Test size:    {TEST_SIZE:.0%}",
        f"Total:        {len(df)}",
        f"Train:        {len(train)}",
        f"Test:         {len(test)}",
        pk_stats(train, "TRAIN"),
        pk_stats(test,  "TEST"),
        "\nData source breakdown (train):",
    ]

    if 'data_source' in train.columns:
        for src, cnt in train['data_source'].value_counts().items():
            summary.append(f"  {src}: {cnt}")
        summary.append("\nData source breakdown (test):")
        for src, cnt in test['data_source'].value_counts().items():
            summary.append(f"  {src}: {cnt}")

    summary_text = '\n'.join(summary)
    print(summary_text)

    # Save
    train.to_excel(TRAIN_OUT, index=False)
    test.to_excel(TEST_OUT, index=False)
    with open(LOG_OUT, 'w') as f:
        f.write(summary_text)

    print(f"\nTrain → {TRAIN_OUT}")
    print(f"Test  → {TEST_OUT}")
    print(f"Log   → {LOG_OUT}")

if __name__ == '__main__':
    run()
