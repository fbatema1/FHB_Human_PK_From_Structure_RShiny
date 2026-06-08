"""
rdkit_features.py
=================
Computes RDKit 2D descriptors and Morgan fingerprints from SMILES.

Outputs:
  - 200 RDKit 2D descriptors (numeric, physicochemical)
  - 2048-bit Morgan fingerprints (radius=2, ECFP4-equivalent)

Pre-filtering (applied to training set only, then applied to test):
  - Remove near-zero variance descriptors (variance < 1e-4 after scaling)
  - Remove highly correlated pairs (Pearson r > 0.95, keep first)

Usage:
    from features.rdkit_features import RDKitFeaturizer
    feat = RDKitFeaturizer()
    feat.fit(train_smiles)           # learns variance/correlation filters
    X_train = feat.transform(train_smiles)
    X_test  = feat.transform(test_smiles)
    feat.save(path)                  # saves filter state for reproducibility
    feat2 = RDKitFeaturizer.load(path)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── Descriptor list ───────────────────────────────────────────────────────────
# Full RDKit descriptor set (~210 descriptors)
_ALL_DESC_NAMES = [name for name, _ in Descriptors.descList]
_CALCULATOR     = MoleculeDescriptors.MolecularDescriptorCalculator(_ALL_DESC_NAMES)

# Morgan fingerprint settings
FP_RADIUS    = 2       # ECFP4 equivalent
FP_NBITS     = 2048
FP_USE_CHIRALITY = True

# Pre-filtering thresholds
VAR_THRESHOLD  = 1e-4   # near-zero variance cutoff (post min-max scale)
CORR_THRESHOLD = 0.95   # Pearson correlation cutoff


def smiles_to_mol(smi: str) -> Optional[object]:
    """Parse SMILES to RDKit mol, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol
    except:
        return None


def compute_descriptors(mol) -> np.ndarray:
    """Compute all RDKit 2D descriptors for a molecule."""
    try:
        vals = _CALCULATOR.CalcDescriptors(mol)
        return np.array(vals, dtype=np.float32)
    except:
        return np.full(len(_ALL_DESC_NAMES), np.nan, dtype=np.float32)


def compute_morgan_fp(mol) -> np.ndarray:
    """Compute Morgan fingerprint as a bit vector."""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_NBITS, useChirality=FP_USE_CHIRALITY
        )
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(FP_NBITS, dtype=np.float32)


class RDKitFeaturizer:
    """
    Featurizer that computes RDKit descriptors + Morgan fingerprints,
    with variance and correlation filtering fitted on the training set.
    """

    def __init__(self):
        self.desc_names         = _ALL_DESC_NAMES.copy()
        self.kept_desc_indices  = None   # set after fit()
        self.kept_desc_names    = None
        self.corr_drop_indices  = None
        self.is_fitted          = False

    # ── Core computation ──────────────────────────────────────────────────────
    def _featurize_one(self, smi: str):
        """Returns (desc_array, fp_array, is_valid)."""
        mol = smiles_to_mol(smi)
        if mol is None:
            return (
                np.full(len(_ALL_DESC_NAMES), np.nan, dtype=np.float32),
                np.zeros(FP_NBITS, dtype=np.float32),
                False
            )
        return compute_descriptors(mol), compute_morgan_fp(mol), True

    def _compute_raw_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """Compute raw descriptor matrix (n_compounds x n_descriptors)."""
        rows = []
        for smi in smiles_list:
            desc, _, _ = self._featurize_one(smi)
            rows.append(desc)
        X = np.vstack(rows)
        # Replace inf and very large values with nan
        X = np.where(np.isfinite(X), X, np.nan)
        return X

    def _compute_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """Compute Morgan fingerprint matrix (n_compounds x FP_NBITS)."""
        rows = []
        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            fp  = compute_morgan_fp(mol) if mol else np.zeros(FP_NBITS, dtype=np.float32)
            rows.append(fp)
        return np.vstack(rows)

    # ── Fit (training set only) ───────────────────────────────────────────────
    def fit(self, smiles_list: List[str], verbose: bool = True) -> 'RDKitFeaturizer':
        """
        Fit variance and correlation filters on the training set.
        Must be called before transform().
        """
        if verbose:
            print(f"[RDKitFeaturizer] Computing descriptors for {len(smiles_list)} compounds...")

        X_raw = self._compute_raw_descriptors(smiles_list)

        # ── Step 1: Remove columns with >20% missing values ───────────────────
        missing_frac = np.mean(np.isnan(X_raw), axis=0)
        keep_nonmissing = missing_frac <= 0.20
        if verbose:
            print(f"  Descriptors with >20% missing: {(~keep_nonmissing).sum()} removed")

        # Impute remaining NaN with column median for filtering purposes
        X_imputed = X_raw[:, keep_nonmissing].copy()
        col_medians = np.nanmedian(X_imputed, axis=0)
        for j in range(X_imputed.shape[1]):
            mask = np.isnan(X_imputed[:, j])
            X_imputed[mask, j] = col_medians[j]

        # ── Step 2: Near-zero variance filter ─────────────────────────────────
        # Scale each column to [0,1] before variance check
        col_min = X_imputed.min(axis=0)
        col_max = X_imputed.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1  # avoid divide by zero
        X_scaled = (X_imputed - col_min) / col_range
        variances = X_scaled.var(axis=0)
        keep_var = variances >= VAR_THRESHOLD
        if verbose:
            print(f"  Near-zero variance removed:    {(~keep_var).sum()}")

        X_filtered = X_imputed[:, keep_var]

        # ── Step 3: Correlation filter ─────────────────────────────────────────
        df_temp  = pd.DataFrame(X_filtered)
        corr_mat = df_temp.corr().abs()
        upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
        keep_corr = np.ones(X_filtered.shape[1], dtype=bool)
        keep_corr[drop_cols] = False
        if verbose:
            print(f"  High correlation (r>{CORR_THRESHOLD}) removed: {(~keep_corr).sum()}")

        # ── Build final index mapping back to original descriptor list ─────────
        original_indices = np.where(keep_nonmissing)[0]
        var_indices      = np.where(keep_var)[0]
        corr_indices     = np.where(keep_corr)[0]

        # Chain the index maps
        after_missing = original_indices
        after_var     = after_missing[var_indices]
        after_corr    = after_var[corr_indices]

        self.kept_desc_indices = after_corr
        self.kept_desc_names   = [_ALL_DESC_NAMES[i] for i in after_corr]
        self.col_medians_raw   = np.nanmedian(X_raw[:, after_corr], axis=0)
        self.is_fitted         = True

        if verbose:
            print(f"  Final descriptors kept:        {len(self.kept_desc_indices)}/{len(_ALL_DESC_NAMES)}")
            print(f"  Morgan FP bits:                {FP_NBITS}")
            print(f"  Total features (desc + FP):    {len(self.kept_desc_indices) + FP_NBITS}")

        return self

    # ── Transform ─────────────────────────────────────────────────────────────
    def transform(
        self,
        smiles_list: List[str],
        include_fp: bool = True,
        top_n_desc: Optional[int] = None
    ) -> np.ndarray:
        """
        Transform SMILES to feature matrix.

        Args:
            smiles_list:  list of SMILES strings
            include_fp:   whether to concatenate Morgan fingerprints
            top_n_desc:   if set, use only the top-N descriptors by SHAP rank
                          (requires feat.shap_ranking to be set)

        Returns:
            X: np.ndarray of shape (n_compounds, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")

        X_raw = self._compute_raw_descriptors(smiles_list)
        X_desc = X_raw[:, self.kept_desc_indices]

        # Impute NaN with training medians
        for j in range(X_desc.shape[1]):
            mask = np.isnan(X_desc[:, j])
            X_desc[mask, j] = self.col_medians_raw[j]

        # Optionally select top-N by SHAP rank
        if top_n_desc is not None and hasattr(self, 'shap_ranking'):
            top_idx = self.shap_ranking[:top_n_desc]
            X_desc  = X_desc[:, top_idx]

        if not include_fp:
            return X_desc.astype(np.float32)

        X_fp = self._compute_fingerprints(smiles_list)
        return np.hstack([X_desc, X_fp]).astype(np.float32)

    def get_feature_names(self, top_n_desc: Optional[int] = None) -> List[str]:
        """Return feature names for the descriptor + FP columns."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        desc_names = self.kept_desc_names
        if top_n_desc is not None and hasattr(self, 'shap_ranking'):
            desc_names = [self.kept_desc_names[i] for i in self.shap_ranking[:top_n_desc]]
        fp_names = [f'Morgan_bit_{i}' for i in range(FP_NBITS)]
        return desc_names + fp_names

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save featurizer state to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[RDKitFeaturizer] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'RDKitFeaturizer':
        """Load featurizer state from disk."""
        feat = cls()
        with open(path, 'rb') as f:
            feat.__dict__.update(pickle.load(f))
        print(f"[RDKitFeaturizer] Loaded ← {path}")
        return feat
