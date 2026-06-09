"""
hybrid/predictor.py
===================
Wadhams hybrid PK predictor.

Routing:
  CL  →  Random Forest  (GMFE 2.256, R² 0.433, within-2fold 60.3%)
  Vd  →  XGBoost        (GMFE 1.815, R² 0.694, within-2fold 71.1%)

Each parameter gets its own conformal predictor for 95% PIs.

Usage:
    predictor = HybridPredictor.load()
    results   = predictor.predict(["CC(C)Cc1ccc(cc1)C(C)C(=O)O"], ci=True)
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.rdkit_features  import RDKitFeaturizer
from models.random_forest     import PKRandomForest
from models.xgboost_model     import PKXGBoost
from evaluation.conformal     import PKConformalPredictor


class HybridPredictor:
    """
    Loads RF (CL) + XGB (Vd) models with their conformal calibrators
    and featurizers, then runs end-to-end inference from SMILES.
    """

    def __init__(
        self,
        rf_cl:      PKRandomForest,
        xgb_vd:     PKXGBoost,
        feat_cl:    RDKitFeaturizer,
        feat_vd:    RDKitFeaturizer,
        cp_cl:      Optional[PKConformalPredictor] = None,
        cp_vd:      Optional[PKConformalPredictor] = None,
    ):
        self.rf_cl   = rf_cl
        self.xgb_vd  = xgb_vd
        self.feat_cl = feat_cl
        self.feat_vd = feat_vd
        self.cp_cl   = cp_cl
        self.cp_vd   = cp_vd

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        models_dir:    Optional[Path] = None,
        conformal_dir: Optional[Path] = None,
        data_dir:      Optional[Path] = None,
    ) -> 'HybridPredictor':
        """Load all components from disk."""
        if models_dir is None:
            models_dir = ROOT / "models" / "saved"
        if conformal_dir is None:
            conformal_dir = models_dir / "conformal"
        if data_dir is None:
            data_dir = ROOT / "data" / "processed"

        print("[HybridPredictor] Loading models...")

        # RF for CL
        rf_cl = PKRandomForest.load(str(models_dir / "rf" / "rf_CL_best.pkl"))
        print("  ✓ RF (CL)")

        # XGB for Vd
        xgb_vd = PKXGBoost.load(str(models_dir / "xgb" / "xgb_Vd_best.pkl"))
        print("  ✓ XGB (Vd)")

        # Featurizers
        feat_cl = RDKitFeaturizer.load(str(data_dir / "featurizer_CL.pkl"))
        feat_vd = RDKitFeaturizer.load(str(data_dir / "featurizer_Vd.pkl"))
        print("  ✓ Featurizers (CL + Vd)")

        # Conformal predictors (optional — if calibration has been run)
        cp_cl, cp_vd = None, None
        cp_cl_path = conformal_dir / "conformal_RF_CL.pkl"
        cp_vd_path = conformal_dir / "conformal_XGB_Vd.pkl"

        if cp_cl_path.exists():
            cp_cl = PKConformalPredictor.load(str(cp_cl_path))
            print("  ✓ Conformal (CL)")
        else:
            print(f"  ⚠ Conformal (CL) not found — PIs will be unavailable")

        if cp_vd_path.exists():
            cp_vd = PKConformalPredictor.load(str(cp_vd_path))
            print("  ✓ Conformal (Vd)")
        else:
            print(f"  ⚠ Conformal (Vd) not found — PIs will be unavailable")

        return cls(rf_cl, xgb_vd, feat_cl, feat_vd, cp_cl, cp_vd)

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(
        self,
        smiles_list: List[str],
        ci: bool = True,
    ) -> List[dict]:
        """
        Predict CL and Vd (with optional 95% PIs) for a list of SMILES.

        Args:
            smiles_list: list of SMILES strings
            ci:          if True, include 95% conformal prediction intervals

        Returns:
            list of dicts, one per compound:
            {
              smiles, model_used,
              CL_pred, CL_lower, CL_upper, CL_log_pred,
              Vd_pred, Vd_lower, Vd_upper, Vd_log_pred,
              thalf_pred, lambdaz_pred,
              status, error
            }
        """
        results = []

        for smi in smiles_list:
            smi = smi.strip()
            try:
                result = self._predict_one(smi, ci=ci)
            except Exception as e:
                result = {
                    "smiles":     smi,
                    "model_used": "hybrid",
                    "status":     "error",
                    "error":      str(e),
                }
            results.append(result)

        return results

    def _predict_one(self, smiles: str, ci: bool) -> dict:
        """Predict for a single SMILES string."""

        # ── CL via RF ─────────────────────────────────────────────────────────
        # feat_idx already encodes the SHAP top-N desc + all FP bits —
        # apply it to the FULL feature matrix (no top_n_desc in transform)
        X_cl = self.feat_cl.transform([smiles], include_fp=True)
        X_cl_sel    = X_cl[:, self.rf_cl.feat_idx]
        cl_log_pred = float(self.rf_cl.predict(X_cl_sel)[0])
        cl_pred     = float(10 ** cl_log_pred)

        cl_lower, cl_upper = None, None
        if ci and self.cp_cl is not None:
            cl_lo_log, cl_hi_log = self.cp_cl.predict_interval(
                np.array([cl_log_pred])
            )
            cl_lower = float(10 ** cl_lo_log[0])
            cl_upper = float(10 ** cl_hi_log[0])

        # ── Vd via XGB ────────────────────────────────────────────────────────
        X_vd = self.feat_vd.transform([smiles], include_fp=True)
        X_vd_sel    = X_vd[:, self.xgb_vd.feat_idx]
        vd_log_pred = float(self.xgb_vd.predict(X_vd_sel)[0])
        vd_pred     = float(10 ** vd_log_pred)

        vd_lower, vd_upper = None, None
        if ci and self.cp_vd is not None:
            vd_lo_log, vd_hi_log = self.cp_vd.predict_interval(
                np.array([vd_log_pred])
            )
            vd_lower = float(10 ** vd_lo_log[0])
            vd_upper = float(10 ** vd_hi_log[0])

        # ── Derived parameters ────────────────────────────────────────────────
        # Unit conversion: CL is mL/min/kg → convert to L/h/kg (* 60/1000)
        # before applying t½ = 0.693 × Vd(L/kg) / CL(L/h/kg)
        cl_L_h_kg = cl_pred * 60 / 1000
        thalf     = round(0.693 * vd_pred / cl_L_h_kg, 3) if cl_pred > 0 else None
        lambdaz   = round(cl_L_h_kg / vd_pred, 6)         if vd_pred > 0 else None

        return {
            "smiles":       smiles,
            "model_used":   "hybrid",
            "CL_pred":      round(cl_pred,     4),
            "CL_lower":     round(cl_lower, 4) if cl_lower is not None else None,
            "CL_upper":     round(cl_upper, 4) if cl_upper is not None else None,
            "CL_log_pred":  round(cl_log_pred, 4),
            "Vd_pred":      round(vd_pred,     4),
            "Vd_lower":     round(vd_lower, 4) if vd_lower is not None else None,
            "Vd_upper":     round(vd_upper, 4) if vd_upper is not None else None,
            "Vd_log_pred":  round(vd_log_pred, 4),
            "thalf_pred":   thalf,
            "lambdaz_pred": lambdaz,
            "status":       "ok",
            "error":        None,
        }
