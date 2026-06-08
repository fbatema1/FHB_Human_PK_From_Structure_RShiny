"""
evaluation/conformal.py
=======================
Split conformal prediction for PK parameter uncertainty quantification.

Method: Split (inductive) conformal prediction
  - Fits on a held-out calibration set (never seen during model training)
  - Distribution-free, model-agnostic, finite-sample coverage guarantee
  - Target coverage: 95% (configurable)

Theory:
  Given calibration residuals r_i = |y_i - ŷ_i| (on log10 scale),
  the conformal quantile q̂ = ceil((n+1)(1-α)) / n quantile of {r_i}.
  For a new compound: prediction interval = [ŷ - q̂, ŷ + q̂]
  Coverage guarantee: P(y_new in PI) >= 1 - α

  On original scale (back-transformed):
    PI_original = [10^(ŷ - q̂), 10^(ŷ + q̂)]

Usage:
    from evaluation.conformal import PKConformalPredictor

    # Calibrate (on a held-out calibration set)
    cp = PKConformalPredictor(coverage=0.95)
    cp.calibrate(y_cal_log, y_pred_cal_log)

    # Predict with intervals
    lower, upper = cp.predict_interval(y_pred_log)          # log10 scale
    lower_orig, upper_orig = cp.predict_interval_original(y_pred_log)  # original scale

    # Save / load
    cp.save('conformal_CL.pkl')
    cp2 = PKConformalPredictor.load('conformal_CL.pkl')
"""

import pickle
import numpy as np
from typing import Tuple


class PKConformalPredictor:
    """
    Split conformal predictor for a single PK parameter.

    Operates on log10-transformed predictions throughout.
    Prediction intervals can be returned in log10 or original scale.
    """

    def __init__(self, coverage: float = 0.95):
        """
        Args:
            coverage: target marginal coverage (default 0.95 = 95% PI)
        """
        assert 0 < coverage < 1, "coverage must be in (0, 1)"
        self.coverage     = coverage
        self.quantile_    = None   # set after calibrate()
        self.n_cal_       = None
        self.is_fitted    = False

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> 'PKConformalPredictor':
        """
        Compute conformal quantile from calibration set residuals.

        Args:
            y_true: true log10-transformed PK values (n_cal,)
            y_pred: predicted log10-transformed PK values (n_cal,)
        Returns:
            self (for chaining)
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"

        n = len(y_true)
        residuals = np.abs(y_true - y_pred)   # conformity scores

        # Conformal quantile: (ceil((n+1)(1-alpha)) / n)-th quantile
        # Equivalent to np.quantile with interpolation for finite-sample guarantee
        alpha = 1.0 - self.coverage
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.quantile_ = float(np.quantile(residuals, level))
        self.n_cal_    = n
        self.is_fitted = True

        print(f"  [Conformal] n_cal={n}  coverage={self.coverage:.0%}  "
              f"quantile={self.quantile_:.4f} log10 units  "
              f"(±{10**self.quantile_:.2f}× on original scale)")
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return prediction intervals on log10 scale.

        Args:
            y_pred: predicted log10 values (n,)
        Returns:
            lower, upper: log10-scale interval bounds, each shape (n,)
        """
        self._check_fitted()
        y_pred = np.asarray(y_pred).ravel()
        return y_pred - self.quantile_, y_pred + self.quantile_

    def predict_interval_original(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return prediction intervals on original scale (back-transformed).

        Args:
            y_pred: predicted log10 values (n,)
        Returns:
            lower, upper: original-scale interval bounds, each shape (n,)
        """
        lower_log, upper_log = self.predict_interval(y_pred)
        return 10 ** lower_log, 10 ** upper_log

    def empirical_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute empirical coverage on a test set.
        Should be >= self.coverage by the conformal guarantee.

        Args:
            y_true: true log10 values
            y_pred: predicted log10 values
        Returns:
            fraction of test compounds whose true value falls in the PI
        """
        self._check_fitted()
        lower, upper = self.predict_interval(y_pred)
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        return float(covered)

    def interval_width_original(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Interval width on original scale: upper - lower.
        Useful for summarising uncertainty across the dataset.
        """
        lower, upper = self.predict_interval_original(y_pred)
        return upper - lower

    def summary(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print a summary of conformal predictor performance."""
        self._check_fitted()
        cov  = self.empirical_coverage(y_true, y_pred)
        lower_o, upper_o = self.predict_interval_original(y_pred)
        med_width = float(np.median(upper_o - lower_o))
        print(f"  Conformal summary:")
        print(f"    Target coverage  : {self.coverage:.0%}")
        print(f"    Empirical coverage: {cov:.1%}")
        print(f"    Quantile (log10) : ±{self.quantile_:.4f}")
        print(f"    Fold multiplier  : ×{10**self.quantile_:.2f}")
        print(f"    Median PI width  : {med_width:.3f} (original scale)")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'PKConformalPredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call calibrate() before predicting.")
