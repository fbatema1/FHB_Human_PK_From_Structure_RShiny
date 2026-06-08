"""
models/xgboost_model.py
=======================
XGBoost model wrapper for PK parameter prediction.

Trains separate XGBoost models for CL and Vd on log10-transformed targets.
Supports early stopping during Optuna trials via eval set.

Usage:
    from models.xgboost_model import PKXGBoost
    model = PKXGBoost(n_estimators=500, learning_rate=0.05, ...)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
"""

import numpy as np
import pickle
from typing import Optional
import xgboost as xgb


class PKXGBoost:
    """
    XGBoost wrapper for a single PK parameter.
    Operates on log10-transformed targets throughout.
    """

    def __init__(
        self,
        n_estimators:      int   = 500,
        learning_rate:     float = 0.05,
        max_depth:         int   = 6,
        min_child_weight:  int   = 1,
        subsample:         float = 0.8,
        colsample_bytree:  float = 0.8,
        gamma:             float = 0.0,
        reg_alpha:         float = 0.0,
        reg_lambda:        float = 1.0,
        early_stopping_rounds: Optional[int] = 50,
        random_state:      int   = 42,
        n_jobs:            int   = -1,
    ):
        self.params = dict(
            n_estimators         = n_estimators,
            learning_rate        = learning_rate,
            max_depth            = max_depth,
            min_child_weight     = min_child_weight,
            subsample            = subsample,
            colsample_bytree     = colsample_bytree,
            gamma                = gamma,
            reg_alpha            = reg_alpha,
            reg_lambda           = reg_lambda,
            random_state         = random_state,
            n_jobs               = n_jobs,
            tree_method          = 'hist',   # fast histogram method
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model      = None
        self.is_fitted  = False
        self.param_name = None

    def fit(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        X_val:      Optional[np.ndarray] = None,
        y_val:      Optional[np.ndarray] = None,
        param_name: str = '',
    ) -> 'PKXGBoost':
        """
        Fit XGBoost on log10-transformed targets.

        Args:
            X:          training feature matrix
            y:          log10-transformed targets
            X_val:      optional validation set for early stopping
            y_val:      optional validation targets
            param_name: 'CL' or 'Vd' — for logging only
        """
        self.param_name = param_name
        self.model = xgb.XGBRegressor(**self.params)

        # XGBoost 3.x: early_stopping_rounds passed at construction, eval_set at fit
        fit_kwargs = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds:
            fit_kwargs['eval_set'] = [(X_val, y_val.ravel())]
            fit_kwargs['verbose']  = False

        self.model.fit(X, y.ravel(), **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log10-transformed values."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        return self.model.predict(X)

    def predict_original_scale(self, X: np.ndarray) -> np.ndarray:
        """Predict and back-transform to original scale."""
        return 10 ** self.predict(X)

    @property
    def feature_importances_(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.feature_importances_

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'PKXGBoost':
        with open(path, 'rb') as f:
            return pickle.load(f)
