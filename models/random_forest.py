"""
models/random_forest.py
=======================
Random Forest model wrapper for PK parameter prediction.

Trains separate RF models for CL and Vd on log10-transformed targets.
Designed to work with RDKitFeaturizer's top_n_desc parameter for
SHAP-guided feature selection during Optuna tuning.

Usage:
    from models.random_forest import PKRandomForest
    model = PKRandomForest(n_estimators=500, max_features='sqrt', ...)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


class PKRandomForest:
    """
    Random Forest wrapper for a single PK parameter.
    Operates on log10-transformed targets throughout.
    """

    def __init__(
        self,
        n_estimators:    int   = 500,
        max_depth:       Optional[int] = None,
        min_samples_leaf: int  = 1,
        min_samples_split: int = 2,
        max_features:    str   = 'sqrt',
        n_jobs:          int   = -1,
        random_state:    int   = 42,
    ):
        self.params = dict(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            min_samples_leaf = min_samples_leaf,
            min_samples_split= min_samples_split,
            max_features     = max_features,
            n_jobs           = n_jobs,
            random_state     = random_state,
        )
        self.model      = None
        self.is_fitted  = False
        self.param_name = None

    def fit(self, X: np.ndarray, y: np.ndarray, param_name: str = '') -> 'PKRandomForest':
        """
        Fit the Random Forest on log10-transformed targets.

        Args:
            X:          feature matrix (n_samples x n_features)
            y:          log10-transformed target values (n_samples,)
            param_name: 'CL' or 'Vd' — for logging only
        """
        self.param_name = param_name
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y.ravel())
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
    def load(cls, path: str) -> 'PKRandomForest':
        with open(path, 'rb') as f:
            return pickle.load(f)
