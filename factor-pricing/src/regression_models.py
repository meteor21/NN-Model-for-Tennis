"""
Forecasting regression models.

Provides a unified interface for OLS, Ridge, Lasso, and Elastic Net
forecasters, along with feature-matrix construction and a comparative
forecasting experiment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ForecastingModel:
    """
    Unified wrapper around OLS / Ridge / Lasso / Elastic Net forecasters.

    All penalised models perform time-series cross-validated alpha selection.

    Parameters
    ----------
    model_type : str
        One of ``'ols'``, ``'ridge'``, ``'lasso'``, ``'elastic_net'``.
    """

    _VALID_TYPES = {"ols", "ridge", "lasso", "elastic_net"}

    def __init__(self, model_type: str = "ridge"):
        if model_type not in self._VALID_TYPES:
            raise ValueError(f"model_type must be one of {self._VALID_TYPES}, got '{model_type}'.")
        self.model_type = model_type
        self._model = None
        self._scaler = StandardScaler()
        self._best_alpha: Optional[float] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
    ) -> "ForecastingModel":
        """
        Fit the selected model type.

        Penalised models choose the regularisation strength via
        *TimeSeriesSplit* cross-validation.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)
        cv_folds : int
            Number of folds for time-series CV (penalised models only).

        Returns
        -------
        self
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)

        # Remove rows with NaN
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid], y[valid]

        X_scaled = self._scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        if self.model_type == "ols":
            self._model = LinearRegression()
            self._model.fit(X_scaled, y)

        elif self.model_type == "ridge":
            alphas = np.logspace(-4, 4, 60)
            self._model = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_mean_squared_error")
            self._model.fit(X_scaled, y)
            self._best_alpha = float(self._model.alpha_)

        elif self.model_type == "lasso":
            self._model = LassoCV(
                cv=tscv, max_iter=5000, tol=1e-4, n_alphas=60, random_state=42
            )
            self._model.fit(X_scaled, y)
            self._best_alpha = float(self._model.alpha_)

        elif self.model_type == "elastic_net":
            self._model = ElasticNetCV(
                cv=tscv, max_iter=5000, tol=1e-4, n_alphas=60,
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                random_state=42,
            )
            self._model.fit(X_scaled, y)
            self._best_alpha = float(self._model.alpha_)

        self._is_fitted = True
        alpha_str = f", α={self._best_alpha:.4g}" if self._best_alpha else ""
        logger.info("ForecastingModel (%s%s) fitted on %d samples.", self.model_type, alpha_str, len(y))
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for *X_test*.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X_test, dtype=float)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    # ------------------------------------------------------------------
    def get_best_alpha(self) -> Optional[float]:
        """
        Return the cross-validated regularisation strength.

        Returns
        -------
        float or None
            ``None`` for OLS (no regularisation).
        """
        return self._best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    factor_returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    lags: List[int] = None,
) -> pd.DataFrame:
    """
    Construct a lagged feature matrix from factor returns and macro variables.

    For each variable, creates lagged versions at each specified horizon.
    Rows with NaN (from lagging) are dropped.

    Parameters
    ----------
    factor_returns_df : pd.DataFrame
        Factor return time series (T × K).
    macro_df : pd.DataFrame
        Macro variable levels / changes (T × M).
    lags : list of int
        Lag horizons in trading days. Defaults to [1, 5, 21].

    Returns
    -------
    pd.DataFrame
        Feature matrix with DatetimeIndex.
    """
    if lags is None:
        lags = [1, 5, 21]

    frames = []

    for lag in lags:
        f_lag = factor_returns_df.shift(lag).add_suffix(f"_lag{lag}")
        m_lag = macro_df.shift(lag).add_suffix(f"_lag{lag}")
        frames.extend([f_lag, m_lag])

    feature_df = pd.concat(frames, axis=1)
    n_before = len(feature_df)
    feature_df = feature_df.dropna()
    logger.info(
        "Feature matrix: %d rows → %d after removing NaN (from lagging). %d features.",
        n_before, len(feature_df), feature_df.shape[1],
    )
    return feature_df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, R², and Information Coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid  = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2   = float(r2_score(yt, yp))
    ic   = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else np.nan
    return {"rmse": rmse, "r2": r2, "ic": ic}


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_forecasting_experiment(
    panel_df: pd.DataFrame,
    latent_factors: pd.DataFrame,
    observed_factors: pd.DataFrame,
    macro_df: pd.DataFrame,
    target: str = "return",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    lags: List[int] = None,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare OLS, Ridge, and Lasso using latent vs observed factor features.

    Constructs lagged feature matrices, splits chronologically into
    train/val/test, and evaluates each model on all three splits.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Full panel with a column named ``<something>_ret`` for the aggregate
        target, or an equal-weighted average is used as the target.
    latent_factors : pd.DataFrame
        PCA-extracted latent factor returns (T × k).
    observed_factors : pd.DataFrame
        Observed FF5 / macro factors (T × K).
    macro_df : pd.DataFrame
        Macro variable levels (T × M).
    target : str
        Either ``'return'`` (equal-weighted portfolio return) or a specific
        column name in *panel_df*.
    train_ratio : float
        Fraction of data used for training.
    val_ratio : float
        Fraction used for validation.
    lags : list of int
        Feature lag horizons.
    cv_folds : int
        CV folds for penalised model alpha selection.

    Returns
    -------
    dict
        Nested dict: ``results[model_name][split] = {rmse, r2, ic}``
        where *model_name* is e.g. ``'ridge_latent'`` and *split* is one of
        ``'train'``, ``'val'``, ``'test'``.
    """
    if lags is None:
        lags = [1, 5, 21]

    # Build target series: equal-weighted portfolio return
    ret_cols = [c for c in panel_df.columns if c.endswith("_ret")]
    if target == "return" or target not in panel_df.columns:
        y_series = panel_df[ret_cols].mean(axis=1)
        y_series.name = "ew_portfolio_return"
    else:
        y_series = panel_df[target]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for feature_label, factor_df in [("latent", latent_factors), ("observed", observed_factors)]:
        feature_matrix = build_feature_matrix(factor_df, macro_df, lags=lags)

        # Align features and target on common index
        common = feature_matrix.index.intersection(y_series.index)
        X_all = feature_matrix.loc[common].values
        y_all = y_series.loc[common].values

        n = len(y_all)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        X_train, y_train = X_all[:n_train],             y_all[:n_train]
        X_val,   y_val   = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
        X_test,  y_test  = X_all[n_train+n_val:],       y_all[n_train+n_val:]

        for model_type in ("ols", "ridge", "lasso"):
            model_name = f"{model_type}_{feature_label}"
            try:
                fm = ForecastingModel(model_type=model_type)
                fm.fit(X_train, y_train, cv_folds=cv_folds)

                results[model_name] = {
                    "train": _metrics(y_train, fm.predict(X_train)),
                    "val":   _metrics(y_val,   fm.predict(X_val)),
                    "test":  _metrics(y_test,  fm.predict(X_test)),
                }
                if fm.get_best_alpha() is not None:
                    logger.info("  %s best alpha = %.4g", model_name, fm.get_best_alpha())
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)
                results[model_name] = {
                    split: {"rmse": np.nan, "r2": np.nan, "ic": np.nan}
                    for split in ("train", "val", "test")
                }

    return results
