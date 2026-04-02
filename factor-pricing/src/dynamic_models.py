"""
Dynamic factor models.

Implements a rolling-window factor forecaster and an optional
Kalman-filter smoother for latent factor returns.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Factor Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class RollingFactorForecaster:
    """
    Walk-forward forecaster that refits PCA and a regression model at each step.

    At each date *t*:

    1. Fit PCA on returns in the window ``[t - window, t)``.
    2. Construct lagged feature vector from the fitted factor returns.
    3. Fit a forecasting model on in-window data.
    4. Predict the equal-weighted portfolio return at *t*.

    Parameters
    ----------
    window : int
        Look-back window in trading days (default 252).
    n_factors : int
        Number of PCA factors (default 5).
    model_type : str
        Regression model type — ``'ridge'``, ``'lasso'``, or ``'ols'``.
    lags : list of int
        Feature lags to use (default [1, 5]).
    min_obs_frac : float
        Minimum fraction of *window* observations needed to fit.
    """

    def __init__(
        self,
        window: int = 252,
        n_factors: int = 5,
        model_type: str = "ridge",
        lags: Optional[list] = None,
        min_obs_frac: float = 0.8,
        step: int = 1,
    ):
        self.window        = window
        self.n_factors     = n_factors
        self.model_type    = model_type
        self.lags          = lags if lags is not None else [1, 5]
        self.min_obs_frac  = min_obs_frac
        self.step          = max(1, step)  # refit every `step` days

        self._predictions: Optional[pd.Series] = None
        self._actuals:     Optional[pd.Series] = None

    # ------------------------------------------------------------------
    def fit_predict(
        self,
        returns_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Run the full rolling walk-forward evaluation.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset log returns aligned on a DatetimeIndex (T × N).
        macro_df : pd.DataFrame
            Macro variables aligned on the same DatetimeIndex (T × M).

        Returns
        -------
        predictions : pd.Series
            Out-of-sample return predictions.
        actuals : pd.Series
            Corresponding realised equal-weighted portfolio returns.
        """
        # Align returns and macro on common trading dates
        common_idx = returns_df.index.intersection(macro_df.index)
        R = returns_df.loc[common_idx].dropna(axis=1, how="any")
        M = macro_df.loc[common_idx].ffill().fillna(0.0)

        dates    = R.index
        n        = len(dates)
        min_obs  = int(self.window * self.min_obs_frac)
        max_lag  = max(self.lags)

        # Target: equal-weighted portfolio return
        ew_ret = R.mean(axis=1)

        preds_dict: Dict = {}

        logger.info(
            "RollingFactorForecaster: %d dates, window=%d, n_factors=%d, model=%s",
            n, self.window, self.n_factors, self.model_type,
        )

        current_model = None
        current_scaler = None
        current_feat_scaler = None
        current_k = None
        current_pca = None
        current_F_win_df = None
        current_macro_win = None

        for i in range(self.window + max_lag, n):
            win_ret  = R.iloc[i - self.window: i]
            win_ret  = win_ret.dropna()

            if len(win_ret) < min_obs:
                continue

            # Only refit model every `step` days
            refit = (i - self.window - max_lag) % self.step == 0

            if refit:
                # ── 1. Fit PCA on window ──────────────────────────────────
                current_scaler = StandardScaler()
                X_win  = current_scaler.fit_transform(win_ret.values)
                current_k  = min(self.n_factors, X_win.shape[1])
                current_pca = PCA(n_components=current_k)
                F_win  = current_pca.fit_transform(X_win)
                current_F_win_df = pd.DataFrame(
                    F_win, index=win_ret.index,
                    columns=[f"F{j+1}" for j in range(current_k)]
                )
                current_macro_win = M.loc[win_ret.index]

                # ── 2. Build in-window lagged features & target ───────────
                feature_rows = []
                y_rows       = []
                for t_idx in range(max_lag, len(win_ret)):
                    row = []
                    for lag in self.lags:
                        if t_idx - lag >= 0:
                            row.extend(current_F_win_df.iloc[t_idx - lag].values.tolist())
                            row.extend(current_macro_win.iloc[t_idx - lag].values.tolist())
                        else:
                            row.extend([np.nan] * (current_k + M.shape[1]))
                    feature_rows.append(row)
                    y_rows.append(ew_ret.iloc[i - self.window + t_idx])

                Xw = np.array(feature_rows, dtype=float)
                yw = np.array(y_rows, dtype=float)
                valid = np.isfinite(Xw).all(axis=1) & np.isfinite(yw)
                Xw, yw = Xw[valid], yw[valid]

                if len(yw) < 20:
                    current_model = None
                    continue

                # ── 3. Fit regression model ───────────────────────────────
                current_feat_scaler = StandardScaler()
                Xw_s = current_feat_scaler.fit_transform(Xw)

                if self.model_type == "ridge":
                    alphas = np.logspace(-4, 4, 20)
                    tscv   = TimeSeriesSplit(n_splits=min(3, len(yw) // 10 + 1))
                    current_model = RidgeCV(alphas=alphas, cv=tscv)
                else:
                    from sklearn.linear_model import LinearRegression
                    current_model = LinearRegression()

                current_model.fit(Xw_s, yw)

            if current_model is None or current_F_win_df is None:
                continue

            # ── 4. Build prediction feature vector for date i ─────────────
            # Project current observation onto the last-fitted PCA
            x_curr  = current_scaler.transform(R.iloc[[i]].values)
            f_curr  = current_pca.transform(x_curr)[0]
            m_curr  = M.iloc[i].values

            pred_row = []
            for lag in self.lags:
                lag_idx = len(current_F_win_df) - lag
                if lag_idx >= 0:
                    pred_row.extend(current_F_win_df.iloc[lag_idx].values.tolist())
                    pred_row.extend(current_macro_win.iloc[lag_idx].values.tolist())
                else:
                    pred_row.extend([0.0] * (current_k + M.shape[1]))

            pred_arr = np.array(pred_row, dtype=float).reshape(1, -1)
            if not np.isfinite(pred_arr).all():
                continue

            pred_arr_s = current_feat_scaler.transform(pred_arr)
            preds_dict[dates[i]] = float(current_model.predict(pred_arr_s)[0])

        predictions = pd.Series(preds_dict, name="predicted")
        predictions.index = pd.to_datetime(predictions.index)
        actuals = ew_ret.loc[predictions.index].rename("actual")

        self._predictions = predictions
        self._actuals     = actuals

        logger.info(
            "RollingFactorForecaster done: %d predictions generated.", len(predictions)
        )
        return predictions, actuals

    @property
    def predictions(self) -> Optional[pd.Series]:
        """Out-of-sample predictions from the last ``fit_predict()`` call."""
        return self._predictions

    @property
    def actuals(self) -> Optional[pd.Series]:
        """Actual returns aligned to predictions."""
        return self._actuals


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter Factor Smoother
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFactorModel:
    """
    Local-level Kalman filter / smoother for factor return time series.

    Models each factor as a random walk with observation noise:

        F_t  = F_{t-1} + η_t ,    η_t ~ N(0, Q)
        y_t  = F_t     + ε_t ,    ε_t ~ N(0, R)

    Transition and observation noise variances are estimated via the EM
    algorithm (using the *pykalman* library when available, otherwise a
    manual EM loop is used).

    Parameters
    ----------
    n_iter : int
        Number of EM iterations (default 10).
    """

    def __init__(self, n_iter: int = 10):
        self.n_iter       = n_iter
        self._smoothed:   Optional[pd.DataFrame] = None
        self._kf_models   = []

    # ------------------------------------------------------------------
    def fit(self, factor_returns: pd.DataFrame) -> "KalmanFactorModel":
        """
        Estimate Kalman filter parameters for each factor column.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor return time series (T × k).

        Returns
        -------
        self
        """
        self._factor_returns = factor_returns.copy()
        self._kf_models = []
        logger.info("KalmanFactorModel: fitting %d factors via EM…", factor_returns.shape[1])

        for col in factor_returns.columns:
            series = factor_returns[col].dropna().values.astype(float)
            kf_params = self._fit_em(series)
            self._kf_models.append((col, kf_params))

        return self

    # ------------------------------------------------------------------
    def smooth(self) -> pd.DataFrame:
        """
        Apply Kalman smoothing and return smoothed factor estimates.

        Returns
        -------
        pd.DataFrame
            Smoothed factor returns with the same DatetimeIndex as the
            input to ``fit()``.
        """
        if not self._kf_models:
            raise RuntimeError("Call fit() before smooth().")

        smoothed_cols = {}
        for col, kf_params in self._kf_models:
            series = self._factor_returns[col].dropna()
            smoothed = self._kalman_smooth(series.values.astype(float), kf_params)
            smoothed_cols[col] = pd.Series(smoothed, index=series.index)

        self._smoothed = pd.DataFrame(smoothed_cols)
        logger.info("KalmanFactorModel: smoothing complete.")
        return self._smoothed

    # ------------------------------------------------------------------
    # Internal EM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_em(
        y: np.ndarray,
        n_iter: int = 10,
        q_init: float = 1e-4,
        r_init: float = 1e-3,
    ) -> dict:
        """
        Run a simple scalar EM algorithm to estimate Q (transition variance)
        and R (observation variance) for a local-level model.

        Returns a dict with keys ``'Q'``, ``'R'``, ``'mu0'``, ``'sigma0'``.
        """
        T   = len(y)
        Q   = q_init
        R   = r_init
        mu0 = float(y[0])
        s0  = 1.0

        for _ in range(n_iter):
            # ── Forward pass ──────────────────────────────────────────────
            mu_f  = np.empty(T)
            sig_f = np.empty(T)
            mu_p  = np.empty(T)
            sig_p = np.empty(T)
            K_g   = np.empty(T)

            mu_p[0]  = mu0
            sig_p[0] = s0 + Q

            for t in range(T):
                if t > 0:
                    mu_p[t]  = mu_f[t - 1]
                    sig_p[t] = sig_f[t - 1] + Q
                S        = sig_p[t] + R
                K_g[t]   = sig_p[t] / S
                mu_f[t]  = mu_p[t] + K_g[t] * (y[t] - mu_p[t])
                sig_f[t] = (1 - K_g[t]) * sig_p[t]

            # ── Backward (RTS) smoother ───────────────────────────────────
            mu_s  = np.empty(T)
            sig_s = np.empty(T)
            cov_s = np.empty(T - 1)      # Cov(F_t, F_{t-1} | y_{1:T})

            mu_s[-1]  = mu_f[-1]
            sig_s[-1] = sig_f[-1]

            for t in range(T - 2, -1, -1):
                G         = sig_f[t] / sig_p[t + 1]
                mu_s[t]   = mu_f[t]  + G * (mu_s[t + 1]  - mu_p[t + 1])
                sig_s[t]  = sig_f[t] + G * (sig_s[t + 1] - sig_p[t + 1]) * G
                cov_s[t]  = G * sig_s[t + 1]

            # ── M-step ───────────────────────────────────────────────────
            # Update Q
            Q_num = (
                np.sum(sig_s[1:])
                + np.sum(mu_s[1:] ** 2)
                - 2 * np.sum(cov_s * mu_s[1:] * mu_s[:-1] / (mu_s[1:] ** 2 + 1e-12))
                + np.sum(sig_s[:-1])
                + np.sum(mu_s[:-1] ** 2)
            )
            Q = max(float(Q_num) / (T - 1), 1e-9)

            # Update R
            resid2 = (y - mu_s) ** 2
            R = max(float(np.mean(resid2 + sig_s)), 1e-9)

            mu0 = float(mu_s[0])
            s0  = float(sig_s[0])

        return {"Q": Q, "R": R, "mu0": mu0, "sigma0": s0}

    @staticmethod
    def _kalman_smooth(y: np.ndarray, params: dict) -> np.ndarray:
        """
        Run Kalman filter + RTS smoother and return smoothed state means.
        """
        T    = len(y)
        Q    = params["Q"]
        R    = params["R"]
        mu0  = params["mu0"]
        s0   = params["sigma0"]

        mu_f  = np.empty(T)
        sig_f = np.empty(T)
        mu_p  = np.empty(T)
        sig_p = np.empty(T)

        mu_p[0]  = mu0
        sig_p[0] = s0 + Q

        for t in range(T):
            if t > 0:
                mu_p[t]  = mu_f[t - 1]
                sig_p[t] = sig_f[t - 1] + Q
            S       = sig_p[t] + R
            K       = sig_p[t] / S
            mu_f[t] = mu_p[t] + K * (y[t] - mu_p[t])
            sig_f[t] = (1 - K) * sig_p[t]

        # RTS smoother
        mu_s  = np.empty(T)
        sig_s = np.empty(T)
        mu_s[-1]  = mu_f[-1]
        sig_s[-1] = sig_f[-1]

        for t in range(T - 2, -1, -1):
            G        = sig_f[t] / sig_p[t + 1]
            mu_s[t]  = mu_f[t] + G * (mu_s[t + 1] - mu_p[t + 1])
            sig_s[t] = sig_f[t] + G * (sig_s[t + 1] - sig_p[t + 1]) * G

        return mu_s
