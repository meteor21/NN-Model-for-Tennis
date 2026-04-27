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


# ─────────────────────────────────────────────────────────────────────────────
# Multivariate Dynamic Factor Model (Kalman DFM)
# ─────────────────────────────────────────────────────────────────────────────

class MultivariateDFM:
    """
    Multivariate Dynamic Factor Model via Kalman filter + EM.

    Models the joint dynamics of K latent factors:

        F_t  = A F_{t-1} + η_t ,   η_t ~ N(0, Q)      [transition]
        y_t  = F_t + ε_t       ,   ε_t ~ N(0, R)      [observation]

    where A is a K×K transition matrix, Q is the K×K process noise
    covariance, and R = diag(r_1, …, r_K) is diagonal observation noise.

    Parameters estimated via the EM algorithm.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations.
    """

    def __init__(self, n_iter: int = 20):
        self.n_iter   = n_iter
        self.A_: Optional[np.ndarray] = None   # K × K transition
        self.Q_: Optional[np.ndarray] = None   # K × K process noise
        self.R_: Optional[np.ndarray] = None   # K × K obs noise (diagonal)
        self._smoothed: Optional[pd.DataFrame] = None
        self._factor_names: list = []

    # ------------------------------------------------------------------
    def fit(self, factor_returns: pd.DataFrame) -> "MultivariateDFM":
        """
        Estimate model parameters on observed factor returns.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor return time series (T × K).

        Returns
        -------
        self
        """
        clean = factor_returns.dropna()
        self._factor_names = list(clean.columns)
        Y = clean.values.T.astype(float)       # K × T
        K, T = Y.shape

        # ── Initialise ──────────────────────────────────────────────────
        A = np.eye(K) * 0.5
        Q = np.cov(Y) * 0.1 + np.eye(K) * 1e-4
        R = np.diag(np.var(Y, axis=1) * 0.5 + 1e-4)
        mu0  = Y[:, 0].copy()
        P0   = np.eye(K)

        logger.info("MultivariateDFM: fitting K=%d factors, T=%d, n_iter=%d", K, T, self.n_iter)

        for iteration in range(self.n_iter):
            # ── E-step: Kalman filter + RTS smoother ──────────────────
            mu_f, P_f, mu_p, P_p = self._kalman_filter(Y, A, Q, R, mu0, P0)
            mu_s, P_s, P_lag     = self._rts_smoother(mu_f, P_f, mu_p, P_p, A)

            # Abort if numerics blew up
            if not np.isfinite(mu_s).all():
                logger.warning("MultivariateDFM: NaN in smoother at iter %d; stopping early.", iteration)
                break

            # ── M-step: update A, Q, R ────────────────────────────────
            # A = (sum_{t=1}^{T-1} E[F_t F_{t-1}']) (sum E[F_{t-1} F_{t-1}'])^{-1}
            S11 = np.zeros((K, K))   # sum E[F_t F_t']   t=1..T-1
            S10 = np.zeros((K, K))   # sum E[F_t F_{t-1}']
            S00 = np.zeros((K, K))   # sum E[F_{t-1} F_{t-1}']

            for t in range(1, T):
                S11 += P_s[:, :, t]     + np.outer(mu_s[:, t],     mu_s[:, t])
                S10 += P_lag[:, :, t-1] + np.outer(mu_s[:, t],     mu_s[:, t-1])
                S00 += P_s[:, :, t-1]   + np.outer(mu_s[:, t-1],   mu_s[:, t-1])

            # Ridge-regularise S00 for stability
            S00 += np.eye(K) * (np.trace(S00) / K) * 1e-4

            A_new = S10 @ np.linalg.solve(S00, np.eye(K))

            # Enforce stationarity: scale down A if spectral radius >= 1
            eigs = np.abs(np.linalg.eigvals(A_new))
            if eigs.max() >= 1.0:
                A_new *= 0.95 / eigs.max()

            # Q = (S11 - A S10') / (T-1)
            Q_new = (S11 - A_new @ S10.T) / (T - 1)
            # Symmetrise and ensure PSD via eigenvalue floor
            Q_new = (Q_new + Q_new.T) / 2
            eigvals_q, eigvecs_q = np.linalg.eigh(Q_new)
            Q_new = eigvecs_q @ np.diag(np.maximum(eigvals_q, 1e-6)) @ eigvecs_q.T

            # R = diag mean squared residual  (diagonal constraint)
            resid2 = np.zeros(K)
            for t in range(T):
                e = Y[:, t] - mu_s[:, t]
                resid2 += e ** 2 + np.diag(P_s[:, :, t])
            R_new = np.diag(np.maximum(resid2 / T, 1e-6))

            # Reject update if NaN appeared
            if not (np.isfinite(A_new).all() and np.isfinite(Q_new).all() and np.isfinite(R_new).all()):
                logger.warning("MultivariateDFM: NaN in M-step at iter %d; keeping previous params.", iteration)
                break

            # Update initial state
            mu0 = mu_s[:, 0].copy()
            P0  = P_s[:, :, 0].copy()

            A, Q, R = A_new, Q_new, R_new

        self.A_ = A
        self.Q_ = Q
        self.R_ = R
        self._mu0 = mu0
        self._P0  = P0
        self._factor_returns = factor_returns.copy()

        logger.info("MultivariateDFM fitted. A eigenvalues: %s",
                    np.round(np.abs(np.linalg.eigvals(A)), 3))
        return self

    # ------------------------------------------------------------------
    def smooth(self) -> pd.DataFrame:
        """
        Return Kalman-smoothed factor estimates.

        Returns
        -------
        pd.DataFrame
            Smoothed factors with same DatetimeIndex as input to ``fit()``.
        """
        if self.A_ is None:
            raise RuntimeError("Call fit() before smooth().")
        clean = self._factor_returns.dropna()
        Y     = clean.values.T.astype(float)

        mu_f, P_f, mu_p, P_p = self._kalman_filter(
            Y, self.A_, self.Q_, self.R_, self._mu0, self._P0
        )
        mu_s, _, _ = self._rts_smoother(mu_f, P_f, mu_p, P_p, self.A_)

        self._smoothed = pd.DataFrame(
            mu_s.T, index=clean.index, columns=self._factor_names
        )
        return self._smoothed

    def impulse_response(self, shock_factor: int = 0, horizon: int = 20) -> pd.DataFrame:
        """
        Compute impulse-response functions to a one-standard-deviation shock.

        Parameters
        ----------
        shock_factor : int
            Index of the factor receiving the shock.
        horizon : int
            Number of periods to trace the response.

        Returns
        -------
        pd.DataFrame
            IRF matrix (horizon × K).
        """
        if self.A_ is None:
            raise RuntimeError("Call fit() first.")
        K    = self.A_.shape[0]
        irf  = np.zeros((horizon, K))
        shock = np.zeros(K)
        shock[shock_factor] = np.sqrt(self.Q_[shock_factor, shock_factor])

        state = shock.copy()
        for h in range(horizon):
            irf[h] = state
            state  = self.A_ @ state

        cols = self._factor_names if self._factor_names else [f"F{i+1}" for i in range(K)]
        return pd.DataFrame(irf, columns=cols,
                            index=pd.RangeIndex(horizon, name="horizon"))

    # ------------------------------------------------------------------
    # Internal Kalman helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kalman_filter(Y, A, Q, R, mu0, P0):
        K, T = Y.shape
        mu_f = np.zeros((K, T))
        P_f  = np.zeros((K, K, T))
        mu_p = np.zeros((K, T))
        P_p  = np.zeros((K, K, T))

        mu_p[:, 0] = A @ mu0
        P_p[:, :, 0] = A @ P0 @ A.T + Q

        for t in range(T):
            if t > 0:
                mu_p[:, t]   = A @ mu_f[:, t-1]
                P_p[:, :, t] = A @ P_f[:, :, t-1] @ A.T + Q

            S = P_p[:, :, t] + R
            try:
                K_gain = P_p[:, :, t] @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K_gain = P_p[:, :, t] @ np.linalg.pinv(S)

            innov       = Y[:, t] - mu_p[:, t]
            mu_f[:, t]  = mu_p[:, t] + K_gain @ innov
            P_f[:, :, t] = (np.eye(K) - K_gain) @ P_p[:, :, t]
            P_f[:, :, t] = (P_f[:, :, t] + P_f[:, :, t].T) / 2

        return mu_f, P_f, mu_p, P_p

    @staticmethod
    def _rts_smoother(mu_f, P_f, mu_p, P_p, A):
        K, T = mu_f.shape
        mu_s  = np.zeros((K, T))
        P_s   = np.zeros((K, K, T))
        P_lag = np.zeros((K, K, T - 1))   # Cov(F_t, F_{t-1} | Y)

        mu_s[:, -1]    = mu_f[:, -1]
        P_s[:, :, -1]  = P_f[:, :, -1]

        for t in range(T - 2, -1, -1):
            try:
                G = P_f[:, :, t] @ np.linalg.inv(P_p[:, :, t+1]) @ A.T
            except np.linalg.LinAlgError:
                G = P_f[:, :, t] @ np.linalg.pinv(P_p[:, :, t+1]) @ A.T

            mu_s[:, t]   = mu_f[:, t] + G @ (mu_s[:, t+1] - mu_p[:, t+1])
            P_s[:, :, t] = P_f[:, :, t] + G @ (P_s[:, :, t+1] - P_p[:, :, t+1]) @ G.T
            P_s[:, :, t] = (P_s[:, :, t] + P_s[:, :, t].T) / 2
            P_lag[:, :, t] = G @ P_s[:, :, t+1]

        return mu_s, P_s, P_lag
