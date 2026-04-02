"""
Factor extraction models.

Implements PCA-based latent factor models, the Bai & Ng (2002) information
criterion for factor number selection, and an observed factor model (OLS
regression on FF5 + macro variables).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Latent Factor Model (PCA)
# ─────────────────────────────────────────────────────────────────────────────

class LatentFactorModel:
    """
    PCA-based latent factor model for asset returns.

    Follows the sklearn fit / transform interface.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to extract.
    """

    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, returns_df: pd.DataFrame, n_factors: Optional[int] = None) -> "LatentFactorModel":
        """
        Fit PCA on the returns DataFrame.

        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame of asset returns (rows = dates, columns = assets).
        n_factors : int, optional
            Override the instance-level n_factors.

        Returns
        -------
        self
        """
        if n_factors is not None:
            self.n_factors = n_factors

        clean = returns_df.dropna(axis=1, how="any").dropna()
        self._feature_names = list(clean.columns)
        self.n_factors = min(self.n_factors, clean.shape[1], clean.shape[0])

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(clean.values)

        self._pca = PCA(n_components=self.n_factors)
        self._pca.fit(X)

        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.components_               = self._pca.components_
        logger.info(
            "LatentFactorModel fitted: %d factors, cumulative variance = %.1f%%",
            self.n_factors,
            np.sum(self.explained_variance_ratio_) * 100,
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Project returns onto the latent factors.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns DataFrame (must contain the fitted asset columns).

        Returns
        -------
        pd.DataFrame
            Factor return time series with DatetimeIndex.
            Columns are labelled ``F1``, ``F2``, …
        """
        self._check_fitted()
        X = self._align_and_scale(returns_df)
        factors = self._pca.transform(X)
        cols = [f"F{i+1}" for i in range(self.n_factors)]
        return pd.DataFrame(factors, index=returns_df.index, columns=cols)

    # ------------------------------------------------------------------
    def get_residuals(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute idiosyncratic (residual) returns after removing factor structure.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Original returns DataFrame.

        Returns
        -------
        pd.DataFrame
            Residual returns with same shape as *returns_df*.
        """
        self._check_fitted()
        common_cols = [c for c in self._feature_names if c in returns_df.columns]
        clean = returns_df[common_cols].dropna()

        X = self._scaler.transform(clean.values)
        reconstructed = self._pca.inverse_transform(self._pca.transform(X))
        residuals = X - reconstructed

        # Un-scale to original units
        residuals_unscaled = residuals * self._scaler.scale_
        return pd.DataFrame(residuals_unscaled, index=clean.index, columns=common_cols)

    # ------------------------------------------------------------------
    def rolling_factors(
        self,
        returns_df: pd.DataFrame,
        window: int = 252,
        n_factors: Optional[int] = None,
        min_obs_frac: float = 0.8,
    ) -> pd.DataFrame:
        """
        Compute rolling PCA factor returns.

        At each date *t*, fits PCA on the preceding *window* observations
        and projects the current return onto those factors.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Full returns history.
        window : int
            Look-back window in trading days.
        n_factors : int, optional
            Number of factors; defaults to ``self.n_factors``.
        min_obs_frac : float
            Minimum fraction of *window* observations required to fit PCA.

        Returns
        -------
        pd.DataFrame
            Rolling factor returns aligned to ``returns_df.index``.
        """
        k = n_factors or self.n_factors
        min_obs = int(window * min_obs_frac)
        clean = returns_df.dropna(axis=1, how="any")
        dates = clean.index
        records = []

        for i in range(window, len(dates)):
            window_data = clean.iloc[i - window: i].dropna()
            if len(window_data) < min_obs:
                logger.debug("Skipping t=%s: only %d obs (need %d)", dates[i], len(window_data), min_obs)
                records.append([np.nan] * k)
                continue

            scaler = StandardScaler()
            X_win = scaler.fit_transform(window_data.values)
            pca = PCA(n_components=min(k, X_win.shape[1]))
            pca.fit(X_win)

            x_curr = scaler.transform(clean.iloc[[i]].values)
            f = pca.transform(x_curr)[0]
            # Pad with NaN if fewer components than k
            if len(f) < k:
                f = np.concatenate([f, [np.nan] * (k - len(f))])
            records.append(f[:k])

        cols = [f"F{i+1}" for i in range(k)]
        result = pd.DataFrame(records, index=dates[window:], columns=cols)
        return result

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _align_and_scale(self, returns_df: pd.DataFrame) -> np.ndarray:
        common_cols = [c for c in self._feature_names if c in returns_df.columns]
        return self._scaler.transform(returns_df[common_cols].values)

    @property
    def pca(self) -> PCA:
        """Expose the underlying sklearn PCA object."""
        self._check_fitted()
        return self._pca

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Bai & Ng (2002) information criterion
# ─────────────────────────────────────────────────────────────────────────────

def bai_ng_criterion(
    returns_df: pd.DataFrame,
    max_factors: int = 15,
    results_dir: str = "results",
    filename: str = "bai_ng_criterion.png",
) -> Dict[str, int]:
    """
    Implement the Bai & Ng (2002) IC1 and IC2 panel information criteria
    for selecting the number of static factors.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns (rows = T, columns = N).
    max_factors : int
        Maximum number of factors to evaluate.
    results_dir : str
        Directory to save the criterion plot.
    filename : str
        Output filename for the plot.

    Returns
    -------
    dict
        ``{'IC1': k1, 'IC2': k2}`` — optimal factor counts under each criterion.

    References
    ----------
    Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate
    Factor Models." *Econometrica*, 70(1), 191–221.
    """
    clean = returns_df.dropna(axis=1, how="any").dropna()
    X = clean.values.astype(float)
    T, N = X.shape
    kmax = min(max_factors, N - 1, T - 1)

    # Standardise
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    ic1_vals, ic2_vals = [], []

    for k in range(1, kmax + 1):
        pca = PCA(n_components=k)
        F   = pca.fit_transform(X_std)          # T × k
        L   = pca.components_.T                  # N × k

        X_hat  = F @ L.T
        resid  = X_std - X_hat
        V_k    = np.mean(resid ** 2)             # mean squared residual

        # Penalty terms from Bai & Ng (2002), equations (8)–(9)
        penalty_base = k * (N + T) / (N * T)
        ic1 = np.log(V_k) + penalty_base * np.log(min(N, T))
        ic2 = np.log(V_k) + penalty_base * np.log(min(N, T) ** 2 / (N + T))

        ic1_vals.append(ic1)
        ic2_vals.append(ic2)

    k_range = np.arange(1, kmax + 1)
    k_ic1   = int(k_range[np.argmin(ic1_vals)])
    k_ic2   = int(k_range[np.argmin(ic2_vals)])

    logger.info("Bai-Ng IC1 optimal k = %d", k_ic1)
    logger.info("Bai-Ng IC2 optimal k = %d", k_ic2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vals, name, k_opt in zip(
        axes, [ic1_vals, ic2_vals], ["IC1", "IC2"], [k_ic1, k_ic2]
    ):
        ax.plot(k_range, vals, "b.-", markersize=8)
        ax.axvline(k_opt, color="red", linestyle="--", label=f"k* = {k_opt}")
        ax.set_xlabel("Number of Factors k")
        ax.set_ylabel("Criterion Value")
        ax.set_title(f"Bai-Ng {name}")
        ax.legend()

    fig.suptitle("Bai & Ng (2002) Factor Selection Criteria", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"IC1": k_ic1, "IC2": k_ic2}


# ─────────────────────────────────────────────────────────────────────────────
# Observed Factor Model (FF5 + macro)
# ─────────────────────────────────────────────────────────────────────────────

class ObservedFactorModel:
    """
    Cross-sectional OLS regression of each asset return on observed factors.

    Uses Fama-French 5 factors plus any additional macro variables provided.

    Parameters
    ----------
    add_constant : bool
        Whether to include an intercept in each asset regression.
    """

    def __init__(self, add_constant: bool = True):
        self.add_constant = add_constant
        self._betas: Optional[pd.DataFrame] = None
        self._fitted: Optional[pd.DataFrame] = None
        self._factor_names: List[str] = []
        self._asset_names:  List[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        returns_df: pd.DataFrame,
        factor_df: pd.DataFrame,
    ) -> "ObservedFactorModel":
        """
        Fit OLS for each asset in *returns_df* on *factor_df*.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset log returns (T × N).
        factor_df : pd.DataFrame
            Observed factor returns / levels (T × K).

        Returns
        -------
        self
        """
        # Align on common dates
        common_idx = returns_df.index.intersection(factor_df.index)
        Y = returns_df.loc[common_idx].dropna(axis=1, how="any")
        X = factor_df.loc[common_idx].dropna(axis=1, how="any")

        # Keep only rows where both Y and X are fully observed
        valid = Y.notna().all(axis=1) & X.notna().all(axis=1)
        Y, X = Y.loc[valid], X.loc[valid]

        self._asset_names  = list(Y.columns)
        self._factor_names = list(X.columns)

        X_np = X.values
        if self.add_constant:
            ones = np.ones((X_np.shape[0], 1))
            X_np = np.hstack([ones, X_np])
            col_labels = ["alpha"] + self._factor_names
        else:
            col_labels = self._factor_names

        # OLS for each asset: β = (X'X)^{-1} X'Y
        XtX_inv = np.linalg.pinv(X_np.T @ X_np)
        B = XtX_inv @ X_np.T @ Y.values  # (K+1) × N

        self._betas  = pd.DataFrame(B, index=col_labels, columns=self._asset_names)
        fitted_vals  = X_np @ B
        self._fitted = pd.DataFrame(fitted_vals, index=Y.index, columns=self._asset_names)

        logger.info(
            "ObservedFactorModel fitted: %d assets, %d factors, %d obs",
            len(self._asset_names), len(self._factor_names), len(Y),
        )
        return self

    # ------------------------------------------------------------------
    def get_factor_exposures(self) -> pd.DataFrame:
        """
        Return the estimated beta matrix.

        Returns
        -------
        pd.DataFrame
            Beta matrix with shape (factors + constant, assets).
        """
        if self._betas is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._betas

    # ------------------------------------------------------------------
    def get_fitted_returns(self) -> pd.DataFrame:
        """
        Return the fitted (in-sample) return values.

        Returns
        -------
        pd.DataFrame
            Fitted returns with DatetimeIndex.
        """
        if self._fitted is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._fitted
