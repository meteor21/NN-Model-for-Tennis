"""
Cross-sectional return prediction.

Shifts from aggregate portfolio forecasting to asset-level rank prediction:
predicts which quintile each asset will fall into next period and evaluates
by rank IC and long-short quintile spread.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional predictor
# ─────────────────────────────────────────────────────────────────────────────

class CrossSectionalPredictor:
    """
    Predicts next-period cross-sectional asset returns using lagged factor
    exposures as features.

    For each rebalancing date the predictor:
    1. Estimates each asset's exposure to the latent factors (betas) via
       a rolling OLS window.
    2. Uses lagged factor return forecasts × betas as predicted returns.
    3. Ranks assets into quintiles and constructs a long-short portfolio.

    Parameters
    ----------
    window : int
        Rolling OLS window for estimating betas (trading days).
    n_quintiles : int
        Number of rank groups (default 5).
    """

    def __init__(self, window: int = 252, n_quintiles: int = 5):
        self.window      = window
        self.n_quintiles = n_quintiles
        self._results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def fit_predict(
        self,
        returns_df: pd.DataFrame,
        factor_returns: pd.DataFrame,
        factor_forecasts: Optional[pd.DataFrame] = None,
        step: int = 21,
    ) -> pd.DataFrame:
        """
        Run walk-forward cross-sectional prediction.

        At each rebalancing date *t*:
        - Estimate betas: B = (F'F)^{-1} F'R  on the look-back window.
        - Predicted return = B × f_{t+1|t}, where f_{t+1|t} is the
          lagged factor return (lag-1) as a naive forecast when
          *factor_forecasts* is not provided.
        - Rank assets into quintiles.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset log returns (T × N).
        factor_returns : pd.DataFrame
            Factor returns (T × K).
        factor_forecasts : pd.DataFrame, optional
            External factor forecasts (T × K). Uses lag-1 if not provided.
        step : int
            Rebalancing frequency in trading days.

        Returns
        -------
        pd.DataFrame
            Results frame with columns:
            ``date``, ``asset``, ``predicted_rank``, ``actual_return``,
            ``quintile``, ``actual_quintile``.
        """
        common = returns_df.index.intersection(factor_returns.index)
        R = returns_df.loc[common].dropna(axis=1, how="any")
        F = factor_returns.loc[common].dropna(axis=1, how="any")
        common2 = R.index.intersection(F.index)
        R, F = R.loc[common2], F.loc[common2]

        dates  = R.index
        n      = len(dates)
        assets = R.columns.tolist()
        K      = F.shape[1]

        records = []
        for i in range(self.window, n - 1, step):
            t_date = dates[i]

            # ── Estimate betas on look-back window ────────────────────
            R_win = R.iloc[i - self.window: i].values   # window × N
            F_win = F.iloc[i - self.window: i].values   # window × K

            # Add intercept
            X_win = np.hstack([np.ones((len(F_win), 1)), F_win])
            try:
                B = np.linalg.lstsq(X_win, R_win, rcond=None)[0]  # (K+1) × N
            except np.linalg.LinAlgError:
                continue

            betas = B[1:]  # K × N  (drop intercept)

            # ── Factor forecast: lag-1 or external ───────────────────
            if factor_forecasts is not None and t_date in factor_forecasts.index:
                f_fcast = factor_forecasts.loc[t_date].values[:K]
            else:
                f_fcast = F.iloc[i - 1].values[:K]   # lag-1 naive

            # ── Predicted return for each asset ────────────────────
            pred_ret = betas.T @ f_fcast   # N-vector

            # ── Actual return at t+1 ──────────────────────────────
            actual_ret = R.iloc[i + 1].values   # N-vector

            # ── Rank into quintiles (1=worst, n_quintiles=best) ───
            pred_rank   = self._quintile_rank(pred_ret)
            actual_rank = self._quintile_rank(actual_ret)

            for j, asset in enumerate(assets):
                records.append({
                    "date":           t_date,
                    "asset":          asset,
                    "predicted_ret":  pred_ret[j],
                    "actual_ret":     actual_ret[j],
                    "pred_quintile":  pred_rank[j],
                    "actual_quintile": actual_rank[j],
                })

        self._results = pd.DataFrame(records)
        logger.info("CrossSectionalPredictor: %d rebalancing dates, %d total records.",
                    len(self._results["date"].unique()), len(self._results))
        return self._results

    # ------------------------------------------------------------------
    def rank_ic(self) -> pd.Series:
        """
        Compute per-date rank IC (Spearman correlation of predicted vs actual).

        Returns
        -------
        pd.Series
            Rank IC time series (one value per rebalancing date).
        """
        if self._results is None:
            raise RuntimeError("Run fit_predict() first.")
        ics = {}
        for date, grp in self._results.groupby("date"):
            if len(grp) < 5:
                continue
            ic, _ = stats.spearmanr(grp["predicted_ret"], grp["actual_ret"])
            ics[date] = ic
        return pd.Series(ics, name="rank_ic").sort_index()

    # ------------------------------------------------------------------
    def quintile_returns(self) -> pd.DataFrame:
        """
        Average return by predicted quintile (for bar chart).

        Returns
        -------
        pd.DataFrame
            Mean and std of actual returns in each predicted quintile.
        """
        if self._results is None:
            raise RuntimeError("Run fit_predict() first.")
        grp = self._results.groupby("pred_quintile")["actual_ret"]
        summary = pd.DataFrame({
            "mean_ret":  grp.mean(),
            "std_ret":   grp.std(),
            "n_obs":     grp.count(),
        })
        summary["t_stat"] = summary["mean_ret"] / (
            summary["std_ret"] / np.sqrt(summary["n_obs"])
        )
        return summary

    # ------------------------------------------------------------------
    def long_short_spread(self) -> pd.Series:
        """
        Time series of long-top-quintile / short-bottom-quintile daily return.

        Returns
        -------
        pd.Series
            Long-short portfolio return at each rebalancing date.
        """
        if self._results is None:
            raise RuntimeError("Run fit_predict() first.")
        ls_returns = {}
        for date, grp in self._results.groupby("date"):
            long_ret  = grp.loc[grp["pred_quintile"] == self.n_quintiles, "actual_ret"].mean()
            short_ret = grp.loc[grp["pred_quintile"] == 1,                 "actual_ret"].mean()
            ls_returns[date] = long_ret - short_ret
        return pd.Series(ls_returns, name="ls_spread").sort_index()

    # ------------------------------------------------------------------
    def plot_quintile_returns(
        self,
        results_dir: str = "results",
        filename: str = "quintile_returns.png",
    ) -> None:
        """
        Bar chart of average actual return by predicted quintile.

        Parameters
        ----------
        results_dir : str
            Output directory.
        filename : str
            Output filename.
        """
        os.makedirs(results_dir, exist_ok=True)
        summary = self.quintile_returns()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of mean returns
        colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850"]
        axes[0].bar(summary.index, summary["mean_ret"] * 252 * 100,
                    color=colors[:len(summary)], edgecolor="black", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Predicted Quintile (1=worst, 5=best)")
        axes[0].set_ylabel("Annualised Mean Return (%)")
        axes[0].set_title("Average Return by Predicted Quintile")

        # Rank IC distribution
        ic = self.rank_ic()
        axes[1].hist(ic.dropna(), bins=30, color="steelblue", edgecolor="white",
                     linewidth=0.3, density=True)
        axes[1].axvline(ic.mean(), color="orange", linewidth=2,
                        label=f"Mean IC = {ic.mean():.3f}")
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Rank IC")
        axes[1].set_title("Rank IC Distribution")
        axes[1].legend()

        fig.suptitle("Cross-Sectional Prediction Performance", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Quintile return plot saved: %s", os.path.join(results_dir, filename))

    # ------------------------------------------------------------------
    def _quintile_rank(self, values: np.ndarray) -> np.ndarray:
        """Map values to quintile labels 1..n_quintiles."""
        n   = len(values)
        idx = np.argsort(np.argsort(values))    # rank 0-based
        return (idx * self.n_quintiles // n + 1).clip(1, self.n_quintiles)


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper
# ─────────────────────────────────────────────────────────────────────────────

def summarise_cross_sectional(predictor: CrossSectionalPredictor) -> Dict:
    """
    Compute and print a summary of cross-sectional prediction performance.

    Parameters
    ----------
    predictor : CrossSectionalPredictor
        A fitted predictor (after ``fit_predict()``).

    Returns
    -------
    dict
        Keys: ``mean_ic``, ``ic_t_stat``, ``ic_positive_pct``,
        ``ls_ann_return``, ``ls_sharpe``, ``quintile_summary``.
    """
    ic = predictor.rank_ic().dropna()
    ls = predictor.long_short_spread().dropna()

    mean_ic    = float(ic.mean())
    ic_t       = float(stats.ttest_1samp(ic, 0).statistic)
    ic_pos_pct = float((ic > 0).mean() * 100)

    ls_ann  = float(ls.mean() * 252)
    ls_std  = float(ls.std()  * np.sqrt(252))
    ls_sh   = ls_ann / ls_std if ls_std > 0 else np.nan

    q_summary = predictor.quintile_returns()

    print("\n── Cross-Sectional Prediction Summary ────────────────────────")
    print(f"  Mean Rank IC     : {mean_ic:.4f}  (t = {ic_t:.2f})")
    print(f"  IC > 0           : {ic_pos_pct:.1f}%  of rebalancing dates")
    print(f"  L/S Ann. Return  : {ls_ann*100:.2f}%")
    print(f"  L/S Sharpe       : {ls_sh:.3f}")
    print("\n  Quintile mean returns (annualised):")
    print((q_summary["mean_ret"] * 252 * 100).round(2).to_string())

    return {
        "mean_ic":        mean_ic,
        "ic_t_stat":      ic_t,
        "ic_positive_pct": ic_pos_pct,
        "ls_ann_return":  ls_ann,
        "ls_sharpe":      ls_sh,
        "quintile_summary": q_summary,
    }
