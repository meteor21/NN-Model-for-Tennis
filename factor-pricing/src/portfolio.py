"""
Portfolio construction and backtesting utilities.

Implements mean-variance optimisation, factor-mimicking portfolios,
a full backtest engine, and comparative performance plots.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mean-variance portfolio
# ─────────────────────────────────────────────────────────────────────────────

def mean_variance_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    regularise: float = 1e-4,
) -> pd.Series:
    """
    Compute mean-variance optimal portfolio weights.

    Solves:  w* = (1 / λ) Σ⁻¹ μ,  then normalises to sum to 1.

    A small diagonal regularisation is added to improve numerical stability.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected return for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    risk_aversion : float
        Scalar risk-aversion parameter λ.
    long_only : bool
        If True, negative weights are clipped to zero before normalisation.
    regularise : float
        Diagonal regularisation added to the covariance matrix.

    Returns
    -------
    pd.Series
        Portfolio weights summing to 1.
    """
    assets = expected_returns.index.intersection(cov_matrix.index)
    mu     = expected_returns.loc[assets].values.astype(float)
    Sigma  = cov_matrix.loc[assets, assets].values.astype(float)

    # Regularise
    Sigma += np.eye(len(assets)) * regularise

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    raw_w = Sigma_inv @ mu / risk_aversion
    w     = pd.Series(raw_w, index=assets)

    if long_only:
        w = w.clip(lower=0)

    total = w.sum()
    if abs(total) < 1e-10:
        logger.warning("Mean-variance weights sum near zero; returning equal weights.")
        w = pd.Series(np.ones(len(assets)) / len(assets), index=assets)
    else:
        w /= total

    return w


# ─────────────────────────────────────────────────────────────────────────────
# Factor-mimicking portfolio
# ─────────────────────────────────────────────────────────────────────────────

def factor_mimicking_portfolio(
    returns_df: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct factor-mimicking portfolios via OLS.

    For each factor f, finds the portfolio w_f such that w_f' R = F_f in-sample
    (i.e. OLS of factor returns on asset returns, then normalise weights).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns (T × N).
    factor_returns : pd.DataFrame
        Factor returns (T × K).

    Returns
    -------
    pd.DataFrame
        Weight matrix (N × K): each column contains the mimicking weights for
        one factor.
    """
    common = returns_df.index.intersection(factor_returns.index)
    R = returns_df.loc[common].dropna(axis=1, how="any").dropna()
    F = factor_returns.loc[R.index].dropna(axis=1, how="any").dropna()

    # Further align
    common2 = R.index.intersection(F.index)
    R, F = R.loc[common2], F.loc[common2]

    weights_dict = {}
    for col in F.columns:
        y = F[col].values
        X = R.values
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if valid.sum() < 20:
            weights_dict[col] = pd.Series(np.ones(R.shape[1]) / R.shape[1], index=R.columns)
            continue

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X[valid], y[valid])
        w = pd.Series(reg.coef_, index=R.columns)

        # Long-only: keep only positive loadings, then normalise to sum = 1
        w = w.clip(lower=0)
        total = w.sum()
        if total < 1e-10:
            # All negative — fall back to equal weight
            w = pd.Series(np.ones(R.shape[1]) / R.shape[1], index=R.columns)
        else:
            w /= total
        weights_dict[col] = w

    return pd.DataFrame(weights_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def backtest_portfolio(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    rebalance_freq: str = "M",
) -> Dict[str, float | pd.Series]:
    """
    Backtest a portfolio given time-varying (or static) weights.

    Parameters
    ----------
    weights_df : pd.DataFrame or pd.Series
        Portfolio weights. If a Series, treated as static (constant over time).
        If a DataFrame, rows are rebalancing dates and columns are assets.
    returns_df : pd.DataFrame
        Asset log returns (daily).
    rebalance_freq : str
        Pandas offset alias for rebalancing frequency (only used when
        *weights_df* is a static Series).

    Returns
    -------
    dict with keys:
        - ``'cumulative_return'`` : pd.Series — cumulative log return
        - ``'sharpe'``            : float — annualised Sharpe ratio
        - ``'max_drawdown'``      : float — maximum drawdown
        - ``'calmar'``            : float — annualised return / |max_drawdown|
        - ``'portfolio_returns'`` : pd.Series — daily portfolio log returns
    """
    # Resolve weight matrix
    if isinstance(weights_df, pd.Series):
        static_w = weights_df
    else:
        static_w = None

    common_assets = returns_df.columns
    if static_w is not None:
        w = static_w.reindex(common_assets).fillna(0.0)
        w_total = w.sum()
        if abs(w_total) > 1e-10:
            w /= w_total

    port_ret_list = []
    port_dates    = []

    for date in returns_df.index:
        if static_w is not None:
            w_today = w
        else:
            # Use the most recent available weights row
            past = weights_df.index[weights_df.index <= date]
            if len(past) == 0:
                continue
            latest = weights_df.loc[past[-1]]
            w_today = latest.reindex(common_assets).fillna(0.0)
            s = w_today.sum()
            if abs(s) > 1e-10:
                w_today /= s

        day_ret = returns_df.loc[date].reindex(common_assets).fillna(0.0)
        port_ret_list.append(float(w_today @ day_ret))
        port_dates.append(date)

    port_returns = pd.Series(port_ret_list, index=port_dates, name="port_return")
    cum_return   = port_returns.cumsum()

    # Metrics
    ann_factor   = 252
    mean_ret     = port_returns.mean()
    std_ret      = port_returns.std()
    sharpe       = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else np.nan

    wealth       = np.exp(cum_return)
    rolling_max  = wealth.cummax()
    drawdown     = (wealth - rolling_max) / rolling_max
    max_dd       = float(drawdown.min())

    ann_ret      = float(mean_ret * ann_factor)
    calmar       = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    return {
        "cumulative_return":  cum_return,
        "portfolio_returns":  port_returns,
        "sharpe":             float(sharpe),
        "max_drawdown":       max_dd,
        "calmar":             float(calmar),
        "annualised_return":  ann_ret,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_portfolios(
    latent_factor_weights: pd.Series,
    observed_factor_weights: pd.Series,
    equal_weights: pd.Series,
    returns_df: pd.DataFrame,
    results_dir: str = "results",
    filename: str = "portfolio_comparison.png",
) -> pd.DataFrame:
    """
    Backtest and plot cumulative returns for three portfolios.

    Parameters
    ----------
    latent_factor_weights : pd.Series
        Weights for the latent (PCA) factor-mimicking portfolio.
    observed_factor_weights : pd.Series
        Weights for the observed (FF5) factor-mimicking portfolio.
    equal_weights : pd.Series
        Equal-weight portfolio weights.
    returns_df : pd.DataFrame
        Asset log returns.
    results_dir : str
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    pd.DataFrame
        Summary metrics table for all three portfolios.
    """
    os.makedirs(results_dir, exist_ok=True)

    portfolios = {
        "Latent Factor": latent_factor_weights,
        "Observed FF5":  observed_factor_weights,
        "Equal Weight":  equal_weights,
    }

    summary_rows = {}
    cum_returns  = {}

    for name, weights in portfolios.items():
        result = backtest_portfolio(weights, returns_df)
        cum_returns[name] = result["cumulative_return"]
        summary_rows[name] = {
            "Ann. Return":  round(result["annualised_return"], 4),
            "Sharpe":       round(result["sharpe"],           4),
            "Max Drawdown": round(result["max_drawdown"],     4),
            "Calmar":       round(result["calmar"],           4),
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["steelblue", "darkorange", "green"]
    for (name, cum), color in zip(cum_returns.items(), colors):
        ax.plot(cum.index, cum.values, label=name, linewidth=1.5, color=color)

    ax.set_title("Portfolio Cumulative Log Returns", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Portfolio comparison plot saved: %s", os.path.join(results_dir, filename))

    summary = pd.DataFrame(summary_rows).T
    print("\n── Portfolio Performance Summary ─────────────────────────────────")
    print(summary.to_string())
    return summary
