"""
Model evaluation utilities.

Provides metric computation, visualisation, model-comparison tables,
Diebold-Mariano tests, and rolling R² plots.
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    y_true : array-like
        Realised values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Keys: ``'rmse'``, ``'mae'``, ``'r2'``, ``'ic'``
        (IC = rank correlation / information coefficient).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "ic": np.nan}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    r2   = float(r2_score(yt, yp))

    if len(yt) > 2:
        ic, _ = stats.spearmanr(yt, yp)
        ic    = float(ic)
    else:
        ic = np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2, "ic": ic}


# ─────────────────────────────────────────────────────────────────────────────
# Prediction vs actuals plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_vs_actuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Predictions vs Actuals",
    results_dir: str = "results",
    filename: Optional[str] = None,
) -> None:
    """
    Plot predicted and actual time series on the same axes.

    Parameters
    ----------
    y_true : pd.Series
        Realised values with DatetimeIndex.
    y_pred : pd.Series
        Predicted values with DatetimeIndex.
    title : str
        Plot title.
    results_dir : str
        Output directory.
    filename : str, optional
        Output filename. Defaults to a sanitised version of *title*.
    """
    os.makedirs(results_dir, exist_ok=True)
    if filename is None:
        safe = title.lower().replace(" ", "_").replace("/", "_")
        filename = f"{safe}.png"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true.index, y_true.values, label="Actual",    color="steelblue",  linewidth=1.0, alpha=0.9)
    ax.plot(y_pred.index, y_pred.values, label="Predicted", color="darkorange", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", os.path.join(results_dir, filename))


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_models_table(
    results_dict: Dict[str, Dict[str, Dict[str, float]]],
    split: str = "test",
) -> pd.DataFrame:
    """
    Print and return a formatted comparison table for a given data split.

    Parameters
    ----------
    results_dict : dict
        Nested dict as returned by ``run_forecasting_experiment()``.
        Structure: ``results[model_name][split] = {rmse, r2, ic, ...}``
    split : str
        Which split to display: ``'train'``, ``'val'``, or ``'test'``.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    rows = {}
    for model_name, splits in results_dict.items():
        if split in splits:
            rows[model_name] = splits[split]

    if not rows:
        logger.warning("No results found for split '%s'.", split)
        return pd.DataFrame()

    table = pd.DataFrame(rows).T
    table = table.sort_values("rmse", ascending=True)

    header = f"\n{'=' * 60}\n  Model Comparison — {split.upper()} split\n{'=' * 60}"
    print(header)
    print(table.round(6).to_string())
    print("=" * 60)
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> Dict[str, float]:
    """
    Harvey, Leybourne & Newbold (1997) modified Diebold-Mariano test.

    Tests H₀: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) and L(·) = squared loss.
    A negative test statistic means model 1 is more accurate.

    Parameters
    ----------
    e1 : array-like
        Forecast errors from model 1.
    e2 : array-like
        Forecast errors from model 2.
    h : int
        Forecast horizon (default 1 for 1-step ahead).

    Returns
    -------
    dict
        ``{'dm_stat': float, 'p_value': float}``
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    # Loss differential
    d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    T = len(d)

    if T < 10:
        logger.warning("DM test: fewer than 10 observations (%d). Result unreliable.", T)
        return {"dm_stat": np.nan, "p_value": np.nan}

    d_bar    = np.mean(d)
    # Newey-West variance estimator with bandwidth h-1
    gamma0   = np.var(d, ddof=1)
    var_d    = gamma0 / T
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        var_d  += 2 * (1 - k / h) * gamma_k / T

    var_d = max(var_d, 1e-15)

    # Modified DM statistic (Harvey et al. 1997)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat    = d_bar / np.sqrt(var_d) * correction
    p_value    = float(2 * stats.t.sf(np.abs(dm_stat), df=T - 1))

    logger.info("DM test: stat = %.4f, p-value = %.4f", dm_stat, p_value)
    return {"dm_stat": float(dm_stat), "p_value": p_value}


# ─────────────────────────────────────────────────────────────────────────────
# Rolling R² plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_r2(
    rolling_predictions_dict: Dict[str, pd.Series],
    actuals: pd.Series,
    window: int = 63,
    results_dir: str = "results",
    filename: str = "rolling_r2.png",
) -> None:
    """
    Plot rolling out-of-sample R² over time for multiple models.

    Parameters
    ----------
    rolling_predictions_dict : dict
        Mapping of model name → predicted pd.Series (DatetimeIndex).
    actuals : pd.Series
        Realised return series (DatetimeIndex).
    window : int
        Rolling window for R² computation in trading days.
    results_dir : str
        Output directory.
    filename : str
        Output filename.
    """
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    for model_name, preds in rolling_predictions_dict.items():
        common = preds.index.intersection(actuals.index)
        if len(common) < window + 1:
            logger.warning("Not enough data for rolling R² of '%s'.", model_name)
            continue

        yt = actuals.loc[common].values
        yp = preds.loc[common].values

        r2_series = []
        idx_series = []
        for i in range(window, len(common)):
            yt_w = yt[i - window: i]
            yp_w = yp[i - window: i]
            mask = np.isfinite(yt_w) & np.isfinite(yp_w)
            if mask.sum() < 10:
                r2_series.append(np.nan)
            else:
                r2_series.append(r2_score(yt_w[mask], yp_w[mask]))
            idx_series.append(common[i])

        ax.plot(idx_series, r2_series, label=model_name, linewidth=1.2)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="R² = 0")
    ax.set_title(f"Rolling Out-of-Sample R² (window = {window} days)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("R²")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rolling R² plot saved: %s", os.path.join(results_dir, filename))
