"""
Exploratory data analysis utilities.

Provides correlation heatmaps, PCA scree plots, summary statistics,
and factor loading displays.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _save_fig(fig: plt.Figure, filepath: str) -> None:
    """Save figure with tight layout and close it."""
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    results_dir: str = "results",
    filename: str = "correlation_heatmap.png",
    figsize: tuple = (16, 14),
) -> None:
    """
    Plot and save a full correlation matrix heatmap for the return series.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns (rows = dates, columns = assets).
    results_dir : str
        Directory where the figure will be saved.
    filename : str
        Output filename.
    figsize : tuple
        Figure size in inches (width, height).
    """
    _ensure_dir(results_dir)
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Asset Return Correlation Matrix", fontsize=14, pad=12)
    _save_fig(fig, os.path.join(results_dir, filename))


# ─────────────────────────────────────────────────────────────────────────────
# PCA scree / cumulative variance
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_scree(
    returns_df: pd.DataFrame,
    n_components: int = 20,
    results_dir: str = "results",
    filename: str = "eigenvalue_scree.png",
) -> None:
    """
    Plot a PCA eigenvalue scree chart.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns.
    n_components : int
        Number of principal components to display.
    results_dir : str
        Output directory.
    filename : str
        Output filename.
    """
    _ensure_dir(results_dir)
    clean = returns_df.dropna(axis=1, how="any").dropna()
    n_components = min(n_components, clean.shape[1], clean.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(clean)

    eigenvalues = pca.explained_variance_
    idx = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, eigenvalues, "bo-", markersize=6, linewidth=1.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="λ = 1")
    ax.set_xlabel("Component Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    ax.set_xticks(idx)
    _save_fig(fig, os.path.join(results_dir, filename))


def plot_cumulative_variance(
    returns_df: pd.DataFrame,
    n_components: int = 20,
    results_dir: str = "results",
    filename: str = "cumulative_variance.png",
) -> None:
    """
    Plot cumulative explained variance ratio from PCA.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of asset returns.
    n_components : int
        Number of components to include.
    results_dir : str
        Output directory.
    filename : str
        Output filename.
    """
    _ensure_dir(results_dir)
    clean = returns_df.dropna(axis=1, how="any").dropna()
    n_components = min(n_components, clean.shape[1], clean.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(clean)

    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    idx = np.arange(1, len(cum_var) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, cum_var, "gs-", markersize=6, linewidth=1.5)
    ax.axhline(80, color="orange", linestyle="--", linewidth=1, label="80 % threshold")
    ax.axhline(95, color="red",    linestyle="--", linewidth=1, label="95 % threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xticks(idx)
    ax.legend()
    _save_fig(fig, os.path.join(results_dir, filename))


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset summary statistics.

    Calculates annualised mean return, annualised standard deviation,
    skewness, excess kurtosis, and annualised Sharpe ratio (assuming
    zero risk-free rate).

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of daily log returns.

    Returns
    -------
    pd.DataFrame
        Summary statistics with one row per asset.
    """
    from scipy.stats import skew, kurtosis

    stats = pd.DataFrame(index=returns_df.columns)
    stats["mean_ann"]  = returns_df.mean() * 252
    stats["std_ann"]   = returns_df.std()  * np.sqrt(252)
    stats["skewness"]  = returns_df.apply(lambda s: skew(s.dropna()))
    stats["kurtosis"]  = returns_df.apply(lambda s: kurtosis(s.dropna()))
    stats["sharpe"]    = stats["mean_ann"] / stats["std_ann"]
    stats["n_obs"]     = returns_df.count()
    return stats.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Factor loadings printer
# ─────────────────────────────────────────────────────────────────────────────

def print_factor_loadings(
    pca_model: PCA,
    feature_names: list,
    n_factors: int = 5,
) -> pd.DataFrame:
    """
    Print and return a formatted DataFrame of PCA factor loadings.

    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        A fitted PCA model.
    feature_names : list of str
        Names of the input features (assets).
    n_factors : int
        Number of factors to display.

    Returns
    -------
    pd.DataFrame
        Loadings DataFrame (rows = assets, columns = PC1 … PCk).
    """
    n_show = min(n_factors, pca_model.n_components_)
    cols   = [f"PC{i+1}" for i in range(n_show)]
    loadings = pd.DataFrame(
        pca_model.components_[:n_show].T,
        index=feature_names,
        columns=cols,
    )
    print("\n── PCA Factor Loadings (top assets by |loading|) ──────────────────")
    for col in cols:
        top = loadings[col].abs().nlargest(5).index.tolist()
        print(f"  {col}: {top}")
    print(loadings.round(4).to_string())
    return loadings
