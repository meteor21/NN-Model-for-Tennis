"""
main.py — Full pipeline orchestration for the factor-pricing project.

Usage
-----
Run with fresh data download:
    python main.py --download

Use pre-cached data, custom results directory:
    python main.py --results_dir my_results/

Steps
-----
1.  Load or download and save the panel dataset.
2.  Run exploratory data analysis (save plots).
3.  Fit latent factor model (PCA); run Bai-Ng criterion.
4.  Fit observed factor model (FF5 + macro).
5.  Build feature matrices; run all forecasting experiments.
6.  Run rolling-window forecaster.
7.  Evaluate models; print comparison table.
8.  Run portfolio backtests; plot results.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# ── Make the src/ package importable when running from the project root ────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import config
from src.data_pipeline import (
    build_panel,
    download_equity_data,
    download_etf_data,
    download_ff_factors,
    download_fred_data,
    load_panel,
    save_panel,
)
from src.eda import (
    plot_correlation_heatmap,
    plot_cumulative_variance,
    plot_eigenvalue_scree,
    print_factor_loadings,
    summary_stats,
)
from src.evaluation import (
    compare_models_table,
    compute_metrics,
    diebold_mariano_test,
    plot_predictions_vs_actuals,
    plot_rolling_r2,
)
from src.factor_models import (
    LatentFactorModel,
    ObservedFactorModel,
    bai_ng_criterion,
)
from src.portfolio import (
    backtest_portfolio,
    compare_portfolios,
    factor_mimicking_portfolio,
    mean_variance_portfolio,
)
from src.regression_models import (
    build_feature_matrix,
    run_forecasting_experiment,
)
from src.dynamic_models import RollingFactorForecaster, KalmanFactorModel


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factor pricing research pipeline."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Re-download all data even if a cached panel already exists.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=config.RESULTS_DIR,
        help="Directory for output plots and metrics (default: results/).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help="Directory for raw and processed data (default: data/).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Column selectors
# ─────────────────────────────────────────────────────────────────────────────

def _ret_columns(df: pd.DataFrame) -> pd.Index:
    """Return the equity / ETF return columns (suffix ``_ret``)."""
    return df.columns[df.columns.str.endswith("_ret")]


def _ff5_columns(df: pd.DataFrame) -> list:
    """Return the Fama-French factor column names present in *df*."""
    candidates = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    return [c for c in candidates if c in df.columns]


def _macro_columns(df: pd.DataFrame) -> list:
    """Return the macro variable column names present in *df*."""
    candidates = list(config.FRED_SERIES.keys())
    return [c for c in candidates if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Data
# ─────────────────────────────────────────────────────────────────────────────

def step_data(args: argparse.Namespace) -> pd.DataFrame:
    """Download (or load) and return the merged panel DataFrame."""
    os.makedirs(args.data_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    panel_path = os.path.join(args.data_dir, config.PANEL_FILE)

    if not args.download and os.path.exists(panel_path):
        logger.info("[1/8] Loading cached panel from %s …", panel_path)
        return load_panel(panel_path)

    logger.info("[1/8] Downloading data …")

    equity_prices = download_equity_data(
        config.EQUITY_TICKERS, config.START_DATE, config.END_DATE
    )
    etf_prices = download_etf_data(
        config.SECTOR_ETFS, config.START_DATE, config.END_DATE
    )
    ff_factors = download_ff_factors(config.START_DATE, config.END_DATE)
    macro_df   = download_fred_data(config.START_DATE, config.END_DATE)

    panel = build_panel(
        equity_prices,
        etf_prices,
        ff_factors,
        macro_df,
        max_missing_frac=config.MAX_MISSING_FRAC,
        max_ffill_days=config.MAX_FFILL_DAYS,
    )
    save_panel(panel, panel_path)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – EDA
# ─────────────────────────────────────────────────────────────────────────────

def step_eda(panel: pd.DataFrame, results_dir: str) -> None:
    """Produce and save all EDA charts."""
    logger.info("[2/8] Running EDA …")
    ret_cols = _ret_columns(panel)
    returns  = panel[ret_cols]

    plot_correlation_heatmap(returns, results_dir=results_dir)
    plot_eigenvalue_scree(returns, n_components=20, results_dir=results_dir)
    plot_cumulative_variance(returns, n_components=20, results_dir=results_dir)

    stats = summary_stats(returns)
    stats_path = os.path.join(results_dir, "summary_stats.csv")
    stats.to_csv(stats_path)
    logger.info("Summary statistics saved to %s.", stats_path)
    print("\n── Summary Statistics (first 10 assets) ──────────────────────────")
    print(stats.head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Latent factor model
# ─────────────────────────────────────────────────────────────────────────────

def step_latent_factors(
    panel: pd.DataFrame,
    results_dir: str,
) -> tuple[LatentFactorModel, pd.DataFrame]:
    """Fit PCA, run Bai-Ng, return fitted model and factor returns."""
    logger.info("[3/8] Fitting latent factor model (PCA) …")
    ret_cols = _ret_columns(panel)
    returns  = panel[ret_cols]

    # Bai-Ng criterion
    bn = bai_ng_criterion(
        returns,
        max_factors=config.MAX_FACTORS,
        results_dir=results_dir,
    )
    logger.info("Bai-Ng optimal factors: IC1=%d, IC2=%d", bn["IC1"], bn["IC2"])

    # Fit PCA with configured N_FACTORS
    lfm = LatentFactorModel(n_factors=config.N_FACTORS)
    lfm.fit(returns)

    factor_returns = lfm.transform(returns)
    print_factor_loadings(lfm.pca, lfm.feature_names, n_factors=config.N_FACTORS)

    logger.info(
        "Latent factors: %d factors, cumulative variance = %.1f%%",
        config.N_FACTORS,
        np.sum(lfm.explained_variance_ratio_) * 100,
    )
    return lfm, factor_returns


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Observed factor model
# ─────────────────────────────────────────────────────────────────────────────

def step_observed_factors(
    panel: pd.DataFrame,
) -> tuple[ObservedFactorModel, pd.DataFrame]:
    """Fit FF5 + macro observed factor model."""
    logger.info("[4/8] Fitting observed factor model (FF5 + macro) …")
    ret_cols   = _ret_columns(panel)
    ff5_cols   = _ff5_columns(panel)
    macro_cols = _macro_columns(panel)

    if not ff5_cols:
        logger.warning("No FF5 columns found in panel; observed model may be uninformative.")

    factor_df = panel[ff5_cols + macro_cols].copy()
    returns   = panel[ret_cols]

    ofm = ObservedFactorModel(add_constant=True)
    ofm.fit(returns, factor_df)

    betas = ofm.get_factor_exposures()
    logger.info("Observed factor betas shape: %s", betas.shape)

    # Return the observed factor series for use in forecasting
    return ofm, factor_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Forecasting experiments
# ─────────────────────────────────────────────────────────────────────────────

def step_forecasting(
    panel: pd.DataFrame,
    latent_factors: pd.DataFrame,
    observed_factors: pd.DataFrame,
) -> dict:
    """Build feature matrices and run all comparative forecasting experiments."""
    logger.info("[5/8] Running forecasting experiments …")

    macro_cols = _macro_columns(panel)
    macro_df   = panel[macro_cols] if macro_cols else pd.DataFrame(index=panel.index)

    results = run_forecasting_experiment(
        panel_df         = panel,
        latent_factors   = latent_factors,
        observed_factors = observed_factors,
        macro_df         = macro_df,
        target           = "return",
        train_ratio      = config.TRAIN_RATIO,
        val_ratio        = config.VAL_RATIO,
        lags             = config.FEATURE_LAGS,
        cv_folds         = 5,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Rolling forecaster
# ─────────────────────────────────────────────────────────────────────────────

def step_rolling(
    panel: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Fit rolling PCA + Ridge forecaster; return predictions and actuals."""
    logger.info("[6/8] Running rolling factor forecaster …")

    ret_cols   = _ret_columns(panel)
    macro_cols = _macro_columns(panel)
    returns    = panel[ret_cols]
    macro_df   = panel[macro_cols] if macro_cols else pd.DataFrame(index=panel.index)

    rff = RollingFactorForecaster(
        window     = config.ROLLING_WINDOW,
        n_factors  = config.N_FACTORS,
        model_type = "ridge",
        lags       = [1, 5],
    )
    predictions, actuals = rff.fit_predict(returns, macro_df)
    logger.info("Rolling forecaster: %d predictions.", len(predictions))
    return predictions, actuals


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def step_evaluation(
    forecast_results: dict,
    rolling_preds: pd.Series,
    rolling_actuals: pd.Series,
    results_dir: str,
) -> None:
    """Print model comparison table; plot predictions and rolling R²."""
    logger.info("[7/8] Evaluating models …")

    # Add rolling model metrics to the results dict
    if len(rolling_preds) > 0:
        rolling_metrics = compute_metrics(rolling_actuals.values, rolling_preds.values)
        forecast_results["rolling_ridge"] = {
            "train": {k: np.nan for k in rolling_metrics},
            "val":   {k: np.nan for k in rolling_metrics},
            "test":  rolling_metrics,
        }

    # Comparison tables
    for split in ("train", "val", "test"):
        compare_models_table(forecast_results, split=split)

    # Prediction plots for rolling model
    if len(rolling_preds) > 0:
        plot_predictions_vs_actuals(
            rolling_actuals,
            rolling_preds,
            title    = "Rolling Ridge: Predictions vs Actuals",
            results_dir = results_dir,
            filename = "rolling_predictions_vs_actuals.png",
        )

        # Diebold-Mariano: compare rolling model against naive zero-forecast
        e_model  = (rolling_actuals - rolling_preds).values
        e_naive  = rolling_actuals.values           # naive: predict 0
        dm_res   = diebold_mariano_test(e_model, e_naive)
        logger.info(
            "DM test (rolling vs naïve): stat=%.4f, p=%.4f",
            dm_res["dm_stat"], dm_res["p_value"],
        )

        # Rolling R²
        plot_rolling_r2(
            {"Rolling Ridge": rolling_preds},
            rolling_actuals,
            window      = 63,
            results_dir = results_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – Portfolios
# ─────────────────────────────────────────────────────────────────────────────

def step_portfolios(
    panel: pd.DataFrame,
    lfm: LatentFactorModel,
    latent_factor_returns: pd.DataFrame,
    ofm: ObservedFactorModel,
    observed_factor_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Build and backtest three portfolios; plot comparison."""
    logger.info("[8/8] Running portfolio backtests …")

    ret_cols = _ret_columns(panel)
    returns  = panel[ret_cols]

    # ── Latent factor-mimicking weights ───────────────────────────────────────
    latent_w_matrix  = factor_mimicking_portfolio(returns, latent_factor_returns)
    # Use the first factor's mimicking weights as a "portfolio"
    latent_weights = latent_w_matrix.iloc[:, 0] if not latent_w_matrix.empty else None

    # ── Observed factor-mimicking weights (first FF5 factor, if available) ────
    obs_factor_cols = [c for c in observed_factor_df.columns if c in panel.columns]
    if obs_factor_cols:
        obs_factors_ret = panel[obs_factor_cols]
        obs_w_matrix = factor_mimicking_portfolio(returns, obs_factors_ret)
        obs_weights  = obs_w_matrix.iloc[:, 0] if not obs_w_matrix.empty else None
    else:
        obs_weights = None

    # ── Equal-weight baseline ─────────────────────────────────────────────────
    ew_weights = pd.Series(
        np.ones(len(returns.columns)) / len(returns.columns),
        index=returns.columns,
    )

    # Fallback if factor mimicking failed
    if latent_weights is None or latent_weights.isna().all():
        latent_weights = ew_weights.copy()
        logger.warning("Latent mimicking weights unavailable; using equal weights as fallback.")

    if obs_weights is None or obs_weights.isna().all():
        obs_weights = ew_weights.copy()
        logger.warning("Observed mimicking weights unavailable; using equal weights as fallback.")

    # ── Mean-variance portfolio (uses latent factor returns as "expected return" proxy)
    try:
        sample_ret = returns.iloc[-config.ROLLING_WINDOW:]
        exp_ret    = sample_ret.mean()
        cov_mat    = sample_ret.cov()
        mv_weights = mean_variance_portfolio(exp_ret, cov_mat, risk_aversion=config.RISK_AVERSION)
    except Exception as exc:
        logger.warning("Mean-variance optimisation failed: %s; using equal weights.", exc)
        mv_weights = ew_weights.copy()

    # ── Compare ───────────────────────────────────────────────────────────────
    summary = compare_portfolios(
        latent_factor_weights   = latent_weights,
        observed_factor_weights = obs_weights,
        equal_weights           = ew_weights,
        returns_df              = returns,
        results_dir             = results_dir,
    )

    # Also backtest mean-variance portfolio and print its metrics
    mv_result = backtest_portfolio(mv_weights, returns)
    print("\n── Mean-Variance Portfolio Metrics ───────────────────────────────")
    for k, v in mv_result.items():
        if not isinstance(v, pd.Series):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save summary to CSV
    summary_path = os.path.join(results_dir, "portfolio_summary.csv")
    summary.to_csv(summary_path)
    logger.info("Portfolio summary saved to %s.", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir,    exist_ok=True)

    # Override config paths with CLI arguments
    config.DATA_DIR    = args.data_dir
    config.RESULTS_DIR = args.results_dir

    logger.info("=" * 60)
    logger.info("Factor Pricing Research Pipeline")
    logger.info("Results dir : %s", args.results_dir)
    logger.info("Data dir    : %s", args.data_dir)
    logger.info("=" * 60)

    # 1. Data
    panel = step_data(args)

    # 2. EDA
    step_eda(panel, args.results_dir)

    # 3. Latent factors
    lfm, latent_factors = step_latent_factors(panel, args.results_dir)

    # 4. Observed factors
    ofm, observed_factors = step_observed_factors(panel)

    # 5. Forecasting
    forecast_results = step_forecasting(panel, latent_factors, observed_factors)

    # 6. Rolling forecaster
    rolling_preds, rolling_actuals = step_rolling(panel)

    # 7. Evaluation
    step_evaluation(forecast_results, rolling_preds, rolling_actuals, args.results_dir)

    # 8. Portfolios
    step_portfolios(panel, lfm, latent_factors, ofm, observed_factors, args.results_dir)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Results saved to '%s'.", args.results_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
