"""
Standalone analysis runner using synthetic data.
Bypasses data_pipeline.py (which requires pandas_datareader/yfinance)
and runs the full EDA, factor, forecasting, evaluation, and portfolio pipeline.
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_analysis")

import config

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[0/8] Generating synthetic panel data …")
np.random.seed(42)

dates = pd.bdate_range(start=config.START_DATE, end=config.END_DATE, freq="B")
T     = len(dates)
logger.info("  %d trading days  (%s – %s)", T, dates[0].date(), dates[-1].date())

K    = 5
N_eq = len(config.EQUITY_TICKERS)
N_etf = len(config.SECTOR_ETFS)
N    = N_eq + N_etf

# ── Factor returns with GARCH-like volatility clustering ─────────────────────
f_vol = np.array([0.18, 0.12, 0.10, 0.08, 0.07]) / np.sqrt(252)
F  = np.zeros((T, K))
h  = np.ones((T, K)) * f_vol**2
for t in range(1, T):
    h[t] = 0.05 * f_vol**2 + 0.10 * F[t-1]**2 + 0.84 * h[t-1]
    F[t] = 0.03 * F[t-1] + np.sqrt(h[t]) * np.random.standard_t(df=6, size=K)

# ── Factor loadings ───────────────────────────────────────────────────────────
np.random.seed(1)
L = np.zeros((N, K))
L[:, 0] = np.random.uniform(0.6, 1.2, N)
L[:, 1] = np.random.uniform(-0.5, 0.5, N)
L[:, 2] = np.random.uniform(-0.4, 0.4, N)
L[:, 3] = np.random.uniform(-0.3, 0.3, N)
L[:, 4] = np.random.uniform(-0.3, 0.3, N)

# Sector ETFs: tighter loadings to sector-specific factors
L[N_eq:, 0] = np.random.uniform(0.8, 1.1, N_etf)   # high market beta

# Idiosyncratic vol
idio_vol = np.random.uniform(0.008, 0.018, N)
eps = np.random.randn(T, N) * idio_vol
R   = F @ L.T + eps

all_tickers = config.EQUITY_TICKERS + config.SECTOR_ETFS
ret_cols    = [t + "_ret" for t in all_tickers]
returns_df  = pd.DataFrame(R, index=dates, columns=ret_cols)

# ── FF5 factors ───────────────────────────────────────────────────────────────
np.random.seed(2)
ff5_df = pd.DataFrame({
    "Mkt-RF": F[:, 0] * 0.8 + np.random.randn(T) * 0.002,
    "SMB":    F[:, 1] * 0.5 + np.random.randn(T) * 0.001,
    "HML":    F[:, 2] * 0.5 + np.random.randn(T) * 0.001,
    "RMW":    F[:, 3] * 0.4 + np.random.randn(T) * 0.001,
    "CMA":    F[:, 4] * 0.4 + np.random.randn(T) * 0.001,
    "RF":     np.random.uniform(0, 0.0001, T),
}, index=dates)

# ── Macro variables ───────────────────────────────────────────────────────────
np.random.seed(3)
vix = np.zeros(T); vix[0] = 18.0
for t in range(1, T):
    vix[t] = 0.97 * vix[t-1] + 0.5 * np.random.randn() + 0.54
vix = np.clip(vix, 9, 80)

yield10 = np.zeros(T); yield10[0] = 2.5
for t in range(1, T):
    yield10[t] = 0.998 * yield10[t-1] + 0.002 * np.random.randn()
yield10 = np.clip(yield10, 0.3, 5.0)

spread = 1.5 + 0.05 * (vix - 18) + np.random.randn(T) * 0.05
spread = np.clip(spread, 0.5, 6.0)

macro_df = pd.DataFrame({
    "VIX":          vix,
    "Yield10":      yield10,
    "CreditSpread": spread,
}, index=dates)

# ── Full panel ────────────────────────────────────────────────────────────────
panel = pd.concat([returns_df, ff5_df, macro_df], axis=1)
panel.index.name = "Date"
logger.info("  Panel shape: %s", panel.shape)

# ── Save parquet ──────────────────────────────────────────────────────────────
panel.to_parquet(os.path.join("data", config.PANEL_FILE))
logger.info("  Panel saved to data/%s", config.PANEL_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EDA
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[2/8] Running EDA …")
from src.eda import (
    plot_correlation_heatmap, plot_eigenvalue_scree,
    plot_cumulative_variance, summary_stats, print_factor_loadings,
)

plot_correlation_heatmap(returns_df, results_dir=RESULTS_DIR)
plot_eigenvalue_scree(returns_df, n_components=20, results_dir=RESULTS_DIR)
plot_cumulative_variance(returns_df, n_components=20, results_dir=RESULTS_DIR)

stats = summary_stats(returns_df)
stats.to_csv(os.path.join(RESULTS_DIR, "summary_stats.csv"))
logger.info("  Summary stats saved.")

# ── Extra EDA: macro time series ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, col, color in zip(axes, ["VIX", "Yield10", "CreditSpread"],
                          ["crimson", "steelblue", "darkorange"]):
    ax.plot(macro_df.index, macro_df[col], color=color, linewidth=0.9)
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
axes[0].set_title("Synthetic Macro Variables (2015–2023)")
axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "macro_series.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Return distribution plot ──────────────────────────────────────────────────
from scipy import stats as scs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ew_ret = returns_df.mean(axis=1)
axes[0].hist(ew_ret, bins=80, color="steelblue", edgecolor="white", linewidth=0.3, density=True)
x = np.linspace(ew_ret.min(), ew_ret.max(), 300)
axes[0].plot(x, scs.norm.pdf(x, ew_ret.mean(), ew_ret.std()), "r--", label="Normal fit")
axes[0].set_title("Equal-Weighted Portfolio Return Distribution")
axes[0].set_xlabel("Daily Return"); axes[0].legend()

axes[1].plot(ew_ret.index, ew_ret.rolling(21).std() * np.sqrt(252), color="darkorange", linewidth=0.9)
axes[1].set_title("Rolling 21-Day Annualised Volatility (EW Portfolio)")
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Annualised Vol")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "return_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Latent Factor Model
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[3/8] Fitting latent factor model (PCA + Bai-Ng) …")
from src.factor_models import LatentFactorModel, bai_ng_criterion

bn = bai_ng_criterion(returns_df, max_factors=config.MAX_FACTORS, results_dir=RESULTS_DIR)
logger.info("  Bai-Ng: IC1=%d, IC2=%d", bn["IC1"], bn["IC2"])

lfm = LatentFactorModel(n_factors=config.N_FACTORS)
lfm.fit(returns_df)
latent_factors = lfm.transform(returns_df)

loadings_df = print_factor_loadings(lfm.pca, lfm.feature_names, n_factors=config.N_FACTORS)

# Plot latent factor returns
fig, axes = plt.subplots(config.N_FACTORS, 1, figsize=(14, 3 * config.N_FACTORS), sharex=True)
colors = ["steelblue", "darkorange", "green", "crimson", "purple"]
for i, col in enumerate(latent_factors.columns):
    axes[i].plot(latent_factors.index, latent_factors[col],
                 color=colors[i], linewidth=0.8, alpha=0.85)
    axes[i].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[i].set_ylabel(col)
    axes[i].grid(alpha=0.25)
    ev = lfm.explained_variance_ratio_[i] * 100
    axes[i].set_title(f"PC{i+1}  (explains {ev:.1f}% of variance)", fontsize=10)
axes[-1].set_xlabel("Date")
fig.suptitle("Latent Factor Returns (PCA)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "latent_factor_returns.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# Factor loadings heatmap
import seaborn as sns
fig, ax = plt.subplots(figsize=(8, max(6, len(loadings_df) * 0.2)))
sns.heatmap(loadings_df, cmap="RdBu_r", center=0, linewidths=0.2,
            yticklabels=True, ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title("PCA Factor Loadings (all assets)", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "factor_loadings_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Observed Factor Model
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[4/8] Fitting observed factor model (FF5 + macro) …")
from src.factor_models import ObservedFactorModel

observed_factors = panel[["Mkt-RF","SMB","HML","RMW","CMA","VIX","Yield10","CreditSpread"]].copy()

ofm = ObservedFactorModel(add_constant=True)
ofm.fit(returns_df, observed_factors)
betas = ofm.get_factor_exposures()

# Beta distribution plots
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "VIX", "Yield10", "CreditSpread"]
for ax, fname in zip(axes.flat, factor_names):
    if fname in betas.index:
        b = betas.loc[fname]
        ax.hist(b, bins=25, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(b.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"μ={b.mean():.3f}")
        ax.set_title(fname); ax.legend(fontsize=8)
        ax.set_xlabel("Beta")
fig.suptitle("Cross-Sectional Beta Distributions (FF5 + Macro)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "observed_factor_betas_dist.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Forecasting Experiments
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[5/8] Running forecasting experiments …")
from src.regression_models import run_forecasting_experiment, build_feature_matrix

forecast_results = run_forecasting_experiment(
    panel_df         = panel,
    latent_factors   = latent_factors,
    observed_factors = observed_factors,
    macro_df         = macro_df,
    train_ratio      = config.TRAIN_RATIO,
    val_ratio        = config.VAL_RATIO,
    lags             = config.FEATURE_LAGS,
    cv_folds         = 5,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Rolling Factor Forecaster
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[6/8] Running rolling factor forecaster …")
from src.dynamic_models import RollingFactorForecaster, KalmanFactorModel

rff = RollingFactorForecaster(
    window=config.ROLLING_WINDOW, n_factors=config.N_FACTORS,
    model_type="ridge", lags=[1, 5],
    step=5,   # refit every 5 days for speed
)
rolling_preds, rolling_actuals = rff.fit_predict(returns_df, macro_df)
logger.info("  %d rolling predictions generated.", len(rolling_preds))

# Kalman smoothing
kf = KalmanFactorModel(n_iter=15)
kf.fit(latent_factors)
smoothed = kf.smooth()

fig, axes = plt.subplots(config.N_FACTORS, 1, figsize=(14, 3 * config.N_FACTORS), sharex=True)
for i, col in enumerate(latent_factors.columns):
    axes[i].plot(latent_factors.index, latent_factors[col],
                 alpha=0.45, color="steelblue", linewidth=0.7, label="Raw PCA")
    axes[i].plot(smoothed.index, smoothed[col],
                 color="darkorange", linewidth=1.5, label="Kalman Smoothed")
    axes[i].set_ylabel(col); axes[i].legend(fontsize=8); axes[i].grid(alpha=0.25)
axes[-1].set_xlabel("Date")
fig.suptitle("Kalman Filter Smoothing of Latent Factors", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "kalman_smoothed_factors.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[7/8] Evaluating models …")
from src.evaluation import (
    compute_metrics, compare_models_table,
    plot_predictions_vs_actuals, diebold_mariano_test, plot_rolling_r2,
)

# Add rolling model to results
if len(rolling_preds) > 0:
    rm = compute_metrics(rolling_actuals.values, rolling_preds.values)
    forecast_results["rolling_ridge"] = {
        "train": {k: np.nan for k in rm},
        "val":   {k: np.nan for k in rm},
        "test":  rm,
    }

print("\n" + "="*65)
print("  FULL MODEL COMPARISON")
print("="*65)
for split in ("train", "val", "test"):
    compare_models_table(forecast_results, split=split)

# Prediction plot
if len(rolling_preds) > 0:
    plot_predictions_vs_actuals(
        rolling_actuals, rolling_preds,
        title="Rolling Ridge: Predictions vs Actuals",
        results_dir=RESULTS_DIR,
        filename="rolling_predictions_vs_actuals.png",
    )

    dm = diebold_mariano_test(
        (rolling_actuals - rolling_preds).values,
        rolling_actuals.values,
    )
    logger.info("  DM test (rolling vs naïve): stat=%.4f, p=%.4f", dm["dm_stat"], dm["p_value"])

    plot_rolling_r2(
        {"Rolling Ridge": rolling_preds},
        rolling_actuals,
        window=63, results_dir=RESULTS_DIR,
    )

# ── Scatter: predicted vs actual (test window) ────────────────────────────────
if len(rolling_preds) > 10:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rolling_actuals, rolling_preds, alpha=0.25, s=8, color="steelblue")
    mn = min(rolling_actuals.min(), rolling_preds.min())
    mx = max(rolling_actuals.max(), rolling_preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect forecast")
    ax.set_xlabel("Actual Return"); ax.set_ylabel("Predicted Return")
    ax.set_title("Rolling Ridge: Predicted vs Actual (scatter)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "predicted_vs_actual_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Portfolios
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("[8/8] Running portfolio backtests …")
from src.portfolio import (
    mean_variance_portfolio, factor_mimicking_portfolio,
    backtest_portfolio, compare_portfolios,
)

latent_w  = factor_mimicking_portfolio(returns_df, latent_factors)
obs_w     = factor_mimicking_portfolio(returns_df, observed_factors[["Mkt-RF","SMB","HML","RMW","CMA"]])

lw = latent_w.iloc[:, 0]
ow = obs_w.iloc[:, 0] if not obs_w.empty else pd.Series(
    np.ones(len(returns_df.columns)) / len(returns_df.columns), index=returns_df.columns)

ew_weights = pd.Series(
    np.ones(len(returns_df.columns)) / len(returns_df.columns),
    index=returns_df.columns,
)

# Mean-variance
sample_ret = returns_df.iloc[-config.ROLLING_WINDOW:]
mv_weights = mean_variance_portfolio(sample_ret.mean(), sample_ret.cov(), risk_aversion=1.0)

# Portfolio comparison
summary = compare_portfolios(lw, ow, ew_weights, returns_df, results_dir=RESULTS_DIR)

# Add MV portfolio to comparison plot
mv_result = backtest_portfolio(mv_weights, returns_df)
lf_result = backtest_portfolio(lw, returns_df)
ow_result = backtest_portfolio(ow, returns_df)
ew_result = backtest_portfolio(ew_weights, returns_df)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

port_dict = {
    "Latent Factor (F1)": lf_result,
    "Observed FF5 (Mkt)": ow_result,
    "Equal Weight":       ew_result,
    "Mean-Variance":      mv_result,
}
colors_p = ["steelblue", "darkorange", "green", "crimson"]

for (name, res), col in zip(port_dict.items(), colors_p):
    axes[0].plot(res["cumulative_return"].index, res["cumulative_return"].values,
                 label=name, linewidth=1.5, color=col)

axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[0].set_title("Portfolio Cumulative Log Returns (2015–2023)", fontsize=12)
axes[0].set_ylabel("Cumulative Log Return"); axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Drawdown plot for EW
wealth_ew = np.exp(ew_result["cumulative_return"])
dd_ew = (wealth_ew - wealth_ew.cummax()) / wealth_ew.cummax()
axes[1].fill_between(dd_ew.index, dd_ew.values, 0, color="crimson", alpha=0.5)
axes[1].set_ylabel("Drawdown"); axes[1].set_xlabel("Date")
axes[1].set_title("Equal-Weight Portfolio Drawdown", fontsize=10)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "portfolio_full_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Metrics summary table ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PORTFOLIO PERFORMANCE SUMMARY")
print("="*65)
rows = {}
for name, res in port_dict.items():
    rows[name] = {
        "Ann. Return (%)":  round(res["annualised_return"] * 100, 2),
        "Sharpe":           round(res["sharpe"],            3),
        "Max Drawdown (%)": round(res["max_drawdown"] * 100, 2),
        "Calmar":           round(res["calmar"],            3),
    }
perf_table = pd.DataFrame(rows).T
print(perf_table.to_string())
perf_table.to_csv(os.path.join(RESULTS_DIR, "portfolio_metrics.csv"))

# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  All plots saved to:", os.path.abspath(RESULTS_DIR))
print("="*65)
print("  Files generated:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"    {f}")
