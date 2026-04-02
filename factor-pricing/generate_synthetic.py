"""
Generate a realistic synthetic panel dataset and run the full analysis pipeline.

Uses a factor model to simulate correlated equity returns with realistic
statistical properties (fat tails, volatility clustering, macro correlation).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("synthetic")

import config

np.random.seed(42)

# ── Date range ────────────────────────────────────────────────────────────────
dates = pd.bdate_range(start=config.START_DATE, end=config.END_DATE, freq="B")
T = len(dates)
logger.info("Generating %d business days of synthetic data (%s – %s)", T, dates[0].date(), dates[-1].date())

# ── Simulate 5 latent factors (with macro regime) ────────────────────────────
K = 5
N_eq  = len(config.EQUITY_TICKERS)
N_etf = len(config.SECTOR_ETFS)
N     = N_eq + N_etf

# Factor volatilities (annualised → daily)
f_vol = np.array([0.18, 0.12, 0.10, 0.08, 0.07]) / np.sqrt(252)

# Factor returns: AR(1) with GARCH-like clustering
F = np.zeros((T, K))
h = np.ones((T, K))  # conditional variance
for t in range(1, T):
    h[t] = 0.05 * f_vol**2 + 0.10 * F[t-1]**2 + 0.84 * h[t-1]
    F[t] = 0.03 * F[t-1] + np.sqrt(h[t]) * np.random.standard_t(df=6, size=K)

factor_df = pd.DataFrame(F, index=dates, columns=[f"F{i+1}" for i in range(K)])

# ── Factor loadings for each asset ───────────────────────────────────────────
# Draw loadings: first factor = market (all positive); rest = style
np.random.seed(1)
L = np.zeros((N, K))
L[:, 0] = np.random.uniform(0.6, 1.2, N)          # market beta
L[:, 1] = np.random.uniform(-0.5, 0.5, N)          # size
L[:, 2] = np.random.uniform(-0.4, 0.4, N)          # value
L[:, 3] = np.random.uniform(-0.3, 0.3, N)          # profitability
L[:, 4] = np.random.uniform(-0.3, 0.3, N)          # investment

# Idiosyncratic vol (daily)
idio_vol = np.random.uniform(0.008, 0.018, N)

# Returns: R = L @ F' + ε
eps = np.random.randn(T, N) * idio_vol
R   = F @ L.T + eps

all_tickers = config.EQUITY_TICKERS + config.SECTOR_ETFS
ret_cols    = [t + "_ret" for t in all_tickers]
returns_df  = pd.DataFrame(R, index=dates, columns=ret_cols)

# ── Simulate FF5 factors ──────────────────────────────────────────────────────
np.random.seed(2)
mkt_rf = F[:, 0] * 0.8 + np.random.randn(T) * 0.002
smb    = F[:, 1] * 0.5 + np.random.randn(T) * 0.001
hml    = F[:, 2] * 0.5 + np.random.randn(T) * 0.001
rmw    = F[:, 3] * 0.4 + np.random.randn(T) * 0.001
cma    = F[:, 4] * 0.4 + np.random.randn(T) * 0.001
rf     = np.random.uniform(0, 0.0001, T)

ff5_df = pd.DataFrame({
    "Mkt-RF": mkt_rf, "SMB": smb, "HML": hml,
    "RMW": rmw,       "CMA": cma, "RF":  rf,
}, index=dates)

# ── Simulate macro variables ──────────────────────────────────────────────────
np.random.seed(3)
# VIX: mean-reverting around 18
vix_raw   = np.zeros(T)
vix_raw[0] = 18.0
for t in range(1, T):
    vix_raw[t] = 0.97 * vix_raw[t-1] + 0.5 * np.random.randn() + 0.54
vix_raw = np.clip(vix_raw, 9, 80)

# 10Y Yield: slow trend + AR
yield10 = np.zeros(T)
yield10[0] = 2.5
for t in range(1, T):
    yield10[t] = 0.998 * yield10[t-1] + 0.002 * np.random.randn()
yield10 = np.clip(yield10, 0.3, 5.0)

# Credit spread: correlated with VIX
spread = 1.5 + 0.05 * (vix_raw - 18) + np.random.randn(T) * 0.05
spread = np.clip(spread, 0.5, 6.0)

macro_df = pd.DataFrame({
    "VIX":          vix_raw,
    "Yield10":      yield10,
    "CreditSpread": spread,
}, index=dates)

# ── Assemble panel ────────────────────────────────────────────────────────────
panel = pd.concat([returns_df, ff5_df, macro_df], axis=1)
panel.index.name = "Date"
panel.sort_index(inplace=True)

os.makedirs("data", exist_ok=True)
from src.data_pipeline import save_panel
save_panel(panel, os.path.join("data", config.PANEL_FILE))

logger.info("Synthetic panel saved: %d rows × %d cols", panel.shape[0], panel.shape[1])
print("\nColumn groups:")
print(f"  Returns:  {len(ret_cols)} columns")
print(f"  FF5:      {ff5_df.shape[1]} columns")
print(f"  Macro:    {macro_df.shape[1]} columns")
print(f"  Total:    {panel.shape[1]} columns")
print(panel.describe().round(4).to_string())
