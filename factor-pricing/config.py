"""
Configuration settings for the factor-pricing research project.
"""

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = "2023-12-31"

# ── 50 large-cap US equity tickers (diversified across sectors) ───────────────
EQUITY_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB", "PNC",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    # Industrials
    "CAT", "BA", "HON", "GE", "MMM", "UPS", "RTX", "LMT", "DE", "EMR",
    # Consumer Discretionary
    "HD", "MCD", "NKE", "SBUX", "TGT",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST",
]

# ── 10 Sector ETFs ────────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]

# ── Factor model settings ─────────────────────────────────────────────────────
N_FACTORS      = 5       # number of PCA factors to extract
MAX_FACTORS    = 15      # upper bound for Bai-Ng criterion search

# ── Train / validation / test split ──────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (0.15)

# ── Rolling window (trading days) ────────────────────────────────────────────
ROLLING_WINDOW = 252

# ── FRED series ──────────────────────────────────────────────────────────────
FRED_SERIES = {
    "VIX":    "VIXCLS",
    "Yield10": "DGS10",
    "CreditSpread": "BAA10YM",
}

# ── Fama-French 5-factor daily data URL ──────────────────────────────────────
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

# ── Data quality thresholds ───────────────────────────────────────────────────
MAX_MISSING_FRAC  = 0.20   # drop columns with > 20 % missing values
MAX_FFILL_DAYS    = 5      # forward-fill up to this many days

# ── Forecasting lags ─────────────────────────────────────────────────────────
FEATURE_LAGS = [1, 5, 21]

# ── Portfolio optimisation ────────────────────────────────────────────────────
RISK_AVERSION = 1.0

# ── Paths (overridden by argparse in main.py) ─────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
PANEL_FILE  = "panel_data.parquet"
