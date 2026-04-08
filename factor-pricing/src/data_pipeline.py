"""
Data acquisition and panel-building pipeline.

Downloads equity prices, sector ETFs, Fama-French 5 factors, and FRED macro
series; merges them into a single clean panel of log returns.
"""

import io
import logging
import os
import zipfile
from typing import List

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Equity / ETF downloads
# ─────────────────────────────────────────────────────────────────────────────

def download_equity_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for a list of equity tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker containing
        adjusted close prices.
    """
    logger.info("Downloading equity data for %d tickers (%s – %s)…", len(tickers), start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    missing_pct = prices.isna().mean()
    bad = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not bad.empty:
        logger.warning("Tickers with missing data:\n%s", bad.to_string())
    return prices


def download_etf_data(etf_tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for a list of sector ETF tickers.

    Parameters
    ----------
    etf_tickers : list of str
        ETF ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ETF.
    """
    logger.info("Downloading ETF data for %d tickers (%s – %s)…", len(etf_tickers), start, end)
    return download_equity_data(etf_tickers, start, end)


# ─────────────────────────────────────────────────────────────────────────────
# Fama-French 5 factors
# ─────────────────────────────────────────────────────────────────────────────

def download_ff_factors(start: str, end: str, url: str = None) -> pd.DataFrame:
    """
    Fetch Fama-French 5-factor daily data from Ken French's Data Library.

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    url : str, optional
        Direct URL to the ZIP file. Defaults to the standard FF5 daily URL.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex DataFrame with columns Mkt-RF, SMB, HML, RMW, CMA, RF
        (values expressed as decimals, i.e. divided by 100).
    """
    from config import FF5_URL
    target_url = url or FF5_URL
    logger.info("Downloading Fama-French 5 factors from %s …", target_url)

    response = requests.get(target_url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            content = f.read().decode("utf-8", errors="replace")

    # Find where the data header starts (skip the text preamble)
    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Mkt-RF") or (
            "," in line and "Mkt" in line
        ):
            header_idx = i
            break
    if header_idx is None:
        # Fallback: skip lines that don't start with a digit
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                header_idx = i - 1
                break

    csv_text = "\n".join(lines[header_idx:])
    ff = pd.read_csv(io.StringIO(csv_text), index_col=0)
    ff.index = ff.index.astype(str).str.strip()

    # Remove footer (annual averages block that starts again after a blank line)
    valid_mask = ff.index.str.match(r"^\d{8}$")
    ff = ff.loc[valid_mask]
    ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
    ff.index.name = "Date"
    ff = ff.apply(pd.to_numeric, errors="coerce") / 100.0
    ff.columns = [c.strip() for c in ff.columns]

    mask = (ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))
    ff = ff.loc[mask].sort_index()
    logger.info("FF5 factors: %d observations, columns: %s", len(ff), list(ff.columns))
    return ff


# ─────────────────────────────────────────────────────────────────────────────
# FRED macro data
# ─────────────────────────────────────────────────────────────────────────────

def download_fred_data(
    start: str,
    end: str,
    series: dict = None,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Fetch macro series from FRED.

    Downloads VIX (VIXCLS), 10-year Treasury yield (DGS10), and the
    Moody's BAA–10Y credit spread (BAA10YM) by default.

    Uses the FRED REST API directly when an *api_key* is provided (or the
    ``FRED_API_KEY`` environment variable is set), which avoids the broken
    ``pandas_datareader`` installation on some systems.  Falls back to
    ``pandas_datareader`` otherwise.

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    series : dict, optional
        Mapping of friendly name → FRED series ID. Defaults to config.FRED_SERIES.
    api_key : str, optional
        FRED API key.  If ``None``, reads ``FRED_API_KEY`` env variable.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex DataFrame with one column per macro series.
    """
    import os
    from config import FRED_SERIES
    target_series = series or FRED_SERIES

    # Resolve API key: argument > env var
    key = api_key or os.environ.get("FRED_API_KEY", "")
    logger.info("Downloading FRED series: %s …", list(target_series.keys()))

    frames = {}

    if key:
        # ── Direct FRED REST API (no pandas_datareader needed) ────────────────
        base = "https://api.stlouisfed.org/fred/series/observations"
        for name, sid in target_series.items():
            try:
                params = {
                    "series_id":        sid,
                    "observation_start": start,
                    "observation_end":   end,
                    "api_key":           key,
                    "file_type":        "json",
                }
                resp = requests.get(base, params=params, timeout=30)
                resp.raise_for_status()
                obs  = resp.json()["observations"]
                s    = pd.Series(
                    {o["date"]: float(o["value"]) if o["value"] != "." else float("nan")
                     for o in obs},
                    name=name,
                )
                s.index = pd.to_datetime(s.index)
                frames[name] = s
                logger.info("  %s (%s): %d obs via FRED API", name, sid, s.notna().sum())
            except Exception as exc:
                logger.warning("  FRED API failed for %s (%s): %s", name, sid, exc)
    else:
        # ── pandas_datareader fallback ────────────────────────────────────────
        for name, sid in target_series.items():
            try:
                s = web.DataReader(sid, "fred", start=start, end=end)
                s.columns = [name]
                frames[name] = s[name]
                logger.info("  %s (%s): %d obs via datareader", name, sid, s[name].notna().sum())
            except Exception as exc:
                logger.warning("  Failed to download %s (%s): %s", name, sid, exc)

    if not frames:
        raise RuntimeError("Could not download any FRED series.")

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# Panel builder
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(
    equity_prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    ff_factors: pd.DataFrame,
    macro_df: pd.DataFrame,
    max_missing_frac: float = 0.20,
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """
    Merge all data sources into a single clean panel.

    Steps:
    1. Compute log returns for equities and ETFs.
    2. Align all frames on the intersection of trading dates.
    3. Forward-fill macro / FF factor series up to *max_ffill_days* days.
    4. Drop columns with missing fraction > *max_missing_frac*.
    5. Drop remaining rows with any NaN.

    Parameters
    ----------
    equity_prices : pd.DataFrame
        Adjusted close prices for equities.
    etf_prices : pd.DataFrame
        Adjusted close prices for ETFs.
    ff_factors : pd.DataFrame
        Fama-French 5 factor returns (decimals).
    macro_df : pd.DataFrame
        FRED macro series (levels, not returns).
    max_missing_frac : float
        Column-level missing threshold above which a column is dropped.
    max_ffill_days : int
        Maximum consecutive days to forward-fill.

    Returns
    -------
    pd.DataFrame
        Clean panel with DatetimeIndex. Columns include:
        - ``<ticker>_ret``   : daily log return for each equity / ETF
        - FF5 factor columns : Mkt-RF, SMB, HML, RMW, CMA, RF
        - macro columns      : VIX, Yield10, CreditSpread
    """
    logger.info("Building panel…")

    # Log returns for prices
    equity_ret = np.log(equity_prices / equity_prices.shift(1)).add_suffix("_ret")
    etf_ret    = np.log(etf_prices   / etf_prices.shift(1)).add_suffix("_ret")

    # Drop the very first row (NaN from log-return)
    equity_ret = equity_ret.iloc[1:]
    etf_ret    = etf_ret.iloc[1:]

    # Forward-fill FF5 and macro (they may have gaps on non-business days)
    ff_filled    = ff_factors.ffill(limit=max_ffill_days)
    macro_filled = macro_df.ffill(limit=max_ffill_days)

    # Align on equity trading dates (inner join to keep only common dates)
    combined = (
        equity_ret
        .join(etf_ret,    how="inner")
        .join(ff_filled,  how="left")
        .join(macro_filled, how="left")
    )
    combined.index = pd.to_datetime(combined.index)
    combined.sort_index(inplace=True)

    total_rows = len(combined)
    logger.info("Combined panel before cleaning: %d rows × %d cols", total_rows, combined.shape[1])

    # Drop columns with too many missing values
    missing_frac = combined.isna().mean()
    bad_cols = missing_frac[missing_frac > max_missing_frac].index.tolist()
    if bad_cols:
        logger.warning("Dropping %d column(s) with >%.0f%% missing: %s",
                       len(bad_cols), max_missing_frac * 100, bad_cols)
        combined.drop(columns=bad_cols, inplace=True)

    # Report remaining missing values before dropping rows
    remaining_na = combined.isna().sum().sum()
    if remaining_na > 0:
        logger.info("Dropping rows with remaining NaN values (%d cells affected).", remaining_na)
        rows_before = len(combined)
        combined.dropna(inplace=True)
        logger.info("Dropped %d rows; %d rows remain.", rows_before - len(combined), len(combined))

    logger.info("Final panel: %d rows × %d cols  (%s – %s)",
                len(combined), combined.shape[1],
                combined.index[0].date(), combined.index[-1].date())
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_panel(df: pd.DataFrame, path: str) -> None:
    """
    Save the panel DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Panel to save.
    path : str
        Destination file path (will create parent directories as needed).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path)
    logger.info("Panel saved to %s  (%d rows × %d cols).", path, len(df), df.shape[1])


def load_panel(path: str) -> pd.DataFrame:
    """
    Load a panel DataFrame from a Parquet file.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    pd.DataFrame
        Loaded panel with DatetimeIndex.
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    logger.info("Panel loaded from %s  (%d rows × %d cols).", path, len(df), df.shape[1])
    return df
