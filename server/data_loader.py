"""
Share-Forge Data Loader.

Fetches Tata Gold ETF (TATAGOLD.NS) historical bars via yfinance and computes
technical indicators. Caches the full dataset to CSV under server/data/.

Data is split at TRAIN_CUTOFF_DATE (2026-03-31, inclusive):
  - load()       returns only bars on or before TRAIN_CUTOFF_DATE
                 ──► used by the trading env, the forecaster, and inference.
  - load_live()  returns only bars on or after LIVE_START_DATE (2026-04-01)
                 ──► exposed on a separate API route. The model never receives
                     these rows during training or single-step inference.

Run as a script to pre-populate the cache:
    python -m server.data_loader
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TICKER = "TATAGOLD.NS"
CACHE_PATH = DATA_DIR / f"{TICKER}.full.csv"
DEFAULT_START = "2015-01-01"

TRAIN_CUTOFF_DATE = pd.Timestamp("2026-03-31")
LIVE_START_DATE = pd.Timestamp("2026-04-01")

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_10", "sma_20", "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "ret_1d",
]


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=1).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(window=n, min_periods=1).mean()
    std = close.rolling(window=n, min_periods=1).std().fillna(0.0)
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]

    df["sma_10"] = _sma(df["close"], 10)
    df["sma_20"] = _sma(df["close"], 20)
    df["ema_12"] = _ema(df["close"], 12)
    df["ema_26"] = _ema(df["close"], 26)
    df["rsi_14"] = _rsi(df["close"], 14)
    macd_line, macd_signal, macd_hist = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    bb_u, bb_m, bb_l = _bollinger(df["close"], 20, 2.0)
    df["bb_upper"] = bb_u
    df["bb_mid"] = bb_m
    df["bb_lower"] = bb_l
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)

    return df.bfill().ffill()


def fetch_raw(ticker: str = TICKER, start: str = DEFAULT_START, end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]
    return df


def _load_full(force_refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force_refresh:
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        return df

    raw = fetch_raw(TICKER)
    enriched = add_indicators(raw)
    cols = ["date"] + FEATURE_COLUMNS
    enriched = enriched[[c for c in cols if c in enriched.columns]]
    enriched["date"] = pd.to_datetime(enriched["date"])
    enriched.to_csv(CACHE_PATH, index=False)
    return enriched


def load(force_refresh: bool = False) -> pd.DataFrame:
    """Training/inference data: every bar with date <= TRAIN_CUTOFF_DATE."""
    df = _load_full(force_refresh)
    df = df[df["date"] <= TRAIN_CUTOFF_DATE].reset_index(drop=True)
    if not df.empty:
        assert df["date"].max() <= TRAIN_CUTOFF_DATE, "training data leaked past cutoff"
    return df


def load_live(force_refresh: bool = False) -> pd.DataFrame:
    """Holdout data: every bar with date >= LIVE_START_DATE. Never given to the model."""
    df = _load_full(force_refresh)
    df = df[df["date"] >= LIVE_START_DATE].reset_index(drop=True)
    return df


def cutoff_summary() -> dict:
    """Quick summary of how many bars sit on each side of the cutoff."""
    try:
        full = _load_full(force_refresh=False)
    except Exception:
        return {
            "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
            "live_start": str(LIVE_START_DATE.date()),
            "train_rows": 0,
            "live_rows": 0,
            "train_last_date": None,
            "live_last_date": None,
            "data_cached": False,
        }
    train = full[full["date"] <= TRAIN_CUTOFF_DATE]
    live = full[full["date"] >= LIVE_START_DATE]
    return {
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "live_start": str(LIVE_START_DATE.date()),
        "train_rows": int(len(train)),
        "live_rows": int(len(live)),
        "train_last_date": str(train["date"].max().date()) if not train.empty else None,
        "live_last_date": str(live["date"].max().date()) if not live.empty else None,
        "data_cached": True,
    }


def slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    out = df.loc[mask].reset_index(drop=True)
    if len(out) < 30:
        raise ValueError(f"Slice [{start}, {end}] returned only {len(out)} rows")
    return out


if __name__ == "__main__":
    print(f"Fetching {TICKER}...")
    full = _load_full(force_refresh=True)
    summary = cutoff_summary()
    print(f"Cached {len(full)} total rows to {CACHE_PATH}")
    print(f"Training rows (<= {summary['train_cutoff']}): {summary['train_rows']} (last: {summary['train_last_date']})")
    print(f"Live rows     (>= {summary['live_start']}): {summary['live_rows']} (last: {summary['live_last_date']})")
