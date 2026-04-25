"""
Share-Forge Price Forecaster.

Two methods available:

  - "gbm" : Geometric Brownian Motion Monte Carlo (default fallback).
  - "ml"  : LSTM regressor trained by `train_forecaster.py`. Produces
            mean + log-std of next-K-day cumulative log return; we Monte
            Carlo from that distribution to get the same percentile bands
            the frontend expects.

Both methods read drift/calibration/inputs only from the training-cutoff
dataset — the forecaster is never given post-cutoff bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from server.data_loader import load

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
ML_FORECASTER_CKPT = CHECKPOINT_DIR / "forecaster.pth"
ML_FORECASTER_STATS = CHECKPOINT_DIR / "forecaster_stats.npz"


HORIZONS: Dict[str, int] = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


@dataclass
class ForecastResult:
    horizon_days: int
    method: str
    mu: float
    sigma: float
    last_close: float
    last_date: str
    history_dates: List[str]
    history_close: List[float]
    forecast_dates: List[str]
    median: List[float]
    p05: List[float]
    p25: List[float]
    p75: List[float]
    p95: List[float]

    def to_dict(self) -> dict:
        return {
            "horizon_days": self.horizon_days,
            "method": self.method,
            "mu_daily": self.mu,
            "sigma_daily": self.sigma,
            "last_close": self.last_close,
            "last_date": self.last_date,
            "history": {
                "dates": self.history_dates,
                "close": self.history_close,
            },
            "forecast": {
                "dates": self.forecast_dates,
                "median": self.median,
                "p05": self.p05,
                "p25": self.p25,
                "p75": self.p75,
                "p95": self.p95,
            },
        }


def _next_trading_dates(start: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    """Skip Saturdays and Sundays — close enough for a daily NSE bar projection."""
    dates: List[pd.Timestamp] = []
    cur = start
    while len(dates) < n:
        cur = cur + pd.Timedelta(days=1)
        if cur.weekday() < 5:
            dates.append(cur)
    return dates


def gbm_forecast(
    horizon_days: int,
    n_simulations: int = 1000,
    lookback_days: int = 120,
    history_lookback: int = 252,
    seed: Optional[int] = 42,
) -> ForecastResult:
    """
    Geometric Brownian Motion forecast.

    Calibrated on log returns of the last `lookback_days` bars of the training
    dataset. Returns history (last `history_lookback` bars) plus forward paths.
    """
    df = load()
    if df.empty:
        raise RuntimeError("Training dataset is empty — run `python -m server.data_loader` first")

    closes = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"])

    if len(closes) < max(lookback_days, history_lookback) + 2:
        lookback_days = min(lookback_days, max(len(closes) - 2, 5))
        history_lookback = min(history_lookback, len(closes))

    log_rets = np.diff(np.log(np.maximum(closes, 1e-8)))
    window = log_rets[-lookback_days:]
    mu = float(window.mean())
    sigma = float(window.std(ddof=1)) if window.size > 1 else 0.01
    sigma = max(sigma, 1e-6)

    last_close = float(closes[-1])
    last_date = dates.iloc[-1]

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu - 0.5 * sigma * sigma, scale=sigma, size=(n_simulations, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = last_close * np.exp(log_paths)

    median = np.median(price_paths, axis=0)
    p05 = np.percentile(price_paths, 5, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    forecast_dates = _next_trading_dates(last_date, horizon_days)

    history_dates = dates.iloc[-history_lookback:].dt.strftime("%Y-%m-%d").tolist()
    history_close = closes[-history_lookback:].tolist()

    return ForecastResult(
        horizon_days=horizon_days,
        method="gbm_monte_carlo",
        mu=mu,
        sigma=sigma,
        last_close=last_close,
        last_date=last_date.strftime("%Y-%m-%d"),
        history_dates=history_dates,
        history_close=history_close,
        forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_dates],
        median=[float(x) for x in median],
        p05=[float(x) for x in p05],
        p25=[float(x) for x in p25],
        p75=[float(x) for x in p75],
        p95=[float(x) for x in p95],
    )


def resolve_horizon(label: str, fallback_days: Optional[int] = None) -> int:
    label = (label or "").upper().strip()
    if label in HORIZONS:
        return HORIZONS[label]
    if fallback_days is not None:
        return int(fallback_days)
    return HORIZONS["1M"]


def ml_forecaster_available() -> bool:
    return ML_FORECASTER_CKPT.exists() and ML_FORECASTER_STATS.exists()


def _load_ml_forecaster():
    """Return (model, stats, config) or None if not available."""
    if not ml_forecaster_available():
        return None
    try:
        import torch
        from ml.forecaster_dataset import NormalizationStats
        from ml.forecaster_model import ForecasterConfig, LSTMForecaster
    except ImportError:
        return None
    try:
        payload = torch.load(str(ML_FORECASTER_CKPT), map_location="cpu")
        cfg = ForecasterConfig(**payload["config"])
        model = LSTMForecaster(cfg).eval()
        model.load_state_dict(payload["state_dict"])
        stats = NormalizationStats.load(ML_FORECASTER_STATS)
        return model, stats, cfg
    except Exception:
        return None


def _ml_window_from_df(df: pd.DataFrame, stats, cfg) -> np.ndarray:
    cols = [c for c in stats.feature_columns if c in df.columns]
    if len(cols) != cfg.n_features:
        raise RuntimeError(
            f"Forecaster trained on {cfg.n_features} features but dataframe has {len(cols)}"
        )
    raw = df[cols].tail(cfg.window_size).to_numpy(dtype=np.float64)
    if len(raw) < cfg.window_size:
        raise RuntimeError(f"Need {cfg.window_size} bars, got {len(raw)}")
    norm = stats.transform(raw).astype(np.float32)
    return norm


def ml_forecast(
    horizon_days: int,
    n_simulations: int = 1000,
    history_lookback: int = 252,
    seed: Optional[int] = 42,
) -> ForecastResult:
    """
    LSTM-driven forecast. The model predicts (mean, log_std) of next-K-day
    cumulative log return; we sample horizon-step paths from a per-step
    Gaussian whose mean and std are derived from those terminal moments.
    """
    bundle = _load_ml_forecaster()
    if bundle is None:
        raise RuntimeError(
            "ML forecaster unavailable — train one with `python train_forecaster.py` first"
        )
    model, stats, cfg = bundle
    import torch

    df = load()
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader`")

    closes = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"])
    last_close = float(closes[-1])
    last_date = dates.iloc[-1]

    norm = _ml_window_from_df(df, stats, cfg)
    with torch.no_grad():
        x = torch.from_numpy(norm).unsqueeze(0)
        mean_terminal, log_std_terminal = model(x)
        mu_terminal = float(mean_terminal.item())
        sigma_terminal = float(torch.exp(log_std_terminal).item())

    K_train = cfg.horizon
    mu_per_step = mu_terminal / max(K_train, 1)
    sigma_per_step = sigma_terminal / np.sqrt(max(K_train, 1))

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu_per_step, scale=sigma_per_step, size=(n_simulations, horizon_days))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = last_close * np.exp(log_paths)

    median = np.median(price_paths, axis=0)
    p05 = np.percentile(price_paths, 5, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    forecast_dates = _next_trading_dates(last_date, horizon_days)
    history_dates = dates.iloc[-history_lookback:].dt.strftime("%Y-%m-%d").tolist()
    history_close = closes[-history_lookback:].tolist()

    return ForecastResult(
        horizon_days=horizon_days,
        method="lstm_ml",
        mu=mu_per_step,
        sigma=sigma_per_step,
        last_close=last_close,
        last_date=last_date.strftime("%Y-%m-%d"),
        history_dates=history_dates,
        history_close=history_close,
        forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_dates],
        median=[float(x) for x in median],
        p05=[float(x) for x in p05],
        p25=[float(x) for x in p25],
        p75=[float(x) for x in p75],
        p95=[float(x) for x in p95],
    )


def evaluate_ml_on_holdout() -> Dict[str, float]:
    """
    Score the trained LSTM forecaster on post-cutoff bars.

    For each holdout bar `t` we feed the model a window of training-cutoff
    features ending exactly at the cutoff (the model never sees post-cutoff
    inputs), project a single point estimate, and compare against the actual
    realized log return at horizon t. Reports MAE, RMSE, and directional
    accuracy.
    """
    bundle = _load_ml_forecaster()
    if bundle is None:
        return {"error": "ML forecaster checkpoint not found"}
    model, stats, cfg = bundle
    import torch

    from server.data_loader import LIVE_START_DATE, load_live

    train_df = load()
    live_df = load_live()
    if train_df.empty or live_df.empty:
        return {"error": "no holdout bars available yet", "n_holdout": int(len(live_df))}

    norm = _ml_window_from_df(train_df, stats, cfg)
    with torch.no_grad():
        x = torch.from_numpy(norm).unsqueeze(0)
        pred_mean, pred_log_std = model(x)
        pred = float(pred_mean.item())

    last_train_close = float(train_df["close"].iloc[-1])
    holdout_closes = live_df["close"].to_numpy(dtype=np.float64)
    realized_terminal_log_return = float(
        np.log(max(holdout_closes[min(cfg.horizon - 1, len(holdout_closes) - 1)], 1e-8))
        - np.log(max(last_train_close, 1e-8))
    )

    abs_err = abs(pred - realized_terminal_log_return)
    sq_err = (pred - realized_terminal_log_return) ** 2
    direction_correct = int((pred >= 0) == (realized_terminal_log_return >= 0))

    return {
        "horizon_days": cfg.horizon,
        "n_holdout_bars_used": int(min(cfg.horizon, len(holdout_closes))),
        "predicted_log_return": pred,
        "realized_log_return": realized_terminal_log_return,
        "abs_error": abs_err,
        "rmse": float(np.sqrt(sq_err)),
        "directional_correct": direction_correct,
        "live_start": str(LIVE_START_DATE.date()),
        "last_train_close": last_train_close,
        "first_holdout_date": live_df["date"].iloc[0].strftime("%Y-%m-%d") if not live_df.empty else None,
    }


def forecast(method: str, horizon_days: int, **kwargs) -> ForecastResult:
    """Dispatch to the requested method, falling back to GBM if ML is missing."""
    method = (method or "gbm").lower()
    common = {k: kwargs[k] for k in ("n_simulations", "seed") if k in kwargs}

    if method == "ml":
        if not ml_forecaster_available():
            raise RuntimeError("ML forecaster not trained — run `python train_forecaster.py` or use method=gbm")
        ml_kwargs = dict(common)
        if "history_lookback" in kwargs:
            ml_kwargs["history_lookback"] = kwargs["history_lookback"]
        return ml_forecast(horizon_days=horizon_days, **ml_kwargs)

    gbm_kwargs = dict(common)
    for k in ("lookback_days", "history_lookback"):
        if k in kwargs:
            gbm_kwargs[k] = kwargs[k]
    return gbm_forecast(horizon_days=horizon_days, **gbm_kwargs)
