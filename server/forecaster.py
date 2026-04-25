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
from typing import Dict, List, Optional, Sequence

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


def chronos_forecast(
    horizon_days: int,
    n_samples: int = 200,
    context_length: int = 64,
    finetuned: bool = False,
    history_lookback: int = 252,
    model_id: str = "amazon/chronos-t5-tiny",
) -> ForecastResult:
    """
    Run Amazon's Chronos foundation model on the most recent `context_length`
    closes from the training-cutoff dataset and return a percentile-banded
    price path.
    """
    try:
        from ml.chronos_forecaster import chronos_available, get_forecaster
    except ImportError as e:
        raise RuntimeError(f"Chronos module unavailable: {e}")

    if not chronos_available():
        raise RuntimeError(
            "chronos-forecasting not installed. Run "
            "`pip install chronos-forecasting`."
        )

    df = load()
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader`")

    closes = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"])
    last_close = float(closes[-1])
    last_date = dates.iloc[-1]

    context = closes[-context_length:].astype(np.float32)
    forecaster = get_forecaster(model_id=model_id, finetuned=finetuned)
    result = forecaster.predict(prices=context, horizon=horizon_days, n_samples=n_samples)

    forecast_dates = _next_trading_dates(last_date, horizon_days)
    history_dates = dates.iloc[-history_lookback:].dt.strftime("%Y-%m-%d").tolist()
    history_close = closes[-history_lookback:].tolist()

    if len(result.p50) >= 2:
        log_rets = np.diff(np.log(np.maximum(np.asarray(result.p50), 1e-8)))
        mu = float(log_rets.mean())
        sigma = float(log_rets.std()) if log_rets.size > 1 else 0.0
    else:
        mu, sigma = 0.0, 0.0

    return ForecastResult(
        horizon_days=horizon_days,
        method=result.method,
        mu=mu,
        sigma=sigma,
        last_close=last_close,
        last_date=last_date.strftime("%Y-%m-%d"),
        history_dates=history_dates,
        history_close=history_close,
        forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_dates],
        median=result.p50,
        p05=result.p05,
        p25=result.p25,
        p75=result.p75,
        p95=result.p95,
    )


def naive_forecast(
    horizon_days: int,
    history_lookback: int = 252,
    seed: Optional[int] = None,
    **_unused,
) -> ForecastResult:
    """
    Naive persistence baseline: predict every future day = last training close.

    Bands grow with sqrt(horizon) using the recent 60-day realized volatility,
    so the prediction acknowledges uncertainty even though the median is flat.
    This is the textbook "you have to beat me" baseline — most price-prediction
    papers report MAPE relative to this.
    """
    df = load()
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader` first")

    closes = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["date"])
    last_close = float(closes[-1])
    last_date = dates.iloc[-1]

    log_rets = np.diff(np.log(np.maximum(closes[-60:], 1e-8)))
    daily_vol = float(log_rets.std()) if log_rets.size > 1 else 0.01
    daily_vol = max(daily_vol, 1e-6)

    median = [last_close] * horizon_days
    p05, p25, p75, p95 = [], [], [], []
    for t in range(horizon_days):
        scale = daily_vol * np.sqrt(t + 1)
        p05.append(last_close * float(np.exp(-1.645 * scale)))
        p25.append(last_close * float(np.exp(-0.674 * scale)))
        p75.append(last_close * float(np.exp(0.674 * scale)))
        p95.append(last_close * float(np.exp(1.645 * scale)))

    forecast_dates = _next_trading_dates(last_date, horizon_days)
    history_dates = dates.iloc[-history_lookback:].dt.strftime("%Y-%m-%d").tolist()
    history_close = closes[-history_lookback:].tolist()

    return ForecastResult(
        horizon_days=horizon_days,
        method="naive_persistence",
        mu=0.0,
        sigma=daily_vol,
        last_close=last_close,
        last_date=last_date.strftime("%Y-%m-%d"),
        history_dates=history_dates,
        history_close=history_close,
        forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_dates],
        median=median,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
    )


def forecast(method: str, horizon_days: int, **kwargs) -> ForecastResult:
    """Dispatch to the requested method, falling back to GBM if ML is missing."""
    method = (method or "gbm").lower()
    common = {k: kwargs[k] for k in ("n_simulations", "seed") if k in kwargs}

    if method == "naive":
        return naive_forecast(
            horizon_days=horizon_days,
            history_lookback=kwargs.get("history_lookback", 252),
            seed=kwargs.get("seed"),
        )

    if method == "ml":
        if not ml_forecaster_available():
            raise RuntimeError("ML forecaster not trained — run `python train_forecaster.py` or use method=gbm")
        ml_kwargs = dict(common)
        if "history_lookback" in kwargs:
            ml_kwargs["history_lookback"] = kwargs["history_lookback"]
        return ml_forecast(horizon_days=horizon_days, **ml_kwargs)

    if method in ("chronos", "chronos_zs", "chronos_ft"):
        finetuned = method == "chronos_ft"
        chronos_kwargs = {}
        if "n_simulations" in kwargs:
            chronos_kwargs["n_samples"] = kwargs["n_simulations"]
        if "history_lookback" in kwargs:
            chronos_kwargs["history_lookback"] = kwargs["history_lookback"]
        if "context_length" in kwargs:
            chronos_kwargs["context_length"] = kwargs["context_length"]
        return chronos_forecast(horizon_days=horizon_days, finetuned=finetuned, **chronos_kwargs)

    gbm_kwargs = dict(common)
    for k in ("lookback_days", "history_lookback"):
        if k in kwargs:
            gbm_kwargs[k] = kwargs[k]
    return gbm_forecast(horizon_days=horizon_days, **gbm_kwargs)


def grade_predictions(
    target_dates: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    n_samples: int = 200,
) -> Dict:
    """
    Hackathon-style grading: for each target_date in the holdout window,
    run each method using only training-cutoff inputs, and compare the
    predicted price to the actual close on that date.

    Returns per-task and per-method aggregate metrics (MAPE, RMSE,
    directional accuracy, calibration of the 95% band).
    """
    from server.data_loader import load_live, TRAIN_CUTOFF_DATE

    methods = methods or ["gbm", "ml", "chronos_zs"]
    if target_dates is None:
        live = load_live()
        target_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(live["date"])]

    if not target_dates:
        return {"error": "no holdout target dates available", "methods": methods}

    df = load()
    last_train_close = float(df["close"].iloc[-1])
    last_train_date = pd.Timestamp(df["date"].iloc[-1])

    live = load_live()
    actuals = {pd.Timestamp(d).strftime("%Y-%m-%d"): float(p) for d, p in zip(live["date"], live["close"])}

    tasks = []
    for date_str in target_dates:
        target = pd.Timestamp(date_str)
        if target <= last_train_date:
            continue
        horizon = max(int((target - last_train_date) / pd.Timedelta(days=1) * 5 / 7), 1)
        actual = actuals.get(date_str)
        if actual is None:
            continue

        per_method = {}
        for m in methods:
            try:
                res = forecast(method=m, horizon_days=horizon, n_simulations=n_samples)
                pred_median = float(res.median[-1])
                pred_p05 = float(res.p05[-1])
                pred_p95 = float(res.p95[-1])
                per_method[m] = {
                    "predicted": pred_median,
                    "p05": pred_p05,
                    "p95": pred_p95,
                    "abs_error": abs(pred_median - actual),
                    "ape": abs(pred_median - actual) / max(actual, 1e-8),
                    "direction_correct": int((pred_median - last_train_close) * (actual - last_train_close) >= 0),
                    "in_95_band": int(pred_p05 <= actual <= pred_p95),
                }
            except Exception as e:
                per_method[m] = {"error": str(e)[:120]}

        tasks.append({
            "target_date": date_str,
            "horizon_days": horizon,
            "actual": actual,
            "last_train_close": last_train_close,
            "predictions": per_method,
        })

    aggregates: Dict[str, Dict] = {}
    for m in methods:
        apes, ses, dirs, cals = [], [], [], []
        for t in tasks:
            p = t["predictions"].get(m, {})
            if "ape" in p:
                apes.append(p["ape"])
                ses.append((p["predicted"] - t["actual"]) ** 2)
                dirs.append(p["direction_correct"])
                cals.append(p["in_95_band"])
        if apes:
            aggregates[m] = {
                "mape": float(np.mean(apes)),
                "rmse": float(np.sqrt(np.mean(ses))),
                "directional_accuracy": float(np.mean(dirs)),
                "calibration_95": float(np.mean(cals)),
                "n_tasks": len(apes),
            }
        else:
            aggregates[m] = {"error": "no valid tasks", "n_tasks": 0}

    return {
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "n_tasks": len(tasks),
        "tasks": tasks,
        "aggregates": aggregates,
    }
