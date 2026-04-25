"""
Share-Forge - FastAPI Application.

Endpoints
─────────
  GET  /api/health           detailed health (status, model, data, db)
  GET  /api/tasks            list available trading tasks
  POST /api/forecast         GBM Monte Carlo forecast (1W / 1M / 3M / 6M / 1Y)
  GET  /api/data             training data (<= 2026-03-31)
  GET  /api/live             holdout data (>= 2026-04-01) — never seen by model
  POST /api/live-action      run trained agent on most recent training-window bars
  POST /api/predict          stateless action prediction from caller-supplied window
  POST /api/backtest         run a full backtest with the trained agent
  GET  /api/history/*        recent persisted predictions / backtests / actions

OpenEnv contract endpoints (/reset, /step, /state, /ws, /health) are provided
by openenv.core's create_app.

Static frontend (Apache ECharts SPA) is served from /. Gradio playground is
mounted at /gradio.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server.http_server import create_app

from models import MarketObservation, TaskDifficulty, TradeAction
from server import database
from server.data_loader import (
    FEATURE_COLUMNS,
    LIVE_START_DATE,
    TRAIN_CUTOFF_DATE,
    cutoff_summary,
    load,
    load_live,
    slice_by_dates,
)
from server.environment import ShareForgeEnvironment
from server.forecaster import (
    HORIZONS,
    evaluate_ml_on_holdout,
    forecast as run_forecast,
    ml_forecaster_available,
    resolve_horizon,
)
from server.grader import compute_metrics, grade
from server.policy_loader import active_source, policy_status, predict as policy_predict
from server.tasks import TASKS, get_task
from server.trading_env import ShareForgeTradingEnv, TradingConfig

ROOT = Path(__file__).parent.parent
FRONTEND_DIR = ROOT / "frontend"

app: FastAPI = create_app(
    ShareForgeEnvironment,
    TradeAction,
    MarketObservation,
    env_name="share_forge",
    max_concurrent_envs=4,
)


@app.on_event("startup")
def _on_startup():
    database.init_db()


# ── Pydantic request/response models ────────────────────────────────────────


class ForecastRequest(BaseModel):
    horizon: Optional[str] = Field(default="1M", description="One of 1W, 1M, 3M, 6M, 1Y")
    horizon_days: Optional[int] = Field(default=None, ge=1, le=2000)
    n_simulations: int = Field(default=1000, ge=100, le=5000)
    lookback_days: int = Field(default=120, ge=20, le=500)
    method: str = Field(default="gbm", description="One of 'gbm' or 'ml'")
    seed: Optional[int] = Field(default=42)


class LiveActionRequest(BaseModel):
    lookback_days: int = Field(default=120, ge=40, le=500)
    is_long: bool = Field(default=False)


class PredictRequest(BaseModel):
    window_features: List[List[float]]
    is_long: bool = False
    session_id: Optional[str] = None


class PredictResponse(BaseModel):
    action: int
    action_name: str
    probabilities: Optional[List[float]] = None
    last_close: Optional[float] = None
    source: str = "policy"


class BacktestRequest(BaseModel):
    task_type: str = Field(default=TaskDifficulty.EASY_LONG_ONLY.value)


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    cutoff = cutoff_summary()
    return {
        "status": "ok",
        "service": "share-forge",
        "version": "1.0.0",
        "policy": policy_status(),
        "ml_forecaster": {
            "available": ml_forecaster_available(),
            "checkpoint": str(ROOT / "checkpoints" / "forecaster.pth"),
        },
        "data": cutoff,
        "database": database.status() | {"counts": database.counts()},
        "frontend_mounted": FRONTEND_DIR.exists(),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ── Forecast ────────────────────────────────────────────────────────────────


@app.post("/api/forecast")
def api_forecast(req: ForecastRequest) -> Dict[str, Any]:
    horizon_days = resolve_horizon(req.horizon or "", fallback_days=req.horizon_days)
    try:
        result = run_forecast(
            method=req.method,
            horizon_days=horizon_days,
            n_simulations=req.n_simulations,
            lookback_days=req.lookback_days,
            seed=req.seed,
        )
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    payload = result.to_dict()
    db_id = database.record_prediction(payload, horizon_label=req.horizon)
    payload["persisted_id"] = db_id
    return payload


@app.get("/api/forecast-eval")
def api_forecast_eval() -> Dict[str, Any]:
    """Score the trained ML forecaster on post-cutoff bars (read-only)."""
    return evaluate_ml_on_holdout()


# ── Training data ──────────────────────────────────────────────────────────


@app.get("/api/data")
def api_data(start: Optional[str] = None, end: Optional[str] = None, columns: Optional[str] = None) -> Dict[str, Any]:
    df = load()
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    cols_requested = [c.strip() for c in columns.split(",")] if columns else None
    available = [c for c in df.columns if c != "date"]
    cols_to_use = cols_requested or available
    bars = [
        {"date": r["date"].strftime("%Y-%m-%d"), **{c: float(r[c]) for c in cols_to_use if c in df.columns}}
        for r in df.to_dict(orient="records")
    ]
    return {
        "ticker": "TATAGOLD.NS",
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "n_bars": len(bars),
        "columns": ["date"] + cols_to_use,
        "bars": bars,
    }


# ── Live (holdout) data — never given to the model ─────────────────────────


@app.get("/api/live")
def api_live() -> Dict[str, Any]:
    df = load_live()
    bars = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]) if "volume" in r and pd.notna(r["volume"]) else 0.0,
        }
        for r in df.to_dict(orient="records")
    ]
    return {
        "ticker": "TATAGOLD.NS",
        "live_start": str(LIVE_START_DATE.date()),
        "n_bars": len(bars),
        "last_date": bars[-1]["date"] if bars else None,
        "bars": bars,
        "note": "These bars are NOT fed to the model during training or single-step inference.",
    }


# ── Live action — run agent on most recent training-window bars ────────────


def _build_window_from_recent(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    feats = df[[c for c in FEATURE_COLUMNS if c in df.columns]].tail(lookback).to_numpy(dtype=np.float32)
    mean = feats.mean(axis=0, keepdims=True)
    std = np.where(feats.std(axis=0, keepdims=True) < 1e-6, 1.0, feats.std(axis=0, keepdims=True))
    norm = (feats - mean) / std
    window = norm[-20:]
    last_row = df.iloc[-1]
    return {
        "window_features": window.tolist(),
        "last_close": float(last_row["close"]),
        "last_date": last_row["date"].strftime("%Y-%m-%d"),
        "history": {
            "dates": pd.to_datetime(df["date"]).tail(lookback).dt.strftime("%Y-%m-%d").tolist(),
            "close": df["close"].tail(lookback).astype(float).tolist(),
            "sma_20": df["sma_20"].tail(lookback).astype(float).tolist() if "sma_20" in df.columns else [],
        },
    }


@app.post("/api/live-action")
def api_live_action(req: LiveActionRequest) -> Dict[str, Any]:
    df = load()
    if df.empty:
        return JSONResponse(
            status_code=503,
            content={"error": "training data unavailable — run `python -m server.data_loader`"},
        )
    bundle = _build_window_from_recent(df, lookback=req.lookback_days)
    action, probs = policy_predict(bundle["window_features"], is_long=req.is_long)
    name = ["HOLD", "BUY", "SELL"][int(action)]
    src = active_source()
    database.record_action(action=int(action), last_close=bundle["last_close"], is_long=req.is_long, session_id=None, source=src)
    return {
        "action": int(action),
        "action_name": name,
        "probabilities": probs,
        "last_close": bundle["last_close"],
        "last_date": bundle["last_date"],
        "history": bundle["history"],
        "source": src,
    }


# ── Stateless prediction ───────────────────────────────────────────────────


@app.post("/api/predict", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def predict_action(req: PredictRequest) -> PredictResponse:
    action, probs = policy_predict(req.window_features, is_long=req.is_long)
    name = ["HOLD", "BUY", "SELL"][int(action)]
    last_close = None
    try:
        last_close = float(req.window_features[-1][3])
    except (IndexError, TypeError, ValueError):
        last_close = None
    if last_close is not None:
        database.record_action(action=int(action), last_close=last_close, is_long=req.is_long, session_id=req.session_id, source="api")
    return PredictResponse(
        action=int(action),
        action_name=name,
        probabilities=probs,
        last_close=last_close,
        source="policy",
    )


# ── Backtest ───────────────────────────────────────────────────────────────


def _run_backtest_internal(task_type: str) -> Dict[str, Any]:
    difficulty = TaskDifficulty(task_type)
    spec = get_task(difficulty)

    df = load()
    df_slice = slice_by_dates(df, spec.start, spec.end).reset_index(drop=True)

    env = ShareForgeTradingEnv(df_slice, TradingConfig())
    obs, _ = env.reset()
    done = False
    while not done:
        raw_window = obs[:, :-2].tolist()
        action, _ = policy_predict(raw_window, is_long=env._is_long)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    equity = env.equity_curve
    prices = df_slice["close"].to_numpy(dtype=np.float64)
    bh = (env.config.initial_cash * (prices / max(prices[0], 1e-8))).tolist()
    bh = bh[: len(equity)]
    summary = compute_metrics(equity, bh, n_trades=env._n_trades)
    graded = grade(summary, spec.grading_mode)

    dates = pd.to_datetime(df_slice["date"]).iloc[: len(equity)].dt.strftime("%Y-%m-%d").tolist()

    summary_dict = {
        "total_return": summary.total_return,
        "sharpe": summary.sharpe,
        "max_drawdown": summary.max_drawdown,
        "calmar": summary.calmar,
        "n_trades": summary.n_trades,
        "final_value": summary.final_value,
        "buy_and_hold_return": summary.buy_and_hold_return,
    }

    return {
        "task_type": difficulty.value,
        "grading_mode": spec.grading_mode,
        "score": float(graded["reward"]),
        "summary": summary_dict,
        "dates": dates,
        "equity_curve": [float(x) for x in equity],
        "buy_and_hold": [float(x) for x in bh],
        "actions": env.action_history,
    }


@app.post("/api/backtest")
def api_backtest(req: BacktestRequest) -> Dict[str, Any]:
    try:
        result = _run_backtest_internal(req.task_type)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    db_id = database.record_backtest(
        task_type=result["task_type"],
        grading_mode=result["grading_mode"],
        summary=result["summary"],
        score=result["score"],
    )
    result["persisted_id"] = db_id
    return result


# ── Tasks ───────────────────────────────────────────────────────────────────


@app.get("/api/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id": spec.task_id,
            "task_type": spec.task_type.value,
            "start": spec.start,
            "end": spec.end,
            "grading_mode": spec.grading_mode,
            "adversarial_shock": spec.adversarial_shock,
        }
        for spec in TASKS.values()
    ]


@app.get("/api/tasks/{task_type}")
def get_task_route(task_type: str):
    try:
        difficulty = TaskDifficulty(task_type)
    except ValueError:
        return JSONResponse(status_code=404, content={"error": f"Unknown task_type '{task_type}'"})
    spec = TASKS[difficulty]
    return {
        "task_id": spec.task_id,
        "task_type": spec.task_type.value,
        "start": spec.start,
        "end": spec.end,
        "instructions": spec.instructions,
        "grading_mode": spec.grading_mode,
    }


# ── Persisted history ──────────────────────────────────────────────────────


@app.get("/api/history/predictions")
def history_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    return database.list_predictions(limit=limit)


@app.get("/api/history/backtests")
def history_backtests(limit: int = 50) -> List[Dict[str, Any]]:
    return database.list_backtests(limit=limit)


@app.get("/api/history/actions")
def history_actions(limit: int = 100) -> List[Dict[str, Any]]:
    return database.list_actions(limit=limit)


# ── Static frontend ─────────────────────────────────────────────────────────


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Gradio playground (optional) ───────────────────────────────────────────


try:
    import gradio as gr
    from server.gradio_ui import create_gradio_app

    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
except Exception:
    pass


def main():
    import os

    import uvicorn
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
