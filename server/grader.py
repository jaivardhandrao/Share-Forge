"""
Share-Forge Grader.

Computes risk-adjusted performance metrics from an equity curve and produces
a single scalar score in roughly [0, 1] depending on the task's grading mode.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np

from models import BacktestSummary


def _safe_array(equity: List[float]) -> np.ndarray:
    arr = np.asarray(equity, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def compute_metrics(
    equity_curve: List[float],
    buy_and_hold_curve: List[float],
    n_trades: int,
    bars_per_year: int = 252,
) -> BacktestSummary:
    eq = _safe_array(equity_curve)
    if eq.size < 2:
        return BacktestSummary()

    returns = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    total_return = float(eq[-1] / eq[0] - 1.0)

    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(bars_per_year)) if std > 1e-9 else 0.0

    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / np.maximum(running_max, 1e-8)
    max_dd = float(-drawdowns.min())

    annualized_return = float((1.0 + total_return) ** (bars_per_year / max(returns.size, 1)) - 1.0)
    calmar = float(annualized_return / max_dd) if max_dd > 1e-9 else 0.0

    bh = _safe_array(buy_and_hold_curve)
    bh_return = float(bh[-1] / bh[0] - 1.0) if bh.size >= 2 else 0.0

    return BacktestSummary(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        n_trades=n_trades,
        final_value=float(eq[-1]),
        buy_and_hold_return=bh_return,
    )


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def grade(summary: BacktestSummary, mode: str) -> Dict[str, float]:
    """Return {'reward': float, 'breakdown': {...}} for a given grading mode."""
    if mode == "total_return":
        excess = summary.total_return - summary.buy_and_hold_return
        reward = _clamp(0.5 + 5.0 * excess)
        return {"reward": reward, "breakdown": {"excess_return": excess, **asdict(summary)}}

    if mode == "sharpe":
        reward = _clamp(summary.sharpe / 3.0)
        return {"reward": reward, "breakdown": {"sharpe_normalized": reward, **asdict(summary)}}

    if mode == "sharpe_turnover":
        turnover_penalty = 0.01 * summary.n_trades
        reward = _clamp(summary.sharpe / 3.0 - turnover_penalty)
        return {"reward": reward, "breakdown": {"turnover_penalty": turnover_penalty, **asdict(summary)}}

    if mode == "composite":
        sharpe_score = _clamp(summary.sharpe / 3.0)
        calmar_score = _clamp(summary.calmar / 3.0)
        dd_score = _clamp(1.0 - summary.max_drawdown)
        reward = 0.4 * sharpe_score + 0.3 * calmar_score + 0.3 * dd_score
        return {
            "reward": _clamp(reward),
            "breakdown": {
                "sharpe_score": sharpe_score,
                "calmar_score": calmar_score,
                "dd_score": dd_score,
                **asdict(summary),
            },
        }

    return {"reward": 0.0, "breakdown": asdict(summary)}
