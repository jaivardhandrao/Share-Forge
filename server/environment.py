"""
Share-Forge OpenEnv Environment.

Wraps the Gymnasium ShareForgeTradingEnv into the OpenEnv Environment
contract (reset / step / state). Each task pins a historical slice of
TATAGOLD.NS data; the agent steps through that slice one bar at a time.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from openenv.core.env_server.interfaces import Environment

from models import (
    BacktestSummary,
    MarketObservation,
    PortfolioSnapshot,
    ShareForgeState,
    TaskDifficulty,
    TradeAction,
)
from server.data_loader import load, slice_by_dates
from server.grader import compute_metrics, grade
from server.tasks import TASKS, get_task
from server.trading_env import ShareForgeTradingEnv, TradingConfig


def _inject_adversarial_shock(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply two synthetic ~8% gap-down shocks at random points to stress-test the agent."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    if n < 50:
        return df
    shock_idxs = rng.choice(np.arange(20, n - 5), size=2, replace=False)
    for idx in shock_idxs:
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df.loc[df.index[idx]:, col] = df.loc[df.index[idx]:, col] * 0.92
    return df


class ShareForgeEnvironment(Environment[TradeAction, MarketObservation, ShareForgeState]):
    """
    Share-Forge OpenEnv wrapper.

    Each episode = one historical slice. Each step = one trading day.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = ShareForgeState()
        self._env: Optional[ShareForgeTradingEnv] = None
        self._task: Optional[TaskDifficulty] = None
        self._df_slice: Optional[pd.DataFrame] = None
        self._initial_obs: Optional[np.ndarray] = None
        self._buy_and_hold_curve: list = []

    def _build_buy_and_hold(self, df: pd.DataFrame, initial_cash: float) -> list:
        prices = df["close"].to_numpy(dtype=np.float64)
        if len(prices) == 0:
            return [initial_cash]
        first = max(float(prices[0]), 1e-8)
        return (initial_cash * (prices / first)).tolist()

    def _to_observation(
        self,
        obs_array: np.ndarray,
        info: Dict[str, Any],
        feedback: Optional[str],
        summary: Optional[BacktestSummary],
        done: bool,
        reward: Optional[float],
    ) -> MarketObservation:
        assert self._task is not None and self._env is not None
        spec = get_task(self._task)
        feature_names = list(self._env.feature_columns) + ["position_flag", "equity_norm"]

        portfolio = PortfolioSnapshot(
            cash=float(info.get("cash", self._env._cash)),
            shares=float(info.get("shares", self._env._shares)),
            position_value=float(info.get("shares", self._env._shares))
            * float(info.get("close_price", self._env.current_close())),
            total_value=float(info.get("portfolio_value", 0.0)),
            is_long=bool(info.get("is_long", self._env._is_long)),
        )

        return MarketObservation(
            done=done,
            reward=reward,
            window_features=obs_array.tolist(),
            feature_names=feature_names,
            portfolio=portfolio,
            last_close=float(info.get("close_price", self._env.current_close())),
            task_type=self._task,
            task_id=spec.task_id,
            instructions=spec.instructions,
            feedback=feedback,
            step_in_episode=int(info.get("step_in_episode", 0)),
            episode_length=int(info.get("episode_length", 0)),
            summary=summary,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_type: str = "easy_long_only",
        **kwargs,
    ) -> MarketObservation:
        difficulty = TaskDifficulty(task_type)
        spec = get_task(difficulty)

        full_df = load()
        df_slice = slice_by_dates(full_df, spec.start, spec.end)
        if spec.adversarial_shock:
            df_slice = _inject_adversarial_shock(df_slice)

        self._task = difficulty
        self._df_slice = df_slice
        self._env = ShareForgeTradingEnv(df_slice, TradingConfig())
        self._buy_and_hold_curve = self._build_buy_and_hold(df_slice, self._env.config.initial_cash)

        obs_array, info = self._env.reset(seed=seed)

        self._state = ShareForgeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_type=difficulty,
            step_in_episode=0,
            episode_length=int(info.get("episode_length", 0)),
            portfolio_value=float(info.get("portfolio_value", 0.0)),
            cumulative_reward=0.0,
            n_trades=0,
        )

        return self._to_observation(
            obs_array=obs_array,
            info=info,
            feedback=None,
            summary=None,
            done=False,
            reward=None,
        )

    def step(self, action: TradeAction, timeout_s: Optional[float] = None, **kwargs) -> MarketObservation:
        if self._env is None or self._task is None:
            return MarketObservation(
                done=True,
                reward=0.0,
                window_features=[],
                feature_names=[],
                portfolio=None,
                instructions="Episode has ended. Call reset() to start a new episode.",
                feedback="No active environment.",
            )

        obs_array, reward, terminated, truncated, info = self._env.step(
            int(action.action),
            trigger_price=action.trigger_price,
        )
        done = bool(terminated or truncated)

        self._state.step_count += 1
        self._state.step_in_episode = int(info.get("step_in_episode", self._state.step_in_episode + 1))
        self._state.portfolio_value = float(info.get("portfolio_value", self._state.portfolio_value))
        self._state.cumulative_reward += float(reward)
        self._state.n_trades = int(info.get("n_trades", self._state.n_trades))

        summary = None
        feedback = None

        if done:
            spec = get_task(self._task)
            equity = self._env.equity_curve
            bh = self._buy_and_hold_curve[: len(equity)] or [self._env.config.initial_cash, self._env.config.initial_cash]
            summary = compute_metrics(equity, bh, n_trades=self._env._n_trades)
            graded = grade(summary, spec.grading_mode)
            reward = float(graded["reward"])
            feedback = (
                f"Episode complete. mode={spec.grading_mode} "
                f"return={summary.total_return:.2%} sharpe={summary.sharpe:.2f} "
                f"max_dd={summary.max_drawdown:.2%} trades={summary.n_trades} "
                f"score={reward:.3f}"
            )

        return self._to_observation(
            obs_array=obs_array,
            info=info,
            feedback=feedback,
            summary=summary,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> ShareForgeState:
        return self._state

    def close(self) -> None:
        self._env = None
        self._df_slice = None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "share_forge",
            "description": "Share-Forge — RL trading environment for Tata Gold ETF (TATAGOLD.NS)",
            "version": "1.0.0",
            "task_types": [t.value for t in TASKS.keys()],
        }
