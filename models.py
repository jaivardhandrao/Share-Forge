"""
Share-Forge Environment - Data Models

Pydantic models for OpenEnv actions, observations, and state.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


class TradeActionType(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class TaskDifficulty(str, Enum):
    EASY_LONG_ONLY = "easy_long_only"
    MEDIUM_VOLATILE = "medium_volatile"
    MEDIUM_SIDEWAYS = "medium_sideways"
    HARD_ADVERSARIAL = "hard_adversarial"


class PortfolioSnapshot(BaseModel):
    cash: float = Field(..., description="Cash balance")
    shares: float = Field(..., description="Shares currently held")
    position_value: float = Field(..., description="Mark-to-market value of shares")
    total_value: float = Field(..., description="cash + position_value")
    is_long: bool = Field(..., description="True if currently holding shares")


class BacktestSummary(BaseModel):
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    final_value: float = 0.0
    buy_and_hold_return: float = 0.0


class TradeAction(Action):
    """Action submitted by the agent on each step."""
    action: int = Field(default=0, ge=0, le=2, description="0=HOLD, 1=BUY, 2=SELL")
    trigger_price: Optional[float] = Field(
        default=None,
        description="Optional GTT trigger price; ignored if next bar never crosses it",
    )


class MarketObservation(Observation):
    """Observation returned by the environment after each step."""
    window_features: List[List[float]] = Field(
        default_factory=list,
        description="Rolling window of feature rows (window_size x n_features)",
    )
    feature_names: List[str] = Field(default_factory=list)
    portfolio: Optional[PortfolioSnapshot] = Field(default=None)
    last_close: float = Field(default=0.0)
    task_type: TaskDifficulty = Field(default=TaskDifficulty.EASY_LONG_ONLY)
    task_id: str = Field(default="")
    instructions: str = Field(default="")
    feedback: Optional[str] = Field(default=None)
    step_in_episode: int = Field(default=0)
    episode_length: int = Field(default=0)
    summary: Optional[BacktestSummary] = Field(default=None)


class ShareForgeState(State):
    task_type: TaskDifficulty = Field(default=TaskDifficulty.EASY_LONG_ONLY)
    step_in_episode: int = Field(default=0)
    episode_length: int = Field(default=0)
    portfolio_value: float = Field(default=0.0)
    cumulative_reward: float = Field(default=0.0)
    n_trades: int = Field(default=0)
