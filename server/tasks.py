"""
Share-Forge Task Definitions.

Each task pins a historical date range for TATAGOLD.NS plus task-specific
instructions and a grading mode. The OpenEnv environment loads a slice of
the cached dataset based on the task type the agent requests in /reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from models import TaskDifficulty


@dataclass
class TaskSpec:
    task_id: str
    task_type: TaskDifficulty
    start: str
    end: str
    instructions: str
    grading_mode: str
    adversarial_shock: bool = False


EASY_INSTRUCTIONS = (
    "TATAGOLD.NS, calm regime. Trade long-only on daily bars. "
    "On each step you observe the last 20 bars of OHLCV plus indicators "
    "(SMA, EMA, RSI, MACD, Bollinger) and your current position. "
    "Action: 0=HOLD, 1=BUY, 2=SELL. Goal: beat buy-and-hold total return."
)

MEDIUM_VOLATILE_INSTRUCTIONS = (
    "TATAGOLD.NS during the COVID crash and recovery. "
    "Drawdowns are large; rebounds are sharp. "
    "Optimize Sharpe ratio rather than raw return. "
    "Excessive turnover is penalized."
)

MEDIUM_SIDEWAYS_INSTRUCTIONS = (
    "TATAGOLD.NS in a sideways consolidation regime. "
    "Buy-and-hold underperforms here. Take positions only when signal is strong. "
    "Score = Sharpe minus turnover penalty."
)

HARD_INSTRUCTIONS = (
    "TATAGOLD.NS with synthetic adversarial shocks injected. "
    "Survive max-drawdown limits. "
    "Score = 0.4 * Sharpe + 0.3 * Calmar + 0.3 * (1 - max_drawdown)."
)


TASKS: Dict[TaskDifficulty, TaskSpec] = {
    TaskDifficulty.EASY_LONG_ONLY: TaskSpec(
        task_id="a1b2c3d4-1111-4000-a000-000000000001",
        task_type=TaskDifficulty.EASY_LONG_ONLY,
        start="2018-01-01",
        end="2019-12-31",
        instructions=EASY_INSTRUCTIONS,
        grading_mode="total_return",
    ),
    TaskDifficulty.MEDIUM_VOLATILE: TaskSpec(
        task_id="a1b2c3d4-2222-4000-a000-000000000002",
        task_type=TaskDifficulty.MEDIUM_VOLATILE,
        start="2020-01-01",
        end="2021-06-30",
        instructions=MEDIUM_VOLATILE_INSTRUCTIONS,
        grading_mode="sharpe",
    ),
    TaskDifficulty.MEDIUM_SIDEWAYS: TaskSpec(
        task_id="a1b2c3d4-3333-4000-a000-000000000003",
        task_type=TaskDifficulty.MEDIUM_SIDEWAYS,
        start="2022-01-01",
        end="2022-12-31",
        instructions=MEDIUM_SIDEWAYS_INSTRUCTIONS,
        grading_mode="sharpe_turnover",
    ),
    TaskDifficulty.HARD_ADVERSARIAL: TaskSpec(
        task_id="a1b2c3d4-4444-4000-a000-000000000004",
        task_type=TaskDifficulty.HARD_ADVERSARIAL,
        start="2023-01-01",
        end="2024-06-30",
        instructions=HARD_INSTRUCTIONS,
        grading_mode="composite",
        adversarial_shock=True,
    ),
}


def get_task(task_type: TaskDifficulty) -> TaskSpec:
    return TASKS[task_type]
