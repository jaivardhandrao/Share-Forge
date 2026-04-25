"""
Share-Forge - Perfect-Hindsight Expert Policy.

Looks `lookahead` bars into the future and picks the optimal action — BUY when
upcoming cumulative return clears a positive threshold, SELL when it falls
below a negative threshold, HOLD otherwise. Used to build a labeled
(observation, action) dataset for behavior cloning.

The expert is purely a label generator: it never runs at inference time and
its inputs come exclusively from the training-cutoff dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


@dataclass
class ExpertConfig:
    lookahead: int = 5
    buy_threshold: float = 0.005
    sell_threshold: float = -0.005


def expert_action(
    log_close: np.ndarray,
    t: int,
    is_long: bool,
    config: ExpertConfig,
) -> int:
    """Pick the optimal action at index `t` using the next `lookahead` bars."""
    n = len(log_close)
    end = min(t + config.lookahead, n - 1)
    if end <= t:
        return ACTION_HOLD
    cum_return = float(log_close[end] - log_close[t])

    if cum_return > config.buy_threshold and not is_long:
        return ACTION_BUY
    if cum_return < config.sell_threshold and is_long:
        return ACTION_SELL
    return ACTION_HOLD


def label_trajectory(
    features_normalized: np.ndarray,
    closes: np.ndarray,
    window_size: int,
    config: ExpertConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk the dataset bar-by-bar with a simulated long/flat position and emit
    (window_features, expert_action) pairs.

    Returns:
        windows : (N, window_size, n_features)
        actions : (N,)  int64
    """
    n = len(closes)
    if n <= window_size + config.lookahead:
        raise ValueError(f"Series too short: need > {window_size + config.lookahead}, got {n}")

    log_close = np.log(np.maximum(closes, 1e-8))
    windows = []
    actions = []
    is_long = False

    for t in range(window_size, n - config.lookahead):
        window = features_normalized[t - window_size : t]
        action = expert_action(log_close, t, is_long, config)
        windows.append(window)
        actions.append(action)

        if action == ACTION_BUY:
            is_long = True
        elif action == ACTION_SELL:
            is_long = False

    return (
        np.stack(windows, axis=0).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def class_weights_from_actions(actions: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights for cross-entropy — counters HOLD imbalance."""
    counts = np.bincount(actions, minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (3.0 * counts)
    return weights.astype(np.float32)
