"""
Share-Forge Trading Environment.

A Gymnasium-compatible single-asset trading environment for Tata Gold ETF.
Observation: rolling window of OHLCV + technical indicators + portfolio flags.
Action: Discrete(3) -> {0: HOLD, 1: BUY, 2: SELL}.
Reward: log return of portfolio value with a small turnover penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from server.data_loader import FEATURE_COLUMNS


@dataclass
class TradingConfig:
    window_size: int = 20
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 5.0
    turnover_penalty: float = 1e-3
    slippage_bps_on_failed_gtt: float = 2.0
    feature_columns: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    use_ml_forecaster: bool = False
    forecaster_checkpoint: Optional[str] = None


class ShareForgeTradingEnv(gym.Env):
    """
    Single-asset trading environment.

    The agent walks one bar at a time through `df`. On each step, it sees the
    last `window_size` rows of features plus its current position, then chooses
    HOLD / BUY / SELL. Reward is the per-step log return of total portfolio value
    minus a turnover penalty.

    When `config.use_ml_forecaster=True`, the env precomputes a one-shot LSTM
    forecast (predicted next-K-day log return) for every bar in `df` and
    appends it as an additional observation channel. This gives the RL policy
    a learned signal that supplements the raw indicators.
    """

    metadata = {"render_modes": ["human"], "name": "share_forge"}

    def __init__(self, df: pd.DataFrame, config: Optional[TradingConfig] = None):
        super().__init__()
        self.config = config or TradingConfig()
        self.feature_columns = [c for c in self.config.feature_columns if c in df.columns]
        if not self.feature_columns:
            raise ValueError("No feature columns found in dataframe")

        self.df = df.reset_index(drop=True).copy()
        self._raw = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        self._normalize_features()

        self._forecast_series: Optional[np.ndarray] = None
        if self.config.use_ml_forecaster:
            self._forecast_series = self._precompute_forecast()

        n_extra = 2 + (1 if self._forecast_series is not None else 0)
        n_features = len(self.feature_columns) + n_extra
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _precompute_forecast(self) -> Optional[np.ndarray]:
        """Run the LSTM forecaster once over every window in the slice."""
        try:
            import torch
            from ml.forecaster_dataset import NormalizationStats
            from ml.forecaster_model import ForecasterConfig, LSTMForecaster
        except ImportError:
            return None

        from pathlib import Path
        ckpt = Path(self.config.forecaster_checkpoint or "checkpoints/forecaster.pth")
        stats_path = Path("checkpoints/forecaster_stats.npz")
        if not ckpt.exists() or not stats_path.exists():
            return None

        try:
            payload = torch.load(str(ckpt), map_location="cpu")
            cfg = ForecasterConfig(**payload["config"])
            model = LSTMForecaster(cfg).eval()
            model.load_state_dict(payload["state_dict"])
            stats = NormalizationStats.load(stats_path)
        except Exception:
            return None

        feat_cols = [c for c in stats.feature_columns if c in self.df.columns]
        if len(feat_cols) != cfg.n_features:
            return None
        raw = self.df[feat_cols].to_numpy(dtype=np.float64)
        x_norm = stats.transform(raw).astype(np.float32)

        n = len(self.df)
        out = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            batch_size = 256
            windows = []
            indices = []
            for t in range(cfg.window_size, n):
                windows.append(x_norm[t - cfg.window_size : t])
                indices.append(t)
                if len(windows) == batch_size:
                    means, _ = model(torch.from_numpy(np.stack(windows)))
                    for i, idx in enumerate(indices):
                        out[idx] = float(means[i].item())
                    windows, indices = [], []
            if windows:
                means, _ = model(torch.from_numpy(np.stack(windows)))
                for i, idx in enumerate(indices):
                    out[idx] = float(means[i].item())
        return out

    def _normalize_features(self) -> None:
        x = self._raw
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        self._features = (x - mean) / std
        self._mean = mean
        self._std = std

    def _reset_state(self) -> None:
        self._t = self.config.window_size
        self._cash = self.config.initial_cash
        self._shares = 0.0
        self._is_long = False
        self._cost_basis = 0.0
        self._n_trades = 0
        self._equity_curve: List[float] = [self.config.initial_cash]
        self._action_history: List[int] = []

    def _close_at(self, idx: int) -> float:
        return float(self.df.iloc[idx]["close"])

    def _next_open_at(self, idx: int) -> float:
        if idx + 1 < len(self.df):
            return float(self.df.iloc[idx + 1]["open"])
        return self._close_at(idx)

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _build_observation(self) -> np.ndarray:
        start = self._t - self.config.window_size
        feats = self._features[start:self._t].copy()
        position_flag = np.full((self.config.window_size, 1), 1.0 if self._is_long else 0.0, dtype=np.float32)
        equity_norm = np.full(
            (self.config.window_size, 1),
            float(self._portfolio_value(self._close_at(self._t - 1)) / self.config.initial_cash) - 1.0,
            dtype=np.float32,
        )
        parts = [feats, position_flag, equity_norm]
        if self._forecast_series is not None:
            forecast_pred = float(self._forecast_series[self._t - 1])
            forecast_col = np.full((self.config.window_size, 1), forecast_pred, dtype=np.float32)
            parts.append(forecast_col)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        obs = self._build_observation()
        info = {
            "portfolio_value": self._portfolio_value(self._close_at(self._t - 1)),
            "step_in_episode": 0,
            "episode_length": len(self.df) - self.config.window_size - 1,
        }
        return obs, info

    def _execute(self, action: int, price: float) -> Tuple[float, bool]:
        """Execute action at `price`. Returns (transaction_cost_dollars, did_trade)."""
        cost_rate = self.config.transaction_cost_bps / 10_000.0
        did_trade = False
        cost = 0.0

        if action == 1 and not self._is_long:
            qty = self._cash / (price * (1 + cost_rate))
            cost = qty * price * cost_rate
            self._cash -= qty * price + cost
            self._shares += qty
            self._cost_basis = price
            self._is_long = True
            did_trade = True
            self._n_trades += 1
        elif action == 2 and self._is_long:
            proceeds = self._shares * price
            cost = proceeds * cost_rate
            self._cash += proceeds - cost
            self._shares = 0.0
            self._cost_basis = 0.0
            self._is_long = False
            did_trade = True
            self._n_trades += 1

        return cost, did_trade

    def _gtt_satisfied(self, idx: int, action: int, trigger: Optional[float]) -> bool:
        if trigger is None:
            return True
        bar = self.df.iloc[idx]
        high, low = float(bar["high"]), float(bar["low"])
        if action == 1:
            return low <= trigger <= high
        if action == 2:
            return low <= trigger <= high
        return True

    def step(self, action: int, trigger_price: Optional[float] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._t >= len(self.df) - 1:
            obs = self._build_observation()
            price = self._close_at(self._t - 1)
            return obs, 0.0, True, False, {"portfolio_value": self._portfolio_value(price)}

        prev_value = self._portfolio_value(self._close_at(self._t - 1))
        next_idx = self._t
        exec_price = self._next_open_at(self._t - 1)

        gtt_ok = self._gtt_satisfied(next_idx, action, trigger_price)
        slippage_penalty = 0.0
        if not gtt_ok:
            action = 0
            slippage_penalty = self.config.slippage_bps_on_failed_gtt / 10_000.0

        cost, did_trade = self._execute(action, exec_price)

        self._t += 1
        new_price = self._close_at(self._t - 1)
        new_value = self._portfolio_value(new_price)
        self._equity_curve.append(new_value)
        self._action_history.append(int(action))

        log_ret = float(np.log(max(new_value, 1e-8) / max(prev_value, 1e-8)))
        turnover = self.config.turnover_penalty if did_trade else 0.0
        reward = log_ret - turnover - slippage_penalty

        terminated = self._t >= len(self.df) - 1
        truncated = False
        obs = self._build_observation()
        info = {
            "portfolio_value": new_value,
            "cash": self._cash,
            "shares": self._shares,
            "is_long": self._is_long,
            "did_trade": did_trade,
            "n_trades": self._n_trades,
            "exec_price": exec_price,
            "close_price": new_price,
            "step_in_episode": self._t - self.config.window_size,
            "episode_length": len(self.df) - self.config.window_size - 1,
        }
        return obs, reward, terminated, truncated, info

    @property
    def equity_curve(self) -> List[float]:
        return list(self._equity_curve)

    @property
    def action_history(self) -> List[int]:
        return list(self._action_history)

    def current_close(self) -> float:
        return self._close_at(self._t - 1)
