"""
Share-Forge ML Forecaster — PyTorch LSTM regressor.

Predicts the cumulative log-return of TATAGOLD.NS over the next K trading days
given a window of OHLCV + technical indicator features. Outputs both a mean
and a log-std so a Gaussian NLL loss gives the model an explicit uncertainty
head — useful for downstream risk-aware trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ForecasterConfig:
    n_features: int = 17
    window_size: int = 20
    horizon: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class LSTMForecaster(nn.Module):
    """
    2-layer LSTM with two linear heads.

    Input  : (batch, window, n_features)
    Output : (mean, log_std) each of shape (batch,)
             representing the predicted next-K-day cumulative log return.
    """

    def __init__(self, config: ForecasterConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.head_mean = nn.Linear(config.hidden_size, 1)
        self.head_log_std = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = self.layer_norm(out[:, -1, :])
        mean = self.head_mean(h).squeeze(-1)
        log_std = self.head_log_std(h).squeeze(-1).clamp(-5.0, 2.0)
        return mean, log_std


def gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically stable Gaussian negative log-likelihood."""
    var = torch.exp(2.0 * log_std)
    nll = 0.5 * (((target - mean) ** 2) / var + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi)))
    return nll.mean()


def directional_accuracy(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of predictions that get the sign of the next-K-day return right."""
    return ((mean.sign() == target.sign()) | (target == 0)).float().mean()
