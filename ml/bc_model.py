"""
Share-Forge - Behavior Cloning Policy.

A small LSTM + MLP classifier that maps a 20-bar window of normalized
features to a 3-way action distribution {HOLD, BUY, SELL}. Trained via
supervised cross-entropy on (window, expert_action) pairs.

This standalone PyTorch policy serves as both:
  1. A baseline trader that the PPO RL agent must beat.
  2. The fallback policy in `server/policy_loader.py` when no PPO checkpoint
     has been trained yet — better than the momentum heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BCPolicyConfig:
    n_features: int = 17
    window_size: int = 20
    hidden_size: int = 64
    num_layers: int = 1
    mlp_hidden: int = 64
    dropout: float = 0.1
    n_actions: int = 3


class BCPolicy(nn.Module):
    """LSTM trunk + MLP head producing action logits."""

    def __init__(self, config: BCPolicyConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1, :])
        return self.head(h)

    @torch.no_grad()
    def act(self, window: np.ndarray, deterministic: bool = True) -> int:
        device = next(self.parameters()).device
        x = torch.from_numpy(window.astype(np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        logits = self.forward(x)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def action_probs(self, window: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        x = torch.from_numpy(window.astype(np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(device)
        logits = self.forward(x)
        return F.softmax(logits, dim=-1).cpu().numpy()[0]


def save_checkpoint(model: BCPolicy, path: str, extra: Optional[dict] = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": vars(model.config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> BCPolicy:
    payload = torch.load(path, map_location=device)
    config = BCPolicyConfig(**payload["config"])
    model = BCPolicy(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
