"""
Share-Forge - PyTorch Dataset for the LSTM forecaster.

Slides a window of `window_size` bars across the training-cutoff dataset and
emits (features, next_K_log_return) pairs. Uses per-column z-score
normalization fit on the training split only — no information from the
post-cutoff holdout leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from server.data_loader import FEATURE_COLUMNS, load


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: List[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std, feature_columns=np.array(self.feature_columns))

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        data = np.load(path, allow_pickle=True)
        return cls(
            mean=data["mean"],
            std=data["std"],
            feature_columns=list(data["feature_columns"]),
        )


def fit_normalization(df: pd.DataFrame, feature_columns: List[str]) -> NormalizationStats:
    arr = df[feature_columns].to_numpy(dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean, std=std, feature_columns=list(feature_columns))


class ForecasterDataset(Dataset):
    """Sliding-window (features → next-K cumulative log return) dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        window_size: int = 20,
        horizon: int = 5,
        stats: Optional[NormalizationStats] = None,
    ):
        self.feature_columns = feature_columns or [c for c in FEATURE_COLUMNS if c in df.columns]
        if not self.feature_columns:
            raise ValueError("No feature columns found in dataframe")
        self.window_size = window_size
        self.horizon = horizon

        if "close" not in df.columns:
            raise ValueError("dataframe must include a 'close' column")

        self.stats = stats or fit_normalization(df, self.feature_columns)
        x_norm = self.stats.transform(df[self.feature_columns].to_numpy(dtype=np.float64)).astype(np.float32)
        log_close = np.log(np.maximum(df["close"].to_numpy(dtype=np.float64), 1e-8))

        self.x = x_norm
        self.log_close = log_close

        self._valid_indices: List[int] = []
        n = len(df)
        for t in range(window_size, n - horizon):
            self._valid_indices.append(t)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._valid_indices[idx]
        window = self.x[t - self.window_size : t]
        target = self.log_close[t + self.horizon - 1] - self.log_close[t - 1]
        return (
            torch.from_numpy(window).float(),
            torch.tensor(float(target), dtype=torch.float32),
        )


def build_train_val_datasets(
    window_size: int = 20,
    horizon: int = 5,
    val_fraction: float = 0.15,
) -> Tuple[ForecasterDataset, ForecasterDataset, NormalizationStats]:
    df = load().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader` first")

    feature_columns = [c for c in FEATURE_COLUMNS if c in df.columns]

    n = len(df)
    split_idx = max(int(n * (1.0 - val_fraction)), window_size + horizon + 1)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx - window_size:].reset_index(drop=True)

    stats = fit_normalization(train_df, feature_columns)

    train_ds = ForecasterDataset(train_df, feature_columns, window_size, horizon, stats)
    val_ds = ForecasterDataset(val_df, feature_columns, window_size, horizon, stats)
    return train_ds, val_ds, stats
