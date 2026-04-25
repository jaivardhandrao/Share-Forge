"""
Share-Forge - Behavior Cloning Pretraining.

Distills a perfect-hindsight expert into a small LSTM policy via supervised
cross-entropy. Output is `checkpoints/bc_policy.pth` — used by
`server.policy_loader` as a fallback policy when no PPO checkpoint exists,
and as a baseline the PPO agent must beat.

The expert has access only to the training-cutoff dataset; no holdout bars
are involved at any stage of label generation.

Usage:
    python train_bc.py --epochs 30 --horizon 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from ml.bc_model import BCPolicy, BCPolicyConfig, save_checkpoint
from ml.expert_policy import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    ExpertConfig,
    class_weights_from_actions,
    label_trajectory,
)
from ml.forecaster_dataset import build_train_val_datasets

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs" / "bc"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_bc_dataset(
    window_size: int,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
):
    """Build (windows, actions) tensors from the training-cutoff dataset."""
    from server.data_loader import FEATURE_COLUMNS, load

    df = load().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader` first")

    feature_columns = [c for c in FEATURE_COLUMNS if c in df.columns]

    raw = df[feature_columns].to_numpy(dtype=np.float64)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    norm = ((raw - mean) / std).astype(np.float32)

    expert_cfg = ExpertConfig(
        lookahead=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    closes = df["close"].to_numpy(dtype=np.float64)
    windows, actions = label_trajectory(norm, closes, window_size=window_size, config=expert_cfg)

    return windows, actions, feature_columns, mean, std


def evaluate(model: BCPolicy, loader: DataLoader, device: str, criterion: nn.Module) -> dict:
    model.eval()
    losses, correct, total = [], 0, 0
    per_class_correct = np.zeros(3, dtype=np.int64)
    per_class_total = np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            for c in range(3):
                mask = y == c
                per_class_total[c] += int(mask.sum().item())
                per_class_correct[c] += int(((preds == y) & mask).sum().item())

    per_class_acc = (per_class_correct / np.maximum(per_class_total, 1)).tolist()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": correct / max(total, 1),
        "per_class_acc": per_class_acc,
    }


def train(args):
    device = auto_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    print(f"Horizon (days): {args.horizon}")

    windows, actions, feature_cols, mean, std = build_bc_dataset(
        window_size=args.window_size,
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    print(f"Total samples: {len(windows)}")
    counts = np.bincount(actions, minlength=3)
    print(f"  HOLD={counts[ACTION_HOLD]}  BUY={counts[ACTION_BUY]}  SELL={counts[ACTION_SELL]}")

    n = len(windows)
    split = int(n * (1.0 - args.val_fraction))
    x_train, y_train = windows[:split], actions[:split]
    x_val, y_val = windows[split:], actions[split:]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = BCPolicyConfig(
        n_features=windows.shape[2],
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )
    model = BCPolicy(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BCPolicy  params={n_params:,}")

    weights = torch.tensor(class_weights_from_actions(y_train), dtype=torch.float32).to(device)
    print(f"Class weights: HOLD={weights[0]:.2f} BUY={weights[1]:.2f} SELL={weights[2]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"bc_h{args.horizon}_ws{args.window_size}_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))

    best_val = float("inf")
    patience_left = args.patience
    best_path = CHECKPOINT_DIR / "bc_policy.pth"
    stats_path = CHECKPOINT_DIR / "bc_stats.npz"
    np.savez(stats_path, mean=mean, std=std, feature_columns=np.array(feature_cols))

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = [], 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss.append(float(loss.item()))
            preds = logits.argmax(dim=-1)
            ep_correct += int((preds == y).sum().item())
            ep_total += int(y.numel())

        scheduler.step()

        train_loss = float(np.mean(ep_loss))
        train_acc = ep_correct / max(ep_total, 1)
        val_metrics = evaluate(model, val_loader, device, criterion)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        writer.add_scalar("val/acc_HOLD", val_metrics["per_class_acc"][0], epoch)
        writer.add_scalar("val/acc_BUY", val_metrics["per_class_acc"][1], epoch)
        writer.add_scalar("val/acc_SELL", val_metrics["per_class_acc"][2], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            patience_left = args.patience
            save_checkpoint(
                model,
                str(best_path),
                extra={
                    "feature_columns": feature_cols,
                    "norm_mean": mean.tolist(),
                    "norm_std": std.tolist(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_acc": val_metrics["acc"],
                    "per_class_acc": val_metrics["per_class_acc"],
                },
            )
        else:
            patience_left -= 1

        marker = " *" if improved else ""
        per = val_metrics["per_class_acc"]
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
            f"[HOLD={per[0]:.2f} BUY={per[1]:.2f} SELL={per[2]:.2f}]{marker}",
            flush=True,
        )

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f})")
            break

    writer.close()
    print(f"\nBC checkpoint saved to {best_path}")
    print(f"BC stats saved to       {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--buy-threshold", type=float, default=0.005)
    parser.add_argument("--sell-threshold", type=float, default=-0.005)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
