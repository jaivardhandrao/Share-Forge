"""
Share-Forge - Train the LSTM Forecaster.

Trains an LSTM regressor with a Gaussian-NLL head to predict the next
K-day cumulative log return of TATAGOLD.NS. Uses MPS on Apple Silicon,
CUDA where available, else CPU. Logs to TensorBoard at runs/forecaster/
and (optionally) to Weights & Biases.

Usage:
    python train_forecaster.py --epochs 50 --horizon 5 --device auto
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.forecaster_dataset import build_train_val_datasets
from ml.forecaster_model import (
    ForecasterConfig,
    LSTMForecaster,
    directional_accuracy,
    gaussian_nll,
)

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs" / "forecaster"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def maybe_init_wandb(run_name: str, config: dict):
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
        wandb.init(project="share-forge-forecaster", name=run_name, config=config, sync_tensorboard=True)
        return wandb
    except Exception as e:
        print(f"[WARN] wandb init failed: {e}", file=sys.stderr)
        return None


def evaluate(model: LSTMForecaster, loader: DataLoader, device: str) -> dict:
    model.eval()
    losses = []
    dir_acc = []
    rmse_terms = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mean, log_std = model(x)
            losses.append(float(gaussian_nll(mean, log_std, y).item()))
            dir_acc.append(float(directional_accuracy(mean, y).item()))
            rmse_terms.append(float(((mean - y) ** 2).mean().item()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dir_acc": float(np.mean(dir_acc)) if dir_acc else float("nan"),
        "rmse": float(math.sqrt(np.mean(rmse_terms))) if rmse_terms else float("nan"),
    }


def train(args):
    device = auto_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    print(f"Horizon (days): {args.horizon}")
    print(f"Window size:    {args.window_size}")

    train_ds, val_ds, stats = build_train_val_datasets(
        window_size=args.window_size,
        horizon=args.horizon,
        val_fraction=args.val_fraction,
    )
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    print(f"Features:      {stats.feature_columns}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = ForecasterConfig(
        n_features=len(stats.feature_columns),
        window_size=args.window_size,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = LSTMForecaster(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:         LSTMForecaster   params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"forecaster_h{args.horizon}_ws{args.window_size}_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    wandb_run = maybe_init_wandb(run_name, vars(args) | {"n_params": n_params})

    best_val = float("inf")
    patience_left = args.patience
    best_path = CHECKPOINT_DIR / "forecaster.pth"
    stats_path = CHECKPOINT_DIR / "forecaster_stats.npz"
    stats.save(stats_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        ep_dir = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            mean, log_std = model(x)
            loss = gaussian_nll(mean, log_std, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
            ep_dir.append(float(directional_accuracy(mean.detach(), y).item()))

        scheduler.step()

        train_metrics = {
            "loss": float(np.mean(ep_losses)),
            "dir_acc": float(np.mean(ep_dir)),
        }
        val_metrics = evaluate(model, val_loader, device)

        writer.add_scalar("train/nll", train_metrics["loss"], epoch)
        writer.add_scalar("train/dir_acc", train_metrics["dir_acc"], epoch)
        writer.add_scalar("val/nll", val_metrics["loss"], epoch)
        writer.add_scalar("val/dir_acc", val_metrics["dir_acc"], epoch)
        writer.add_scalar("val/rmse", val_metrics["rmse"], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            patience_left = args.patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": vars(config),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_dir_acc": val_metrics["dir_acc"],
                    "feature_columns": stats.feature_columns,
                },
                best_path,
            )
        else:
            patience_left -= 1

        marker = " *" if improved else ""
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train nll={train_metrics['loss']:.4f} dir={train_metrics['dir_acc']:.3f}  "
            f"val nll={val_metrics['loss']:.4f} dir={val_metrics['dir_acc']:.3f} "
            f"rmse={val_metrics['rmse']:.4f}{marker}",
            flush=True,
        )

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best val nll={best_val:.4f})")
            break

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    print(f"\nBest checkpoint saved to {best_path}")
    print(f"Normalization stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--horizon", type=int, default=5, help="Trading days to forecast ahead")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
