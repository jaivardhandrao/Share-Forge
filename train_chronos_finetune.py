"""
Share-Forge - Fine-tune Chronos-T5-tiny on TATAGOLD.NS LOCALLY.

Designed for Apple Silicon (M-series, MPS) but auto-detects CUDA / CPU.
Trains the underlying T5 encoder-decoder on the same next-token-prediction
objective Chronos was pretrained with, just specialised to TATAGOLD's
training-cutoff history. Saves a HuggingFace-format checkpoint to
`checkpoints/chronos/` which the server picks up via method='chronos_ft'.

Usage on Mac:
    pip install chronos-forecasting transformers accelerate datasets
    python train_chronos_finetune.py --epochs 30

Hardware notes:
    chronos-t5-tiny  (8M params)  — fits easily on 16GB M-series MPS, ~10 min
    chronos-t5-mini  (20M params) — ~20 min on M-series
    chronos-t5-small (46M params) — ~40 min on M-series
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "chronos"
RUNS_DIR = ROOT / "runs" / "chronos"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chronos_pipeline(model_id: str, device: str):
    try:
        from chronos import ChronosPipeline
    except ImportError as e:
        print("chronos-forecasting not installed. Run: pip install chronos-forecasting", file=sys.stderr)
        raise SystemExit(1) from e
    return ChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=torch.float32)


class WindowDataset(Dataset):
    """
    Slides over the training-cutoff close series and emits
    (context_tokens, target_tokens) tuples using the Chronos tokenizer.
    """

    def __init__(self, prices: np.ndarray, tokenizer, context_length: int, prediction_length: int):
        self.prices = prices.astype(np.float32)
        self.tokenizer = tokenizer
        self.ctx = context_length
        self.pred = prediction_length

        n = len(prices)
        self.indices = list(range(self.ctx, n - self.pred + 1))
        if len(self.indices) < 16:
            raise ValueError(
                f"Need at least 16 windows, got {len(self.indices)}. "
                f"Reduce --context-length / --prediction-length."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        ctx = torch.tensor(self.prices[t - self.ctx : t], dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(self.prices[t : t + self.pred], dtype=torch.float32).unsqueeze(0)
        ctx_tokens, ctx_attn, scale = self.tokenizer.context_input_transform(ctx)
        tgt_tokens, _ = self.tokenizer.label_input_transform(tgt, scale)
        return {
            "input_ids": ctx_tokens.squeeze(0),
            "attention_mask": ctx_attn.squeeze(0),
            "labels": tgt_tokens.squeeze(0),
        }


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(args):
    device = auto_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    print(f"Model:  {args.model_id}")

    pipeline = load_chronos_pipeline(args.model_id, device)

    # Chronos's tokenizer has a baked-in prediction_length (default 64). Override
    # it to match our requested target length so label_input_transform doesn't
    # assert. This is safe — fine-tuning works at any prediction_length.
    if hasattr(pipeline.tokenizer, "config"):
        pipeline.tokenizer.config.prediction_length = args.prediction_length
        print(f"Set tokenizer prediction_length -> {args.prediction_length}")
    if hasattr(pipeline.model, "config") and hasattr(pipeline.model.config, "prediction_length"):
        pipeline.model.config.prediction_length = args.prediction_length

    model = pipeline.model.model
    tokenizer = pipeline.tokenizer

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    from server.data_loader import load
    df = load().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Training dataset empty — run `python -m server.data_loader` first.")
    prices = df["close"].to_numpy(dtype=np.float64)
    print(f"Train rows: {len(prices)}  range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

    split = int(len(prices) * (1.0 - args.val_fraction))
    train_prices = prices[:split]
    val_prices = prices[max(split - args.context_length, 0):]

    train_ds = WindowDataset(train_prices, tokenizer, args.context_length, args.prediction_length)
    val_ds = WindowDataset(val_prices, tokenizer, args.context_length, args.prediction_length)
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = f"chronos_ft_{int(time.time())}"
        writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    except ImportError:
        writer = None

    model.train()
    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        ep_losses: List[float] = []
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(ep_losses))
        val_loss = evaluate(model, val_loader, device)

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            patience_left = args.patience
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(CHECKPOINT_DIR))
            tokenizer.save_pretrained(str(CHECKPOINT_DIR)) if hasattr(tokenizer, "save_pretrained") else None
        else:
            patience_left -= 1

        marker = " *" if improved else ""
        print(
            f"[epoch {epoch:03d}/{args.epochs}] train loss={train_loss:.4f}  "
            f"val loss={val_loss:.4f}{marker}",
            flush=True,
        )

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f}).")
            break

    if writer:
        writer.close()

    print(f"\nBest checkpoint saved to {CHECKPOINT_DIR}")
    print("Server picks it up automatically via method='chronos_ft' in /api/forecast.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="amazon/chronos-t5-tiny",
                        help="HF model ID or local checkpoint path. "
                             "Examples: amazon/chronos-t5-tiny, amazon/chronos-t5-mini, "
                             "amazon/chronos-t5-small, amazon/chronos-t5-base, "
                             "or a local path like checkpoints/chronos_multistock")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--prediction-length", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
