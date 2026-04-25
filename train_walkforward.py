"""
Share-Forge - Walk-Forward Fine-Tuning of Chronos.

For each step T in the training-cutoff window:
  1. Fine-tune the model on bars [0..T-1]
  2. Predict the close at bar T
  3. Compare prediction to actual at bar T
  4. Move T forward by `--retrain-every` days
  5. Repeat until cutoff

This simulates production deployment: the model only ever sees past bars,
predicts forward, then incorporates the new ground truth and continues.

Configurable for any yfinance-supported ticker. Saves the final model per
ticker so each stock has its own specialised checkpoint.

Usage:
    python3 train_walkforward.py --ticker TATASILVER.NS --retrain-every 10
    python3 train_walkforward.py --ticker TATAGOLD.NS   --retrain-every 5

Output:
    checkpoints/chronos_<ticker_safe>/   — final fine-tuned model
    walkforward_<ticker_safe>.csv        — per-step prediction log
    walkforward_<ticker_safe>.json       — summary metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
CHECKPOINT_ROOT = ROOT / "checkpoints"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF_DATE = pd.Timestamp("2026-03-31")


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def safe_ticker(ticker: str) -> str:
    return ticker.lower().replace(".", "_").replace("/", "_")


def fetch_close_series(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Fetch daily close bars for any yfinance-supported ticker."""
    import yfinance as yf
    df = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= TRAIN_CUTOFF_DATE].reset_index(drop=True)
    return df


class ChronosWindowDataset(Dataset):
    def __init__(self, prices: np.ndarray, tokenizer, context_length: int, prediction_length: int):
        self.prices = prices.astype(np.float32)
        self.tokenizer = tokenizer
        self.ctx = context_length
        self.pred = prediction_length

        n = len(prices)
        self.indices = list(range(self.ctx, n - self.pred + 1))

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


def fine_tune_step(
    pipeline,
    prices_so_far: np.ndarray,
    epochs_per_step: int,
    batch_size: int,
    lr: float,
    context_length: int,
    prediction_length: int,
    device: str,
) -> float:
    """One walk-forward fine-tuning step. Returns final train loss."""
    model = pipeline.model.model
    tokenizer = pipeline.tokenizer

    if len(prices_so_far) < context_length + prediction_length + 4:
        return float("nan")

    ds = ChronosWindowDataset(prices_so_far, tokenizer, context_length, prediction_length)
    if len(ds) < 4:
        return float("nan")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    last_loss = float("nan")
    for _ in range(epochs_per_step):
        ep_losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_losses.append(float(out.loss.item()))
        if ep_losses:
            last_loss = float(np.mean(ep_losses))
    model.eval()
    return last_loss


def predict_next(pipeline, context_prices: np.ndarray, num_samples: int = 100) -> float:
    """Predict the next bar from the most recent context."""
    ctx = torch.tensor(context_prices, dtype=torch.float32)
    try:
        forecast = pipeline.predict(ctx, prediction_length=1, num_samples=num_samples)
    except TypeError:
        forecast = pipeline.predict(ctx, prediction_length=1)

    if hasattr(forecast, "cpu"):
        arr = forecast[0].cpu().float().numpy()
    elif isinstance(forecast, (list, tuple)):
        arr = np.asarray(forecast[0])
    else:
        arr = np.asarray(forecast)
        if arr.ndim == 3:
            arr = arr[0]
    return float(np.median(arr, axis=0)[0])


def walkforward(args):
    device = auto_device() if args.device == "auto" else args.device
    print(f"Ticker:           {args.ticker}")
    print(f"Device:           {device}")
    print(f"Retrain every:    {args.retrain_every} bars")
    print(f"Epochs per step:  {args.epochs_per_step}")
    print(f"Context length:   {args.context_length}")

    df = fetch_close_series(args.ticker)
    closes = df["close"].to_numpy(dtype=np.float64)
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    print(f"Fetched {len(closes)} bars: {dates[0]} -> {dates[-1]}")

    if len(closes) < args.context_length + args.retrain_every + 5:
        print("Not enough bars for walk-forward. Need at least context+retrain_every+5 bars.")
        sys.exit(1)

    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("chronos-forecasting not installed. Run: pip install chronos-forecasting", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model_id}...")
    pipeline = ChronosPipeline.from_pretrained(args.model_id, device_map=device, torch_dtype=torch.float32)

    if hasattr(pipeline.tokenizer, "config"):
        pipeline.tokenizer.config.prediction_length = args.prediction_length
    if hasattr(pipeline.model, "config") and hasattr(pipeline.model.config, "prediction_length"):
        pipeline.model.config.prediction_length = args.prediction_length

    log: List[Dict] = []
    start_idx = max(args.context_length + args.warmup, 60)
    print(f"Walk-forward range: bar {start_idx} -> {len(closes) - 1}")
    print(f"Total steps: {(len(closes) - 1 - start_idx) // args.retrain_every + 1}")
    print()

    t_start = time.time()
    for step_num, t in enumerate(range(start_idx, len(closes) - 1, args.retrain_every), start=1):
        train_loss = fine_tune_step(
            pipeline,
            prices_so_far=closes[:t],
            epochs_per_step=args.epochs_per_step,
            batch_size=args.batch_size,
            lr=args.lr,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            device=device,
        )

        ctx_start = max(0, t - args.context_length)
        predicted = predict_next(pipeline, closes[ctx_start:t], num_samples=args.num_samples)
        actual = float(closes[t])
        error_pct = abs(predicted - actual) / max(actual, 1e-8) * 100.0

        log_row = {
            "step": step_num,
            "bar_index": t,
            "date": dates[t],
            "n_train_bars": t,
            "train_loss": train_loss,
            "predicted": predicted,
            "actual": actual,
            "abs_error": abs(predicted - actual),
            "error_pct": error_pct,
            "direction_correct": int((predicted - closes[t - 1]) * (actual - closes[t - 1]) >= 0),
        }
        log.append(log_row)
        print(
            f"[step {step_num:03d} | bar {t:04d} | {dates[t]}] "
            f"train_loss={train_loss:.4f}  "
            f"pred=₹{predicted:.2f}  actual=₹{actual:.2f}  err={error_pct:.2f}%  "
            f"dir={'✓' if log_row['direction_correct'] else '✗'}",
            flush=True,
        )

    elapsed = time.time() - t_start

    out_dir = CHECKPOINT_ROOT / f"chronos_{safe_ticker(args.ticker)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline.model.model.save_pretrained(str(out_dir))
    if hasattr(pipeline.tokenizer, "save_pretrained"):
        pipeline.tokenizer.save_pretrained(str(out_dir))

    log_df = pd.DataFrame(log)
    csv_path = ROOT / f"walkforward_{safe_ticker(args.ticker)}.csv"
    log_df.to_csv(csv_path, index=False)

    summary = {
        "ticker": args.ticker,
        "n_steps": len(log),
        "n_train_bars_total": int(closes.size),
        "first_date": dates[0],
        "last_date": dates[-1],
        "retrain_every": args.retrain_every,
        "epochs_per_step": args.epochs_per_step,
        "elapsed_seconds": elapsed,
        "metrics": {
            "mape_pct": float(log_df["error_pct"].mean()) if len(log_df) else 0.0,
            "median_error_pct": float(log_df["error_pct"].median()) if len(log_df) else 0.0,
            "rmse_pct": float(np.sqrt((log_df["error_pct"] ** 2).mean())) if len(log_df) else 0.0,
            "directional_accuracy": float(log_df["direction_correct"].mean()) if len(log_df) else 0.0,
        },
        "checkpoint_dir": str(out_dir),
        "log_csv": str(csv_path),
    }
    json_path = ROOT / f"walkforward_{safe_ticker(args.ticker)}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s")
    print(f"  MAPE:                 {summary['metrics']['mape_pct']:.2f}%")
    print(f"  Median error:         {summary['metrics']['median_error_pct']:.2f}%")
    print(f"  Directional accuracy: {summary['metrics']['directional_accuracy'] * 100:.1f}%")
    print(f"  Final model:          {out_dir}")
    print(f"  Step log:             {csv_path}")
    print(f"  Summary:              {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TATASILVER.NS",
                        help="Any yfinance-supported ticker. Examples: TATAGOLD.NS, TATASILVER.NS, RELIANCE.NS, INFY.NS")
    parser.add_argument("--model-id", default="amazon/chronos-t5-tiny",
                        choices=[
                            "amazon/chronos-t5-tiny",
                            "amazon/chronos-t5-mini",
                            "amazon/chronos-t5-small",
                            "amazon/chronos-t5-base",
                        ])
    parser.add_argument("--retrain-every", type=int, default=10,
                        help="Re-fine-tune every N bars (lower = more frequent updates, slower)")
    parser.add_argument("--epochs-per-step", type=int, default=1,
                        help="Fine-tune epochs per step (lower = faster, less overfit)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--prediction-length", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=60,
                        help="Initial bars to skip before walk-forward starts")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Monte Carlo samples per prediction")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    walkforward(args)


if __name__ == "__main__":
    main()
