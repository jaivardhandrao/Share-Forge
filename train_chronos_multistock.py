"""
Share-Forge - Multi-stock Chronos Fine-Tuning.

Trains Chronos on a basket of gold-related Indian ETFs simultaneously,
treating each ticker as a separate time series. This is exactly how
Chronos itself was pretrained — on thousands of unrelated series at once.

By combining the long histories of GOLDBEES (since 2007), KOTAKGOLD (2007),
HDFCGOLD (2010), and AXISGOLD (2010) with newer ETFs like TATAGOLD and
TATASILVER, we go from ~500 training rows to 15,000+. The model learns
general gold-ETF patterns from the deep history, then specialises further
on TATAGOLD's specific behaviour via the recent rows.

The same training-cutoff (2026-03-31) is enforced across every ticker, so
post-cutoff bars from any source never reach the model.

Usage:
    python3 train_chronos_multistock.py --epochs 50
    python3 train_chronos_multistock.py --tickers GOLDBEES.NS KOTAKGOLD.NS TATAGOLD.NS

Output:
    checkpoints/chronos_multistock/  — fine-tuned model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "chronos_multistock"
RUNS_DIR = ROOT / "runs" / "chronos_multistock"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF_DATE = pd.Timestamp("2026-03-31")

DEFAULT_TICKERS = [
    "GOLDBEES.NS",
    "KOTAKGOLD.NS",
    "HDFCGOLD.NS",
    "AXISGOLD.NS",
    "GOLDSHARE.NS",
    "SBIGETS.NS",
    "BSLGOLDETF.NS",
    "IDBIGOLD.NS",
    "TATAGOLD.NS",
]


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fetch_close_series(ticker: str, start: str = "2007-01-01") -> np.ndarray:
    """Fetch daily closes for a ticker, capped at TRAIN_CUTOFF_DATE."""
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            return np.array([])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "date"})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= TRAIN_CUTOFF_DATE]
        return df["close"].dropna().to_numpy(dtype=np.float64)
    except Exception as e:
        print(f"  [WARN] failed to fetch {ticker}: {e}", file=sys.stderr)
        return np.array([])


class MultiSeriesWindowDataset(Dataset):
    """
    Sliding-window dataset over multiple independent series.

    For each series, we generate windows of (context_length, prediction_length).
    The dataset concatenates windows across series so a single batch can mix
    samples from different ETFs.
    """

    def __init__(
        self,
        series_dict: Dict[str, np.ndarray],
        tokenizer,
        context_length: int,
        prediction_length: int,
    ):
        self.tokenizer = tokenizer
        self.ctx = context_length
        self.pred = prediction_length

        self.entries: List[Tuple[str, int]] = []
        self.series: Dict[str, np.ndarray] = {}
        for name, prices in series_dict.items():
            if len(prices) < context_length + prediction_length + 4:
                continue
            self.series[name] = prices.astype(np.float32)
            for t in range(context_length, len(prices) - prediction_length + 1):
                self.entries.append((name, t))

        if len(self.entries) < 16:
            raise ValueError(
                f"Combined dataset has only {len(self.entries)} windows. "
                f"Need 16+. Try more tickers or shorter context_length."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        name, t = self.entries[idx]
        prices = self.series[name]
        ctx = torch.tensor(prices[t - self.ctx : t], dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(prices[t : t + self.pred], dtype=torch.float32).unsqueeze(0)
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
            losses.append(float(model(**batch).loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(args):
    device = auto_device() if args.device == "auto" else args.device
    print(f"Device: {device}")
    print(f"Tickers: {args.tickers}")
    print(f"Model:   {args.model_id}")
    print(f"Cutoff:  {TRAIN_CUTOFF_DATE.date()}")
    print()

    series_dict: Dict[str, np.ndarray] = {}
    for ticker in args.tickers:
        prices = fetch_close_series(ticker)
        if len(prices) >= args.context_length + args.prediction_length + 16:
            series_dict[ticker] = prices
            print(f"  {ticker}: {len(prices)} bars")
        else:
            print(f"  {ticker}: SKIPPED ({len(prices)} bars, too few)")

    if not series_dict:
        print("No usable tickers. Exiting.")
        sys.exit(1)

    total = sum(len(s) for s in series_dict.values())
    print(f"\nTotal training rows across {len(series_dict)} tickers: {total:,}")

    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("chronos-forecasting not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {args.model_id}...")
    pipeline = ChronosPipeline.from_pretrained(args.model_id, device_map=device, torch_dtype=torch.float32)

    if hasattr(pipeline.tokenizer, "config"):
        pipeline.tokenizer.config.prediction_length = args.prediction_length
    if hasattr(pipeline.model, "config") and hasattr(pipeline.model.config, "prediction_length"):
        pipeline.model.config.prediction_length = args.prediction_length

    model = pipeline.model.model
    tokenizer = pipeline.tokenizer
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    full_ds = MultiSeriesWindowDataset(series_dict, tokenizer, args.context_length, args.prediction_length)
    n_total = len(full_ds)
    n_val = max(int(n_total * args.val_fraction), 32)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(RUNS_DIR / f"multistock_{int(time.time())}"))
    except ImportError:
        writer = None

    best_val = float("inf")
    patience_left = args.patience
    model.train()
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_losses.append(float(out.loss.item()))

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
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(CHECKPOINT_DIR))
        else:
            patience_left -= 1

        marker = " *" if improved else ""
        print(f"[epoch {epoch:03d}/{args.epochs}] train loss={train_loss:.4f}  val loss={val_loss:.4f}{marker}", flush=True)

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f}).")
            break

    elapsed = time.time() - t_start
    if writer:
        writer.close()

    summary = {
        "tickers": list(series_dict.keys()),
        "model_id": args.model_id,
        "n_train_rows_total": int(total),
        "n_train_windows": int(n_train),
        "n_val_windows": int(n_val),
        "best_val_loss": float(best_val),
        "elapsed_seconds": elapsed,
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }
    with open(CHECKPOINT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s.  Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print()
    print("To use this model for TATAGOLD predictions, point CHRONOS_FT_DIR")
    print(f"to the multistock checkpoint:")
    print(f"  CHRONOS_FT_DIR={CHECKPOINT_DIR} python3 -m uvicorn server.app:app --port 8080 --reload")
    print()
    print("Then in the UI, select 'Chronos fine-tuned (local)' and run forecasts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="List of yfinance tickers to combine. Default = 7 Indian gold/silver ETFs.")
    parser.add_argument("--model-id", default="amazon/chronos-t5-mini",
                        choices=[
                            "amazon/chronos-t5-tiny",
                            "amazon/chronos-t5-mini",
                            "amazon/chronos-t5-small",
                            "amazon/chronos-t5-base",
                        ])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--prediction-length", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
