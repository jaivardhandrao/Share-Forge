---
title: Share-Forge
emoji: "📈"
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 80
pinned: true
license: mit
---

# Share-Forge

**RL trading agent for Tata Gold ETF (TATAGOLD.NS)** — built for the Meta PyTorch OpenEnv Hackathon at Scaler School of Technology.

A three-stage ML pipeline: an LSTM **price forecaster** trained with Gaussian-NLL, a **behavior-cloned LSTM policy** distilled from a perfect-hindsight expert, and a **PPO + LSTM RL agent** that fine-tunes on top of the BC initialization with the forecaster's prediction injected into its observation space. Plus an OpenEnv-compliant server, an Apache ECharts frontend with 1W / 1M / 3M / 6M / 1Y forecasts, and a Postgres-backed history layer — all wired into a single `docker compose up`.

## Quick Start

```bash
docker compose up --build
```

Then open **http://localhost** (port 80). Postgres is brought up first, the Share-Forge service waits on its healthcheck, and the ECharts UI is served at `/`.

### Local development (without Docker)

```bash
pip install -r server/requirements.txt
python -m server.data_loader              # fetch + cache TATAGOLD.NS

# ── ML pipeline (in order) ───────────────────────────────────────────
python train_forecaster.py --epochs 50    # 1. LSTM forecaster (PyTorch + MPS)
python train_bc.py        --epochs 30     # 2. BC policy (supervised cross-entropy)
python train.py --timesteps 200000 \      # 3. PPO RL agent (uses forecaster signal)
                --use-ml-forecaster

DATABASE_URL=sqlite:///./share_forge.db PORT=8080 \
    python -m uvicorn server.app:app --host 0.0.0.0 --port 8080 --reload
```

The server reads whichever checkpoints exist and uses the best available policy automatically — `PPO > BC > momentum heuristic`. So a freshly-cloned repo produces sensible actions even before any training, and each stage measurably improves them.

When `DATABASE_URL` is unset, the app falls back to a local SQLite file. When Postgres is unreachable, the app still starts and just skips persistence — so `/api/health` continues to respond.

## Hard Data Cutoff

The model is trained and queried only on bars dated **on or before 2026-03-31** (`TRAIN_CUTOFF_DATE`). Bars dated **on or after 2026-04-01** (`LIVE_START_DATE`) are exposed on a separate route (`GET /api/live`) for visual comparison only — they are never fed to the trading env, the forecaster, or the policy.

```python
# server/data_loader.py
TRAIN_CUTOFF_DATE = pd.Timestamp("2026-03-31")
LIVE_START_DATE   = pd.Timestamp("2026-04-01")

load()       # bars <= 2026-03-31  (used by env / forecaster / policy)
load_live()  # bars >= 2026-04-01  (exposed via /api/live, never seen by model)
```

`load()` asserts on every call that the returned frame's max date is ≤ the cutoff, so a misconfigured cache cannot leak post-cutoff rows into the model.

## Architecture

```
                                 yfinance (full history)
                                          │
                                          ▼
                           server/data/TATAGOLD.NS.full.csv
                              │                       │
              load() ≤ 2026-03-31           load_live() ≥ 2026-04-01
                  │                                   │
                  │                                   ▼
                  │                             GET /api/live  (display only,
                  │                                            never touches model)
                  │
                  ├──► ml/forecaster_dataset ── train_forecaster.py ──► checkpoints/forecaster.pth
                  │           │                  (LSTM + Gaussian NLL)
                  │           │
                  │           └────► /api/forecast?method=ml   /api/forecast-eval
                  │
                  ├──► ml/expert_policy ── train_bc.py ──► checkpoints/bc_policy.pth
                  │           │            (supervised cross-entropy)
                  │           │
                  │           └────► policy fallback when no PPO checkpoint
                  │
                  ▼
        ShareForgeTradingEnv ◄─── (optional) forecaster.pth signal injected as obs feature
                  │
                  └──► train.py ── RecurrentPPO + LSTM ──► checkpoints/ppo_share_forge.zip
                                                                  │
                                                                  ▼
                                              policy_loader.predict
                                              (PPO > BC > heuristic chain)
                                                                  │
                                                                  ▼
                                              database (predictions, backtests, actions_log)
                                                                  │
                                                                  ▼
                                  Apache ECharts SPA at  /
                                  Gradio playground at   /gradio
                                  OpenEnv WS/REST at     /reset /step /ws
```

## ML Pipeline

### Stage 1 — LSTM forecaster (`train_forecaster.py`)

A 2-layer LSTM with two linear heads (`mean`, `log_std`) predicts the cumulative log return of TATAGOLD.NS over the next K trading days. Trained with Gaussian negative log-likelihood so the model learns its own uncertainty.

| Detail | Value |
|---|---|
| Input | `(batch, 20, 17)` — 20-day window, 17 features |
| Loss | `0.5·((y − μ)² / e^{2 log σ} + 2 log σ + log 2π)` |
| Metrics | NLL, RMSE, directional accuracy |
| Optim | AdamW + cosine annealing + grad-clip 1.0 + early stop |
| Logs | TensorBoard at `runs/forecaster/`, optional WandB |
| Output | `checkpoints/forecaster.pth`, `checkpoints/forecaster_stats.npz` |

### Stage 2 — Behavior cloning (`train_bc.py`)

A perfect-hindsight expert (`ml/expert_policy.py`) labels every bar with the action `BUY` / `SELL` / `HOLD` that maximises the next-K-day return given the current position. A small LSTM policy is then trained via supervised cross-entropy with inverse-frequency class weights to counter the natural HOLD imbalance.

| Detail | Value |
|---|---|
| Input | `(batch, 20, 17)` |
| Output | 3-way action logits |
| Loss | Weighted cross-entropy |
| Metrics | Per-class accuracy {HOLD, BUY, SELL} |
| Output | `checkpoints/bc_policy.pth`, `checkpoints/bc_stats.npz` |

### Stage 3 — RL fine-tuning (`train.py --use-ml-forecaster`)

`RecurrentPPO` from sb3-contrib with an LSTM policy. The Gymnasium env optionally pre-computes the trained LSTM forecaster's prediction at every bar and appends it as an extra observation channel — so the RL agent learns on top of a learned signal, not just raw indicators. The policy fallback chain (PPO > BC > heuristic) means the BC policy stays useful as a baseline even after PPO finishes.

| Detail | Value |
|---|---|
| Algorithm | RecurrentPPO (sb3-contrib) |
| Policy | `MlpLstmPolicy` (LSTM hidden=64, 2-layer MLP) |
| Reward | `log(V_t/V_{t-1}) − λ·turnover − GTT_slippage` |
| Obs (default) | `(20, 19)` |
| Obs (`--use-ml-forecaster`) | `(20, 20)` |
| Logs | TensorBoard at `runs/`, optional WandB |
| Output | `checkpoints/ppo_share_forge.zip` |

### Holdout evaluation (`/api/forecast-eval`)

The LSTM forecaster predicts the cumulative log return over its training horizon using only training-cutoff inputs, then we score that single prediction against the realized post-cutoff bars (`load_live()`) — strictly read-only on the holdout side. Reports MAE, RMSE, and directional accuracy. The frontend's "Eval LSTM on holdout" button surfaces this directly.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | service + model + data + DB status |
| `/health` | GET | OpenEnv minimal healthcheck |
| `/api/forecast` | POST | Forecast for `horizon ∈ {1W, 1M, 3M, 6M, 1Y}` with `method ∈ {gbm, ml}` |
| `/api/forecast-eval` | GET | Score the trained LSTM forecaster on post-cutoff bars |
| `/api/data` | GET | training bars (≤ 2026-03-31) |
| `/api/live` | GET | holdout bars (≥ 2026-04-01) — never seen by model |
| `/api/live-action` | POST | run agent on most recent training bars |
| `/api/predict` | POST | stateless action from caller-supplied window |
| `/api/backtest` | POST | run a full task backtest |
| `/api/tasks` | GET | available trading tasks |
| `/api/history/predictions` | GET | recent persisted forecasts |
| `/api/history/backtests` | GET | recent persisted backtests |
| `/api/history/actions` | GET | recent persisted live-action calls |
| `/reset`, `/step`, `/state`, `/ws` | — | OpenEnv contract |
| `/docs` | GET | Swagger UI |
| `/` | GET | ECharts frontend |
| `/gradio` | GET | Gradio playground |

## Frontend

Pure HTML + CSS + vanilla JS, with Apache ECharts loaded from CDN. Five tabs:

1. **Forecast** — pick `1W / 1M / 3M / 6M / 1Y`, view median + 5/25/75/95 percentile bands.
2. **Live Action** — run the trained agent on recent training-window bars; shows BUY / HOLD / SELL.
3. **Backtest** — pick a task, run end-to-end, plot agent equity vs. buy-and-hold.
4. **Holdout** — render `/api/live` bars (post-cutoff) for visual comparison.
5. **History** — last 50 forecasts / backtests / actions from Postgres.

The header shows live pills for service health, data cutoff, and DB status; they refresh every 30 s.

## Asset, Observation, Action, Reward

- **Asset**: `TATAGOLD.NS` (Tata Gold ETF, NSE), daily bars.
- **Observation**: rolling 20-bar window of OHLCV + SMA(10/20) + EMA(12/26) + RSI(14) + MACD + Bollinger + position flag + normalized equity.
- **Action**: `Discrete(3)` — `0=HOLD, 1=BUY, 2=SELL`, with optional GTT trigger price.
- **Reward**: `log(V_t / V_{t-1}) − λ·turnover − GTT_slippage`.

## Tasks

| Tier | Task | Period | Grading |
|---|---|---|---|
| Easy | `easy_long_only` | 2018-2019 | Total return excess vs. buy-and-hold |
| Medium | `medium_volatile` | 2020-2021 | Sharpe ratio |
| Medium | `medium_sideways` | 2022 | Sharpe minus turnover penalty |
| Hard | `hard_adversarial` | 2023-Jun 2024 + synthetic shocks | 0.4·Sharpe + 0.3·Calmar + 0.3·(1 − MaxDD) |

All task date ranges sit safely before the 2026-03-31 cutoff.

## Database

Three tables auto-created on startup via SQLAlchemy `create_all`:

- `predictions` — every `/api/forecast` call (horizon, last close, terminal percentiles, full payload).
- `backtests` — every `/api/backtest` call (task, score, Sharpe, max drawdown, return, trade count).
- `actions_log` — every `/api/predict` and `/api/live-action` call.

Default URL inside Docker: `postgresql+psycopg2://shareforge:shareforge_dev@postgres:5432/shareforge`. Override with `DATABASE_URL`.

## Training Logs

- TensorBoard — `tensorboard --logdir runs/`
- Weights & Biases — `WANDB_API_KEY=… python train.py` streams metrics to a shareable dashboard. Falls back silently if unset.
- Stdout — every PPO rollout prints `ep_rew_mean / ep_len_mean / timesteps`.

## Tech Stack

| Component | Technology |
|---|---|
| RL | Stable-Baselines3 + sb3-contrib (RecurrentPPO with LSTM) |
| Supervised learning | PyTorch (LSTM forecaster + BC policy) |
| Imitation source | Perfect-hindsight expert (`ml/expert_policy.py`) |
| Deep learning | PyTorch (MPS / CUDA / CPU) |
| Environment | Gymnasium |
| Forecasting | LSTM (Gaussian NLL) + GBM Monte Carlo fallback |
| Data | yfinance (cached CSV, hard cutoff at 2026-03-31) |
| Backend | FastAPI + Uvicorn + OpenEnv |
| Database | PostgreSQL + SQLAlchemy 2.x |
| Frontend | Apache ECharts (vanilla JS) |
| Container | Docker + docker compose |

## License

MIT
