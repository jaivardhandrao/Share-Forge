"""
Share-Forge - Gradio UI.

Three tabs:
  1. Live Action - fetches recent TATAGOLD.NS bars, runs the trained model,
     shows BUY / HOLD / SELL plus a price chart.
  2. Backtest - replays a chosen historical task with the trained model and
     plots the resulting equity curve vs. buy-and-hold.
  3. Training Logs - quick pointer to TensorBoard / WandB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from server.data_loader import FEATURE_COLUMNS, add_indicators, load, slice_by_dates
from server.grader import compute_metrics, grade
from server.policy_loader import predict
from server.tasks import TASKS, get_task
from server.trading_env import ShareForgeTradingEnv, TradingConfig
from models import TaskDifficulty


def _live_action_tab():
    def run_live(lookback_days: int):
        try:
            df = load()
        except Exception as exc:
            return f"Failed to load data: {exc}", None

        recent = df.tail(int(lookback_days)).reset_index(drop=True)
        if len(recent) < 20:
            return f"Not enough data ({len(recent)} bars). Need at least 20.", None

        feats = recent[[c for c in FEATURE_COLUMNS if c in recent.columns]]
        x = feats.to_numpy(dtype=np.float32)
        mean = x.mean(axis=0, keepdims=True)
        std = np.where(x.std(axis=0, keepdims=True) < 1e-6, 1.0, x.std(axis=0, keepdims=True))
        x_norm = (x - mean) / std
        window = x_norm[-20:]

        action, probs = predict(window.tolist(), is_long=False)
        name = ["HOLD", "BUY", "SELL"][int(action)]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent["date"],
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            name="TATAGOLD.NS",
        ))
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["sma_20"],
            mode="lines", name="SMA(20)",
        ))
        fig.update_layout(
            title=f"TATAGOLD.NS - last {lookback_days} bars",
            xaxis_rangeslider_visible=False,
            height=500,
        )

        prob_str = ""
        if probs is not None:
            prob_str = f"  P(HOLD)={probs[0]:.2f}  P(BUY)={probs[1]:.2f}  P(SELL)={probs[2]:.2f}"
        last_close = float(recent["close"].iloc[-1])
        result = f"### Recommended Action: **{name}**\n\n- Last close: ₹{last_close:.2f}{prob_str}"
        return result, fig

    with gr.Tab("Live Action"):
        gr.Markdown("Run the trained PPO agent on the most recent cached bars of TATAGOLD.NS.")
        with gr.Row():
            lookback = gr.Slider(40, 250, value=120, step=10, label="Lookback days")
            run_btn = gr.Button("Run Agent", variant="primary")
        result_md = gr.Markdown()
        chart = gr.Plot()
        run_btn.click(fn=run_live, inputs=[lookback], outputs=[result_md, chart])


def _backtest_tab():
    def run_backtest(task_value: str) -> Tuple[str, gr.Plot]:
        try:
            difficulty = TaskDifficulty(task_value)
        except ValueError:
            return f"Unknown task '{task_value}'", None

        spec = get_task(difficulty)
        try:
            df = load()
            df_slice = slice_by_dates(df, spec.start, spec.end).reset_index(drop=True)
        except Exception as exc:
            return f"Failed to load data: {exc}", None

        env = ShareForgeTradingEnv(df_slice, TradingConfig())
        obs, info = env.reset()

        done = False
        while not done:
            action, _ = predict(obs[:, :-2].tolist(), is_long=env._is_long)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        equity = env.equity_curve
        prices = df_slice["close"].to_numpy(dtype=np.float64)
        bh = (env.config.initial_cash * (prices / max(prices[0], 1e-8))).tolist()
        bh = bh[: len(equity)]

        summary = compute_metrics(equity, bh, n_trades=env._n_trades)
        graded = grade(summary, spec.grading_mode)

        dates = pd.to_datetime(df_slice["date"]).iloc[: len(equity)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name="Agent"))
        fig.add_trace(go.Scatter(x=dates, y=bh, mode="lines", name="Buy & Hold"))
        fig.update_layout(
            title=f"Backtest: {difficulty.value} ({spec.start} → {spec.end})",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (INR)",
            height=500,
        )

        report = (
            f"### Backtest Results - {difficulty.value}\n\n"
            f"- **Score (mode={spec.grading_mode})**: {graded['reward']:.3f}\n"
            f"- Total return: {summary.total_return:.2%}\n"
            f"- Buy-and-hold return: {summary.buy_and_hold_return:.2%}\n"
            f"- Sharpe ratio: {summary.sharpe:.2f}\n"
            f"- Max drawdown: {summary.max_drawdown:.2%}\n"
            f"- Calmar ratio: {summary.calmar:.2f}\n"
            f"- Number of trades: {summary.n_trades}\n"
            f"- Final value: ₹{summary.final_value:,.0f}"
        )
        return report, fig

    with gr.Tab("Backtest"):
        gr.Markdown("Replay a historical task with the trained agent and compare against buy-and-hold.")
        task_dd = gr.Dropdown(
            choices=[t.value for t in TASKS.keys()],
            value=TaskDifficulty.EASY_LONG_ONLY.value,
            label="Task",
        )
        run_btn = gr.Button("Run Backtest", variant="primary")
        report_md = gr.Markdown()
        chart = gr.Plot()
        run_btn.click(fn=run_backtest, inputs=[task_dd], outputs=[report_md, chart])


def _logs_tab():
    with gr.Tab("Training Logs"):
        gr.Markdown(
            "**Training is run locally** with `python train.py --timesteps 200000`.\n\n"
            "- TensorBoard: `tensorboard --logdir runs/` then open http://localhost:6006\n"
            "- Weights and Biases: set `WANDB_API_KEY` before training to stream metrics to a shareable dashboard.\n"
            "- Final checkpoint: `checkpoints/ppo_share_forge.zip` — auto-loaded by /predict."
        )


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="Share-Forge", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Share-Forge\n"
            "RL trading agent for **Tata Gold ETF (TATAGOLD.NS)**. "
            "PPO + LSTM, trained with PyTorch, served via OpenEnv."
        )
        _live_action_tab()
        _backtest_tab()
        _logs_tab()
    return demo


if __name__ == "__main__":
    create_gradio_app().launch(server_name="0.0.0.0", server_port=7860)
