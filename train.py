"""
Share-Forge - PPO Training Script.

Trains a Recurrent PPO (LSTM policy) on the Share-Forge trading environment
for Tata Gold ETF, using PyTorch on Apple Silicon (MPS), CUDA, or CPU.

Logs go to TensorBoard under runs/ and, if WANDB_API_KEY is set, to Weights
and Biases for shareable dashboards. Saves the final model to
checkpoints/ppo_share_forge.zip.

Usage:
    python train.py --timesteps 200000 --device auto
    python train.py --task easy_long_only --timesteps 100000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from server.data_loader import load, slice_by_dates
from server.tasks import TASKS, get_task
from models import TaskDifficulty
from server.trading_env import ShareForgeTradingEnv, TradingConfig

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_env(task_type: TaskDifficulty, use_ml_forecaster: bool = False):
    spec = get_task(task_type)
    df = load()
    df_slice = slice_by_dates(df, spec.start, spec.end)

    def _thunk():
        env = ShareForgeTradingEnv(
            df_slice,
            TradingConfig(use_ml_forecaster=use_ml_forecaster),
        )
        return Monitor(env)

    return _thunk


class StdoutLogger(BaseCallback):
    """Prints concise per-rollout metrics so training is visible without TensorBoard."""

    def __init__(self, log_every: int = 1):
        super().__init__()
        self.log_every = log_every
        self._rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts % self.log_every != 0:
            return
        ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        ep_lens = [ep["l"] for ep in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_l = float(np.mean(ep_lens)) if ep_lens else 0.0
        print(
            f"[ROLLOUT {self._rollouts:04d}] "
            f"timesteps={self.num_timesteps} "
            f"ep_rew_mean={mean_r:+.4f} ep_len_mean={mean_l:.1f}",
            flush=True,
        )


def maybe_init_wandb(run_name: str, config: dict):
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
        wandb.init(project="share-forge", name=run_name, config=config, sync_tensorboard=True)
        return wandb
    except Exception as e:
        print(f"[WARN] wandb init failed: {e}")
        return None


def train(
    task_type: str,
    timesteps: int,
    device: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    use_ml_forecaster: bool,
):
    difficulty = TaskDifficulty(task_type)
    print(f"Task:           {difficulty.value}")
    print(f"Device:         {device}")
    print(f"Timesteps:      {timesteps}")
    print(f"Seed:           {seed}")
    print(f"ML forecaster:  {'on' if use_ml_forecaster else 'off'}")

    env_fn = make_env(difficulty, use_ml_forecaster=use_ml_forecaster)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    run_name = f"ppo_lstm_{difficulty.value}{'_ml' if use_ml_forecaster else ''}"
    wandb_run = maybe_init_wandb(run_name, {
        "algo": "RecurrentPPO",
        "policy": "MlpLstmPolicy",
        "task": difficulty.value,
        "timesteps": timesteps,
        "device": device,
        "use_ml_forecaster": use_ml_forecaster,
    })

    policy_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        net_arch=[64, 64],
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(RUNS_DIR),
        seed=seed,
        device=device,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(timesteps // 4, 1000),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=run_name,
        save_vecnormalize=True,
    )
    stdout_cb = StdoutLogger(log_every=1)

    model.learn(
        total_timesteps=timesteps,
        callback=[ckpt_cb, stdout_cb],
        tb_log_name=run_name,
        progress_bar=False,
    )

    final_path = CHECKPOINT_DIR / "ppo_share_forge.zip"
    model.save(str(final_path))
    vec_env.save(str(CHECKPOINT_DIR / "vec_normalize.pkl"))
    print(f"\nSaved final model to {final_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy_long_only", choices=[t.value for t in TASKS.keys()])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-ml-forecaster",
        action="store_true",
        help="Augment the observation with the trained LSTM forecaster's prediction",
    )
    args = parser.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    train(
        task_type=args.task,
        timesteps=args.timesteps,
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_ml_forecaster=args.use_ml_forecaster,
    )


if __name__ == "__main__":
    main()
