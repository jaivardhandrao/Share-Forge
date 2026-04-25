"""
Share-Forge - Inference Script.

MANDATORY environment variables:
    API_BASE_URL  Optional, only used if a sentiment LLM is plugged in.
    MODEL_NAME    Model identifier label.
    HF_TOKEN      Hugging Face / API key.
    IMAGE_NAME    Local Docker image for the Share-Forge environment.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from client import ShareForgeEnv
from models import MarketObservation, TradeAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "share-forge-ppo-lstm")

BENCHMARK = "share_forge"
SUCCESS_THRESHOLD = 0.5
CHECKPOINT_PATH = Path(os.getenv("SHARE_FORGE_CHECKPOINT", "checkpoints/ppo_share_forge.zip"))

TASK_TYPES = [
    "easy_long_only",
    "medium_volatile",
    "medium_sideways",
    "hard_adversarial",
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        from sb3_contrib import RecurrentPPO
        _model = RecurrentPPO.load(str(CHECKPOINT_PATH), device="cpu")
        return _model
    except Exception as exc:
        print(f"  [WARN] Failed to load checkpoint: {exc}", file=sys.stderr)
        return None


def _heuristic(window: np.ndarray, is_long: bool) -> int:
    """Momentum baseline used when no checkpoint is loaded."""
    if window.size == 0:
        return 0
    short_ma = float(window[-5:, 3].mean()) if window.shape[0] >= 5 else float(window[-1, 3])
    long_ma = float(window[:, 3].mean())
    if short_ma > long_ma * 1.005 and not is_long:
        return 1
    if short_ma < long_ma * 0.995 and is_long:
        return 2
    return 0


def predict_action(obs: MarketObservation, lstm_state, episode_starts) -> Tuple[int, object, np.ndarray]:
    window = np.asarray(obs.window_features, dtype=np.float32)
    if window.size == 0:
        return 0, lstm_state, episode_starts

    model = _load_model()
    if model is None:
        is_long = bool(obs.portfolio.is_long) if obs.portfolio else False
        return _heuristic(window, is_long), lstm_state, episode_starts

    obs_batch = window[None, ...]
    try:
        action, lstm_state = model.predict(
            obs_batch,
            state=lstm_state,
            episode_start=episode_starts,
            deterministic=True,
        )
        episode_starts = np.zeros((1,), dtype=bool)
        return int(np.asarray(action).flatten()[0]), lstm_state, episode_starts
    except Exception as exc:
        print(f"  [WARN] predict failed, falling back: {exc}", file=sys.stderr)
        is_long = bool(obs.portfolio.is_long) if obs.portfolio else False
        return _heuristic(window, is_long), lstm_state, episode_starts


async def run_episode(env: ShareForgeEnv, task_type: str) -> float:
    result = await env.reset(task_type=task_type)
    log_start(task=task_type, env=BENCHMARK, model=MODEL_NAME)

    step_num = 0
    rewards: List[float] = []
    final_score = 0.0
    success = False
    lstm_state = None
    episode_starts = np.ones((1,), dtype=bool)

    try:
        while not result.done:
            step_num += 1
            obs: MarketObservation = result.observation
            error = None

            try:
                action_int, lstm_state, episode_starts = predict_action(obs, lstm_state, episode_starts)
                action = TradeAction(action=int(action_int))
                result = await env.step(action)
                reward = float(result.reward) if result.reward is not None else 0.0
                rewards.append(reward)
                action_desc = ["HOLD", "BUY", "SELL"][int(action_int)]
            except Exception as exc:
                error = str(exc).replace("\n", " ")
                reward = 0.0
                rewards.append(reward)
                action_desc = "error"

            log_step(step=step_num, action=action_desc, reward=reward, done=result.done, error=error)

        if rewards:
            final_score = max(0.0, min(1.0, float(rewards[-1])))
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[WARN] Episode failed: {exc}", file=sys.stderr)

    finally:
        log_end(success=success, steps=step_num, score=final_score, rewards=rewards)

    return final_score


async def main() -> None:
    print("=" * 60, file=sys.stderr)
    print("Share-Forge - Inference", file=sys.stderr)
    print(f"Model:      {MODEL_NAME}", file=sys.stderr)
    print(f"Checkpoint: {CHECKPOINT_PATH} (exists={CHECKPOINT_PATH.exists()})", file=sys.stderr)
    print(f"Image:      {IMAGE_NAME}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    env = await ShareForgeEnv.from_docker_image(IMAGE_NAME, env_vars={"PORT": "8000"})
    results = {}

    try:
        for task_type in TASK_TYPES:
            print(f"\n--- Running {task_type.upper()} ---", file=sys.stderr)
            score = await run_episode(env, task_type)
            results[task_type] = score
            print(f"  Score: {score:.4f}", file=sys.stderr)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr)

    avg_score = sum(results.values()) / len(results) if results else 0.0
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("FINAL RESULTS:", file=sys.stderr)
    for task_type, s in results.items():
        print(f"  {task_type:25s}: {s:.4f}", file=sys.stderr)
    print(f"  {'AVERAGE':25s}: {avg_score:.4f}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
