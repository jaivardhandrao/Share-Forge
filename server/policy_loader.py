"""
Share-Forge Policy Loader.

Lazy-loads the best available policy at first call and caches it. Fallback
chain (in priority order):

    1. PPO checkpoint   ── checkpoints/ppo_share_forge.zip
    2. BC checkpoint    ── checkpoints/bc_policy.pth
    3. Momentum baseline ── always available, deterministic SMA crossover

Used by /api/predict, /api/live-action, the Gradio UI, and the inference
runner. The exposed `predict()` always returns (action, optional_probs,
source) where `source ∈ {ppo, bc, heuristic}` so the frontend can show
which policy answered.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
PPO_CHECKPOINT = CHECKPOINT_DIR / "ppo_share_forge.zip"
BC_CHECKPOINT = CHECKPOINT_DIR / "bc_policy.pth"

_ppo = None
_bc = None
_load_attempted = {"ppo": False, "bc": False}
_active_source: str = "heuristic"


def _try_load_ppo():
    global _ppo
    if _load_attempted["ppo"]:
        return _ppo
    _load_attempted["ppo"] = True

    ckpt = Path(os.getenv("SHARE_FORGE_CHECKPOINT", str(PPO_CHECKPOINT)))
    if not ckpt.exists():
        return None
    try:
        from sb3_contrib import RecurrentPPO
        _ppo = RecurrentPPO.load(str(ckpt), device="cpu")
        return _ppo
    except Exception:
        try:
            from stable_baselines3 import PPO
            _ppo = PPO.load(str(ckpt), device="cpu")
            return _ppo
        except Exception:
            return None


def _try_load_bc():
    global _bc
    if _load_attempted["bc"]:
        return _bc
    _load_attempted["bc"] = True

    ckpt = Path(os.getenv("SHARE_FORGE_BC_CHECKPOINT", str(BC_CHECKPOINT)))
    if not ckpt.exists():
        return None
    try:
        from ml.bc_model import load_checkpoint
        _bc = load_checkpoint(str(ckpt), device="cpu")
        return _bc
    except Exception:
        return None


def active_source() -> str:
    """Which policy answered the last predict() call."""
    return _active_source


def policy_status() -> dict:
    return {
        "ppo": {"checkpoint": str(PPO_CHECKPOINT), "loaded": _ppo is not None, "exists": PPO_CHECKPOINT.exists()},
        "bc": {"checkpoint": str(BC_CHECKPOINT), "loaded": _bc is not None, "exists": BC_CHECKPOINT.exists()},
        "active_source": _active_source,
    }


def _heuristic(window: np.ndarray, is_long: bool) -> Tuple[int, List[float]]:
    """SMA crossover baseline used when no trained policy is available."""
    if window.size == 0:
        return 0, [1.0, 0.0, 0.0]
    last = window[-1]
    short_ma = float(window[-5:, 3].mean()) if window.shape[0] >= 5 else float(last[3])
    long_ma = float(window[:, 3].mean())
    if short_ma > long_ma * 1.005 and not is_long:
        return 1, [0.1, 0.8, 0.1]
    if short_ma < long_ma * 0.995 and is_long:
        return 2, [0.1, 0.1, 0.8]
    return 0, [0.8, 0.1, 0.1]


def predict(
    window_features: List[List[float]],
    is_long: bool,
) -> Tuple[int, Optional[List[float]]]:
    """
    Returns (action, probs).

    `window_features` is the raw feature window (without position/equity columns).
    The PPO policy expects an obs that includes position+equity columns, so we
    re-attach them here. The BC policy works on the raw window directly.
    """
    global _active_source

    arr = np.asarray(window_features, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        _active_source = "heuristic"
        return 0, [1.0, 0.0, 0.0]

    ppo = _try_load_ppo()
    if ppo is not None:
        try:
            pos_col = np.full((arr.shape[0], 1), 1.0 if is_long else 0.0, dtype=np.float32)
            eq_col = np.zeros((arr.shape[0], 1), dtype=np.float32)
            obs = np.concatenate([arr, pos_col, eq_col], axis=1).astype(np.float32)
            obs_batch = obs[None, ...]
            action, _ = ppo.predict(obs_batch, deterministic=True)
            _active_source = "ppo"
            return int(np.asarray(action).flatten()[0]), None
        except Exception:
            pass

    bc = _try_load_bc()
    if bc is not None:
        try:
            probs = bc.action_probs(arr)
            action = int(np.argmax(probs))
            _active_source = "bc"
            return action, [float(p) for p in probs]
        except Exception:
            pass

    _active_source = "heuristic"
    action, probs = _heuristic(arr, is_long)
    return action, probs
