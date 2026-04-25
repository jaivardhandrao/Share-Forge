"""
Share-Forge - OpenEnv SDK Client.

EnvClient subclass that talks to the Share-Forge FastAPI server over WebSocket.
Compatible with EnvClient.from_docker_image() and EnvClient.from_env().
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import MarketObservation, TradeAction


class ShareForgeEnv(EnvClient[TradeAction, MarketObservation, State]):
    """
    Client for the Share-Forge trading environment.

    Example:
        >>> env = await ShareForgeEnv.from_docker_image("share-forge:latest")
        >>> result = await env.reset(task_type="easy_long_only")
        >>> action = TradeAction(action=1)  # BUY
        >>> result = await env.step(action)
        >>> print(result.reward, result.observation.portfolio.total_value)
    """

    def _step_payload(self, action: TradeAction) -> Dict[str, Any]:
        return {
            "action": int(action.action),
            "trigger_price": action.trigger_price,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MarketObservation]:
        obs_data = payload.get("observation", {}) or {}

        observation = MarketObservation(
            window_features=obs_data.get("window_features", []),
            feature_names=obs_data.get("feature_names", []),
            portfolio=obs_data.get("portfolio"),
            last_close=obs_data.get("last_close", 0.0),
            task_type=obs_data.get("task_type", "easy_long_only"),
            task_id=obs_data.get("task_id", ""),
            instructions=obs_data.get("instructions", ""),
            feedback=obs_data.get("feedback"),
            step_in_episode=obs_data.get("step_in_episode", 0),
            episode_length=obs_data.get("episode_length", 0),
            summary=obs_data.get("summary"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
