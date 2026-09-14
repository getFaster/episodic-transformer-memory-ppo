"""Deterministically seeded Memory-Gym environment adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np


class MemoryGymEnv:
    """Adapter around Gymnasium that requires an explicit episode seed."""

    def __init__(
        self,
        env_id: str,
        reset_options: Mapping[str, Any] | None = None,
        *,
        render_mode: str | None = None,
        make_env: Callable[..., Any] | None = None,
    ) -> None:
        if make_env is None:
            import gymnasium as gym
            import memory_gym  # noqa: F401 - registers environments

            make_env = gym.make
        self._env = make_env(env_id, disable_env_checker=True, render_mode=render_mode)
        self._reset_options = dict(reset_options or {})
        for key in ("start-seed", "num-seeds", "seed"):
            self._reset_options.pop(key, None)

    @property
    def observation_space(self) -> Any:
        space = self._env.observation_space
        shape = tuple(space.shape)
        if len(shape) == 3 and shape[-1] in (1, 3, 4):
            return SimpleNamespace(shape=(shape[-1], shape[0], shape[1]))
        return space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def max_episode_steps(self) -> int:
        value = getattr(self._env, "max_episode_steps", None)
        if value is None:
            value = getattr(getattr(self._env, "spec", None), "max_episode_steps", None)
        if value is None:
            raise AttributeError("environment does not expose max_episode_steps")
        return int(value)

    def reset(self, *, seed: int) -> np.ndarray:
        observation, _ = self._env.reset(
            seed=int(seed), options=dict(self._reset_options)
        )
        return self._normalize_observation(observation)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if isinstance(action, (list, tuple)) and len(action) == 1:
            action = action[0]
        observation, reward, terminated, truncated, info = self._env.step(action)
        return (
            self._normalize_observation(observation),
            float(reward),
            bool(terminated or truncated),
            dict(info),
        )

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _normalize_observation(observation: Any) -> np.ndarray:
        original = np.asarray(observation)
        array = original
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = np.moveaxis(array, -1, 0)
        array = array.astype(np.float32, copy=False)
        if array.size and np.issubdtype(original.dtype, np.integer):
            array = array / 255.0
        return array


def mortar_reset_options(
    base_options: Mapping[str, Any], command_count: int
) -> dict[str, Any]:
    options = dict(base_options)
    for key in ("start-seed", "num-seeds", "seed"):
        options.pop(key, None)
    options["command_count"] = [int(command_count)]
    return options
