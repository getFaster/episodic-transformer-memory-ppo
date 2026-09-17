"""MiniGrid Memory-S9 task adapter used by the generic PPO runtime.

This module deliberately keeps MiniGrid-specific behaviour at the task
boundary.  In particular, the task owns condition sampling and its random
number generator, rather than letting workers independently use NumPy's
process-global RNG.  That makes a resumed sweep allocate exactly the same
delay conditions as an uninterrupted one.

``minigrid`` is imported only when a real environment is constructed.  This
keeps package import and CPU-only unit tests independent of the optional
MiniGrid backend.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata as package_metadata
from typing import Any, Literal

import numpy as np

DelayCondition = Literal[32, 64, 96, 128]
DelayBackend = Literal["minigrid", "gym_minigrid"]
DELAY_CONDITIONS: tuple[DelayCondition, ...] = (32, 64, 96, 128)


@dataclass(frozen=True)
class MiniGridTaskConfig:
    """Immutable construction contract for one Memory-S9 task stream.

    ``delay`` is implemented by a transparent, action-ignored bridge between
    the initial cue observation and the first reference-path decision action.
    It is intentionally in the adapter instead of a fork of MiniGrid's
    ``MemoryEnv`` so backend upgrades do not alter calibrated task semantics.
    """

    task_seed: int
    backend: DelayBackend = "minigrid"
    environment_id: str | None = None
    view_size: int = 3
    tile_size: int = 28
    delay_conditions: tuple[DelayCondition, ...] = DELAY_CONDITIONS
    render_mode: str | None = None

    def __post_init__(self) -> None:
        if self.task_seed < 0:
            raise ValueError("task_seed must be nonnegative")
        if self.backend not in ("minigrid", "gym_minigrid"):
            raise ValueError("backend must be 'minigrid' or 'gym_minigrid'")
        if self.view_size < 3 or self.view_size % 2 == 0:
            raise ValueError("view_size must be an odd integer of at least three")
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive")
        if tuple(self.delay_conditions) != DELAY_CONDITIONS:
            raise ValueError("delay_conditions must be exactly (32, 64, 96, 128)")

    @property
    def resolved_environment_id(self) -> str:
        # Both maintained MiniGrid 3.1.0 and gym-minigrid 1.0.2 register the
        # historical no-separator spelling.  Keep this exact ID: the dashed
        # form is not registered by the maintained package.
        if self.environment_id is not None:
            return self.environment_id
        return "MiniGrid-MemoryS9-v0"


@dataclass(frozen=True)
class ReferencePath:
    """Reference-path calibration for an adapter delay condition.

    ``bridge_actions`` are intentionally ignored by :class:`MiniGridTask`.
    The first following action is the reference decision action.  This is the
    expected calibration for an adapter condition; the runtime separately
    records the observed cue-to-decision gap as ``actual_delay``.
    """

    adapter_bridge_length: DelayCondition
    cue_timestep: int
    decision_timestep: int
    bridge_actions: tuple[int, ...]

    @property
    def cue_to_decision_delay(self) -> int:
        return self.decision_timestep - self.cue_timestep


@dataclass(frozen=True)
class MiniGridEpisodeRecord:
    """Environment-neutral episode result with MiniGrid provenance fields."""

    environment_seed: int
    adapter_bridge_length: int
    cue_timestep: int
    decision_timestep: int
    actual_delay: int
    backend: str
    reward: float
    episodic_return: float
    episode_length: int
    success: int
    terminated: bool
    truncated: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class DelayAllocator:
    """Checkpointable uniform condition allocator shared by an env factory.

    A single allocator is the source of training-condition randomness.  This
    matters when a runtime constructs multiple environments: making every
    environment seed its own generator would accidentally give every worker
    the same condition sequence.  The runtime should checkpoint this object's
    ``state_dict`` once per committed update.
    """

    def __init__(self, seed: int) -> None:
        if int(seed) < 0:
            raise ValueError("delay allocator seed must be nonnegative")
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.allocations = 0

    def next(self) -> DelayCondition:
        self.allocations += 1
        return DELAY_CONDITIONS[int(self._rng.integers(len(DELAY_CONDITIONS)))]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "allocations": self.allocations,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or int(state.get("seed", -1)) != self.seed:
            raise ValueError("checkpoint delay allocator does not match runtime")
        try:
            allocations = int(state["allocations"])
            rng_state = copy.deepcopy(state["rng_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid delay allocator checkpoint state") from error
        if allocations < 0:
            raise ValueError("delay allocator allocations must be nonnegative")
        generator = np.random.default_rng()
        try:
            generator.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("invalid delay allocator RNG state") from error
        self._rng = generator
        self.allocations = allocations


def reference_path(delay: DelayCondition | int) -> ReferencePath:
    """Return the calibrated, backend-independent cue-to-decision path.

    A reset emits the cue at logical step zero.  The adapter then emits exactly
    ``delay`` masked bridge observations.  The next policy action is the first
    action applied to the wrapped Memory-S9 environment (the decision action
    in this task contract).  This avoids relying on MiniGrid internals while
    making the measured delay independently testable.
    """

    checked = _checked_delay(delay)
    return ReferencePath(
        adapter_bridge_length=checked,
        cue_timestep=0,
        decision_timestep=int(checked),
        bridge_actions=(0,) * int(checked),
    )


class MiniGridTask:
    """Gymnasium-native standardized adapter for the MiniGrid Memory S9 task.

    ``reset(seed=...)`` returns a normalized ``float32`` CHW image and samples
    a delay uniformly from the task-local RNG.  Evaluation can supply
    ``delay=...`` to pin an axis condition without advancing that RNG.
    ``step`` always returns ``(observation, reward, done, info)`` and attaches
    an :class:`MiniGridEpisodeRecord` when the episode completes.
    """

    def __init__(
        self,
        config: MiniGridTaskConfig | None = None,
        *,
        task_seed: int | None = None,
        backend: DelayBackend = "minigrid",
        environment_id: str | None = None,
        view_size: int = 3,
        tile_size: int = 28,
        render_mode: str | None = None,
        make_env: Callable[..., Any] | None = None,
        delay_allocator: DelayAllocator | None = None,
        fixed_delay: DelayCondition | int | None = None,
    ) -> None:
        if config is None:
            if task_seed is None:
                raise TypeError("task_seed is required when config is omitted")
            config = MiniGridTaskConfig(
                task_seed=int(task_seed),
                backend=backend,
                environment_id=environment_id,
                view_size=view_size,
                tile_size=tile_size,
                render_mode=render_mode,
            )
        elif task_seed is not None:
            raise TypeError("pass either config or task_seed, not both")
        self.config = config
        self.delay_allocator = delay_allocator or DelayAllocator(config.task_seed)
        self._fixed_delay = None if fixed_delay is None else _checked_delay(fixed_delay)
        self._make_env = make_env
        self._env, self.backend = self._create_environment()
        self._delay: DelayCondition | None = None
        self._environment_seed: int | None = None
        self._bridge_remaining = 0
        self._cue_timestep: int | None = None
        self._decision_timestep: int | None = None
        self._cue_observation: np.ndarray | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._done = True
        self._last_record: MiniGridEpisodeRecord | None = None

    @property
    def observation_space(self) -> Any:
        space = self._env.observation_space
        shape = tuple(getattr(space, "shape", ()))
        if len(shape) == 3 and shape[-1] in (1, 3, 4):
            # A simple namespace would discard bounds/dtype.  The task runtime
            # only requires shape, so preserve the native space when already
            # channel-first and expose a light-weight normalized view otherwise.
            from types import SimpleNamespace

            return SimpleNamespace(shape=(shape[-1], shape[0], shape[1]))
        return space

    @property
    def action_space(self) -> Any:
        # Memory-S9 only needs MiniGrid's native left/right/forward actions
        # (0/1/2).  The hash-pinned checkpoint has a three-logit policy head;
        # exposing MiniGrid's full seven-action space would make strict
        # checkpoint loading fail before the transfer gate can run.
        from types import SimpleNamespace

        return SimpleNamespace(n=3)

    @property
    def max_episode_steps(self) -> int:
        base = getattr(self._env, "max_episode_steps", None)
        if base is None:
            base = getattr(getattr(self._env, "spec", None), "max_episode_steps", None)
        if base is None:
            base = getattr(getattr(self._env, "unwrapped", None), "max_steps", None)
        if base is None:
            raise AttributeError("environment does not expose max_episode_steps")
        return int(base) + max(DELAY_CONDITIONS)

    @property
    def current_adapter_bridge_length(self) -> int | None:
        return self._delay

    @property
    def last_episode_record(self) -> MiniGridEpisodeRecord | None:
        return self._last_record

    def reset(
        self,
        *,
        seed: int,
        adapter_bridge_length: DelayCondition | int | None = None,
    ) -> np.ndarray:
        """Begin an episode, sampling or pinning its adapter bridge length."""

        if adapter_bridge_length is None and self._fixed_delay is not None:
            selected = self._fixed_delay
        elif adapter_bridge_length is None:
            selected = self.delay_allocator.next()
        else:
            selected = _checked_delay(adapter_bridge_length)
        observation, _ = self._env.reset(seed=int(seed))
        self._delay = selected
        self._environment_seed = int(seed)
        self._bridge_remaining = int(selected)
        # The reset observation is the cue at logical timestep zero.  The
        # first action forwarded to Memory-S9 is the decision action.
        self._cue_timestep = 0
        self._decision_timestep = None
        self._cue_observation = self._normalize_observation(observation)
        self._episode_return = 0.0
        self._episode_length = 0
        self._done = False
        self._last_record = None
        return self._cue_observation

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._done or self._delay is None or self._environment_seed is None:
            raise RuntimeError("reset(seed=...) must be called before step")
        action = _scalar_action(action)
        if not isinstance(action, (int, np.integer)) or int(action) not in (0, 1, 2):
            raise ValueError("MiniGrid Memory-S9 actions must be left/right/forward (0, 1, 2)")
        if self._bridge_remaining:
            self._bridge_remaining -= 1
            self._episode_length += 1
            info = self._metadata_info()
            info["delay_phase"] = True
            return self._masked_observation(), 0.0, False, info

        # Capture the observed boundary rather than copying the requested
        # bridge length: _episode_length counts completed bridge timesteps.
        if self._decision_timestep is None:
            self._decision_timestep = self._episode_length
        observation, reward, terminated, truncated, raw_info = self._env.step(action)
        self._episode_length += 1
        self._episode_return += float(reward)
        self._done = bool(terminated or truncated)
        info = dict(raw_info or {})
        info.update(self._metadata_info())
        info["delay_phase"] = False
        if self._done:
            success = _goal_success(bool(terminated), float(reward), info)
            record = MiniGridEpisodeRecord(
                environment_seed=self._environment_seed,
                adapter_bridge_length=int(self._delay),
                cue_timestep=self._required_cue_timestep(),
                decision_timestep=self._required_decision_timestep(),
                actual_delay=(
                    self._required_decision_timestep() - self._required_cue_timestep()
                ),
                backend=self.backend,
                reward=float(reward),
                episodic_return=self._episode_return,
                episode_length=self._episode_length,
                success=int(success),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            self._last_record = record
            info.update(record.asdict())
            info["episode_record"] = record.asdict()
        return self._normalize_observation(observation), float(reward), self._done, info

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint just task allocation state; active episodes are not resumable."""

        return {
            "version": 1,
            "config": asdict(self.config),
            "delay_allocator": self.delay_allocator.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported MiniGrid task state version")
        saved_config = state.get("config")
        if saved_config != asdict(self.config):
            raise ValueError("checkpoint MiniGrid task config does not match runtime")
        try:
            allocator_state = state["delay_allocator"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid MiniGrid task checkpoint state") from error
        self.delay_allocator.load_state_dict(allocator_state)

    def reference_path(self, delay: DelayCondition | int | None = None) -> ReferencePath:
        return reference_path(self._delay if delay is None else delay)  # type: ignore[arg-type]

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        return self._env.render()

    def _create_environment(self) -> tuple[Any, str]:
        if self._make_env is not None:
            return (
                self._make_env(
                    self.config.resolved_environment_id,
                    render_mode=self.config.render_mode,
                ),
                "injected",
            )
        if self.config.backend == "minigrid":
            return _make_maintained_environment(self.config)
        return _make_legacy_environment(self.config)

    def _masked_observation(self) -> np.ndarray:
        if self._cue_observation is None:
            raise RuntimeError("task bridge has no cue observation")
        return np.zeros_like(self._cue_observation)

    def _metadata_info(self) -> dict[str, Any]:
        assert self._delay is not None
        return {
            "adapter_bridge_length": int(self._delay),
            "cue_timestep": self._cue_timestep,
            "decision_timestep": self._decision_timestep,
            "actual_delay": (
                None
                if self._decision_timestep is None or self._cue_timestep is None
                else self._decision_timestep - self._cue_timestep
            ),
            "backend": self.backend,
            "environment_seed": self._environment_seed,
        }

    def _required_cue_timestep(self) -> int:
        if self._cue_timestep is None:
            raise RuntimeError("episode has no recorded cue timestep")
        return self._cue_timestep

    def _required_decision_timestep(self) -> int:
        if self._decision_timestep is None:
            raise RuntimeError("episode completed before a decision action")
        return self._decision_timestep

    @staticmethod
    def _normalize_observation(observation: Any) -> np.ndarray:
        if isinstance(observation, Mapping):
            if "image" not in observation:
                raise ValueError("MiniGrid observation mapping must contain 'image'")
            observation = observation["image"]
        original = np.asarray(observation)
        array = original
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = np.moveaxis(array, -1, 0)
        array = array.astype(np.float32, copy=False)
        if array.size and np.issubdtype(original.dtype, np.integer):
            array = array / 255.0
        return array


class MiniGridTaskFactory:
    """Pickle-friendly environment factory with a reproducible task RNG seed."""

    def __init__(self, config: MiniGridTaskConfig, *, make_env: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._make_env = make_env
        self.allocator = DelayAllocator(config.task_seed)

    def __call__(self) -> MiniGridTask:
        return MiniGridTask(
            self.config,
            make_env=self._make_env,
            delay_allocator=self.allocator,
        )

    def for_adapter_bridge_length(
        self, adapter_bridge_length: DelayCondition | int
    ) -> FixedDelayMiniGridTaskFactory:
        """Return an evaluator factory pinned to one adapter bridge length.

        The condition is validated here so evaluator setup fails before it
        constructs a policy or loads a checkpoint.  Environments from this
        factory can use generic ``reset(seed=...)`` calls without consuming the
        training task allocator.
        """

        return FixedDelayMiniGridTaskFactory(
            self, _checked_delay(adapter_bridge_length)
        )


class FixedDelayMiniGridTaskFactory:
    """Generic factory adapter which fixes a MiniGrid evaluation axis value."""

    def __init__(self, parent: MiniGridTaskFactory, adapter_bridge_length: DelayCondition) -> None:
        self.parent = parent
        self.adapter_bridge_length = adapter_bridge_length

    @property
    def allocator(self) -> DelayAllocator:
        return self.parent.allocator

    def __call__(self) -> MiniGridTask:
        return MiniGridTask(
            self.parent.config,
            make_env=self.parent._make_env,
            delay_allocator=self.parent.allocator,
            fixed_delay=self.adapter_bridge_length,
        )


def make_env(
    config: MiniGridTaskConfig,
    *,
    make_environment: Callable[..., Any] | None = None,
) -> MiniGridTask:
    """Convenience constructor used by experiment registries."""

    return MiniGridTask(config, make_env=make_environment)


def _make_maintained_environment(config: MiniGridTaskConfig) -> tuple[Any, str]:
    try:
        import gymnasium as gym
        import minigrid
        from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MiniGrid backend 'minigrid' is unavailable; install minigrid==3.1.0 "
            "or explicitly select backend='gym_minigrid' for the transfer fallback"
        ) from error
    # Set the base environment's view size before rendering.  Applying
    # ViewSizeWrapper outside RGBImgPartialObsWrapper leaves the renderer at
    # the base 7x7 view and silently produces 196x196 images.
    env = gym.make(
        config.resolved_environment_id,
        render_mode=config.render_mode,
        agent_view_size=config.view_size,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=config.tile_size)
    env = ImgObsWrapper(env)
    return env, f"minigrid=={getattr(minigrid, '__version__', _distribution_version('minigrid'))}"


def _make_legacy_environment(config: MiniGridTaskConfig) -> tuple[Any, str]:
    try:
        import gym
        import gym_minigrid
        from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MiniGrid backend 'gym_minigrid' is unavailable; install gym-minigrid==1.0.2"
        ) from error
    env = gym.make(config.resolved_environment_id, agent_view_size=config.view_size)
    env = RGBImgPartialObsWrapper(env, tile_size=config.tile_size)
    env = ImgObsWrapper(env)
    return env, f"gym-minigrid=={getattr(gym_minigrid, '__version__', _distribution_version('gym-minigrid'))}"


def _distribution_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "unknown"


def _checked_delay(delay: DelayCondition | int | None) -> DelayCondition:
    if delay not in DELAY_CONDITIONS:
        raise ValueError("delay must be one of 32, 64, 96, or 128")
    return int(delay)  # type: ignore[return-value]


def _scalar_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        if action.size != 1:
            raise ValueError("MiniGrid actions must contain exactly one scalar")
        return action.reshape(-1)[0].item()
    if isinstance(action, (list, tuple)):
        if len(action) != 1:
            raise ValueError("MiniGrid actions must contain exactly one scalar")
        return action[0]
    return action


def _goal_success(terminated: bool, reward: float, info: Mapping[str, Any]) -> bool:
    """Use terminal reward as the portable MiniGrid goal-success definition."""

    if "success" in info:
        return bool(info["success"])
    if "is_success" in info:
        return bool(info["is_success"])
    return bool(terminated and reward > 0.0)
