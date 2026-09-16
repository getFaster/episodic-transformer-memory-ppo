from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from episodic_moba_ppo.minigrid_task import (
    DELAY_CONDITIONS,
    DelayAllocator,
    MiniGridTask,
    MiniGridTaskConfig,
    MiniGridTaskFactory,
    reference_path,
)


class FakeMiniGrid:
    observation_space = SimpleNamespace(shape=(3, 5, 5))
    action_space = SimpleNamespace(n=3)
    max_episode_steps = 9

    def __init__(self) -> None:
        self.seed = None
        self.actions: list[int] = []
        self.closed = False

    def reset(self, *, seed: int):
        self.seed = seed
        return np.full((5, 5, 3), 255, dtype=np.uint8), {}

    def step(self, action: int):
        self.actions.append(action)
        return np.zeros((5, 5, 3), dtype=np.uint8), 1.0, True, False, {"source": "fake"}

    def close(self) -> None:
        self.closed = True

    def render(self):
        return "frame"


def new_task(seed: int = 9):
    raw = FakeMiniGrid()
    task = MiniGridTask(task_seed=seed, make_env=lambda *args, **kwargs: raw)
    return task, raw


def test_reset_uses_local_rng_and_state_restores_future_condition_allocation():
    task, _ = new_task()
    observed = []
    for environment_seed in range(4):
        task.reset(seed=environment_seed)
        observed.append(task.current_delay)
    state = task.state_dict()
    expected = []
    for environment_seed in range(4, 8):
        task.reset(seed=environment_seed)
        expected.append(task.current_delay)

    restored, _ = new_task()
    restored.load_state_dict(state)
    actual = []
    for environment_seed in range(4, 8):
        restored.reset(seed=environment_seed)
        actual.append(restored.current_delay)

    assert all(delay in DELAY_CONDITIONS for delay in observed)
    assert actual == expected


def test_factory_shares_one_checkpointable_delay_allocator_across_environments():
    created: list[FakeMiniGrid] = []

    def make_fake(*args, **kwargs):
        raw = FakeMiniGrid()
        created.append(raw)
        return raw

    factory = MiniGridTaskFactory(MiniGridTaskConfig(task_seed=10), make_env=make_fake)
    first, second = factory(), factory()
    assert first.delay_allocator is second.delay_allocator is factory.allocator
    before = factory.allocator.state_dict()
    first.reset(seed=1)
    second.reset(seed=2)
    after = factory.allocator.state_dict()
    assert after["allocations"] == before["allocations"] + 2
    restored = DelayAllocator(10)
    restored.load_state_dict(after)
    assert restored.next() == factory.allocator.next()


def test_fixed_delay_factory_works_with_generic_reset_without_consuming_allocator():
    factory = MiniGridTaskFactory(
        MiniGridTaskConfig(task_seed=10), make_env=lambda *args, **kwargs: FakeMiniGrid()
    )
    evaluator_task = factory.for_delay(96)()
    evaluator_task.reset(seed=10_000)
    assert evaluator_task.current_delay == 96
    assert factory.allocator.allocations == 0


def test_delay_bridge_is_calibrated_and_does_not_step_base_environment():
    task, raw = new_task()
    cue = task.reset(seed=10_000, delay=32)
    assert cue.shape == (3, 5, 5)
    assert np.all(cue == 1.0)
    for index in range(32):
        observation, reward, done, info = task.step([2])
        assert reward == 0.0
        assert not done
        assert info["delay_phase"] is True
        assert info["actual_delay"] == 32
        assert np.all(observation == 0.0)
        assert raw.actions == []
    observation, reward, done, info = task.step(np.array([2]))
    assert observation.shape == (3, 5, 5)
    assert reward == 1.0
    assert done is True
    assert raw.actions == [2]
    assert info["delay_phase"] is False
    assert info["success"] == 1
    assert info["actual_delay"] == 32
    assert info["backend"] == "injected"
    assert info["episode_record"]["episode_length"] == 33


def test_goal_label_requires_positive_terminal_reward_when_backend_has_no_label():
    task, raw = new_task()
    raw.step = lambda action: (np.zeros((5, 5, 3), dtype=np.uint8), 0.0, True, False, {})
    task.reset(seed=2, delay=32)
    for _ in range(32):
        task.step(0)
    _, _, done, info = task.step(0)
    assert done
    assert info["success"] == 0


def test_reference_paths_match_every_requested_delay():
    for delay in DELAY_CONDITIONS:
        path = reference_path(delay)
        assert path.cue_to_decision_delay == delay
        assert len(path.bridge_actions) == delay
    with pytest.raises(ValueError, match="32, 64, 96, or 128"):
        reference_path(33)


def test_legacy_backend_id_and_load_rejects_different_task_contract():
    config = MiniGridTaskConfig(task_seed=3, backend="gym_minigrid")
    assert config.resolved_environment_id == "MiniGrid-MemoryS9-v0"
    task, _ = new_task()
    state = task.state_dict()
    other, _ = new_task(seed=4)
    with pytest.raises(ValueError, match="config"):
        other.load_state_dict(state)
