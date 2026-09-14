from types import SimpleNamespace

import numpy as np

from episodic_moba_ppo.environment import MemoryGymEnv, mortar_reset_options


class FakeGymEnv:
    observation_space = SimpleNamespace(shape=(2, 3, 3))
    action_space = SimpleNamespace(n=5)
    max_episode_steps = 9

    def __init__(self):
        self.reset_call = None
        self.closed = False

    def reset(self, *, seed, options):
        self.reset_call = (seed, options)
        return np.full((2, 3, 3), 255, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((2, 3, 3), dtype=np.uint8), 0.5, False, True, {"x": 1}

    def close(self):
        self.closed = True

    def render(self):
        return "frame"


def test_reset_is_explicit_and_strips_legacy_seed_selectors():
    raw = FakeGymEnv()
    env = MemoryGymEnv(
        "fake",
        {"start-seed": 0, "num-seeds": 10, "seed": 3, "command_count": [10]},
        make_env=lambda *args, **kwargs: raw,
    )
    observation = env.reset(seed=10_007)
    assert raw.reset_call == (10_007, {"command_count": [10]})
    assert observation.shape == (3, 2, 3)
    assert observation.dtype == np.float32
    assert np.all(observation == 1.0)
    assert env.observation_space.shape == (3, 2, 3)


def test_step_treats_truncation_as_done_and_flattens_single_action():
    raw = FakeGymEnv()
    env = MemoryGymEnv("fake", make_env=lambda *args, **kwargs: raw)
    _, reward, done, info = env.step([2])
    assert reward == 0.5
    assert done is True
    assert info == {"x": 1}


def test_mortar_options_fix_command_count_without_mutating_input():
    original = {
        "start-seed": 1,
        "num-seeds": 2,
        "command_count": [10],
        "explosion_delay": [5],
    }
    result = mortar_reset_options(original, 40)
    assert result == {"command_count": [40], "explosion_delay": [5]}
    assert original["command_count"] == [10]
