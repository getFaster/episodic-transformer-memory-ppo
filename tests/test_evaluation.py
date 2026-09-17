from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from episodic_moba_ppo.commands import eval_pretrained
from episodic_moba_ppo.evaluation import (
    EVALUATION_SEEDS,
    EpisodeRecord,
    EpisodeSpec,
    baseline_document,
    episode_specs,
    evaluate_policy,
    normalized_return,
    paired_action_seed,
)


class FakePolicy:
    def __init__(self):
        self.resets = 0
        self.generators = []

    def reset(self):
        self.resets += 1

    def act(self, observation, generator):
        self.generators.append(generator)
        return 0


class FakeEnv:
    def __init__(self):
        self.seed = None
        self.closed = False

    def reset(self, *, seed):
        self.seed = seed
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        return (
            np.zeros(1),
            0.5,
            True,
            {
                "reward": 0.5,
                "length": 7,
                "success": 0,
                "commands_completed": 0.5,
            },
        )

    def close(self):
        self.closed = True


def test_episode_specs_are_paired_stable_and_reserved():
    specs = episode_specs(2)
    assert len(specs) == 100
    assert specs[0].environment_seed == 10_000
    assert specs[-1].environment_seed == 10_049
    assert paired_action_seed(10_003, 1) == paired_action_seed(10_003, 1)
    with pytest.raises(ValueError):
        episode_specs(1, [9_999])


def test_evaluate_policy_records_fraction_and_count():
    policy = FakePolicy()
    created = []

    def factory():
        env = FakeEnv()
        created.append(env)
        return env

    records = evaluate_policy(
        policy=policy,
        env_factory=factory,
        command_count=10,
        specs=[EpisodeSpec(10_000, 0, 123)],
        arm="trxl",
        model_seed=1,
        generator_factory=lambda seed: ("generator", seed),
    )
    assert records[0].normalized_return == pytest.approx(0.5)
    assert records[0].commands_completed_count == 5
    assert records[0].episode_length == 7
    assert policy.generators == [("generator", 123)]
    assert created[0].closed


def test_baseline_gate_requires_full_protocol_and_applies_both_thresholds():
    template = EpisodeRecord(
        arm="pretrained_trxl",
        model_seed=None,
        command_count=10,
        environment_seed=10_000,
        action_repeat=0,
        action_seed=1,
        reward=1.0,
        normalized_return=1.0,
        success=1,
        episode_length=118,
        commands_completed_fraction=1.0,
        commands_completed_count=10,
    )
    records = [
        replace(
            template,
            environment_seed=seed,
            action_repeat=repeat,
            action_seed=paired_action_seed(seed, repeat),
        )
        for seed in EVALUATION_SEEDS
        for repeat in range(2)
    ]
    document = baseline_document(records, source_commit="abc", checkpoint_sha256="def")
    assert document["passed"] is True
    assert document["protocol"]["model_max_episode_steps"] == 119
    records[0] = replace(records[0], success=0, normalized_return=0.0, reward=0.0)
    document = baseline_document(
        records,
        source_commit="abc",
        checkpoint_sha256="def",
        success_threshold=0.995,
        normalized_return_threshold=0.995,
    )
    assert document["passed"] is False


def test_normalized_return_rejects_invalid_reward_scale():
    with pytest.raises(ValueError):
        normalized_return(0.0, 10, reward_command_success=0.0)


def test_pretrained_command_preserves_historical_environment_and_model_horizon(
    monkeypatch, tmp_path
):
    repo_root = Path(__file__).resolve().parents[1]
    observed = {"environments": [], "policy_kwargs": None, "document": None}

    class ProbeEnv:
        def __init__(self, name, reset_options):
            observed["environments"].append((name, reset_options))

        def reset(self, *, seed):
            assert seed == 10_000

        def close(self):
            pass

    class ProbePolicy:
        def __init__(self, state_dict, legacy_config, env, **kwargs):
            observed["policy_kwargs"] = kwargs

    template = EpisodeRecord(
        arm="pretrained_trxl",
        model_seed=None,
        command_count=10,
        environment_seed=10_000,
        action_repeat=0,
        action_seed=1,
        reward=1.0,
        normalized_return=1.0,
        success=1,
        episode_length=119,
        commands_completed_fraction=1.0,
        commands_completed_count=10,
    )
    records = [
        replace(
            template,
            environment_seed=seed,
            action_repeat=repeat,
            action_seed=paired_action_seed(seed, repeat),
        )
        for seed in EVALUATION_SEEDS
        for repeat in range(2)
    ]

    monkeypatch.setattr(eval_pretrained, "MemoryGymEnv", ProbeEnv)
    monkeypatch.setattr(eval_pretrained, "LegacyTrXLPolicy", ProbePolicy)
    monkeypatch.setattr(
        eval_pretrained,
        "load_legacy_checkpoint",
        lambda checkpoint, expected_hash: ({}, {"transformer": {}}),
    )
    monkeypatch.setattr(eval_pretrained, "evaluate_policy", lambda **kwargs: records)
    monkeypatch.setattr(
        eval_pretrained,
        "atomic_write_json",
        lambda path, document: observed.__setitem__("document", document),
    )

    result = eval_pretrained.main(
        [
            "--config",
            str(repo_root / "configs" / "pretrained_eval.yaml"),
            "--repo-root",
            str(repo_root),
            "--output",
            str(tmp_path / "baseline_reference.json"),
        ]
    )

    assert result == 0
    assert observed["environments"][0][1]["explosion_delay"] == [5]
    assert observed["policy_kwargs"] == {"model_max_episode_steps": 119}
    assert observed["document"]["protocol"]["model_max_episode_steps"] == 119
