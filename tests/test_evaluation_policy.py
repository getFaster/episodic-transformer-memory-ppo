from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import model as legacy_model_module
from episodic_moba_ppo.checkpoint import CheckpointIntegrityError, CheckpointStore
from episodic_moba_ppo.commands import evaluate
from episodic_moba_ppo.config import TrainConfig, load_config
from episodic_moba_ppo.evaluation import EpisodeRecord
from episodic_moba_ppo.evaluation_policy import (
    TrainingCheckpoint,
    TrainedLongHistoryPolicy,
    load_training_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> TrainConfig:
    loaded = load_config(ROOT / "configs" / "trxl_moba_command40.yaml")
    assert isinstance(loaded, TrainConfig)
    return loaded


def _payload(config: TrainConfig, update: int = 3) -> dict:
    return {
        "model": {},
        "config": config.model_dump(mode="json"),
        "provenance": config.provenance.model_dump(mode="json"),
        "counters": {
            "completed_update": update,
            "global_step": update * 16_384,
        },
    }


def test_training_checkpoint_accepts_directory_and_payload_path(tmp_path):
    directory = CheckpointStore(tmp_path).commit(3, _payload(_config()), {})

    from_directory = load_training_checkpoint(directory)
    from_payload = load_training_checkpoint(directory / "training_state.pt")

    assert from_directory.update == 3
    assert from_payload.payload_sha256 == from_directory.payload_sha256
    assert from_directory.config.arm == "trxl_moba"


def test_training_checkpoint_rejects_markerless_payload(tmp_path):
    directory = tmp_path / "update-00003"
    directory.mkdir()
    torch.save(_payload(_config()), directory / "training_state.pt")

    with pytest.raises(CheckpointIntegrityError, match="missing commit_success"):
        load_training_checkpoint(directory / "training_state.pt")


def test_training_checkpoint_rejects_provenance_disagreement(tmp_path):
    payload = _payload(_config())
    payload["provenance"] = {**payload["provenance"], "checkpoint_path": "other"}
    directory = CheckpointStore(tmp_path).commit(3, payload, {})

    with pytest.raises(CheckpointIntegrityError, match="provenance disagrees"):
        load_training_checkpoint(directory)


def test_stateful_policy_uses_saved_arm_and_resets_episode_trace(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, config, observation_space, action_shape, max_steps):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.max_steps = max_steps

        def enable_lora(self, **kwargs):
            pass

        def forward_long_history(self, obs, contexts, *, arm, attention_config):
            assert arm == "trxl_moba"
            assert contexts[0].dense_states.device.type == "cpu"
            assert contexts[0].dense_timesteps.device.type == "cpu"
            assert all(block.states.device.type == "cpu" for block in contexts[0].old_blocks)
            timestep = contexts[0].query_timestep
            probs = torch.tensor([[0.0, 1.0]], device=obs.device)
            branch = torch.distributions.Categorical(probs=probs)
            memories = torch.full((1, 3, 384), float(timestep), device=obs.device)
            return [branch], torch.zeros(1, device=obs.device), memories, ()

    monkeypatch.setattr(legacy_model_module, "ActorCriticModel", FakeModel)
    monkeypatch.setattr(
        "episodic_moba_ppo.evaluation_policy.load_legacy_checkpoint",
        lambda *args: ({"marker": torch.zeros(())}, {}),
    )
    config = _config()
    checkpoint = TrainingCheckpoint(
        directory=ROOT,
        payload_path=ROOT / "unused",
        payload_sha256="a" * 64,
        update=31,
        payload={"model": {"marker": torch.zeros(())}},
        config=config,
    )
    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(4,)),
        action_space=SimpleNamespace(n=2),
        max_episode_steps=900,
    )
    policy = TrainedLongHistoryPolicy(
        checkpoint,
        env,
        repo_root=ROOT,
        expected_arm="trxl_moba",
        expected_model_seed=1,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(4)

    assert policy.act(np.zeros(4, dtype=np.float32), generator) == [1]
    assert policy._trace.reference().query_timestep == 1
    # The second decision has an actual dense-history token.  It remains in
    # the CPU trace/context; CUDA transfer is owned by the transformer after
    # routing, so full episode history is never copied by the evaluator.
    assert policy.act(np.zeros(4, dtype=np.float32), generator) == [1]
    assert policy._trace.reference().query_timestep == 2
    assert all(not parameter.requires_grad for parameter in policy.model.parameters())
    policy.reset()
    assert policy._trace.reference().query_timestep == 0


def test_evaluate_cli_locks_protocol_and_records_checkpoint(monkeypatch, tmp_path):
    observed_options = []
    observed_specs = []
    document = {}

    class FakeEnv:
        def __init__(self, env_id, options):
            assert env_id == "MortarMayhem-Grid-v0"
            observed_options.append(options)

        def reset(self, *, seed):
            assert seed == 10_000

        def close(self):
            pass

    class FakePolicy:
        checkpoint_sha256 = "b" * 64
        checkpoint_update = 31

    def factory(checkpoint, env, **kwargs):
        assert kwargs["expected_arm"] == "trxl_moba"
        assert kwargs["expected_model_seed"] == 1
        return FakePolicy()

    def fake_evaluate_policy(**kwargs):
        specs = tuple(kwargs["specs"])
        observed_specs.append(specs)
        command_count = kwargs["command_count"]
        return [
            EpisodeRecord(
                arm=kwargs["arm"],
                model_seed=kwargs["model_seed"],
                command_count=command_count,
                environment_seed=10_000,
                action_repeat=0,
                action_seed=specs[0].action_seed,
                reward=float(command_count) / 10,
                normalized_return=1.0,
                success=1,
                episode_length=1,
                commands_completed_fraction=1.0,
                commands_completed_count=command_count,
                checkpoint_sha256=kwargs["checkpoint_sha256"],
                checkpoint_update=kwargs["checkpoint_update"],
            )
        ]

    monkeypatch.setattr(evaluate, "MemoryGymEnv", FakeEnv)
    monkeypatch.setattr(evaluate, "_load_factory", lambda _: factory)
    monkeypatch.setattr(evaluate, "evaluate_policy", fake_evaluate_policy)
    monkeypatch.setattr(
        evaluate,
        "atomic_write_json",
        lambda path, value: document.update(value),
    )

    result = evaluate.main(
        [
            "--checkpoint",
            str(tmp_path / "update-00031"),
            "--arm",
            "trxl_moba",
            "--model-seed",
            "1",
            "--output",
            str(tmp_path / "evaluation.json"),
            "--repo-root",
            str(ROOT),
        ]
    )

    assert result == 0
    assert [options["command_count"] for options in observed_options] == [
        [10],
        [20],
        [30],
        [40],
        [50],
        [60],
        [80],
    ]
    assert all(options["explosion_delay"] == [5] for options in observed_options)
    assert all(len(specs) == 150 for specs in observed_specs)
    assert document["checkpoint_sha256"] == "b" * 64
    assert document["checkpoint_update"] == 31
