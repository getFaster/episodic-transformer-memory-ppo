import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from episodic_moba_ppo.checkpoint import MARKER_NAME, CheckpointStore
from episodic_moba_ppo.moba_retrieval import MobaSelection
from episodic_moba_ppo.ppo import MuonWithAdamWHeads
from episodic_moba_ppo.training import LinearUpdateScheduler, TrainingRuntime


class TinyEnv:
    def __init__(self) -> None:
        self.episode_step = 0
        self.reset_seeds: list[int] = []
        self.closed = False

    def reset(self, *, seed: int):
        assert 0 <= seed <= 9_999
        self.reset_seeds.append(seed)
        self.episode_step = 0
        return np.array([seed % 7, 0], dtype=np.float32)

    def step(self, action: int):
        assert action in (0, 1)
        self.episode_step += 1
        done = self.episode_step == 3
        observation = np.array([action, self.episode_step], dtype=np.float32)
        return observation, float(action), done, {"length": self.episode_step}

    def close(self) -> None:
        self.closed = True


class TinyLongHistoryModel(torch.nn.Module):
    def __init__(self, fail_at_call=None) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 4)
        self.policy = torch.nn.Linear(4, 2)
        self.value = torch.nn.Linear(4, 1)
        self.forward_calls = 0
        self.batch_sizes: list[int] = []
        self.fail_at_call = fail_at_call

    def forward_long_history(self, observations, contexts, *, arm, attention_config):
        assert arm == "trxl_moba"
        assert attention_config.budget == 256
        assert len(contexts) == observations.shape[0]
        self.forward_calls += 1
        if self.forward_calls == self.fail_at_call:
            raise RuntimeError("injected training failure")
        self.batch_sizes.append(observations.shape[0])
        hidden = torch.tanh(self.encoder(observations))
        memories = hidden[:, None, :].expand(-1, 3, -1)
        return (
            [Categorical(logits=self.policy(hidden))],
            self.value(hidden).squeeze(1),
            memories,
            (),
        )


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_heterogeneous_optimizer_groups_keep_independent_lr_schedules() -> None:
    muon_parameter = torch.nn.Parameter(torch.zeros(1, 1))
    head_parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = MuonWithAdamWHeads(
        torch.optim.SGD([muon_parameter], lr=0.02),
        torch.optim.AdamW([head_parameter], lr=0.0001),
    )
    scheduler = LinearUpdateScheduler(
        optimizer,
        SimpleNamespace(
            muon=SimpleNamespace(initial_lr=0.02, final_lr=0.00067),
            adamw_heads=SimpleNamespace(initial_lr=0.0001, final_lr=0.00000335),
        ),
    )

    assert scheduler.step_to(0) == pytest.approx((0.02, 0.0001))
    assert scheduler.step_to(61) == pytest.approx((0.00067, 0.00000335))
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        (0.00067, 0.00000335)
    )


class DiagnosticTinyModel(TinyLongHistoryModel):
    def forward_long_history(self, observations, contexts, *, arm, attention_config):
        policy, value, memories, _ = super().forward_long_history(
            observations, contexts, arm=arm, attention_config=attention_config
        )
        empty = torch.empty(0, dtype=torch.long)
        routing = tuple(
            tuple(
                MobaSelection(
                    context_indices=empty,
                    context_timesteps=empty,
                    dense_indices=empty,
                    selected_block_indices=empty,
                    selected_block_ranges=(),
                    candidate_block_indices=empty,
                    routing_scores=torch.empty(0),
                )
                for _ in contexts
            )
            for _ in range(3)
        )
        return policy, value, memories, routing


class RecordingLogger:
    def __init__(self) -> None:
        self.logged = []
        self.records = []
        self.artifacts = []
        self.histograms = []

    @property
    def identity(self):
        return {"backend": "recording"}

    def log(self, values, *, step):
        self.logged.append((dict(values), step))

    def log_run_metadata(self, values):
        pass

    def log_records(self, name, records, *, step):
        self.records.append((name, records, step))

    def log_histogram(self, name, values, *, step):
        self.histograms.append((name, list(values), step))

    def log_artifact(self, path, *, name, metadata):
        self.artifacts.append((path, name, dict(metadata)))

    def finish(self):
        pass


class TinyConfig(SimpleNamespace):
    def model_dump(self, *, mode):
        assert mode == "json"
        return {"task": "tiny-test"}


class TinyProvenance(SimpleNamespace):
    def model_dump(self, *, mode):
        assert mode == "json"
        return {"source": "tiny-test"}


def _config():
    return TinyConfig(
        arm="trxl_moba",
        transformer=SimpleNamespace(num_blocks=3, embed_dim=4),
        attention=SimpleNamespace(
            budget=256,
            dense_recent=2,
            search_horizon=8,
            retrieval=SimpleNamespace(block_size=2),
        ),
        seeds=SimpleNamespace(model=1),
        ppo=SimpleNamespace(
            worker_steps=4,
            effective_minibatch_size=8,
            microbatch_size=2,
            gamma=0.9,
            gae_lambda=0.8,
            epochs=1,
            clip_range=0.1,
            value_loss_coefficient=0.5,
            entropy_beta_initial=0.0,
            entropy_beta_final=0.0,
            max_grad_norm=1.0,
            updates=1,
        ),
        checkpointing=SimpleNamespace(every_updates=5, milestone_updates=[1]),
        provenance=TinyProvenance(),
    )


def test_tiny_one_rollout_one_update_recomputes_each_microbatch(tmp_path) -> None:
    model = TinyLongHistoryModel()
    optimizer = CountingSGD(model.parameters())
    environments = [TinyEnv(), TinyEnv()]
    runtime = TrainingRuntime(
        config=_config(),
        model=model,
        optimizer=optimizer,
        environments=environments,
        rollout_steps=4,
        effective_minibatch_size=8,
        microbatch_size=2,
        local_checkpoints=CheckpointStore(tmp_path / "local"),
        drive_checkpoints=CheckpointStore(tmp_path / "drive"),
    )

    runtime.run(updates=1)

    assert runtime.completed_update == 1
    assert runtime.global_step == 8
    assert optimizer.step_calls == 1
    # Four sampling calls + bootstrap + four PPO microbatch forwards.
    assert model.forward_calls == 9
    assert model.batch_sizes[-4:] == [2, 2, 2, 2]
    assert all(environment.closed for environment in environments)
    assert all(len(environment.reset_seeds) == 2 for environment in environments)
    assert (tmp_path / "local" / "update-00001" / MARKER_NAME).is_file()
    assert (tmp_path / "drive" / "update-00001" / MARKER_NAME).is_file()


def test_runtime_reports_progress_from_local_checkpoint(tmp_path) -> None:
    model = TinyLongHistoryModel()
    progress_path = tmp_path / "progress" / "seed1.json"
    runtime = TrainingRuntime(
        config=_config(),
        model=model,
        optimizer=CountingSGD(model.parameters()),
        environments=[TinyEnv(), TinyEnv()],
        rollout_steps=4,
        effective_minibatch_size=8,
        microbatch_size=2,
        local_checkpoints=CheckpointStore(tmp_path / "local"),
        progress_path=progress_path,
    )

    runtime.run(updates=1)

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["phase"] == "checkpoint"
    assert progress["completed_update"] == 1
    assert progress["checkpoint_global_step"] == 8
    assert progress["checkpoint_path"].endswith("local/update-00001")
    assert progress["discard_score"] == 0


def test_rollout_progress_logs_before_optimizer(monkeypatch) -> None:
    monkeypatch.setattr(
        "episodic_moba_ppo.training.ROLLOUT_LOG_INTERVAL_ENV_STEPS", 2
    )
    logger = RecordingLogger()
    model = TinyLongHistoryModel()
    runtime = TrainingRuntime(
        config=_config(),
        model=model,
        optimizer=CountingSGD(model.parameters()),
        environments=[TinyEnv(), TinyEnv()],
        logger=logger,
        rollout_steps=4,
        effective_minibatch_size=8,
        microbatch_size=2,
    )

    runtime.run(updates=1)

    assert [step for _, step in logger.logged] == [2, 4, 6, 8, 8]
    assert all(
        "losses/policy_loss" not in metrics for metrics, _ in logger.logged[:-1]
    )
    assert "losses/policy_loss" in logger.logged[-1][0]


def test_exception_promotes_latest_completed_update_only(tmp_path) -> None:
    config = _config()
    config.ppo.updates = 2
    config.checkpointing.milestone_updates = [31]
    model = TinyLongHistoryModel(fail_at_call=10)
    optimizer = CountingSGD(model.parameters())
    runtime = TrainingRuntime(
        config=config,
        model=model,
        optimizer=optimizer,
        environments=[TinyEnv(), TinyEnv()],
        rollout_steps=4,
        effective_minibatch_size=8,
        microbatch_size=2,
        local_checkpoints=CheckpointStore(tmp_path / "local"),
        drive_checkpoints=CheckpointStore(tmp_path / "drive"),
    )

    with pytest.raises(RuntimeError, match="injected training failure"):
        runtime.run(updates=2)

    assert optimizer.step_calls == 1
    assert (tmp_path / "local" / "update-00001" / MARKER_NAME).is_file()
    assert not (tmp_path / "local" / "update-00002").exists()
    assert (tmp_path / "drive" / "update-00001" / MARKER_NAME).is_file()


def test_runtime_logs_compact_routing_and_samples_detail_artifact(tmp_path) -> None:
    config = _config()
    config.diagnostics = SimpleNamespace(
        enabled=True,
        detailed_routing_sample_rate=1.0,
        artifact_name="routing-details",
        output_path="ignored-by-injection.csv",
    )
    model = DiagnosticTinyModel()
    logger = RecordingLogger()
    output = tmp_path / "results" / "routing-details.csv"
    runtime = TrainingRuntime(
        config=config,
        model=model,
        optimizer=CountingSGD(model.parameters()),
        environments=[TinyEnv(), TinyEnv()],
        logger=logger,
        rollout_steps=4,
        effective_minibatch_size=8,
        microbatch_size=2,
        routing_output_path=output,
    )

    runtime.run(updates=1)

    assert len(logger.logged) == 2
    rollout_metrics, rollout_step = logger.logged[0]
    update_metrics, update_step = logger.logged[1]
    assert rollout_step == update_step == 8
    rollout_required = {
        "charts/episodic_return",
        "charts/episodic_length",
        "charts/success_rate",
        "charts/global_step",
        "losses/explained_variance",
        "moba/selected_distance_mean",
        "moba/selected_distance_p90",
        "moba/fraction_beyond_recent_window",
        "moba/selection_entropy",
        "moba/unique_blocks_selected",
        "moba/retrieved_attention_mass",
        "moba/useful_retrieval_rate",
        "perf/env_steps_per_sec",
        "perf/rollout_time_sec",
    }
    update_required = {
        "charts/global_step",
        "losses/policy_loss",
        "losses/value_loss",
        "losses/entropy",
        "losses/approx_kl",
        "losses/clip_fraction",
        "lora/grad_norm",
        "lora/parameter_delta_norm",
        "perf/update_time_sec",
        "perf/gpu_memory_peak_mb",
    }
    assert rollout_required <= rollout_metrics.keys()
    assert update_required <= update_metrics.keys()
    assert rollout_metrics["routing/layer_0/query_count"] == 8
    assert rollout_metrics["routing/layer_0/no_eligible_fraction"] == 1.0
    assert logger.artifacts == [
        (
            str(output),
            "routing-details",
            {"arm": "trxl_moba", "model_seed": 1, "update": 1, "sample_rate": 1.0},
        )
    ]
    assert output.is_file()
