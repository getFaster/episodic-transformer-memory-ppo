"""Synchronous long-history PPO training runtime."""

from __future__ import annotations

import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from episodic_moba_ppo.checkpoint import CheckpointStore
from episodic_moba_ppo.diagnostics import RoutingDiagnostics
from episodic_moba_ppo.episodic_memory import TraceRef, TraceRegistry
from episodic_moba_ppo.logging import NoOpLogger, RunLogger
from episodic_moba_ppo.ppo import accumulate_effective_minibatch, linear_learning_rate
from episodic_moba_ppo.runtime import (
    assemble_checkpoint_state,
    restore_checkpoint_state,
)


class TrainingInterrupted(RuntimeError):
    pass


class TrainingSeedAllocator:
    """A checkpointable seed stream confined to the training range 0..9999."""

    def __init__(self, model_seed: int, start: int = 0, count: int = 10_000) -> None:
        if start < 0 or count < 1 or start + count > 10_000:
            raise ValueError("training seed pool must remain within 0..9999")
        self._rng = np.random.default_rng(int(model_seed))
        self.start = int(start)
        self.count = int(count)

    def next(self) -> int:
        return int(self._rng.integers(self.start, self.start + self.count))

    def state_dict(self) -> dict[str, Any]:
        return dict(self._rng.bit_generator.state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.bit_generator.state = dict(state)


class LinearUpdateScheduler:
    """Set a single LR for every optimizer step in one PPO update."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.update = 0

    def step_to(self, update: int) -> float:
        self.update = int(update)
        lr = linear_learning_rate(self.update)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, int]:
        return {"update": self.update}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step_to(int(state["update"]))


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    references: list[TraceRef]
    episode_records: list[dict[str, Any]]

    def __len__(self) -> int:
        return self.observations.shape[0]


class TrainingRuntime:
    """A genuine, intentionally simple synchronous PPO implementation."""

    def __init__(
        self,
        *,
        config: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        environments: Sequence[Any],
        device: torch.device | str = "cpu",
        local_checkpoints: CheckpointStore | None = None,
        drive_checkpoints: CheckpointStore | None = None,
        logger: RunLogger | None = None,
        rollout_steps: int | None = None,
        effective_minibatch_size: int | None = None,
        microbatch_size: int | None = None,
        routing_output_path: str | Path | None = None,
    ) -> None:
        if not environments:
            raise ValueError("at least one environment is required")
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.envs = list(environments)
        self.device = torch.device(device)
        self.logger = logger or NoOpLogger()
        self.local_checkpoints = local_checkpoints
        self.drive_checkpoints = drive_checkpoints
        self.rollout_steps = rollout_steps or int(config.ppo.worker_steps)
        self.effective_minibatch_size = effective_minibatch_size or int(
            config.ppo.effective_minibatch_size
        )
        self.microbatch_size = microbatch_size or int(config.ppo.microbatch_size)
        self.seed_allocator = TrainingSeedAllocator(
            config.seeds.model,
            getattr(config.seeds, "environment_start", 0),
            getattr(config.seeds, "environment_count", 10_000),
        )
        self.traces = TraceRegistry(
            config.transformer.num_blocks, config.transformer.embed_dim
        )
        self.scheduler = LinearUpdateScheduler(optimizer)
        self.completed_update = 0
        self.global_step = 0
        self._stop_requested = False
        self._observations: list[np.ndarray] = []
        self._initialized = False
        diagnostics = getattr(config, "diagnostics", None)
        self.routing_diagnostics = None
        if (
            diagnostics is not None
            and diagnostics.enabled
            and config.arm == "trxl_moba"
        ):
            self.routing_diagnostics = RoutingDiagnostics(
                sample_rate=diagnostics.detailed_routing_sample_rate,
                arm=config.arm,
                model_seed=config.seeds.model,
                random_seed=config.seeds.model,
            )
        configured_output = getattr(diagnostics, "output_path", None)
        self.routing_output_path = Path(
            routing_output_path or configured_output or "results/routing-details.csv"
        )

    def _initialize_environments(self) -> None:
        self._observations = []
        self.traces = TraceRegistry(
            self.config.transformer.num_blocks, self.config.transformer.embed_dim
        )
        for worker, env in enumerate(self.envs):
            self._observations.append(env.reset(seed=self.seed_allocator.next()))
            self.traces.start_worker(worker)
        self._initialized = True

    def _contexts(self, references: Sequence[TraceRef]) -> list[Any]:
        retrieval = self.config.attention.retrieval
        return [
            self.traces.context(
                reference,
                dense_recent=self.config.attention.dense_recent,
                search_horizon=self.config.attention.search_horizon,
                block_size=retrieval.block_size,
                # only summaries, dense tokens, and selected blocks reach the accelerator
                device="cpu",
            )
            for reference in references
        ]

    def _forward(
        self,
        observations: torch.Tensor,
        references: Sequence[TraceRef],
        *,
        record_routing: bool = False,
    ):
        contexts = self._contexts(references)
        result = self.model.forward_long_history(
            observations.to(self.device, dtype=torch.float32),
            contexts,
            arm=self.config.arm,
            attention_config=self.config.attention,
        )
        if record_routing and self.routing_diagnostics is not None:
            self.routing_diagnostics.observe(
                result[3], references, update=self.completed_update + 1
            )
        return result

    def collect_rollout(self) -> RolloutBatch:
        if not self._initialized:
            self._initialize_environments()
        workers = len(self.envs)
        observations: list[list[torch.Tensor]] = [[] for _ in self.envs]
        actions: list[list[torch.Tensor]] = [[] for _ in self.envs]
        log_probs: list[list[torch.Tensor]] = [[] for _ in self.envs]
        values: list[list[torch.Tensor]] = [[] for _ in self.envs]
        rewards = torch.zeros((workers, self.rollout_steps), dtype=torch.float32)
        dones = torch.zeros((workers, self.rollout_steps), dtype=torch.bool)
        references: list[list[TraceRef]] = [[] for _ in self.envs]
        records: list[dict[str, Any]] = []

        for timestep in range(self.rollout_steps):
            if self._stop_requested:
                raise TrainingInterrupted(
                    "interrupted during rollout; partial rollout discarded"
                )
            refs = [self.traces.active_trace(w).reference() for w in range(workers)]
            obs_tensor = torch.as_tensor(
                np.stack(self._observations), dtype=torch.float32
            )
            with torch.no_grad():
                policy, value, new_memories, _ = self._forward(
                    obs_tensor, refs, record_routing=True
                )
                sampled = torch.stack([branch.sample() for branch in policy], dim=1)
                old_lp = torch.stack(
                    [branch.log_prob(sampled[:, i]) for i, branch in enumerate(policy)],
                    dim=1,
                )
            for worker, env in enumerate(self.envs):
                observations[worker].append(obs_tensor[worker].cpu())
                actions[worker].append(sampled[worker].cpu())
                log_probs[worker].append(old_lp[worker].cpu())
                values[worker].append(value[worker].cpu())
                references[worker].append(refs[worker])
                self.traces.append(worker, new_memories[worker])
                action = sampled[worker]
                env_action: Any = (
                    int(action.item())
                    if action.numel() == 1
                    else action.cpu().numpy()
                )
                next_obs, reward, done, info = env.step(env_action)
                rewards[worker, timestep] = float(reward)
                dones[worker, timestep] = bool(done)
                if done:
                    records.append(dict(info))
                    next_obs = env.reset(seed=self.seed_allocator.next())
                    self.traces.reset_worker(worker)
                self._observations[worker] = next_obs

        refs = [self.traces.active_trace(w).reference() for w in range(workers)]
        obs_tensor = torch.as_tensor(np.stack(self._observations), dtype=torch.float32)
        with torch.no_grad():
            _, bootstrap, _, _ = self._forward(obs_tensor, refs)
        value_tensor = torch.stack([torch.stack(row) for row in values])
        advantages = torch.zeros_like(value_tensor)
        last_advantage = torch.zeros(workers)
        last_value = bootstrap.detach().cpu()
        for timestep in reversed(range(self.rollout_steps)):
            mask = (~dones[:, timestep]).float()
            delta = rewards[:, timestep] + self.config.ppo.gamma * last_value * mask
            delta -= value_tensor[:, timestep]
            last_advantage = delta + (
                self.config.ppo.gamma
                * self.config.ppo.gae_lambda
                * last_advantage
                * mask
            )
            advantages[:, timestep] = last_advantage
            last_value = value_tensor[:, timestep]

        return RolloutBatch(
            observations=torch.stack(
                [torch.stack(row) for row in observations]
            ).flatten(0, 1),
            actions=torch.stack([torch.stack(row) for row in actions]).flatten(0, 1),
            old_log_probs=torch.stack(
                [torch.stack(row) for row in log_probs]
            ).flatten(0, 1),
            values=value_tensor.flatten(),
            advantages=advantages.flatten(),
            references=[reference for row in references for reference in row],
            episode_records=records,
        )

    def _optimize_effective_minibatch(self, batch: Mapping[str, Any]) -> float:
        clip_range = self.config.ppo.clip_range
        value_coefficient = self.config.ppo.value_loss_coefficient
        progress = self.completed_update / max(1, self.config.ppo.updates - 1)
        beta = self.config.ppo.entropy_beta_initial + progress * (
            self.config.ppo.entropy_beta_final - self.config.ppo.entropy_beta_initial
        )

        def loss_fn(microbatch: Mapping[str, Any], normalized: torch.Tensor):
            policy, value, _, _ = self._forward(
                microbatch["observations"], microbatch["references"]
            )
            new_log_probs = torch.stack(
                [
                    branch.log_prob(microbatch["actions"][:, index])
                    for index, branch in enumerate(policy)
                ],
                dim=1,
            )
            log_ratio = new_log_probs - microbatch["old_log_probs"]
            ratio = log_ratio.exp()
            expanded_advantage = normalized[:, None].expand_as(ratio)
            surrogate = torch.minimum(
                ratio * expanded_advantage,
                ratio.clamp(1 - clip_range, 1 + clip_range) * expanded_advantage,
            )
            returns = microbatch["values"] + microbatch["advantages"]
            clipped_value = microbatch["values"] + (
                value - microbatch["values"]
            ).clamp(-clip_range, clip_range)
            value_loss = torch.maximum(
                (value - returns).square(), (clipped_value - returns).square()
            ).mean()
            entropy = torch.stack([branch.entropy() for branch in policy], dim=1)
            return (
                -surrogate.mean()
                + value_coefficient * value_loss
                - beta * entropy.sum(1).mean()
            )

        result = accumulate_effective_minibatch(
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            loss_fn=loss_fn,
            microbatch_size=self.microbatch_size,
            expected_size=self.effective_minibatch_size,
            max_grad_norm=self.config.ppo.max_grad_norm,
        )
        return result.loss

    def optimize_rollout(self, rollout: RolloutBatch) -> float:
        if len(rollout) % self.effective_minibatch_size:
            raise ValueError("rollout must divide into complete effective minibatches")
        losses: list[float] = []
        for _ in range(self.config.ppo.epochs):
            permutation = torch.randperm(len(rollout))
            for start in range(0, len(rollout), self.effective_minibatch_size):
                if self._stop_requested:
                    raise TrainingInterrupted(
                        "interrupted during PPO; partial update discarded"
                    )
                indices = permutation[start : start + self.effective_minibatch_size]
                selected = indices.tolist()
                batch = {
                    "observations": rollout.observations[indices].to(self.device),
                    "actions": rollout.actions[indices].to(self.device),
                    "old_log_probs": rollout.old_log_probs[indices].to(self.device),
                    "values": rollout.values[indices].to(self.device),
                    "advantages": rollout.advantages[indices].to(self.device),
                    "references": [rollout.references[index] for index in selected],
                }
                losses.append(self._optimize_effective_minibatch(batch))
        return float(np.mean(losses))

    def _checkpoint(self) -> Path | None:
        if self.local_checkpoints is None:
            return None
        payload = assemble_checkpoint_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            completed_update=self.completed_update,
            global_step=self.global_step,
            config=self.config,
            wandb_identity=self.logger.identity,
            provenance=self.config.provenance.model_dump(mode="json"),
            seed_streams={"environment": self.seed_allocator.state_dict()},
        )
        path = self.local_checkpoints.commit(
            self.completed_update, payload, {"global_step": self.global_step}
        )
        milestones = set(self.config.checkpointing.milestone_updates)
        if self.drive_checkpoints is not None and (
            self.completed_update % self.config.checkpointing.every_updates == 0
            or self.completed_update in milestones
        ):
            self.drive_checkpoints.promote_from(path)
        return path

    def resume(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        discontinuity = restore_checkpoint_state(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        counters = payload["counters"]
        self.completed_update = int(counters["completed_update"])
        self.global_step = int(counters["global_step"])
        self.seed_allocator.load_state_dict(payload["seed_streams"]["environment"])
        self._initialize_environments()
        self.logger.log({"resume/discontinuity": 1}, step=self.global_step)
        self.logger.log_records(
            "resume/discontinuities", [discontinuity], step=self.global_step
        )
        return discontinuity

    def _flush_diagnostics(self) -> dict[str, float]:
        if self.routing_diagnostics is None:
            return {}
        metrics = self.routing_diagnostics.metrics(reset=True)
        written = self.routing_diagnostics.write_details(self.routing_output_path)
        if written is not None:
            self.logger.log_artifact(
                str(written),
                name=self.config.diagnostics.artifact_name,
                metadata={
                    "arm": self.config.arm,
                    "model_seed": self.config.seeds.model,
                    "update": self.completed_update,
                    "sample_rate": (
                        self.config.diagnostics.detailed_routing_sample_rate
                    ),
                },
            )
        return metrics

    def request_stop(self, *_: Any) -> None:
        self._stop_requested = True

    def _promote_latest_completed(self) -> None:
        if self.drive_checkpoints is None or self.local_checkpoints is None:
            return
        latest = self.local_checkpoints.recover_latest()
        target = self.drive_checkpoints.update_directory(latest.update)
        if (target / "commit_success.json").is_file():
            self.drive_checkpoints.validate(target)
            return
        self.drive_checkpoints.promote_from(latest.directory)

    def run(self, updates: int | None = None) -> None:
        target = int(updates or self.config.ppo.updates)
        previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in previous_handlers:
            signal.signal(signum, self.request_stop)
        try:
            while self.completed_update < target:
                self.scheduler.step_to(self.completed_update)
                rollout = self.collect_rollout()
                loss = self.optimize_rollout(rollout)
                self.completed_update += 1
                self.global_step += len(self.envs) * self.rollout_steps
                self._checkpoint()
                routing_metrics = self._flush_diagnostics()
                episode_records = [
                    {
                        **record,
                        "arm": self.config.arm,
                        "model_seed": self.config.seeds.model,
                        "update": self.completed_update,
                    }
                    for record in rollout.episode_records
                ]
                self.logger.log_records(
                    "episodes/records", episode_records, step=self.global_step
                )
                self.logger.log(
                    {
                        "ppo/loss": loss,
                        "ppo/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "ppo/value_mean": float(rollout.values.mean()),
                        "ppo/advantage_mean": float(rollout.advantages.mean()),
                        "episodes": len(rollout.episode_records),
                        **routing_metrics,
                    },
                    step=self.global_step,
                )
                self.traces.prune_closed()
                if self._stop_requested:
                    self._promote_latest_completed()
                    break
        except BaseException:
            try:
                self._promote_latest_completed()
            except Exception:
                pass
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            for env in self.envs:
                env.close()
            self.logger.finish()
