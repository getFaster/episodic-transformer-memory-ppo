"""Fail-closed gates, provenance capture, and production runtime assembly."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from episodic_moba_ppo.config import MiniGridTrainConfig, TrainConfig
from episodic_moba_ppo.extension_gate import require_extension_gate
from episodic_moba_ppo.moba_retrieval import USEFUL_ATTENTION_THRESHOLD


class BaselineGateError(RuntimeError):
    pass


def runtime_metadata(
    *,
    repo_root: str | Path,
    config: TrainConfig | MiniGridTrainConfig,
    trainable_names: list[str],
    trainable_count: int,
) -> dict[str, Any]:
    dependencies = {}
    for distribution in (
        "torch",
        "numpy",
        "gymnasium",
        "memory-gym",
        "peft",
        "wandb",
    ):
        try:
            dependencies[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            dependencies[distribution] = "unavailable"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "resolved_config": config.model_dump(mode="json"),
        "git_commit": git_commit,
        "upstream_commit": config.provenance.upstream_commit,
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "logging_definitions": {
            "fraction_beyond_recent_window": (
                "selected block distance > attention.dense_recent"
            ),
            "retrieved_attention_mass": (
                "mean attention mass on retrieved tokens across heads and layers"
            ),
            "useful_retrieval_attention_threshold": USEFUL_ATTENTION_THRESHOLD,
            "retrieval_distance_histogram_every_updates": 5,
            "rollout_log_interval_environment_steps": 2_048,
        },
    }


@dataclass(frozen=True)
class ResumeDiscontinuity:
    resumed_update: int
    environments_reset: bool = True
    episodic_memories_reset: bool = True
    partial_work_discarded: bool = True
    timestamp_utc: str = ""


def require_baseline_gate(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    gate_path = Path(path)
    if not gate_path.is_file():
        raise BaselineGateError(f"baseline gate artifact is missing: {gate_path}")
    try:
        document = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineGateError(
            f"baseline gate artifact is invalid: {gate_path}"
        ) from error
    if document.get("passed") is not True:
        raise BaselineGateError("baseline gate did not pass; training is forbidden")
    protocol = document.get("protocol")
    expected_protocol = {
        "command_count": 10,
        "model_max_episode_steps": 119,
        "environment_seeds": [10_000, 10_049],
        "action_repeats": 2,
        "action_sampling": "paired_stochastic",
        "normalized_return": (
            "reward / (command_count * reward_command_success + "
            "reward_episode_success)"
        ),
    }
    if protocol != expected_protocol:
        raise BaselineGateError(
            "baseline gate protocol does not match the locked protocol"
        )
    summary = document.get("summary", {})
    thresholds = document.get("thresholds", {})
    try:
        success_rate = float(summary["success_rate"])
        mean_return = float(summary["mean_normalized_return"])
        success_threshold = float(thresholds["success_rate"])
        return_threshold = float(thresholds["mean_normalized_return"])
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineGateError("baseline gate is missing validated metrics") from error
    if not all(math.isfinite(value) for value in (success_rate, mean_return)):
        raise BaselineGateError("baseline gate metrics must be finite")
    if success_threshold != 0.95 or return_threshold != 0.95:
        raise BaselineGateError("baseline gate thresholds must both equal 0.95")
    if success_rate < success_threshold or mean_return < return_threshold:
        raise BaselineGateError(
            "baseline metrics do not satisfy their recorded thresholds"
        )
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 100:
        raise BaselineGateError("baseline gate requires exactly 100 episode records")
    try:
        pairs = {
            (int(row["environment_seed"]), int(row["action_repeat"]))
            for row in episodes
        }
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineGateError("baseline gate has invalid episode records") from error
    expected_pairs = {
        (seed, repeat) for seed in range(10_000, 10_050) for repeat in range(2)
    }
    if pairs != expected_pairs:
        raise BaselineGateError("baseline gate episode coverage is incomplete")
    provenance = document.get("provenance", {})
    if (
        expected_checkpoint_sha256
        and provenance.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise BaselineGateError(
            "baseline checkpoint hash does not match training config"
        )
    if (
        expected_source_commit
        and provenance.get("source_commit") != expected_source_commit
    ):
        raise BaselineGateError("baseline source commit does not match training config")
    return document


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore process RNGs; CUDA state is required only on a CUDA resume."""

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
    except KeyError as error:
        raise ValueError(f"checkpoint RNG state is missing {error.args[0]}") from error
    if torch.cuda.is_available():
        if "torch_cuda" not in state:
            raise ValueError("CUDA resume requires saved CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def assemble_checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    completed_update: int,
    global_step: int,
    config: TrainConfig | Mapping[str, Any],
    wandb_identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    seed_streams: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if completed_update < 0 or global_step < 0:
        raise ValueError("completed counters must be nonnegative")
    if hasattr(config, "model_dump"):
        config_data = config.model_dump(mode="json")
    else:
        config_data = dict(config)
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
        "seed_streams": dict(seed_streams or {}),
        "counters": {
            "completed_update": int(completed_update),
            "global_step": int(global_step),
        },
        "config": config_data,
        "wandb": dict(wandb_identity),
        "provenance": dict(provenance),
    }


def resume_discontinuity(completed_update: int) -> dict[str, Any]:
    return asdict(
        ResumeDiscontinuity(
            resumed_update=int(completed_update),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
    )


def restore_checkpoint_state(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    steps_per_update: int = 16_384,
) -> dict[str, Any]:
    """Restore only committed-update state and describe the mandatory reset.

    Environment state and episodic traces are intentionally absent from the
    checkpoint contract.  The caller must create fresh environments and an
    empty :class:`TraceRegistry` before collecting another complete rollout.
    """

    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler_state = payload["scheduler"]
        counters = payload["counters"]
        rng_state = payload["rng"]
    except KeyError as error:
        raise ValueError(f"checkpoint payload is missing {error.args[0]}") from error
    if scheduler is None:
        if scheduler_state is not None:
            raise ValueError(
                "checkpoint has scheduler state but runtime has no scheduler"
            )
    elif scheduler_state is None:
        raise ValueError("runtime scheduler requires checkpoint scheduler state")
    else:
        scheduler.load_state_dict(scheduler_state)
    completed_update = int(counters["completed_update"])
    global_step = int(counters["global_step"])
    if completed_update < 0 or global_step != completed_update * steps_per_update:
        raise ValueError("checkpoint counters do not describe a completed PPO update")
    restore_rng_state(rng_state)
    return resume_discontinuity(completed_update)


@dataclass(frozen=True)
class TrainingTaskAssembly:
    """Environment factory plus optional checkpointable task-local state."""

    factory: Any
    task_state: Any | None = None


def _mortar_task_factory(config: TrainConfig) -> TrainingTaskAssembly:
    """Retain the historical Mortar adapter behind the task-factory boundary."""

    from episodic_moba_ppo.environment import MemoryGymEnv, mortar_reset_options

    options = mortar_reset_options(
        {
            "agent_scale": config.environment.agent_scale,
            "arena_size": config.environment.arena_size,
            "allowed_commands": config.environment.allowed_commands,
            "explosion_duration": [config.environment.explosion_duration],
            "explosion_delay": [config.environment.explosion_delay],
            "reward_command_failure": config.environment.reward_command_failure,
            "reward_command_success": config.environment.reward_command_success,
            "reward_episode_success": config.environment.reward_episode_success,
        },
        config.environment.command_count,
    )
    return TrainingTaskAssembly(
        factory=lambda: MemoryGymEnv(config.environment.name, options)
    )


def _minigrid_task_factory(config: MiniGridTrainConfig) -> TrainingTaskAssembly:
    """Resolve all MiniGrid details at the task boundary, not in PPO."""

    from episodic_moba_ppo.minigrid_task import MiniGridTaskConfig, MiniGridTaskFactory

    backend = {
        "minigrid==3.1.0": "minigrid",
        "gym-minigrid==1.0.2": "gym_minigrid",
    }[config.environment.backend]
    factory = MiniGridTaskFactory(
        MiniGridTaskConfig(
            task_seed=config.environment.task_rng_seed,
            backend=backend,
            environment_id=config.environment.name,
            delay_conditions=tuple(config.environment.delay_conditions),
        )
    )
    return TrainingTaskAssembly(factory=factory, task_state=factory.allocator)


_TASK_FACTORY_REGISTRY: dict[type[Any], Any] = {
    TrainConfig: _mortar_task_factory,
    MiniGridTrainConfig: _minigrid_task_factory,
}


def _resolve_training_task(
    config: TrainConfig | MiniGridTrainConfig,
) -> TrainingTaskAssembly:
    try:
        return _TASK_FACTORY_REGISTRY[type(config)](config)
    except KeyError as error:
        raise TypeError(f"no task factory is registered for {type(config).__name__}") from error


def require_minigrid_transfer_gate(
    path: str | Path,
    *,
    config: MiniGridTrainConfig,
) -> dict[str, Any]:
    """Validate the recorded untouched-checkpoint transfer measurement.

    The evaluator owns backend fallback; this runtime only accepts a completed
    artifact whose metrics, coverage, and checkpoint provenance match the
    resolved experiment.  Thresholds are enforced only when the resolved
    experiment explicitly asks for them; provenance and episode coverage are
    always required.
    """

    gate_path = Path(path)
    if not gate_path.is_file():
        raise BaselineGateError(f"MiniGrid transfer gate artifact is missing: {gate_path}")
    try:
        document = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineGateError("MiniGrid transfer gate artifact is invalid") from error
    if config.transfer_gate.enforce_thresholds and document.get("passed") is not True:
        raise BaselineGateError("MiniGrid transfer gate did not pass; training is forbidden")
    summary = document.get("summary", {})
    thresholds = document.get("thresholds", {})
    try:
        success_rate = float(summary["success_rate"])
        mean_return = float(summary["mean_return"])
        success_threshold = float(thresholds["success_rate"])
        return_threshold = float(thresholds["mean_return"])
    except (KeyError, TypeError, ValueError) as error:
        raise BaselineGateError("MiniGrid transfer gate is missing validated metrics") from error
    if not all(math.isfinite(value) for value in (success_rate, mean_return)):
        raise BaselineGateError("MiniGrid transfer gate metrics must be finite")
    if (
        success_threshold != config.transfer_gate.minimum_success_rate
        or return_threshold != config.transfer_gate.minimum_mean_return
    ):
        raise BaselineGateError("MiniGrid transfer gate thresholds do not match config")
    if config.transfer_gate.enforce_thresholds and (
        success_rate < success_threshold or mean_return < return_threshold
    ):
        raise BaselineGateError("MiniGrid transfer gate metrics do not meet thresholds")
    episodes = document.get("episodes")
    expected_episodes = (
        config.transfer_gate.environment_count * config.transfer_gate.action_rng_repeats
    )
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise BaselineGateError("MiniGrid transfer gate has incomplete episode records")
    provenance = document.get("provenance", {})
    if provenance.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise BaselineGateError("MiniGrid transfer gate checkpoint hash does not match config")
    return document


def _require_mortar_training_gate(config: TrainConfig, path: Path) -> None:
    require_baseline_gate(
        path,
        expected_checkpoint_sha256=config.provenance.checkpoint_sha256,
        expected_source_commit=config.provenance.upstream_commit,
    )


_TRAINING_GATE_REGISTRY: dict[type[Any], Any] = {
    TrainConfig: _require_mortar_training_gate,
}


def _require_training_gate(config: TrainConfig | MiniGridTrainConfig, path: Path | None) -> None:
    if isinstance(config, MiniGridTrainConfig):
        # Transfer/smoke measurements are explicit preflight commands.  A user
        # may run the full MiniGrid experiment without producing one first.
        return
    if path is None:
        raise BaselineGateError("Mortar training requires a baseline gate artifact")
    try:
        _TRAINING_GATE_REGISTRY[type(config)](config, path)
    except KeyError as error:
        raise TypeError(f"no baseline gate is registered for {type(config).__name__}") from error


def build_training_runtime(
    config: TrainConfig | MiniGridTrainConfig,
    baseline_gate_path: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    progress_path: str | Path | None = None,
):
    """Build the production 32-environment runtime after required checks pass."""

    from episodic_moba_ppo.checkpoint import (
        CheckpointStore,
        load_legacy_checkpoint,
    )
    from episodic_moba_ppo.logging import NoOpLogger, WandbLogger
    from episodic_moba_ppo.lora import freeze_for_lora
    from episodic_moba_ppo.ppo import create_muon_optimizer
    from episodic_moba_ppo.training import TrainingRuntime

    root = Path(repo_root).resolve()
    gate_path = None if baseline_gate_path is None else Path(baseline_gate_path)
    if gate_path is not None and not gate_path.is_absolute():
        gate_path = root / gate_path
    _require_training_gate(config, gate_path)
    if config.ppo.updates == 62:
        # The config schema requires this path for the extension stage.  The
        # artifact itself is revalidated here, before checkpoint deserialization
        # or environment creation, and relative paths are rooted at the repo.
        require_extension_gate(
            config.ppo.extension_gate_artifact,
            repo_root=root,
            expected_checkpoint_sha256=config.provenance.checkpoint_sha256,
            expected_source_commit=config.provenance.upstream_commit,
        )
    checkpoint = config.provenance.verify_checkpoint(root)
    if config.drive.enabled and config.drive.root == "REPLACE_ME":
        raise ValueError("drive.root must be configured before training")
    state_dict, legacy_config = load_legacy_checkpoint(
        checkpoint, config.provenance.checkpoint_sha256
    )
    task_assembly = _resolve_training_task(config)
    environments = [task_assembly.factory() for _ in range(config.ppo.workers)]
    try:
        from model import ActorCriticModel

        # Memory-Gym computes command-count-dependent episode capacity at reset.
        # This probe reset is explicit and does not consume the checkpointed
        # TrainingSeedAllocator stream created later by TrainingRuntime.
        environments[0].reset(seed=config.environment.seed_start)
        action_space = environments[0].action_space
        if hasattr(action_space, "n"):
            action_shape = (int(action_space.n),)
        elif hasattr(action_space, "nvec"):
            action_shape = tuple(int(value) for value in action_space.nvec)
        else:
            raise TypeError("unsupported action space")
        torch.manual_seed(config.seeds.model)
        model = ActorCriticModel(
            legacy_config,
            environments[0].observation_space,
            action_shape,
            environments[0].max_episode_steps,
        )
        model.load_state_dict(state_dict, strict=True)
        model.enable_lora(
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
        )
        # Policy cardinality differs between Mortar and MiniGrid.  The frozen
        # set is verified by ``freeze_for_lora``; derive its size from the
        # instantiated model instead of retaining Mortar's old 73,728 constant.
        trainable_names = freeze_for_lora(model)
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print("trainable parameters:")
        for name in trainable_names:
            print(f"  {name}")
        print(f"trainable parameter count: {trainable_count}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = create_muon_optimizer(
            list(model.named_parameters()),
            muon_lr=config.optimizer.muon.initial_lr,
            adamw_heads_lr=config.optimizer.adamw_heads.initial_lr,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            ns_steps=config.optimizer.ns_steps,
            weight_decay=config.optimizer.weight_decay,
            adjust_lr_fn=config.optimizer.adjust_lr_fn,
        )
        if config.wandb.enabled and config.wandb.mode != "disabled":
            logger = WandbLogger(
                entity=config.wandb.entity,
                project=config.wandb.project,
                name=config.wandb.run_name,
                run_id=config.wandb.run_id,
                mode=config.wandb.mode,
                config=config.model_dump(mode="json"),
            )
        else:
            logger = NoOpLogger()
        logger.log_run_metadata(
            runtime_metadata(
                repo_root=root,
                config=config,
                trainable_names=trainable_names,
                trainable_count=trainable_count,
            )
        )
        local = CheckpointStore(root / config.checkpointing.local_dir)
        drive = None
        if config.drive.enabled:
            drive_root = Path(config.drive.root)
            drive = CheckpointStore(drive_root / config.checkpointing.drive_dir)
        runtime = TrainingRuntime(
            config=config,
            model=model,
            optimizer=optimizer,
            environments=environments,
            device=device,
            local_checkpoints=local,
            drive_checkpoints=drive,
            logger=logger,
            routing_output_path=root / config.diagnostics.output_path,
            progress_path=progress_path,
            task_state=task_assembly.task_state,
        )
        if config.checkpointing.resume_from:
            resume_path = Path(config.checkpointing.resume_from)
            if not resume_path.is_absolute():
                resume_path = root / resume_path
            store = CheckpointStore(resume_path.parent)
            store.validate(resume_path)
            payload = store.load(resume_path / "training_state.pt")
            runtime.resume(payload)
        return runtime
    except Exception:
        for environment in environments:
            environment.close()
        raise
