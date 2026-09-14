"""Stateful evaluation policy for committed PPO training checkpoints."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from episodic_moba_ppo.checkpoint import (
    MARKER_NAME,
    PAYLOAD_NAME,
    CheckpointIntegrityError,
    CheckpointStore,
    load_legacy_checkpoint,
    sha256_file,
)
from episodic_moba_ppo.config import TrainConfig
from episodic_moba_ppo.episodic_memory import EpisodeTrace


def _legacy_actor_critic(repo_root: str | Path) -> type[torch.nn.Module]:
    """Import the checkout's legacy model, irrespective of the caller's cwd.

    Training checkpoints deliberately contain only the trained state, not a
    pickled model object.  Evaluation must therefore reconstruct the model
    from the same checkout that supplied the hash-pinned upstream checkpoint.
    """

    root = Path(repo_root).resolve()
    model_path = root / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"legacy model.py not found under --repo-root: {root}")
    # Insert at position zero even if the root is already present farther down
    # sys.path; an installed package or another checkout must not win.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != root]
    sys.path.insert(0, str(root))
    module = importlib.import_module("model")
    return module.ActorCriticModel


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Validated, loadable state for one completed PPO update."""

    directory: Path
    payload_path: Path
    payload_sha256: str
    update: int
    payload: Mapping[str, Any]
    config: TrainConfig


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Validate a marker-last directory (or its ``training_state.pt`` path)."""

    supplied = Path(path).resolve()
    if supplied.is_dir():
        directory = supplied
        payload_path = directory / PAYLOAD_NAME
    elif supplied.is_file() and supplied.name == PAYLOAD_NAME:
        directory = supplied.parent
        payload_path = supplied
    else:
        raise CheckpointIntegrityError(
            "checkpoint must be an update directory or its training_state.pt"
        )

    # A direct payload path is accepted for convenience, but never bypasses
    # the durable marker contract.
    if not (directory / MARKER_NAME).is_file():
        raise CheckpointIntegrityError(f"missing {MARKER_NAME}: {directory}")
    store = CheckpointStore(directory.parent)
    marker = store.validate(directory)
    loaded = store.load(payload_path)
    if not isinstance(loaded, Mapping):
        raise CheckpointIntegrityError("training checkpoint payload must be a mapping")
    try:
        config = TrainConfig.model_validate(loaded["config"])
        counters = loaded["counters"]
        update = int(counters["completed_update"])
        global_step = int(counters["global_step"])
        model_state = loaded["model"]
        payload_provenance = loaded["provenance"]
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointIntegrityError(
            "training checkpoint is missing valid evaluation metadata"
        ) from error
    if not isinstance(model_state, Mapping) or not isinstance(payload_provenance, Mapping):
        raise CheckpointIntegrityError("checkpoint model/provenance must be mappings")
    if update != int(marker["update"]):
        raise CheckpointIntegrityError("payload update disagrees with commit marker")
    expected_steps = update * config.ppo.workers * config.ppo.worker_steps
    if global_step != expected_steps:
        raise CheckpointIntegrityError(
            "checkpoint counters do not describe a completed PPO update"
        )
    config_provenance = config.provenance.model_dump(mode="json")
    if dict(payload_provenance) != config_provenance:
        raise CheckpointIntegrityError(
            "checkpoint payload provenance disagrees with its resolved config"
        )
    return TrainingCheckpoint(
        directory=directory,
        payload_path=payload_path,
        payload_sha256=sha256_file(payload_path),
        update=update,
        payload=loaded,
        config=config,
    )


def _action_shape(action_space: Any) -> tuple[int, ...]:
    if hasattr(action_space, "n"):
        return (int(action_space.n),)
    if hasattr(action_space, "nvec"):
        return tuple(int(value) for value in action_space.nvec)
    raise TypeError("unsupported action space")


class TrainedLongHistoryPolicy:
    """No-gradient actor with detached, episode-local long-history state."""

    def __init__(
        self,
        checkpoint: TrainingCheckpoint,
        env: Any,
        *,
        repo_root: str | Path = Path(__file__).resolve().parents[2],
        expected_arm: str | None = None,
        expected_model_seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        root = Path(repo_root).resolve()
        ActorCriticModel = _legacy_actor_critic(root)

        config = checkpoint.config
        if expected_arm is not None and config.arm != expected_arm:
            raise ValueError(
                f"checkpoint arm is {config.arm!r}, requested {expected_arm!r}"
            )
        if expected_model_seed is not None and config.seeds.model != expected_model_seed:
            raise ValueError(
                "checkpoint model seed does not match --model-seed: "
                f"{config.seeds.model} != {expected_model_seed}"
            )
        base_checkpoint = config.provenance.verify_checkpoint(root)
        base_state, legacy_config = load_legacy_checkpoint(
            base_checkpoint, config.provenance.checkpoint_sha256
        )
        model = ActorCriticModel(
            legacy_config,
            env.observation_space,
            _action_shape(env.action_space),
            env.max_episode_steps,
        )
        # Re-establish the exact frozen base before attaching PEFT modules.
        # The second strict load below then restores the committed adapters
        # and independently verifies the training checkpoint's full key set.
        model.load_state_dict(base_state, strict=True)
        model.enable_lora(
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
        )
        model.load_state_dict(checkpoint.payload["model"], strict=True)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.config = config
        self.checkpoint_sha256 = checkpoint.payload_sha256
        self.checkpoint_update = checkpoint.update
        self.reset()

    def reset(self) -> None:
        self._trace = EpisodeTrace(
            trace_id=0,
            num_layers=self.config.transformer.num_blocks,
            width=self.config.transformer.embed_dim,
        )

    def act(self, observation: np.ndarray, action_generator: Any) -> list[int]:
        retrieval = self.config.attention.retrieval
        reference = self._trace.reference()
        context = self._trace.context(
            reference.query_timestep,
            dense_recent=self.config.attention.dense_recent,
            search_horizon=self.config.attention.search_horizon,
            block_size=retrieval.block_size,
            # Episode traces and all old token blocks stay resident on CPU.
            # forward_long_history transfers only dense tokens, summaries,
            # and the selected old tokens to the model device.
            device="cpu",
        )
        obs = torch.as_tensor(
            observation[None], dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            policy, _, new_memories, _ = self.model.forward_long_history(
                obs,
                [context],
                arm=self.config.arm,
                attention_config=self.config.attention,
            )
        self._trace.append(new_memories[0])

        # Sample on CPU so paired seeds have identical generator semantics on
        # CPU and CUDA, matching the pretrained-baseline evaluator.
        return [
            int(
                torch.multinomial(
                    branch.probs.detach().to(device="cpu"),
                    1,
                    generator=action_generator,
                ).item()
            )
            for branch in policy
        ]


def load_evaluation_policy(
    checkpoint_path: str | Path,
    env: Any,
    *,
    repo_root: str | Path = Path(__file__).resolve().parents[2],
    expected_arm: str | None = None,
    expected_model_seed: int | None = None,
    device: str | torch.device | None = None,
) -> TrainedLongHistoryPolicy:
    """Factory used by the packaged ``evaluate`` command."""

    checkpoint = load_training_checkpoint(checkpoint_path)
    return TrainedLongHistoryPolicy(
        checkpoint,
        env,
        repo_root=repo_root,
        expected_arm=expected_arm,
        expected_model_seed=expected_model_seed,
        device=device,
    )
