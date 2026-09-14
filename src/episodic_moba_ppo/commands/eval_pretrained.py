"""Evaluate the hash-pinned upstream checkpoint and enforce the baseline gate."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from episodic_moba_ppo.checkpoint import load_legacy_checkpoint
from episodic_moba_ppo.config import PretrainedEvalConfig, load_config
from episodic_moba_ppo.environment import MemoryGymEnv, mortar_reset_options
from episodic_moba_ppo.evaluation import (
    atomic_write_json,
    baseline_document,
    episode_specs,
    evaluate_policy,
)

PINNED_SOURCE_COMMIT = "acc5cb2e0cdc87bd7b5ce857d270a72a66146e0b"
PINNED_CHECKPOINT_SHA256 = (
    "a356543eebd1a3571fe0923c4222894f021874202ce93fb9e43da9192ffd65b5"
)


class LegacyTrXLPolicy:
    def __init__(
        self,
        state_dict: Any,
        config: dict[str, Any],
        env: MemoryGymEnv,
        *,
        model_max_episode_steps: int | None = None,
    ) -> None:
        import torch

        # Root imports are deliberately lazy: unit tests for evaluation and
        # persistence do not need the legacy model or Memory-Gym installed.
        from model import ActorCriticModel

        action_space = env.action_space
        if hasattr(action_space, "n"):
            action_shape = (int(action_space.n),)
        elif hasattr(action_space, "nvec"):
            action_shape = tuple(int(value) for value in action_space.nvec)
        else:
            raise TypeError("unsupported action space")
        self._torch = torch
        self._config = config
        self._max_episode_steps = (
            env.max_episode_steps
            if model_max_episode_steps is None
            else int(model_max_episode_steps)
        )
        self._model = ActorCriticModel(
            config, env.observation_space, action_shape, self._max_episode_steps
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
        self._memory_length = int(config["transformer"]["memory_length"])
        self._num_blocks = int(config["transformer"]["num_blocks"])
        self._embed_dim = int(config["transformer"]["embed_dim"])
        self._mask = torch.tril(
            torch.ones((self._memory_length, self._memory_length)), diagonal=-1
        )
        repetitions = torch.repeat_interleave(
            torch.arange(self._memory_length).unsqueeze(0),
            self._memory_length - 1,
            dim=0,
        ).long()
        sliding = torch.stack(
            [
                torch.arange(index, index + self._memory_length)
                for index in range(self._max_episode_steps - self._memory_length + 1)
            ]
        ).long()
        self._indices = torch.cat((repetitions, sliding))
        self.reset()

    def reset(self) -> None:
        self._memory = self._torch.zeros(
            (1, self._max_episode_steps, self._num_blocks, self._embed_dim),
            dtype=self._torch.float32,
        )
        self._timestep = 0

    def act(self, observation: np.ndarray, action_generator: Any) -> list[int]:
        torch = self._torch
        indices = self._indices[self._timestep].unsqueeze(0)
        memory = self._memory[0, indices]
        mask_index = max(0, min(self._timestep, self._memory_length - 1))
        mask = self._mask[mask_index].unsqueeze(0)
        obs = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.inference_mode():
            policy, _, new_memory = self._model(obs, memory, mask, indices)
        self._memory[:, self._timestep] = new_memory
        self._timestep += 1
        return [
            int(torch.multinomial(branch.probs, 1, generator=action_generator).item())
            for branch in policy
        ]


def _torch_generator(seed: int) -> Any:
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrained_eval.yaml", type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--output", type=Path, help="Override the configured output path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    run_config = load_config(args.config)
    if not isinstance(run_config, PretrainedEvalConfig):
        raise TypeError("eval-pretrained requires a task: eval-pretrained config")
    checkpoint = run_config.provenance.verify_checkpoint(repo_root)
    state_dict, legacy_config = load_legacy_checkpoint(
        checkpoint, run_config.provenance.checkpoint_sha256
    )
    legacy_config = copy.deepcopy(legacy_config)
    env_config = run_config.environment
    reset_options = mortar_reset_options(
        {
            "agent_scale": env_config.agent_scale,
            "arena_size": env_config.arena_size,
            "allowed_commands": env_config.allowed_commands,
            "explosion_duration": [env_config.explosion_duration],
            "explosion_delay": [env_config.explosion_delay],
            "reward_command_failure": env_config.reward_command_failure,
            "reward_command_success": env_config.reward_command_success,
            "reward_episode_success": env_config.reward_episode_success,
        },
        env_config.command_count,
    )

    def env_factory() -> MemoryGymEnv:
        return MemoryGymEnv(env_config.name, reset_options)

    probe = env_factory()
    try:
        probe.reset(seed=10_000)
        policy = LegacyTrXLPolicy(
            state_dict,
            legacy_config,
            probe,
            model_max_episode_steps=run_config.evaluation.model_max_episode_steps,
        )
    finally:
        probe.close()
    records = evaluate_policy(
        policy=policy,
        env_factory=env_factory,
        command_count=10,
        specs=episode_specs(2),
        arm="pretrained_trxl",
        model_seed=None,
        generator_factory=_torch_generator,
        reward_command_success=float(reset_options["reward_command_success"]),
        reward_episode_success=float(reset_options["reward_episode_success"]),
        checkpoint_sha256=run_config.provenance.checkpoint_sha256,
    )
    document = baseline_document(
        records,
        source_commit=run_config.provenance.upstream_commit,
        checkpoint_sha256=run_config.provenance.checkpoint_sha256,
        success_threshold=run_config.evaluation.minimum_success_rate,
        normalized_return_threshold=(
            run_config.evaluation.minimum_mean_normalized_return
        ),
    )
    output = args.output or Path(run_config.evaluation.output_path)
    atomic_write_json(output, document)
    print(f"baseline gate {'passed' if document['passed'] else 'failed'}: {output}")
    return 0 if document["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
