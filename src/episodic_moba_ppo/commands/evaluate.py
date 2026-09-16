"""Run the fixed held-out command-count sweep for a trained policy."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from episodic_moba_ppo.environment import MemoryGymEnv, mortar_reset_options
from episodic_moba_ppo.evaluation import (
    aggregate_records,
    atomic_write_json,
    episode_specs,
    evaluate_policy,
)
from episodic_moba_ppo.logging import NoOpLogger, WandbLogger


def _load_factory(specification: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("policy factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), attribute)


def _torch_generator(seed: int) -> Any:
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arm", choices=("trxl", "trxl_moba"), required=True)
    parser.add_argument("--model-seed", choices=(1, 2, 3), required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--policy-factory",
        default="episodic_moba_ppo.evaluation_policy:load_evaluation_policy",
        help="Callable(checkpoint_path, env, **metadata) returning EvaluationPolicy",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default="episodic-moba-ppo")
    parser.add_argument("--wandb-run-name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    policy_factory = _load_factory(args.policy_factory)
    checkpoint_hash: str | None = None
    checkpoint_update: int | None = None
    records = []
    for command_count in (10, 20, 30, 40, 50, 60, 80):
        options = mortar_reset_options(
            {
                "agent_scale": 0.25,
                "arena_size": 5,
                "allowed_commands": 5,
                "explosion_duration": [2],
                "explosion_delay": [5],
                "reward_command_failure": 0.0,
                "reward_command_success": 0.1,
                "reward_episode_success": 0.0,
            },
            command_count,
        )

        def env_factory(options: dict[str, Any] = options) -> MemoryGymEnv:
            return MemoryGymEnv("MortarMayhem-Grid-v0", options)

        probe = env_factory()
        try:
            probe.reset(seed=10_000)
            policy = policy_factory(
                args.checkpoint,
                probe,
                repo_root=repo_root,
                expected_arm=args.arm,
                expected_model_seed=args.model_seed,
            )
            policy_hash = getattr(policy, "checkpoint_sha256", None)
            policy_update = getattr(policy, "checkpoint_update", None)
            if policy_hash is None or policy_update is None:
                raise ValueError(
                    "evaluation policy must expose checkpoint_sha256 and checkpoint_update"
                )
            if checkpoint_hash is None:
                checkpoint_hash = str(policy_hash)
                checkpoint_update = int(policy_update)
            elif (checkpoint_hash, checkpoint_update) != (
                str(policy_hash),
                int(policy_update),
            ):
                raise ValueError("policy factory returned inconsistent checkpoints")
        finally:
            probe.close()
        records.extend(
            evaluate_policy(
                policy=policy,
                env_factory=env_factory,
                command_count=command_count,
                specs=episode_specs(3),
                arm=args.arm,
                model_seed=args.model_seed,
                generator_factory=_torch_generator,
                checkpoint_sha256=checkpoint_hash,
                checkpoint_update=checkpoint_update,
            )
        )
    summary = aggregate_records(records)
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_update": checkpoint_update,
            "arm": args.arm,
            "model_seed": args.model_seed,
            "summary_by_command_count": summary,
            "episodes": [asdict(record) for record in records],
        },
    )
    logger = NoOpLogger()
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=(
                args.wandb_run_name
                or f"{args.arm}-seed{args.model_seed}-evaluation"
            ),
            run_id=None,
            mode=args.wandb_mode,
            config={
                "task": "evaluate",
                "arm": args.arm,
                "model_seed": args.model_seed,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_update": checkpoint_update,
                "memory_delay_axis": "command_count",
            },
        )
    evaluation_step = int(checkpoint_update or 0) * 16_384
    logger.log_records(
        "eval/success_by_memory_delay",
        [
            {
                "memory_delay": int(command_count),
                "success_rate": values["success_rate"],
                "episodes": values["episodes"],
            }
            for command_count, values in summary.items()
        ],
        step=evaluation_step,
    )
    logger.log({"charts/global_step": evaluation_step}, step=evaluation_step)
    logger.finish()
    print(f"evaluation complete: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
