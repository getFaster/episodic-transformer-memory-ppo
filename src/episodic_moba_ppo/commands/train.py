"""Validate a training request and start the configured PPO runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from episodic_moba_ppo.config import MiniGridTrainConfig, TrainConfig, load_config
from episodic_moba_ppo.runtime import build_training_runtime
from episodic_moba_ppo.training import TrainingInterrupted


def resolve_wandb_run_id(
    *, entity: str | None, project: str, run_name: str
) -> str | None:
    import wandb

    path = f"{entity}/{project}" if entity else project
    resumable_states = {"running", "crashed", "failed", "killed"}
    runs = wandb.Api().runs(path=path)
    resumable_runs = [
        run for run in runs if run.name == run_name and run.state in resumable_states
    ]
    if resumable_runs:
        return resumable_runs[0].id
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", choices=("mortar", "minigrid"), help="Optional task assertion")
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=Path("results/baseline_reference.json"),
        help="Mortar baseline artifact; ignored for MiniGrid training.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--progress-path",
        type=Path,
        help="Optional atomic local progress snapshot for launcher supervision",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = load_config(args.config)
    if not isinstance(config, (TrainConfig, MiniGridTrainConfig)):
        raise TypeError("train requires a train or train-minigrid configuration")
    if args.env == "minigrid" and not isinstance(config, MiniGridTrainConfig):
        raise ValueError("--env minigrid requires task: train-minigrid")
    if args.env == "mortar" and not isinstance(config, TrainConfig):
        raise ValueError("--env mortar requires task: train")
    run_name = f"{config.wandb.run_name}{config.seeds.model}"
    run_id = None
    if config.wandb.enabled and config.wandb.mode == "online":
        run_id = resolve_wandb_run_id(
            entity=config.wandb.entity,
            project=config.wandb.project,
            run_name=run_name,
        )
    config = config.model_copy(
        update={
            "wandb": config.wandb.model_copy(update={"run_name": run_name}),
        }
    )
    gate = args.baseline_reference if isinstance(config, TrainConfig) else None
    runtime = build_training_runtime(
        config,
        gate,
        repo_root=repo_root,
        progress_path=args.progress_path,
        wandb_run_id=run_id,
    )
    try:
        runtime.run()
    except TrainingInterrupted as error:
        print(str(error), file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
