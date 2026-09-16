"""Validate a training request and start the configured PPO runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from episodic_moba_ppo.config import MiniGridTrainConfig, TrainConfig, load_config
from episodic_moba_ppo.runtime import build_training_runtime
from episodic_moba_ppo.training import TrainingInterrupted


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
    gate = args.baseline_reference if isinstance(config, TrainConfig) else None
    runtime = build_training_runtime(
        config,
        gate,
        repo_root=repo_root,
        progress_path=args.progress_path,
    )
    try:
        runtime.run()
    except TrainingInterrupted as error:
        print(str(error), file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
