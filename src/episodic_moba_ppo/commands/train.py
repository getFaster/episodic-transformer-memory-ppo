"""Validate a training request and start the gated runtime when available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from episodic_moba_ppo.config import TrainConfig, load_config
from episodic_moba_ppo.runtime import build_training_runtime
from episodic_moba_ppo.training import TrainingInterrupted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=Path("results/baseline_reference.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = load_config(args.config)
    if not isinstance(config, TrainConfig):
        raise TypeError("train requires a task: train configuration")
    runtime = build_training_runtime(
        config, args.baseline_reference, repo_root=repo_root
    )
    try:
        runtime.run()
    except TrainingInterrupted as error:
        print(str(error), file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
