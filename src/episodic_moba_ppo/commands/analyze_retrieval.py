"""Generate descriptive routing distributions from a sampled routing CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from episodic_moba_ppo.analysis import analyze_routing_csv
from episodic_moba_ppo.config import AnalyzeRetrievalConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if not isinstance(config, AnalyzeRetrievalConfig):
        raise TypeError("analyze-retrieval requires a task: analyze-retrieval config")
    outputs = analyze_routing_csv(
        config.routing_artifact_path, config.output_csv, config.output_dir
    )
    print(f"routing summary: {outputs.summary_csv}")
    print(f"retrieval distances: {outputs.retrieval_distances_csv}")
    print(f"selected block ages: {outputs.block_ages_csv}")
    print(f"routing scores: {outputs.routing_scores_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
