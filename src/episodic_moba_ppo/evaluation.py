"""Paired, reproducible policy evaluation and baseline-gate utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

EVALUATION_SEEDS = tuple(range(10_000, 10_050))


class EvaluationPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: np.ndarray, action_generator: Any) -> Any: ...


@dataclass(frozen=True)
class EpisodeSpec:
    environment_seed: int
    action_repeat: int
    action_seed: int


@dataclass(frozen=True)
class EpisodeRecord:
    arm: str
    model_seed: int | None
    command_count: int
    environment_seed: int
    action_repeat: int
    action_seed: int
    reward: float
    normalized_return: float
    success: int
    episode_length: int
    commands_completed_fraction: float
    commands_completed_count: int
    checkpoint_sha256: str | None = None
    checkpoint_update: int | None = None


def paired_action_seed(environment_seed: int, action_repeat: int) -> int:
    payload = f"episodic-moba-eval-v1:{environment_seed}:{action_repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def episode_specs(
    repeats: int, seeds: Sequence[int] = EVALUATION_SEEDS
) -> tuple[EpisodeSpec, ...]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    checked = tuple(int(seed) for seed in seeds)
    if any(seed < 10_000 or seed > 10_049 for seed in checked):
        raise ValueError("evaluation seeds must be in reserved range 10000..10049")
    if len(set(checked)) != len(checked):
        raise ValueError("evaluation seeds must be unique")
    return tuple(
        EpisodeSpec(seed, repeat, paired_action_seed(seed, repeat))
        for seed in checked
        for repeat in range(repeats)
    )


def normalized_return(
    reward: float,
    command_count: int,
    reward_command_success: float = 0.1,
    reward_episode_success: float = 0.0,
) -> float:
    maximum = command_count * reward_command_success + reward_episode_success
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("maximum configured episodic reward must be positive")
    return float(reward) / maximum


def evaluate_policy(
    *,
    policy: EvaluationPolicy,
    env_factory: Callable[[], Any],
    command_count: int,
    specs: Iterable[EpisodeSpec],
    arm: str,
    model_seed: int | None,
    generator_factory: Callable[[int], Any],
    reward_command_success: float = 0.1,
    reward_episode_success: float = 0.0,
    checkpoint_sha256: str | None = None,
    checkpoint_update: int | None = None,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for spec in specs:
        env = env_factory()
        try:
            observation = env.reset(seed=spec.environment_seed)
            policy.reset()
            generator = generator_factory(spec.action_seed)
            done = False
            accumulated_reward = 0.0
            steps = 0
            info: dict[str, Any] = {}
            while not done:
                action = policy.act(observation, generator)
                observation, reward, done, info = env.step(action)
                accumulated_reward += float(reward)
                steps += 1
            reward = float(info.get("reward", accumulated_reward))
            length = int(info.get("length", steps))
            success = int(bool(info.get("success", False)))
            completed_fraction = float(info.get("commands_completed", success))
            records.append(
                EpisodeRecord(
                    arm=arm,
                    model_seed=model_seed,
                    command_count=command_count,
                    environment_seed=spec.environment_seed,
                    action_repeat=spec.action_repeat,
                    action_seed=spec.action_seed,
                    reward=reward,
                    normalized_return=normalized_return(
                        reward,
                        command_count,
                        reward_command_success,
                        reward_episode_success,
                    ),
                    success=success,
                    episode_length=length,
                    commands_completed_fraction=completed_fraction,
                    commands_completed_count=int(
                        round(completed_fraction * command_count)
                    ),
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_update=checkpoint_update,
                )
            )
        finally:
            env.close()
    return records


def aggregate_records(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate zero episodes")
    grouped: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.command_count].append(record)
    result: dict[str, Any] = {}
    for command_count, group in sorted(grouped.items()):
        result[str(command_count)] = {
            "episodes": len(group),
            "mean_reward": statistics.fmean(x.reward for x in group),
            "median_reward": statistics.median(x.reward for x in group),
            "success_rate": statistics.fmean(x.success for x in group),
            "mean_normalized_return": statistics.fmean(
                x.normalized_return for x in group
            ),
            "mean_episode_length": statistics.fmean(x.episode_length for x in group),
            "mean_commands_completed": statistics.fmean(
                x.commands_completed_count for x in group
            ),
        }
    return result


def baseline_document(
    records: Sequence[EpisodeRecord],
    *,
    source_commit: str,
    checkpoint_sha256: str,
    success_threshold: float = 0.95,
    normalized_return_threshold: float = 0.95,
) -> dict[str, Any]:
    aggregates = aggregate_records(records)
    if set(aggregates) != {"10"}:
        raise ValueError("baseline gate requires only command_count=10 records")
    expected = {(seed, repeat) for seed in EVALUATION_SEEDS for repeat in range(2)}
    actual = {(r.environment_seed, r.action_repeat) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("baseline gate requires 50 held-out seeds x 2 repeats")
    summary = aggregates["10"]
    passed = (
        summary["success_rate"] >= success_threshold
        and summary["mean_normalized_return"] >= normalized_return_threshold
    )
    return {
        "schema_version": 1,
        "protocol": {
            "command_count": 10,
            "model_max_episode_steps": 119,
            "environment_seeds": [10_000, 10_049],
            "action_repeats": 2,
            "action_sampling": "paired_stochastic",
            "normalized_return": "reward / (command_count * reward_command_success + reward_episode_success)",
        },
        "provenance": {
            "source_commit": source_commit,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "thresholds": {
            "success_rate": success_threshold,
            "mean_normalized_return": normalized_return_threshold,
        },
        "summary": summary,
        "passed": passed,
        "episodes": [asdict(record) for record in records],
    }


def atomic_write_json(path: str | os.PathLike[str], document: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
