"""Run the untouched MiniGrid Memory-S9 transfer gate with backend fallback."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

from episodic_moba_ppo.checkpoint import load_legacy_checkpoint
from episodic_moba_ppo.commands.eval_pretrained import LegacyTrXLPolicy
from episodic_moba_ppo.config import MiniGridTrainConfig, load_config
from episodic_moba_ppo.evaluation import atomic_write_json
from episodic_moba_ppo.minigrid_task import MiniGridTaskConfig, MiniGridTaskFactory


def _action_seed(environment_seed: int, repeat: int) -> int:
    payload = f"minigrid-s9-transfer-v1:{environment_seed}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _evaluate(config: MiniGridTrainConfig, state: Any, legacy: dict[str, Any], backend: str) -> dict[str, Any]:
    factory = MiniGridTaskFactory(MiniGridTaskConfig(
        task_seed=config.environment.task_rng_seed, backend=backend, delay_conditions=tuple(config.environment.delay_conditions)))
    probe = factory()
    try:
        probe.reset(
            seed=config.transfer_gate.environment_start, adapter_bridge_length=32
        )
        policy = LegacyTrXLPolicy(state, legacy, probe)
    finally:
        probe.close()
    episodes = []
    for seed in range(config.transfer_gate.environment_start, config.transfer_gate.environment_start + config.transfer_gate.environment_count):
        for repeat in range(config.transfer_gate.action_rng_repeats):
            env = factory.for_adapter_bridge_length(32)()
            try:
                observation = env.reset(seed=seed)
                policy.reset()
                generator = torch.Generator(device="cpu").manual_seed(_action_seed(seed, repeat))
                done = False
                reward = 0.0
                while not done:
                    observation, step_reward, done, info = env.step(policy.act(observation, generator))
                    reward += float(step_reward)
                episodes.append({"environment_seed": seed, "action_repeat": repeat, "action_seed": _action_seed(seed, repeat), "reward": reward, "success": int(info.get("success", False)), "episode_length": int(info.get("episode_length", 0)), "adapter_bridge_length": int(info["adapter_bridge_length"]), "cue_timestep": int(info["cue_timestep"]), "decision_timestep": int(info["decision_timestep"]), "actual_delay": int(info["actual_delay"]), "backend": str(info["backend"])})
            finally:
                env.close()
    success = sum(row["success"] for row in episodes) / len(episodes)
    mean_return = sum(row["reward"] for row in episodes) / len(episodes)
    return {"backend": backend, "episodes": episodes, "summary": {"episodes": len(episodes), "success_rate": success, "mean_return": mean_return}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/minigrid_delay_trxl.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-failed-threshold",
        action="store_true",
        help="Write a complete below-threshold measurement and exit successfully.",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    config = load_config(args.config)
    if not isinstance(config, MiniGridTrainConfig):
        raise TypeError("eval-minigrid-pretrained requires task: train-minigrid")
    checkpoint = config.provenance.verify_checkpoint(root)
    state, legacy = load_legacy_checkpoint(checkpoint, config.provenance.checkpoint_sha256)
    attempts = []
    for backend in ("minigrid", "gym_minigrid"):
        try:
            result = _evaluate(config, state, copy.deepcopy(legacy), backend)
        except Exception as error:
            attempts.append({"backend": backend, "error": str(error)})
            continue
        attempts.append({"backend": backend, **result["summary"]})
        passed = (result["summary"]["success_rate"] >= config.transfer_gate.minimum_success_rate and result["summary"]["mean_return"] >= config.transfer_gate.minimum_mean_return)
        document = {"schema_version": 1, "passed": passed, "protocol": {"task": "minigrid-memory-s9", "environment_seeds": [10000, 10049], "action_repeats": 3, "action_sampling": "paired_stochastic", "adapter_bridge_length": 32}, "provenance": {"source_commit": config.provenance.upstream_commit, "checkpoint_sha256": config.provenance.checkpoint_sha256}, "thresholds": {"success_rate": config.transfer_gate.minimum_success_rate, "mean_return": config.transfer_gate.minimum_mean_return}, "summary": result["summary"], "backend": result["backend"], "attempts": attempts, "episodes": result["episodes"]}
        output = args.output or root / config.transfer_gate.output_path
        atomic_write_json(output, document)
        if passed:
            print(f"MiniGrid transfer gate passed: {output}")
            return 0
        if args.allow_failed_threshold:
            print(f"MiniGrid transfer measurement recorded below threshold: {output}")
            return 0
        return 2
    output = args.output or root / config.transfer_gate.output_path
    atomic_write_json(output, {"schema_version": 1, "passed": False, "attempts": attempts, "episodes": []})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
