"""Progress reporting and cgroup monitoring for concurrent seed launches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


MIB = 1024**2
SIZE_SUFFIXES = {
    "": 1,
    "K": 1024,
    "M": MIB,
    "G": 1024**3,
    "T": 1024**4,
}


def parse_memory_size(value: str) -> int:
    """Parse the binary size suffixes accepted by systemd resource controls."""

    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("memory size cannot be empty")
    suffix = normalized[-1] if normalized[-1].isalpha() else ""
    number = normalized[:-1] if suffix else normalized
    if suffix not in SIZE_SUFFIXES:
        raise ValueError(f"unsupported memory size suffix: {suffix}")
    try:
        amount = float(number)
    except ValueError as error:
        raise ValueError(f"invalid memory size: {value}") from error
    result = int(amount * SIZE_SUFFIXES[suffix])
    if result <= 0:
        raise ValueError("memory size must be positive")
    return result


@dataclass(frozen=True)
class TrainingProgress:
    schema_version: int
    pid: int
    model_seed: int
    completed_update: int
    checkpoint_global_step: int
    checkpoint_path: str | None
    phase: str
    rollout_steps: int
    rollout_total: int
    ppo_minibatches_completed: int
    ppo_minibatches_total: int
    discard_score: float
    timestamp_utc: str


class ProgressReporter:
    """Atomically publish small, local progress snapshots for a launcher."""

    def __init__(self, path: str | Path, *, model_seed: int) -> None:
        self.path = Path(path)
        self.model_seed = int(model_seed)

    def write(
        self,
        *,
        completed_update: int,
        checkpoint_global_step: int,
        checkpoint_path: str | Path | None,
        phase: str,
        rollout_steps: int,
        rollout_total: int,
        ppo_minibatches_completed: int = 0,
        ppo_minibatches_total: int = 0,
    ) -> TrainingProgress:
        fraction = (
            ppo_minibatches_completed / ppo_minibatches_total
            if ppo_minibatches_total
            else 0.0
        )
        discard_score = float(rollout_steps + rollout_total * fraction)
        progress = TrainingProgress(
            schema_version=1,
            pid=os.getpid(),
            model_seed=self.model_seed,
            completed_update=int(completed_update),
            checkpoint_global_step=int(checkpoint_global_step),
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            phase=phase,
            rollout_steps=int(rollout_steps),
            rollout_total=int(rollout_total),
            ppo_minibatches_completed=int(ppo_minibatches_completed),
            ppo_minibatches_total=int(ppo_minibatches_total),
            discard_score=discard_score,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(asdict(progress), sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)
        return progress


@dataclass(frozen=True)
class SeedProcess:
    seed: int
    pid: int
    progress_path: Path
    cgroup_path: Path | None = None


def read_progress(path: Path) -> TrainingProgress | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return TrainingProgress(**document)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def progress_is_fresh(path: Path, *, max_age_seconds: float = 300.0) -> bool:
    try:
        return time.time() - path.stat().st_mtime <= max_age_seconds
    except OSError:
        return False


def process_is_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (OSError, IndexError):
        return False
    return state != "Z"


def seed_is_alive(process: SeedProcess) -> bool:
    if process.cgroup_path is None:
        return process_is_alive(process.pid)
    try:
        fields = dict(
            line.split(maxsplit=1)
            for line in (process.cgroup_path / "cgroup.events")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except (OSError, ValueError):
        return process_is_alive(process.pid)
    return fields.get("populated") == "1"


def choose_victim(processes: Sequence[SeedProcess]) -> SeedProcess | None:
    """Choose least post-checkpoint work, with newest seed as the tie-breaker."""

    active = [process for process in processes if seed_is_alive(process)]
    if not active:
        return None
    known = [
        (
            read_progress(process.progress_path)
            if progress_is_fresh(process.progress_path)
            else None,
            process,
        )
        for process in active
    ]
    usable = [(progress, process) for progress, process in known if progress]
    if not usable:
        return max(active, key=lambda process: process.seed)
    _, victim = min(
        usable,
        key=lambda item: (item[0].discard_score, -item[1].seed),
    )
    return victim


def _read_integer(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def verify_cgroup(cgroup: Path, expected_limit: int) -> None:
    actual_limit = _read_integer(cgroup / "memory.max")
    swap_limit = (cgroup / "memory.swap.max").read_text(encoding="utf-8").strip()
    if actual_limit != expected_limit:
        raise RuntimeError(
            f"cgroup memory.max is {actual_limit}, expected {expected_limit}"
        )
    if swap_limit != "0":
        raise RuntimeError(f"cgroup memory.swap.max is {swap_limit}, expected 0")


def _signal_group(process: SeedProcess, signum: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def monitor(
    *,
    cgroup: Path,
    memory_limit: int,
    processes: Sequence[SeedProcess],
    poll_seconds: float = 0.25,
    grace_seconds: float = 10.0,
) -> int:
    trigger = memory_limit - 256 * MIB
    resume = memory_limit - 512 * MIB
    if resume <= 0:
        raise ValueError("memory limit must exceed 512 MiB")
    verify_cgroup(cgroup, memory_limit)
    print(
        "Memory guard active: "
        f"limit={memory_limit} trigger={trigger} resume={resume} cgroup={cgroup}",
        flush=True,
    )
    victims = 0
    while any(seed_is_alive(process) for process in processes):
        current = _read_integer(cgroup / "memory.current")
        if current < trigger:
            time.sleep(poll_seconds)
            continue
        victim = choose_victim(processes)
        if victim is None:
            break
        survivors = [
            process
            for process in processes
            if process.pid != victim.pid and seed_is_alive(process)
        ]
        for survivor in survivors:
            _signal_group(survivor, signal.SIGSTOP)
        try:
            progress = read_progress(victim.progress_path)
            print(
                "Memory guard selecting "
                f"seed={victim.seed} pid={victim.pid} memory={current} "
                f"checkpoint={progress.checkpoint_path if progress else None} "
                f"discard_score={progress.discard_score if progress else 'unknown'}",
                flush=True,
            )
            _signal_group(victim, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while seed_is_alive(victim) and time.monotonic() < deadline:
                time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
            if seed_is_alive(victim):
                print(
                    f"Memory guard escalating seed={victim.seed} to SIGKILL",
                    flush=True,
                )
                _signal_group(victim, signal.SIGKILL)
            resume_deadline = time.monotonic() + grace_seconds
            while (
                _read_integer(cgroup / "memory.current") >= resume
                and any(seed_is_alive(process) for process in survivors)
                and time.monotonic() < resume_deadline
            ):
                time.sleep(poll_seconds)
        finally:
            for survivor in survivors:
                _signal_group(survivor, signal.SIGCONT)
        victims += 1
    return victims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgroup", required=True, type=Path)
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument(
        "--seed-process",
        action="append",
        default=[],
        metavar="SEED:PID:PROGRESS_PATH:CGROUP_PATH",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processes = []
    for value in args.seed_process:
        seed, pid, path, cgroup = value.split(":", 3)
        processes.append(SeedProcess(int(seed), int(pid), Path(path), Path(cgroup)))
    if not processes:
        raise ValueError("at least one --seed-process is required")
    monitor(
        cgroup=args.cgroup,
        memory_limit=parse_memory_size(args.memory_limit),
        processes=processes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
