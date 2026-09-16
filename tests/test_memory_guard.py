import json
import os
import signal
import time
from pathlib import Path

import pytest

from episodic_moba_ppo.memory_guard import (
    MIB,
    ProgressReporter,
    SeedProcess,
    choose_victim,
    monitor,
    parse_memory_size,
    verify_cgroup,
)


def test_parse_memory_size_uses_binary_suffixes() -> None:
    assert parse_memory_size("18G") == 18 * 1024**3
    assert parse_memory_size("768M") == 768 * MIB
    with pytest.raises(ValueError, match="positive"):
        parse_memory_size("0")


def test_progress_reporter_writes_atomic_discard_score(tmp_path) -> None:
    path = tmp_path / "progress" / "seed2.json"
    reporter = ProgressReporter(path, model_seed=2)

    reporter.write(
        completed_update=4,
        checkpoint_global_step=65_536,
        checkpoint_path=tmp_path / "checkpoints" / "update-00004",
        phase="ppo",
        rollout_steps=16_384,
        rollout_total=16_384,
        ppo_minibatches_completed=6,
        ppo_minibatches_total=24,
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["model_seed"] == 2
    assert document["completed_update"] == 4
    assert document["discard_score"] == 20_480
    assert not list(path.parent.glob(".*.tmp"))


def _write_progress(path: Path, *, seed: int, score: float) -> None:
    ProgressReporter(path, model_seed=seed).write(
        completed_update=1,
        checkpoint_global_step=16_384,
        checkpoint_path=f"update-{seed}",
        phase="rollout",
        rollout_steps=int(score),
        rollout_total=16_384,
    )


def test_choose_victim_uses_least_progress_then_newest(tmp_path, monkeypatch) -> None:
    processes = [
        SeedProcess(seed, 100 + seed, tmp_path / f"seed{seed}.json")
        for seed in (1, 2, 3)
    ]
    for process, score in zip(processes, (4096, 2048, 2048), strict=True):
        _write_progress(process.progress_path, seed=process.seed, score=score)
    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard.process_is_alive", lambda pid: True
    )

    assert choose_victim(processes) == processes[2]


def test_choose_victim_does_not_prefer_stale_progress(tmp_path, monkeypatch) -> None:
    processes = [
        SeedProcess(seed, 100 + seed, tmp_path / f"seed{seed}.json")
        for seed in (1, 2, 3)
    ]
    _write_progress(processes[0].progress_path, seed=1, score=0)
    old = time.time() - 600
    os.utime(processes[0].progress_path, (old, old))
    _write_progress(processes[1].progress_path, seed=2, score=100)
    _write_progress(processes[2].progress_path, seed=3, score=200)
    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard.process_is_alive", lambda pid: True
    )

    assert choose_victim(processes) == processes[1]


def test_verify_cgroup_checks_hard_limit_and_zero_swap(tmp_path) -> None:
    limit = 18 * 1024**3
    (tmp_path / "memory.max").write_text(str(limit), encoding="utf-8")
    (tmp_path / "memory.swap.max").write_text("0", encoding="utf-8")
    verify_cgroup(tmp_path, limit)

    (tmp_path / "memory.swap.max").write_text("max", encoding="utf-8")
    with pytest.raises(RuntimeError, match="memory.swap.max"):
        verify_cgroup(tmp_path, limit)


def test_monitor_stops_survivors_and_terminates_least_progress(
    tmp_path, monkeypatch
) -> None:
    processes = [
        SeedProcess(seed, 100 + seed, tmp_path / f"seed{seed}.json")
        for seed in (1, 2, 3)
    ]
    for process, score in zip(processes, (300, 100, 200), strict=True):
        _write_progress(process.progress_path, seed=process.seed, score=score)
    alive = {process.pid: True for process in processes}
    signals = []
    resumed = 0

    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard.verify_cgroup", lambda *args: None
    )
    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard.seed_is_alive",
        lambda process: alive[process.pid],
    )
    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard._read_integer",
        lambda path: 800 * MIB if alive[102] else 0,
    )

    def fake_signal(process, signum):
        nonlocal resumed
        signals.append((process.seed, signum))
        if signum == signal.SIGTERM:
            alive[process.pid] = False
        elif signum == signal.SIGCONT:
            resumed += 1
            if resumed == 2:
                alive.update({pid: False for pid in alive})

    monkeypatch.setattr(
        "episodic_moba_ppo.memory_guard._signal_group", fake_signal
    )
    monkeypatch.setattr("episodic_moba_ppo.memory_guard.time.sleep", lambda _: None)

    assert (
        monitor(
            cgroup=tmp_path,
            memory_limit=1024 * MIB,
            processes=processes,
            poll_seconds=0.001,
            grace_seconds=0.01,
        )
        == 1
    )
    assert signals == [
        (1, signal.SIGSTOP),
        (3, signal.SIGSTOP),
        (2, signal.SIGTERM),
        (1, signal.SIGCONT),
        (3, signal.SIGCONT),
    ]
