"""Detached, episode-local Transformer memory traces.

The rollout buffer stores lightweight :class:`TraceRef` objects.  Hidden states
are stored once per episode on CPU and are sliced only when a PPO microbatch is
materialized.  This keeps episode identity explicit and makes cross-episode or
future retrieval impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TraceRef:
    trace_id: int
    query_timestep: int


@dataclass(frozen=True)
class HistoricalBlock:
    """A clipped slice of one fixed episode-aligned block."""

    block_index: int
    # Token bodies deliberately remain detached on CPU until this block is
    # selected by the current Q/K router.
    states: torch.Tensor
    timesteps: torch.Tensor


@dataclass(frozen=True)
class EpisodeContext:
    """History eligible for a query, with old token bodies retained on CPU."""

    trace_id: int
    query_timestep: int
    dense_states: torch.Tensor
    dense_timesteps: torch.Tensor
    old_blocks: tuple[HistoricalBlock, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.old_blocks)


class EpisodeTrace:
    """A contiguous CPU trace with shape ``[time, layer, width]``."""

    def __init__(self, trace_id: int, num_layers: int, width: int) -> None:
        if num_layers < 1 or width < 1:
            raise ValueError("num_layers and width must be positive")
        self.trace_id = int(trace_id)
        self.num_layers = int(num_layers)
        self.width = int(width)
        self._states: list[torch.Tensor] = []
        self._closed = False

    def __len__(self) -> int:
        return len(self._states)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def states(self) -> torch.Tensor:
        if not self._states:
            return torch.empty((0, self.num_layers, self.width), dtype=torch.float32)
        return torch.stack(self._states)

    def append(
        self, hidden_states: torch.Tensor, *, timestep: int | None = None
    ) -> TraceRef:
        if self._closed:
            raise RuntimeError("cannot append to a closed episode trace")
        expected = len(self)
        actual = expected if timestep is None else int(timestep)
        if actual != expected:
            raise ValueError(
                "episode timesteps must be contiguous: "
                f"expected {expected}, got {actual}"
            )
        if tuple(hidden_states.shape) != (self.num_layers, self.width):
            raise ValueError(
                "hidden state must have shape "
                f"({self.num_layers}, {self.width}), got {tuple(hidden_states.shape)}"
            )
        state = hidden_states.detach().to(device="cpu", dtype=torch.float32).clone()
        state.requires_grad_(False)
        self._states.append(state)
        return TraceRef(self.trace_id, actual)

    def reference(self, query_timestep: int | None = None) -> TraceRef:
        timestep = len(self) if query_timestep is None else int(query_timestep)
        if timestep < 0 or timestep > len(self):
            raise ValueError("query timestep must address the existing trace prefix")
        return TraceRef(self.trace_id, timestep)

    def close(self) -> None:
        self._closed = True

    def context(
        self,
        query_timestep: int,
        *,
        dense_recent: int,
        search_horizon: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> EpisodeContext:
        if min(dense_recent, search_horizon, block_size) < 1:
            raise ValueError("context sizes must be positive")
        query = int(query_timestep)
        if query < 0 or query > len(self):
            raise ValueError("query timestep cannot be future or outside this trace")

        history_start = max(0, query - search_horizon)
        dense_start = max(history_start, query - dense_recent)
        all_states = self.states
        dense_states = all_states[dense_start:query].to(device=device)
        dense_times = torch.arange(dense_start, query, dtype=torch.long, device=device)

        first_block = history_start // block_size
        last_exclusive = dense_start
        last_block = (last_exclusive + block_size - 1) // block_size
        blocks: list[HistoricalBlock] = []
        for block_index in range(first_block, last_block):
            start = max(history_start, block_index * block_size)
            end = min(last_exclusive, (block_index + 1) * block_size)
            if start >= end:
                continue
            blocks.append(
                HistoricalBlock(
                    block_index=block_index,
                    states=all_states[start:end],
                    timesteps=torch.arange(start, end, dtype=torch.long),
                )
            )
        return EpisodeContext(
            trace_id=self.trace_id,
            query_timestep=query,
            dense_states=dense_states,
            dense_timesteps=dense_times,
            old_blocks=tuple(blocks),
        )


class TraceRegistry:
    """Deduplicated episode traces and per-worker active-episode identities."""

    def __init__(self, num_layers: int, width: int) -> None:
        self.num_layers = int(num_layers)
        self.width = int(width)
        self._next_id = 0
        self._traces: dict[int, EpisodeTrace] = {}
        self._active_by_worker: dict[int, int] = {}

    def _new_trace(self) -> EpisodeTrace:
        trace = EpisodeTrace(self._next_id, self.num_layers, self.width)
        self._traces[trace.trace_id] = trace
        self._next_id += 1
        return trace

    def start_worker(self, worker_id: int) -> EpisodeTrace:
        worker = int(worker_id)
        if worker in self._active_by_worker:
            raise RuntimeError(f"worker {worker} already has an active episode")
        trace = self._new_trace()
        self._active_by_worker[worker] = trace.trace_id
        return trace

    def active_trace(self, worker_id: int) -> EpisodeTrace:
        try:
            return self._traces[self._active_by_worker[int(worker_id)]]
        except KeyError as error:
            raise KeyError(f"worker {worker_id} has no active episode") from error

    def append(self, worker_id: int, hidden_states: torch.Tensor) -> TraceRef:
        return self.active_trace(worker_id).append(hidden_states)

    def reset_worker(self, worker_id: int) -> EpisodeTrace:
        worker = int(worker_id)
        previous = self.active_trace(worker)
        previous.close()
        trace = self._new_trace()
        self._active_by_worker[worker] = trace.trace_id
        return trace

    def get(self, trace_id: int) -> EpisodeTrace:
        try:
            return self._traces[int(trace_id)]
        except KeyError as error:
            raise KeyError(f"unknown episode trace {trace_id}") from error

    def prune_closed(self) -> int:
        """Drop closed traces after their completed rollout has been optimized."""

        active = set(self._active_by_worker.values())
        removable = [
            trace_id
            for trace_id, trace in self._traces.items()
            if trace.closed and trace_id not in active
        ]
        for trace_id in removable:
            del self._traces[trace_id]
        return len(removable)

    def context(
        self,
        reference: TraceRef,
        *,
        dense_recent: int,
        search_horizon: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> EpisodeContext:
        return self.get(reference.trace_id).context(
            reference.query_timestep,
            dense_recent=dense_recent,
            search_horizon=search_horizon,
            block_size=block_size,
            device=device,
        )
