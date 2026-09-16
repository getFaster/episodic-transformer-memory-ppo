"""Deterministic fixed-block MoBA retrieval for a single layer and query."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Projection = Callable[[torch.Tensor], torch.Tensor]
USEFUL_ATTENTION_THRESHOLD = 1e-6


@dataclass(frozen=True)
class MobaSelection:
    context_indices: torch.Tensor
    context_timesteps: torch.Tensor
    dense_indices: torch.Tensor
    selected_block_indices: torch.Tensor
    selected_block_ranges: tuple[tuple[int, int], ...]
    candidate_block_indices: torch.Tensor
    routing_scores: torch.Tensor
    retrieved_attention_mass: float = 0.0
    useful_retrieval: bool = False

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_block_indices.numel())


@dataclass(frozen=True)
class BlockSelection:
    """Selection over already summarized fixed episode blocks."""

    candidate_positions: torch.Tensor
    selected_block_indices: torch.Tensor
    routing_scores: torch.Tensor


def select_moba_blocks(
    query: torch.Tensor,
    normalized_block_summaries: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    *,
    project_queries: Projection,
    project_keys: Projection,
    retrieved_blocks: int,
) -> BlockSelection:
    """Route using only small block summaries under the current Q/K weights."""
    if normalized_block_summaries.ndim != 2:
        raise ValueError("block summaries must have shape (blocks, embed_dim)")
    if candidate_block_indices.ndim != 1 or candidate_block_indices.numel() != normalized_block_summaries.shape[0]:
        raise ValueError("candidate block indices must align with summaries")
    if retrieved_blocks < 0:
        raise ValueError("retrieved_blocks cannot be negative")
    if candidate_block_indices.numel() > 1 and not bool(
        torch.all(candidate_block_indices[1:] > candidate_block_indices[:-1])
    ):
        raise ValueError("candidate blocks must be strictly chronological")

    if not candidate_block_indices.numel():
        empty = candidate_block_indices.detach()
        return BlockSelection(
            candidate_positions=empty,
            selected_block_indices=empty,
            routing_scores=normalized_block_summaries.new_empty((0,)),
        )

    query_heads = project_queries(query.reshape(1, 1, -1))[0, 0]
    key_heads = project_keys(normalized_block_summaries.unsqueeze(0))[0]
    per_head_scores = torch.einsum("hd,bhd->bh", query_heads, key_heads)
    scores = (per_head_scores / query_heads.shape[-1] ** 0.5).amax(dim=1)
    ranked = torch.argsort(scores.detach(), descending=True, stable=True)
    chosen = ranked[: min(retrieved_blocks, int(ranked.numel()))]
    chosen = torch.sort(chosen).values
    return BlockSelection(
        candidate_positions=chosen.detach(),
        selected_block_indices=candidate_block_indices.index_select(0, chosen).detach(),
        routing_scores=scores,
    )


def select_moba_context(
    query: torch.Tensor,
    normalized_history: torch.Tensor,
    history_timesteps: torch.Tensor,
    query_timestep: int,
    *,
    project_queries: Projection,
    project_keys: Projection,
    dense_recent: int = 128,
    search_horizon: int = 2560,
    block_size: int = 16,
    retrieved_blocks: int = 8,
    attention_budget: int = 256,
) -> MobaSelection:
    """Select old fixed blocks plus dense recent history.

    ``normalized_history`` contains content-only attention inputs: normalized
    tokens for pre-norm blocks, raw tokens for post-norm blocks. Its block means
    are projected under the current K weights, equivalent to averaging their
    bias-free K projections.
    """
    if normalized_history.ndim != 2:
        raise ValueError("normalized_history must have shape (time, embed_dim)")
    if history_timesteps.ndim != 1 or history_timesteps.numel() != normalized_history.shape[0]:
        raise ValueError("history_timesteps must align one-to-one with history")
    if dense_recent < 0 or search_horizon <= 0 or block_size <= 0 or retrieved_blocks < 0:
        raise ValueError("invalid routing budget")
    if dense_recent + retrieved_blocks * block_size > attention_budget:
        raise ValueError("configured retrieval can exceed the attention budget")
    if history_timesteps.numel() > 1 and not bool(torch.all(history_timesteps[1:] > history_timesteps[:-1])):
        raise ValueError("history_timesteps must be strictly chronological")

    device = history_timesteps.device
    lower_bound = max(0, int(query_timestep) - search_horizon)
    eligible = torch.nonzero(
        (history_timesteps >= lower_bound) & (history_timesteps < int(query_timestep)),
        as_tuple=False,
    ).flatten()
    dense_count = min(dense_recent, int(eligible.numel()))
    dense_indices = eligible[-dense_count:] if dense_count else eligible[:0]
    old_indices = eligible[:-dense_count] if dense_count else eligible

    if old_indices.numel():
        old_block_ids = torch.div(
            history_timesteps.index_select(0, old_indices), block_size, rounding_mode="floor"
        )
        candidate_blocks = torch.unique_consecutive(old_block_ids)
        members = [old_indices[old_block_ids == block_id] for block_id in candidate_blocks]
        summaries = torch.stack(
            [normalized_history.index_select(0, indices).mean(dim=0) for indices in members]
        )
        block_selection = select_moba_blocks(
            query,
            summaries,
            candidate_blocks,
            project_queries=project_queries,
            project_keys=project_keys,
            retrieved_blocks=retrieved_blocks,
        )
        chosen = block_selection.candidate_positions
        scores = block_selection.routing_scores
        selected_blocks = block_selection.selected_block_indices
        selected_old = torch.cat([members[int(index)] for index in chosen.tolist()]) if chosen.numel() else old_indices[:0]
        selected_ranges = tuple(
            (
                int(history_timesteps[members[int(index)][0]].item()),
                int(history_timesteps[members[int(index)][-1]].item()),
            )
            for index in chosen.tolist()
        )
    else:
        candidate_blocks = torch.empty(0, dtype=torch.long, device=device)
        selected_blocks = candidate_blocks
        selected_old = old_indices
        scores = normalized_history.new_empty((0,))
        selected_ranges = ()

    context_indices = torch.cat((selected_old, dense_indices))
    if context_indices.numel() > attention_budget:
        raise AssertionError("MoBA selection exceeded the attention budget")
    return MobaSelection(
        context_indices=context_indices.detach(),
        context_timesteps=history_timesteps.index_select(0, context_indices).detach(),
        dense_indices=dense_indices.detach(),
        selected_block_indices=selected_blocks.detach(),
        selected_block_ranges=selected_ranges,
        candidate_block_indices=candidate_blocks.detach(),
        routing_scores=scores,
    )
