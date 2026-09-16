"""Low-overhead aggregation and sampling of MoBA routing diagnostics."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from episodic_moba_ppo.episodic_memory import TraceRef

THRESHOLDS = (256, 512, 1024)
DETAIL_FIELDS = (
    "arm",
    "model_seed",
    "update",
    "episode_id",
    "query_id",
    "layer",
    "query_timestep",
    "selected",
    "candidate_blocks",
    "block_index",
    "block_start",
    "block_end",
    "routing_score",
    "retrieval_distance",
)


@dataclass(frozen=True)
class QueryRoutingSummary:
    layer: int
    query_timestep: int
    selected_indices: tuple[int, ...]
    selected_ranges: tuple[tuple[int, int], ...]
    selected_scores: tuple[float, ...]
    candidate_count: int
    mean_distance: float
    max_distance: int
    fractions_outside: Mapping[int, float]

    @property
    def no_eligible(self) -> bool:
        return self.candidate_count == 0


@dataclass
class _LayerAggregate:
    queries: int = 0
    no_eligible: int = 0
    candidate_total: int = 0
    selected_total: int = 0
    distance_total: int = 0
    distance_max: int = 0
    outside: Counter[int] = field(default_factory=Counter)
    selected_indices: Counter[int] = field(default_factory=Counter)
    score_total: float = 0.0
    score_count: int = 0
    retrieved_attention_mass_total: float = 0.0


@dataclass(frozen=True)
class MobaMetrics:
    values: Mapping[str, float]
    retrieval_distances: tuple[int, ...]


def summarize_selection(
    selection: Any, *, layer: int, query_timestep: int
) -> QueryRoutingSummary:
    selected = tuple(int(value) for value in selection.selected_block_indices.tolist())
    ranges = tuple(
        (int(start), int(end)) for start, end in selection.selected_block_ranges
    )
    if len(selected) != len(ranges):
        raise ValueError("selected block indices and ranges must align")
    if selected:
        positions = torch.searchsorted(
            selection.candidate_block_indices,
            selection.selected_block_indices
        )
        scores = tuple(
            float(value)
            for value in selection.routing_scores.detach()
            .index_select(0, positions)
            .cpu()
            .tolist()
        )
    else:
        scores = ()
    distances = tuple(query_timestep - end for _, end in ranges)
    if any(distance <= 0 for distance in distances):
        raise ValueError("retrieved blocks must be strictly historical")
    count = len(distances)
    fractions = {
        threshold: (
            sum(distance > threshold for distance in distances) / count
            if count
            else 0.0
        )
        for threshold in THRESHOLDS
    }
    return QueryRoutingSummary(
        layer=int(layer),
        query_timestep=int(query_timestep),
        selected_indices=selected,
        selected_ranges=ranges,
        selected_scores=scores,
        candidate_count=int(selection.candidate_count),
        mean_distance=sum(distances) / count if count else 0.0,
        max_distance=max(distances, default=0),
        fractions_outside=fractions,
    )


class RoutingDiagnostics:
    def __init__(
        self,
        *,
        sample_rate: float,
        arm: str,
        model_seed: int,
        random_seed: int = 0,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        self.sample_rate = float(sample_rate)
        self.arm = arm
        self.model_seed = int(model_seed)
        self._random = random.Random(random_seed)
        self._layers: dict[int, _LayerAggregate] = defaultdict(_LayerAggregate)
        self._detail_rows: list[dict[str, Any]] = []
        self._retrieval_distances: list[int] = []
        self._query_keys: set[tuple[int | str, int]] = set()
        self._useful_query_keys: set[tuple[int | str, int]] = set()

    def observe(
        self,
        routing: Sequence[Sequence[Any | None]],
        references: Sequence[TraceRef],
        *,
        update: int,
    ) -> None:
        for layer, selections in enumerate(routing):
            if len(selections) != len(references):
                raise ValueError("routing batch and trace references must align")
            for reference, selection in zip(references, selections, strict=True):
                if selection is None:
                    continue
                summary = summarize_selection(
                    selection,
                    layer=layer,
                    query_timestep=reference.query_timestep,
                )
                aggregate = self._layers[layer]
                aggregate.queries += 1
                aggregate.no_eligible += int(summary.no_eligible)
                aggregate.candidate_total += summary.candidate_count
                distances = [
                    summary.query_timestep - end
                    for _, end in summary.selected_ranges
                ]
                query_key = (reference.trace_id, reference.query_timestep)
                self._query_keys.add(query_key)
                if selection.useful_retrieval:
                    self._useful_query_keys.add(query_key)
                aggregate.selected_total += len(distances)
                aggregate.distance_total += sum(distances)
                self._retrieval_distances.extend(distances)
                aggregate.distance_max = max(
                    aggregate.distance_max, summary.max_distance
                )
                for threshold in THRESHOLDS:
                    aggregate.outside[threshold] += sum(
                        distance > threshold for distance in distances
                    )
                aggregate.selected_indices.update(summary.selected_indices)
                aggregate.score_total += sum(summary.selected_scores)
                aggregate.score_count += len(summary.selected_scores)
                aggregate.retrieved_attention_mass_total += float(
                    selection.retrieved_attention_mass
                )
                if self._random.random() < self.sample_rate:
                    self._sample_rows(summary, reference, update)

    def _sample_rows(
        self, summary: QueryRoutingSummary, reference: TraceRef, update: int
    ) -> None:
        common = {
            "arm": self.arm,
            "model_seed": self.model_seed,
            "update": int(update),
            "episode_id": reference.trace_id,
            "query_id": f"{reference.trace_id}:{reference.query_timestep}",
            "layer": summary.layer,
            "query_timestep": summary.query_timestep,
            "candidate_blocks": summary.candidate_count,
        }
        if not summary.selected_indices:
            self._detail_rows.append(
                {
                    **common,
                    "selected": False,
                    "block_index": "",
                    "block_start": "",
                    "block_end": "",
                    "routing_score": "",
                    "retrieval_distance": "",
                }
            )
            return
        for block, (start, end), score in zip(
            summary.selected_indices,
            summary.selected_ranges,
            summary.selected_scores,
            strict=True,
        ):
            self._detail_rows.append(
                {
                    **common,
                    "selected": True,
                    "block_index": block,
                    "block_start": start,
                    "block_end": end,
                    "routing_score": score,
                    "retrieval_distance": summary.query_timestep - end,
                }
            )

    @staticmethod
    def _entropy_and_diversity(counts: Counter[int], total: int) -> tuple[float, float]:
        if total == 0:
            return 0.0, 0.0
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
        normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        return normalized, len(counts) / total

    def metrics(self, *, reset: bool = False) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for layer, aggregate in sorted(self._layers.items()):
            prefix = f"routing/layer_{layer}"
            selected = aggregate.selected_total
            entropy, diversity = self._entropy_and_diversity(
                aggregate.selected_indices, selected
            )
            metrics.update(
                {
                    f"{prefix}/query_count": float(aggregate.queries),
                    f"{prefix}/candidate_count_mean": aggregate.candidate_total
                    / max(1, aggregate.queries),
                    f"{prefix}/no_eligible_fraction": aggregate.no_eligible
                    / max(1, aggregate.queries),
                    f"{prefix}/retrieval_distance_mean": aggregate.distance_total
                    / max(1, selected),
                    f"{prefix}/retrieval_distance_max": float(
                        aggregate.distance_max
                    ),
                    f"{prefix}/selection_entropy": entropy,
                    f"{prefix}/selection_diversity": diversity,
                    f"{prefix}/selected_score_mean": aggregate.score_total
                    / max(1, aggregate.score_count),
                }
            )
            for threshold in THRESHOLDS:
                metrics[f"{prefix}/fraction_outside_{threshold}"] = (
                    aggregate.outside[threshold] / max(1, selected)
                )
        if reset:
            self._layers.clear()
        return metrics

    def moba_metrics(self, *, dense_recent: int, reset: bool = False) -> MobaMetrics:
        """Return layer-aggregated, bounded W&B diagnostics for one update."""
        distances = tuple(self._retrieval_distances)
        selected_total = sum(layer.selected_total for layer in self._layers.values())
        query_total = sum(layer.queries for layer in self._layers.values())
        counts: Counter[int] = Counter()
        for layer in self._layers.values():
            counts.update(layer.selected_indices)
        entropy, _ = self._entropy_and_diversity(counts, selected_total)
        attention_mass = sum(
            layer.retrieved_attention_mass_total for layer in self._layers.values()
        )
        values = {
            "moba/selected_distance_mean": (
                sum(distances) / len(distances) if distances else 0.0
            ),
            "moba/selected_distance_p90": (
                float(torch.quantile(torch.tensor(distances, dtype=torch.float32), 0.9))
                if distances
                else 0.0
            ),
            "moba/fraction_beyond_recent_window": (
                sum(distance > dense_recent for distance in distances) / len(distances)
                if distances
                else 0.0
            ),
            "moba/selection_entropy": entropy,
            "moba/unique_blocks_selected": selected_total / max(1, query_total),
            "moba/retrieved_attention_mass": attention_mass / max(1, query_total),
            "moba/useful_retrieval_rate": len(self._useful_query_keys)
            / max(1, len(self._query_keys)),
        }
        result = MobaMetrics(values=values, retrieval_distances=distances)
        if reset:
            self._layers.clear()
            self._retrieval_distances.clear()
            self._query_keys.clear()
            self._useful_query_keys.clear()
        return result

    def write_details(self, path: str | Path) -> Path | None:
        if not self._detail_rows:
            return None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        append = destination.is_file()
        with destination.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS)
            if not append:
                writer.writeheader()
            writer.writerows(self._detail_rows)
        self._detail_rows.clear()
        return destination
