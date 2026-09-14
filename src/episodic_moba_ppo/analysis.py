"""Descriptive analysis of sampled MoBA routing rows."""

from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVENANCE_COLUMNS = ("arm", "model_seed", "update", "episode_id", "query_id")
THRESHOLDS = (256, 512, 1024)


class RoutingAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class RoutingRow:
    layer: int
    query_timestep: int
    selected: bool
    candidate_blocks: int
    block_index: int | None
    block_start: int | None
    block_end: int | None
    routing_score: float | None
    retrieval_distance: int | None
    provenance: dict[str, str]

    @property
    def block_age(self) -> float | None:
        if self.block_start is None or self.block_end is None:
            return None
        return self.query_timestep - (self.block_start + self.block_end) / 2

    @property
    def query_key(self) -> tuple[str, ...]:
        return tuple(self.provenance.get(key, "") for key in PROVENANCE_COLUMNS) + (
            str(self.layer),
            str(self.query_timestep),
        )


@dataclass(frozen=True)
class AnalysisOutputs:
    summary_csv: Path
    retrieval_distances_csv: Path
    block_ages_csv: Path
    routing_scores_csv: Path


def _first(row: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _integer(
    row: Mapping[str, str], names: Sequence[str], *, required: bool
) -> int | None:
    value = _first(row, names)
    if not value:
        if required:
            raise RoutingAnalysisError(f"missing required column/value: {names[0]}")
        return None
    try:
        return int(value)
    except ValueError as error:
        raise RoutingAnalysisError(
            f"{names[0]} must be an integer, got {value!r}"
        ) from error


def _floating(row: Mapping[str, str], names: Sequence[str]) -> float | None:
    value = _first(row, names)
    if not value:
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise RoutingAnalysisError(
            f"{names[0]} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise RoutingAnalysisError(f"{names[0]} must be finite")
    return result


def _boolean(row: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = _first(row, (name,)).lower()
    if not value:
        return default
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise RoutingAnalysisError(f"{name} must be true or false, got {value!r}")


def parse_routing_row(row: Mapping[str, str]) -> RoutingRow:
    selected = _boolean(row, "selected", default=True)
    layer = _integer(row, ("layer",), required=True)
    query_timestep = _integer(row, ("query_timestep",), required=True)
    candidate_blocks = _integer(row, ("candidate_blocks",), required=True)
    assert (
        layer is not None
        and query_timestep is not None
        and candidate_blocks is not None
    )
    if layer < 0 or query_timestep < 0 or candidate_blocks < 0:
        raise RoutingAnalysisError(
            "layer, query_timestep, and candidate_blocks must be nonnegative"
        )

    block_index = _integer(
        row, ("block_index", "selected_block_index"), required=selected
    )
    block_start = _integer(row, ("block_start", "selected_start"), required=selected)
    block_end = _integer(row, ("block_end", "selected_end"), required=selected)
    routing_score = _floating(row, ("routing_score", "score"))
    supplied_distance = _integer(row, ("retrieval_distance",), required=False)
    if selected:
        assert (
            block_index is not None
            and block_start is not None
            and block_end is not None
        )
        if candidate_blocks == 0:
            raise RoutingAnalysisError("a selected block requires candidate_blocks > 0")
        if block_index < 0 or block_start < 0 or block_end < block_start:
            raise RoutingAnalysisError("invalid selected block index/range")
        if block_end >= query_timestep:
            raise RoutingAnalysisError("selected blocks must be strictly historical")
        derived_distance = query_timestep - block_end
        if supplied_distance is not None and supplied_distance != derived_distance:
            raise RoutingAnalysisError(
                "retrieval_distance disagrees with query_timestep - block_end"
            )
        retrieval_distance = derived_distance
        if routing_score is None:
            raise RoutingAnalysisError("selected rows require routing_score")
    else:
        if any(
            value is not None
            for value in (block_index, block_start, block_end, routing_score)
        ):
            raise RoutingAnalysisError(
                "unselected rows must leave block fields and score blank"
            )
        retrieval_distance = None
    provenance = {
        name: str(row.get(name, "") or "").strip() for name in PROVENANCE_COLUMNS
    }
    return RoutingRow(
        layer=layer,
        query_timestep=query_timestep,
        selected=selected,
        candidate_blocks=candidate_blocks,
        block_index=block_index,
        block_start=block_start,
        block_end=block_end,
        routing_score=routing_score,
        retrieval_distance=retrieval_distance,
        provenance=provenance,
    )


def read_routing_csv(path: str | os.PathLike[str]) -> list[RoutingRow]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RoutingAnalysisError("routing CSV has no header")
        rows = [parse_routing_row(row) for row in reader]
    if not rows:
        raise RoutingAnalysisError("routing CSV contains no data rows")
    return rows


def _entropy_and_diversity(rows: Sequence[RoutingRow]) -> tuple[float, float]:
    indices = [row.block_index for row in rows if row.selected]
    if not indices:
        return 0.0, 0.0
    counts = Counter(indices)
    total = len(indices)
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return normalized_entropy, len(counts) / total


def descriptive_summary(rows: Sequence[RoutingRow]) -> list[dict[str, Any]]:
    if not rows:
        raise RoutingAnalysisError("cannot summarize zero routing rows")
    summaries: list[dict[str, Any]] = []
    scopes: list[tuple[str, Sequence[RoutingRow]]] = [("all", rows)]
    by_layer: dict[int, list[RoutingRow]] = defaultdict(list)
    for row in rows:
        by_layer[row.layer].append(row)
    scopes.extend(
        (f"layer:{layer}", group) for layer, group in sorted(by_layer.items())
    )

    for scope, group in scopes:
        selected = [row for row in group if row.selected]
        query_groups: dict[tuple[str, ...], list[RoutingRow]] = defaultdict(list)
        for row in group:
            query_groups[row.query_key].append(row)
        candidates = [
            max(item.candidate_blocks for item in query)
            for query in query_groups.values()
        ]
        distances = [
            int(row.retrieval_distance)
            for row in selected
            if row.retrieval_distance is not None
        ]
        scores = [
            float(row.routing_score)
            for row in selected
            if row.routing_score is not None
        ]
        entropy, diversity = _entropy_and_diversity(selected)
        metrics: dict[str, int | float | str] = {
            "query_count": len(query_groups),
            "selected_block_count": len(selected),
            "no_eligible_query_fraction": sum(value == 0 for value in candidates)
            / len(candidates),
            "mean_candidate_blocks": statistics.fmean(candidates),
            "selection_entropy_normalized": entropy,
            "selected_block_unique_fraction": diversity,
            "retrieval_distance_definition": "query_timestep - block_end",
        }
        if distances:
            metrics["mean_retrieval_distance"] = statistics.fmean(distances)
            metrics["max_retrieval_distance"] = max(distances)
            for threshold in THRESHOLDS:
                metrics[f"fraction_retrieval_distance_outside_{threshold}"] = sum(
                    distance > threshold for distance in distances
                ) / len(distances)
        if scores:
            metrics.update(
                routing_score_mean=statistics.fmean(scores),
                routing_score_min=min(scores),
                routing_score_max=max(scores),
            )
        summaries.extend(
            {"scope": scope, "metric": metric, "value": value}
            for metric, value in metrics.items()
        )
    return summaries


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def analyze_routing_csv(
    input_path: str | os.PathLike[str],
    summary_csv: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> AnalysisOutputs:
    rows = read_routing_csv(input_path)
    summary_path = Path(summary_csv)
    output = Path(output_dir)
    retrieval_path = output / "retrieval_distance_distribution.csv"
    ages_path = output / "selected_block_age_distribution.csv"
    scores_path = output / "routing_score_distribution.csv"
    passthrough = list(PROVENANCE_COLUMNS)

    _atomic_write_csv(
        summary_path, ("scope", "metric", "value"), descriptive_summary(rows)
    )
    selected = [row for row in rows if row.selected]
    common_fields = passthrough + ["layer", "query_timestep", "block_index"]
    _atomic_write_csv(
        retrieval_path,
        common_fields + ["retrieval_distance"],
        (
            {
                **row.provenance,
                "layer": row.layer,
                "query_timestep": row.query_timestep,
                "block_index": row.block_index,
                "retrieval_distance": row.retrieval_distance,
            }
            for row in selected
        ),
    )
    _atomic_write_csv(
        ages_path,
        common_fields + ["block_start", "block_end", "selected_block_age"],
        (
            {
                **row.provenance,
                "layer": row.layer,
                "query_timestep": row.query_timestep,
                "block_index": row.block_index,
                "block_start": row.block_start,
                "block_end": row.block_end,
                "selected_block_age": row.block_age,
            }
            for row in selected
        ),
    )
    _atomic_write_csv(
        scores_path,
        common_fields + ["routing_score"],
        (
            {
                **row.provenance,
                "layer": row.layer,
                "query_timestep": row.query_timestep,
                "block_index": row.block_index,
                "routing_score": row.routing_score,
            }
            for row in selected
        ),
    )
    return AnalysisOutputs(summary_path, retrieval_path, ages_path, scores_path)
