import csv

import pytest
import torch

from episodic_moba_ppo.analysis import read_routing_csv
from episodic_moba_ppo.diagnostics import (
    RoutingDiagnostics,
    summarize_selection,
)
from episodic_moba_ppo.episodic_memory import TraceRef
from episodic_moba_ppo.moba_retrieval import MobaSelection


def _selection():
    return MobaSelection(
        context_indices=torch.arange(4),
        context_timesteps=torch.tensor([32, 47, 80, 95]),
        dense_indices=torch.empty(0, dtype=torch.long),
        selected_block_indices=torch.tensor([2, 5]),
        selected_block_ranges=((32, 47), (80, 95)),
        candidate_block_indices=torch.tensor([2, 4, 5]),
        routing_scores=torch.tensor([0.1, 0.2, 0.3], requires_grad=True),
    )


def _empty_selection():
    empty_long = torch.empty(0, dtype=torch.long)
    return MobaSelection(
        context_indices=empty_long,
        context_timesteps=empty_long,
        dense_indices=empty_long,
        selected_block_indices=empty_long,
        selected_block_ranges=(),
        candidate_block_indices=empty_long,
        routing_scores=torch.empty(0),
    )


def test_query_summary_contains_locked_compact_diagnostics() -> None:
    summary = summarize_selection(_selection(), layer=1, query_timestep=800)

    assert summary.layer == 1
    assert summary.selected_indices == (2, 5)
    assert summary.selected_ranges == ((32, 47), (80, 95))
    assert summary.selected_scores == pytest.approx((0.1, 0.3))
    assert summary.candidate_count == 3
    assert summary.mean_distance == 729
    assert summary.max_distance == 753
    assert summary.fractions_outside == {256: 1.0, 512: 1.0, 1024: 0.0}


def test_collector_aggregates_and_writes_analysis_compatible_rows(tmp_path) -> None:
    collector = RoutingDiagnostics(
        sample_rate=1.0, arm="trxl_moba", model_seed=2, random_seed=4
    )
    collector.observe(
        ((_selection(), _empty_selection()),),
        (TraceRef(7, 800), TraceRef(8, 100)),
        update=3,
    )

    metrics = collector.metrics(reset=True)
    assert metrics["routing/layer_0/query_count"] == 2
    assert metrics["routing/layer_0/candidate_count_mean"] == 1.5
    assert metrics["routing/layer_0/no_eligible_fraction"] == 0.5
    assert metrics["routing/layer_0/retrieval_distance_mean"] == 729
    assert metrics["routing/layer_0/retrieval_distance_max"] == 753
    assert metrics["routing/layer_0/fraction_outside_512"] == 1.0
    assert metrics["routing/layer_0/selection_entropy"] == pytest.approx(1.0)
    assert metrics["routing/layer_0/selection_diversity"] == pytest.approx(1.0)

    path = collector.write_details(tmp_path / "routing-details.csv")
    assert path is not None
    parsed = read_routing_csv(path)
    assert len(parsed) == 3
    assert [row.selected for row in parsed] == [True, True, False]
    assert parsed[0].provenance["query_id"] == "7:800"
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    assert raw[0]["arm"] == "trxl_moba"
    assert raw[0]["model_seed"] == "2"


def test_zero_sampling_keeps_compact_metrics_without_artifact(tmp_path) -> None:
    collector = RoutingDiagnostics(
        sample_rate=0.0, arm="trxl_moba", model_seed=1
    )
    collector.observe(((_selection(),),), (TraceRef(1, 800),), update=1)
    assert collector.metrics()["routing/layer_0/query_count"] == 1
    assert collector.write_details(tmp_path / "routing.csv") is None
    assert not (tmp_path / "routing.csv").exists()
