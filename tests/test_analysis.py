import csv

import pytest

from episodic_moba_ppo.analysis import (
    RoutingAnalysisError,
    analyze_routing_csv,
    descriptive_summary,
    parse_routing_row,
    read_routing_csv,
)


FIELDS = [
    "arm",
    "model_seed",
    "update",
    "episode_id",
    "layer",
    "query_timestep",
    "selected",
    "candidate_blocks",
    "block_index",
    "block_start",
    "block_end",
    "routing_score",
    "retrieval_distance",
]


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows():
    provenance = {
        "arm": "trxl_moba",
        "model_seed": "1",
        "update": "31",
        "episode_id": "episode-a",
    }
    return [
        {
            **provenance,
            "layer": 0,
            "query_timestep": 800,
            "selected": "true",
            "candidate_blocks": 20,
            "block_index": 6,
            "block_start": 96,
            "block_end": 111,
            "routing_score": 0.75,
            "retrieval_distance": 689,
        },
        {
            **provenance,
            "layer": 0,
            "query_timestep": 800,
            "selected": "true",
            "candidate_blocks": 20,
            "block_index": 25,
            "block_start": 400,
            "block_end": 415,
            "routing_score": -0.25,
            "retrieval_distance": 385,
        },
        {
            **provenance,
            "episode_id": "episode-b",
            "layer": 0,
            "query_timestep": 100,
            "selected": "false",
            "candidate_blocks": 0,
            "block_index": "",
            "block_start": "",
            "block_end": "",
            "routing_score": "",
            "retrieval_distance": "",
        },
    ]


def metric_map(summary, scope="all"):
    return {row["metric"]: row["value"] for row in summary if row["scope"] == scope}


def test_descriptive_summary_derives_distances_and_threshold_fractions():
    rows = [parse_routing_row(row) for row in sample_rows()]
    metrics = metric_map(descriptive_summary(rows))
    assert metrics["query_count"] == 2
    assert metrics["selected_block_count"] == 2
    assert metrics["no_eligible_query_fraction"] == pytest.approx(0.5)
    assert metrics["mean_retrieval_distance"] == pytest.approx(537)
    assert metrics["max_retrieval_distance"] == 689
    assert metrics["fraction_retrieval_distance_outside_256"] == 1.0
    assert metrics["fraction_retrieval_distance_outside_512"] == 0.5
    assert metrics["fraction_retrieval_distance_outside_1024"] == 0.0
    assert metrics["routing_score_mean"] == pytest.approx(0.25)


def test_analyzer_writes_nonlossy_descriptive_distributions(tmp_path):
    source = tmp_path / "routing.csv"
    write_rows(source, sample_rows())
    outputs = analyze_routing_csv(
        source, tmp_path / "summary.csv", tmp_path / "distributions"
    )
    assert outputs.summary_csv.is_file()
    assert outputs.retrieval_distances_csv.is_file()
    assert outputs.block_ages_csv.is_file()
    assert outputs.routing_scores_csv.is_file()

    with outputs.retrieval_distances_csv.open(newline="") as stream:
        distances = list(csv.DictReader(stream))
    assert [int(row["retrieval_distance"]) for row in distances] == [689, 385]
    assert all(row["arm"] == "trxl_moba" for row in distances)

    with outputs.block_ages_csv.open(newline="") as stream:
        ages = list(csv.DictReader(stream))
    assert [float(row["selected_block_age"]) for row in ages] == [696.5, 392.5]

    text = outputs.summary_csv.read_text(encoding="utf-8").lower()
    assert "significant" not in text
    assert "superior" not in text
    assert "positive" not in text
    assert "negative" not in text


def test_retrieval_distance_is_validated_when_supplied():
    row = sample_rows()[0]
    row["retrieval_distance"] = 1
    with pytest.raises(RoutingAnalysisError, match="disagrees"):
        parse_routing_row(row)


def test_future_or_dense_overlap_block_is_rejected():
    row = sample_rows()[0]
    row["block_end"] = 800
    row["retrieval_distance"] = 0
    with pytest.raises(RoutingAnalysisError, match="strictly historical"):
        parse_routing_row(row)


def test_empty_csv_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    write_rows(path, [])
    with pytest.raises(RoutingAnalysisError, match="no data rows"):
        read_routing_csv(path)
