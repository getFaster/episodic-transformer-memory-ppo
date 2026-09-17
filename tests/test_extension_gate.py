import json
from pathlib import Path

import pytest

from episodic_moba_ppo.extension_gate import (
    ExtensionGateError,
    require_extension_gate,
    validate_extension_gate,
)

COMMIT = "a" * 40
CHECKPOINT = "b" * 64


def _set_passed_false(document: dict) -> None:
    document["passed"] = False


def _set_wrong_hash(document: dict) -> None:
    document["provenance"]["checkpoint_sha256"] = "c" * 64


def _remove_early_window(document: dict) -> None:
    document["runs"][0]["metrics"] = [
        metric
        for metric in document["runs"][0]["metrics"]
        if metric["step"] >= 400000
    ]


def _artifact() -> dict:
    runs = []
    for arm in ("trxl", "trxl_moba"):
        for seed, early, late in ((1, 0.10, 0.25), (2, 0.20, 0.35), (3, 0.30, 0.45)):
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "metrics": [
                        {"step": 300000, "training_success": early},
                        {"step": 399999, "training_success": early},
                        {"step": 400000, "training_success": late},
                        {"step": 507904, "training_success": late},
                    ],
                }
            )
    return {
        "schema_version": 1,
        "task": "extension-gate",
        "passed": True,
        "provenance": {
            "source_commit": COMMIT,
            "checkpoint_sha256": CHECKPOINT,
        },
        "windows": {
            "early": {
                "start_step": 300000,
                "end_step": 400000,
                "end_inclusive": False,
            },
            "late": {
                "start_step": 400000,
                "end_step": 507904,
                "end_inclusive": True,
            },
        },
        "minimum_improvement": 0.05,
        "runs": runs,
    }


def test_valid_extension_gate_recomputes_across_seed_medians() -> None:
    summary = validate_extension_gate(
        _artifact(),
        expected_checkpoint_sha256=CHECKPOINT,
        expected_source_commit=COMMIT,
    )

    assert summary["passed"] is True
    assert summary["arms"]["trxl"]["early_median"] == pytest.approx(0.2)
    assert summary["arms"]["trxl"]["late_median"] == pytest.approx(0.35)
    assert summary["arms"]["trxl_moba"]["improvement"] == pytest.approx(0.15)


def test_extension_gate_resolves_relative_path_against_repo_root(tmp_path: Path) -> None:
    artifact_path = tmp_path / "results" / "extension_gate.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")

    summary = require_extension_gate(
        "results/extension_gate.json",
        repo_root=tmp_path,
        expected_checkpoint_sha256=CHECKPOINT,
        expected_source_commit=COMMIT,
    )

    assert summary["artifact_path"] == str(artifact_path.resolve())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda doc: doc["runs"].pop(), "incomplete arm/seed coverage"),
        (_remove_early_window, "both gate windows"),
        (_set_passed_false, "did not pass"),
        (_set_wrong_hash, "does not match"),
        (lambda doc: doc["windows"]["early"].__setitem__("end_step", 400001), "locked"),
    ],
)
def test_extension_gate_fails_closed_for_invalid_artifacts(mutate, match: str) -> None:
    document = _artifact()
    mutate(document)
    with pytest.raises(ExtensionGateError, match=match):
        validate_extension_gate(
            document,
            expected_checkpoint_sha256=CHECKPOINT,
            expected_source_commit=COMMIT,
        )


def test_extension_gate_rejects_unknown_fields_and_invalid_success_values() -> None:
    document = _artifact()
    document["unexpected"] = 1
    with pytest.raises(ExtensionGateError, match="unknown"):
        validate_extension_gate(document)

    document = _artifact()
    document["runs"][0]["metrics"][0]["training_success"] = 1.1
    with pytest.raises(ExtensionGateError, match="between 0 and 1"):
        validate_extension_gate(document)
