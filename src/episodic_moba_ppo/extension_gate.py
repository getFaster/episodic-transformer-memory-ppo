"""Validation for the 62-update training extension gate.

The gate is intentionally computed from raw per-seed training-success rows,
instead of trusting a producer-supplied aggregate.  An artifact is valid only
when it contains both arms and exactly the paired model seeds (1, 2, 3), with
at least one sample for every seed in each locked window.  Each seed's window
value is the arithmetic mean of its samples; the arm value is the median of
those three per-seed means.

Canonical artifact schema (JSON):

.. code-block:: json

  {
    "schema_version": 1,
    "task": "extension-gate",
    "passed": true,
    "provenance": {
      "source_commit": "<40 lowercase hex characters>",
      "checkpoint_sha256": "<64 lowercase hex characters>"
    },
    "windows": {
      "early": {"start_step": 300000, "end_step": 400000, "end_inclusive": false},
      "late": {"start_step": 400000, "end_step": 507904, "end_inclusive": true}
    },
    "minimum_improvement": 0.05,
    "runs": [
      {"arm": "trxl", "seed": 1,
       "metrics": [{"step": 300000, "training_success": 0.1}]}
    ]
  }

``runs`` must contain one record for each arm/seed pair.  Unknown top-level
fields and malformed records are rejected, so a stale or partial artifact
cannot silently authorize the 62-update stage.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any


class ExtensionGateError(RuntimeError):
    """Raised when a 62-update extension artifact cannot authorize training."""


ARMS = ("trxl", "trxl_moba")
MODEL_SEEDS = (1, 2, 3)
EARLY_START = 300_000
EARLY_END = 400_000
LATE_START = 400_000
LATE_END = 507_904
MINIMUM_IMPROVEMENT = 0.05
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _fail(message: str) -> None:
    raise ExtensionGateError(f"extension gate artifact is invalid: {message}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        _fail(f"{name} has " + ", ".join(details))


def _integer(value: Any, name: str) -> int:
    # bool is an int subclass, but is not a valid step/seed value.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _validate_provenance(
    raw: Any,
    *,
    expected_checkpoint_sha256: str | None,
    expected_source_commit: str | None,
) -> dict[str, str]:
    provenance = _mapping(raw, "provenance")
    _exact_keys(provenance, {"source_commit", "checkpoint_sha256"}, "provenance")
    source_commit = provenance["source_commit"]
    checkpoint_sha256 = provenance["checkpoint_sha256"]
    if not isinstance(source_commit, str) or not _COMMIT_RE.fullmatch(source_commit):
        _fail("provenance.source_commit must be a lowercase 40-character Git hash")
    if not isinstance(checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
        checkpoint_sha256
    ):
        _fail("provenance.checkpoint_sha256 must be a lowercase 64-character SHA-256")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        _fail("provenance source commit does not match training config")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        _fail("provenance checkpoint hash does not match training config")
    return {"source_commit": source_commit, "checkpoint_sha256": checkpoint_sha256}


def _validate_windows(raw: Any) -> None:
    windows = _mapping(raw, "windows")
    _exact_keys(windows, {"early", "late"}, "windows")
    expected = {
        "early": (EARLY_START, EARLY_END, False),
        "late": (LATE_START, LATE_END, True),
    }
    for name, (start, end, inclusive) in expected.items():
        window = _mapping(windows[name], f"windows.{name}")
        _exact_keys(
            window,
            {"start_step", "end_step", "end_inclusive"},
            f"windows.{name}",
        )
        if (
            _integer(window["start_step"], f"windows.{name}.start_step") != start
            or _integer(window["end_step"], f"windows.{name}.end_step") != end
            or window["end_inclusive"] is not inclusive
        ):
            _fail(
                f"windows.{name} must be the locked "
                f"[{start},{end}{']' if inclusive else ')'} window"
            )


def _window_value(metrics: list[tuple[int, float]], *, late: bool) -> float:
    if late:
        values = [
            value for step, value in metrics if LATE_START <= step <= LATE_END
        ]
    else:
        values = [value for step, value in metrics if EARLY_START <= step < EARLY_END]
    if not values:
        _fail("every arm/seed run must contain samples in both gate windows")
    return float(sum(values) / len(values))


def validate_extension_gate(
    document: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate and summarize a parsed extension-gate artifact.

    The returned dictionary contains the recomputed per-arm window medians and
    improvements.  It is safe for callers to use for diagnostics because the
    artifact's claimed ``passed`` value is checked against the recomputation.
    """

    if not isinstance(document, Mapping):
        _fail("top level must be an object")
    _exact_keys(
        document,
        {
            "schema_version",
            "task",
            "passed",
            "provenance",
            "windows",
            "minimum_improvement",
            "runs",
        },
        "top level",
    )
    if _integer(document["schema_version"], "schema_version") != 1:
        _fail("schema_version must be exactly 1")
    if document["task"] != "extension-gate":
        _fail("task must be extension-gate")
    if document["passed"] is not True:
        raise ExtensionGateError("extension gate did not pass; 62 updates are forbidden")
    if (
        _number(document["minimum_improvement"], "minimum_improvement")
        != MINIMUM_IMPROVEMENT
    ):
        _fail("minimum_improvement must be exactly 0.05")
    provenance = _validate_provenance(
        document["provenance"],
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_source_commit=expected_source_commit,
    )
    _validate_windows(document["windows"])

    runs = document["runs"]
    if not isinstance(runs, list):
        _fail("runs must be an array")
    expected_pairs = {(arm, seed) for arm in ARMS for seed in MODEL_SEEDS}
    observed_pairs: set[tuple[str, int]] = set()
    values: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ARMS}
    for index, raw_run in enumerate(runs):
        run = _mapping(raw_run, f"runs[{index}]")
        _exact_keys(run, {"arm", "seed", "metrics"}, f"runs[{index}]")
        arm = run["arm"]
        seed = _integer(run["seed"], f"runs[{index}].seed")
        if arm not in ARMS:
            _fail(f"runs[{index}].arm must be one of {list(ARMS)}")
        if seed not in MODEL_SEEDS:
            _fail(f"runs[{index}].seed must be one of {list(MODEL_SEEDS)}")
        pair = (arm, seed)
        if pair in observed_pairs:
            _fail(f"duplicate run for arm={arm}, seed={seed}")
        observed_pairs.add(pair)
        metrics = run["metrics"]
        if not isinstance(metrics, list) or not metrics:
            _fail(f"runs[{index}].metrics must be a non-empty array")
        parsed: list[tuple[int, float]] = []
        seen_steps: set[int] = set()
        for metric_index, raw_metric in enumerate(metrics):
            metric = _mapping(raw_metric, f"runs[{index}].metrics[{metric_index}]")
            _exact_keys(
                metric,
                {"step", "training_success"},
                f"runs[{index}].metrics[{metric_index}]",
            )
            step = _integer(
                metric["step"], f"runs[{index}].metrics[{metric_index}].step"
            )
            value = _number(
                metric["training_success"],
                f"runs[{index}].metrics[{metric_index}].training_success",
            )
            if step < 0:
                _fail("metric steps must be nonnegative")
            if not 0.0 <= value <= 1.0:
                _fail("training_success must be between 0 and 1")
            if step in seen_steps:
                _fail(f"duplicate metric step {step} for arm={arm}, seed={seed}")
            seen_steps.add(step)
            parsed.append((step, value))
        early = _window_value(parsed, late=False)
        late = _window_value(parsed, late=True)
        values[arm][seed] = {
            "early": early,
            "late": late,
            "improvement": late - early,
        }

    missing = expected_pairs - observed_pairs
    extra = observed_pairs - expected_pairs
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing runs {sorted(missing)}")
        if extra:
            details.append(f"unexpected runs {sorted(extra)}")
        _fail("incomplete arm/seed coverage (" + ", ".join(details) + ")")

    summary: dict[str, dict[str, Any]] = {}
    all_passed = True
    for arm in ARMS:
        early_median = float(median(values[arm][seed]["early"] for seed in MODEL_SEEDS))
        late_median = float(median(values[arm][seed]["late"] for seed in MODEL_SEEDS))
        improvement = late_median - early_median
        arm_passed = improvement >= MINIMUM_IMPROVEMENT
        all_passed = all_passed and arm_passed
        summary[arm] = {
            "per_seed": values[arm],
            "early_median": early_median,
            "late_median": late_median,
            "improvement": improvement,
            "passed": arm_passed,
        }
    if not all_passed:
        raise ExtensionGateError(
            "extension gate metrics do not show the required 0.05 improvement for both arms"
        )
    return {
        "schema_version": 1,
        "provenance": provenance,
        "minimum_improvement": MINIMUM_IMPROVEMENT,
        "arms": summary,
        "passed": True,
    }


def require_extension_gate(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    expected_checkpoint_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Load, validate, and require a 62-update extension artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = Path(repo_root) / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise ExtensionGateError(f"extension gate artifact is missing: {artifact_path}")
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExtensionGateError(
            f"extension gate artifact is unreadable or invalid: {artifact_path}"
        ) from error
    summary = validate_extension_gate(
        document,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_source_commit=expected_source_commit,
    )
    summary["artifact_path"] = str(artifact_path)
    return summary
