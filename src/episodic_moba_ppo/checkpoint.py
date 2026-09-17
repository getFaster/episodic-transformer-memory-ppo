"""Checkpoint integrity, legacy loading, and marker-last persistence."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT_VERSION = 1
PAYLOAD_NAME = "training_state.pt"
MARKER_NAME = "commit_success.json"


class CheckpointError(RuntimeError):
    pass


class CheckpointIntegrityError(CheckpointError):
    pass


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: str | os.PathLike[str], expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise CheckpointIntegrityError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_legacy_checkpoint(
    path: str | os.PathLike[str], expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Verify then load the trusted, pinned upstream pickle checkpoint."""

    verify_file_hash(path, expected_sha256)
    with open(path, "rb") as stream:
        loaded = pickle.load(stream)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise CheckpointIntegrityError(
            "legacy checkpoint is not a (state_dict, config) tuple"
        )
    state_dict, config = loaded
    if not isinstance(state_dict, Mapping) or not isinstance(config, dict):
        raise CheckpointIntegrityError("legacy checkpoint has invalid component types")
    return state_dict, config


def _default_dump(payload: Any, path: Path) -> None:
    import torch

    torch.save(payload, path)


def _default_load(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class RecoveredCheckpoint:
    update: int
    directory: Path
    marker: dict[str, Any]
    payload: Any


class CheckpointStore:
    """Immutable update directories committed by writing a hash marker last."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        dump: Callable[[Any, Path], None] = _default_dump,
        load: Callable[[Path], Any] = _default_load,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.dump = dump
        self.load = load
        self.fault_hook = fault_hook or (lambda _: None)

    def update_directory(self, update: int) -> Path:
        if update < 0:
            raise ValueError("update must be nonnegative")
        return self.root / f"update-{update:05d}"

    def commit(self, update: int, payload: Any, metadata: Mapping[str, Any]) -> Path:
        target = self.update_directory(update)
        self.root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (target / MARKER_NAME).exists():
                raise FileExistsError(f"immutable checkpoint already exists: {target}")
            # A markerless directory was never committed. Reuse it so a Drive
            # copy interrupted before its final marker can be retried safely.
            for stale in target.glob(".*.tmp"):
                stale.unlink()
        else:
            target.mkdir()
        temporary_payload = target / f".{PAYLOAD_NAME}.{uuid.uuid4().hex}.tmp"
        self.dump(payload, temporary_payload)
        self._fsync_file(temporary_payload)
        os.replace(temporary_payload, target / PAYLOAD_NAME)
        self.fault_hook("payload_committed")

        payload_path = target / PAYLOAD_NAME
        marker = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "update": int(update),
            "metadata": dict(metadata),
            "payloads": [
                {
                    "name": PAYLOAD_NAME,
                    "size": payload_path.stat().st_size,
                    "sha256": sha256_file(payload_path),
                }
            ],
        }
        marker_tmp = target / f".{MARKER_NAME}.{uuid.uuid4().hex}.tmp"
        with open(marker_tmp, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.fault_hook("before_marker")
        os.replace(marker_tmp, target / MARKER_NAME)
        self._fsync_directory(target)
        self.fault_hook("marker_committed")
        return target

    def validate(self, directory: str | os.PathLike[str]) -> dict[str, Any]:
        path = Path(directory)
        marker_path = path / MARKER_NAME
        if not marker_path.is_file():
            raise CheckpointIntegrityError(f"missing {MARKER_NAME}: {path}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointIntegrityError(
                f"invalid checkpoint marker: {path}"
            ) from error
        if marker.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointIntegrityError("unsupported checkpoint format version")
        try:
            expected_name = f"update-{int(marker['update']):05d}"
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointIntegrityError("invalid checkpoint update") from error
        if path.name != expected_name:
            raise CheckpointIntegrityError(
                "checkpoint update disagrees with directory name"
            )
        payloads = marker.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            raise CheckpointIntegrityError("checkpoint marker has no payloads")
        for descriptor in payloads:
            name = descriptor.get("name")
            if not isinstance(name, str) or Path(name).name != name:
                raise CheckpointIntegrityError("unsafe payload filename")
            payload_path = path / name
            if not payload_path.is_file():
                raise CheckpointIntegrityError(f"missing checkpoint payload: {name}")
            if payload_path.stat().st_size != descriptor.get("size"):
                raise CheckpointIntegrityError(f"payload size mismatch: {name}")
            verify_file_hash(payload_path, str(descriptor.get("sha256", "")))
        return marker

    def recover_latest(self) -> RecoveredCheckpoint:
        if not self.root.exists():
            raise CheckpointError(f"checkpoint store does not exist: {self.root}")
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        for path in self.root.glob("update-[0-9][0-9][0-9][0-9][0-9]"):
            if not path.is_dir():
                continue
            try:
                marker = self.validate(path)
            except CheckpointIntegrityError:
                continue
            candidates.append((int(marker["update"]), path, marker))
        for update, directory, marker in sorted(candidates, reverse=True):
            try:
                payload = self.load(directory / PAYLOAD_NAME)
            except Exception:
                continue
            return RecoveredCheckpoint(update, directory, marker, payload)
        raise CheckpointError("no valid loadable committed checkpoint found")

    def promote_from(self, source_directory: str | os.PathLike[str]) -> Path:
        source = Path(source_directory)
        source_marker = self.validate(source)
        update = int(source_marker["update"])
        payload = self.load(source / PAYLOAD_NAME)
        metadata = dict(source_marker.get("metadata", {}))
        metadata["promoted_from"] = str(source)
        return self.commit(update, payload, metadata)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
