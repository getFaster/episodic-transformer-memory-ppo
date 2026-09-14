"""Small experiment logging seam with a no-op implementation for tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class RunLogger(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def log(self, values: Mapping[str, Any], *, step: int) -> None: ...

    def log_run_metadata(self, values: Mapping[str, Any]) -> None: ...

    def log_records(
        self, name: str, records: list[Mapping[str, Any]], *, step: int
    ) -> None: ...

    def log_artifact(
        self, path: str, *, name: str, metadata: Mapping[str, Any]
    ) -> None: ...

    def finish(self) -> None: ...


class NoOpLogger:
    @property
    def identity(self) -> Mapping[str, Any]:
        return {"backend": "disabled"}

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        del values, step

    def log_run_metadata(self, values: Mapping[str, Any]) -> None:
        del values

    def log_records(
        self, name: str, records: list[Mapping[str, Any]], *, step: int
    ) -> None:
        del name, records, step

    def log_artifact(
        self, path: str, *, name: str, metadata: Mapping[str, Any]
    ) -> None:
        del path, name, metadata

    def finish(self) -> None:
        return None


class WandbLogger:
    def __init__(
        self,
        *,
        entity: str,
        project: str,
        name: str,
        run_id: str | None,
        mode: str,
        config: Mapping[str, Any],
    ) -> None:
        import wandb

        self._run = wandb.init(
            entity=entity,
            project=project,
            name=name,
            id=run_id,
            resume="allow" if run_id else None,
            mode=mode,
            config=dict(config),
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"backend": "wandb", "run_id": self._run.id, "name": self._run.name}

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        self._run.log(dict(values), step=step)

    def log_run_metadata(self, values: Mapping[str, Any]) -> None:
        self._run.config.update(dict(values), allow_val_change=True)

    def log_records(
        self, name: str, records: list[Mapping[str, Any]], *, step: int
    ) -> None:
        if not records:
            return
        import wandb

        columns = sorted({key for record in records for key in record})
        table = wandb.Table(
            columns=columns,
            data=[[record.get(column) for column in columns] for record in records],
        )
        self._run.log({name: table}, step=step)

    def log_artifact(
        self, path: str, *, name: str, metadata: Mapping[str, Any]
    ) -> None:
        import wandb

        artifact = wandb.Artifact(name=name, type="routing", metadata=dict(metadata))
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()
