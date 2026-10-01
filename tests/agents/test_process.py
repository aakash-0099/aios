"""
Tests for ProcessMetadata.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import pytest

from aios.agents.process import ProcessMetadata
from aios.core.exceptions import ValidationError
from aios.core.ids import TaskID

_UNSET: object = object()


def make_metadata(
    task_id: TaskID | None = None,
    worker_id: str = "worker-1",
    started_at: Any = _UNSET,
    retry_count: int = 0,
) -> ProcessMetadata:
    """Build ProcessMetadata with sensible defaults."""

    return ProcessMetadata(
        task_id=task_id if task_id is not None else TaskID.generate(),
        worker_id=worker_id,
        started_at=(
            datetime.now(timezone.utc)
            if started_at is _UNSET
            else started_at
        ),
        retry_count=retry_count,
    )


def test_construction_with_valid_values() -> None:
    task_id = TaskID.generate()
    started_at = datetime.now(timezone.utc)

    metadata = ProcessMetadata(
        task_id=task_id,
        worker_id="worker-1",
        started_at=started_at,
        retry_count=2,
    )

    assert metadata.task_id == task_id
    assert metadata.worker_id == "worker-1"
    assert metadata.started_at == started_at
    assert metadata.retry_count == 2


def test_default_retry_count_is_zero() -> None:
    metadata = make_metadata()

    assert metadata.retry_count == 0


def test_negative_retry_count_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_metadata(retry_count=-1)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_metadata(started_at=datetime.now())


def test_none_started_at_is_allowed() -> None:
    metadata = make_metadata(started_at=None)

    assert metadata.started_at is None


def test_task_id_must_be_task_id() -> None:
    with pytest.raises(ValidationError):
        make_metadata(task_id="not-a-task-id")  # type: ignore[arg-type]


def test_worker_id_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        make_metadata(worker_id="")


def test_metadata_is_immutable() -> None:
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.retry_count = 5  # type: ignore[misc]


def test_with_retry_increments_without_mutating() -> None:
    metadata = make_metadata(retry_count=1)

    retried = metadata.with_retry()

    assert retried.retry_count == 2
    assert metadata.retry_count == 1
    assert retried.task_id == metadata.task_id
    assert retried.worker_id == metadata.worker_id
    assert retried.started_at == metadata.started_at


def test_to_dict_from_dict_round_trip() -> None:
    metadata = make_metadata(retry_count=3)

    restored = ProcessMetadata.from_dict(metadata.to_dict())

    assert restored == metadata


def test_round_trip_with_unstarted_process() -> None:
    metadata = make_metadata(started_at=None)

    restored = ProcessMetadata.from_dict(metadata.to_dict())

    assert restored == metadata
    assert restored.started_at is None


def test_from_dict_rejects_missing_field() -> None:
    data = make_metadata().to_dict()
    del data["worker_id"]

    with pytest.raises(ValidationError):
        ProcessMetadata.from_dict(data)


def test_from_dict_rejects_malformed_timestamp() -> None:
    with pytest.raises(ValidationError):
        ProcessMetadata.from_dict(
            {
                "task_id": str(TaskID.generate()),
                "worker_id": "worker-1",
                "started_at": "not-a-timestamp",
                "retry_count": 0,
            }
        )


def test_from_dict_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        ProcessMetadata.from_dict(
            {
                "task_id": str(TaskID.generate()),
                "worker_id": "worker-1",
                "started_at": datetime.now().isoformat(),
                "retry_count": 0,
            }
        )


def test_from_dict_rejects_negative_retry_count() -> None:
    with pytest.raises(ValidationError):
        ProcessMetadata.from_dict(
            {
                "task_id": str(TaskID.generate()),
                "worker_id": "worker-1",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": -1,
            }
        )
