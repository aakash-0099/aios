"""
Execution process metadata.

``ProcessMetadata`` records the runtime facts about one execution of
a task: which task it belongs to, when it started, which worker runs
it, and how many times it has been retried. It is immutable: updates
such as ``with_retry`` return a new instance.

This module deliberately does not know about ``Task`` or status
transitions; it only carries data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from uuid import UUID

from aios.core.exceptions import ValidationError
from aios.core.ids import TaskID
from aios.core.validation import require_non_empty_string


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    """
    Require a timezone-aware datetime; naive datetimes are rejected.
    """

    if not isinstance(value, datetime):
        raise ValidationError(f"{field_name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field_name} must be timezone-aware.")


@dataclass(frozen=True)
class ProcessMetadata:
    """
    Immutable runtime metadata for one task execution.

    Fields:
    - task_id: the TaskID of the task being executed.
    - worker_id: identifier of the worker running the process.
    - started_at: timezone-aware UTC start time, or ``None`` when the
      process has not started yet.
    - retry_count: number of retries so far; must be >= 0.
    """

    task_id: TaskID
    worker_id: str
    started_at: datetime | None = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        """
        Validate the ProcessMetadata contract.
        """

        if not isinstance(self.task_id, TaskID):
            raise ValidationError("task_id must be a TaskID.")

        require_non_empty_string(self.worker_id, "worker_id")

        if self.started_at is not None:
            _require_aware_datetime(self.started_at, "started_at")

        if not isinstance(self.retry_count, int):
            raise ValidationError("retry_count must be an integer.")

        if self.retry_count < 0:
            raise ValidationError("retry_count must be >= 0.")

    def with_retry(self) -> ProcessMetadata:
        """
        Return a copy with ``retry_count`` incremented by one.
        """

        return replace(self, retry_count=self.retry_count + 1)

    def to_dict(self) -> dict[str, Any]:
        """
        Flatten into a JSON-safe dictionary (ISO-8601 timestamps).
        """

        return {
            "task_id": str(self.task_id),
            "worker_id": self.worker_id,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessMetadata:
        """
        Rebuild from ``to_dict``'s output.

        Raises ``ValidationError`` if ``data`` is missing a required
        key or holds a malformed value.
        """

        try:
            started_at_raw = data["started_at"]
            started_at = (
                datetime.fromisoformat(started_at_raw)
                if started_at_raw is not None
                else None
            )

            return cls(
                task_id=TaskID(UUID(data["task_id"])),
                worker_id=data["worker_id"],
                started_at=started_at,
                retry_count=data["retry_count"],
            )
        except KeyError as exc:
            raise ValidationError(
                f"Serialized process metadata is missing field: {exc}."
            ) from exc
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                f"Serialized process metadata is malformed: {exc}."
            ) from exc
