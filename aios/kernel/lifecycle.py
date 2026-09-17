"""
System-call lifecycle tracking.

Phase 2 gives every dispatched system call an independent
lifecycle: it is created, it starts, and it ends in either
completion or failure. This module defines that lifecycle so
`SysCall` does not have to manage timing and status inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from aios.core.exceptions import ValidationError


class SysCallStatus(str, Enum):
    """
    Lifecycle state of a dispatched system call.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SysCallLifecycle:
    """
    Timing and status information for one `SysCall`.

    A `SysCall` owns exactly one `SysCallLifecycle`. The kernel
    and runtime never mutate timing directly; they only read it.
    """

    status: SysCallStatus = SysCallStatus.PENDING

    created_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    start_time: datetime | None = None

    end_time: datetime | None = None

    def mark_started(self) -> None:
        """
        Record that execution has begun.
        """

        if self.status != SysCallStatus.PENDING:
            raise ValidationError(
                "Only a pending syscall can be started."
            )

        self.start_time = datetime.now(timezone.utc)
        self.status = SysCallStatus.RUNNING

    def mark_completed(self) -> None:
        """
        Record successful completion.
        """

        if self.status != SysCallStatus.RUNNING:
            raise ValidationError(
                "Only a running syscall can complete."
            )

        self.end_time = datetime.now(timezone.utc)
        self.status = SysCallStatus.COMPLETED

    def mark_failed(self) -> None:
        """
        Record that execution ended in failure.
        """

        if self.status != SysCallStatus.RUNNING:
            raise ValidationError(
                "Only a running syscall can fail."
            )

        self.end_time = datetime.now(timezone.utc)
        self.status = SysCallStatus.FAILED

    @property
    def duration_seconds(self) -> float | None:
        """
        Wall-clock execution time, once the syscall has ended.
        """

        if self.start_time is None or self.end_time is None:
            return None

        return (self.end_time - self.start_time).total_seconds()

    @property
    def is_finished(self) -> bool:
        """
        Whether the syscall has reached a terminal state.
        """

        return self.status in (
            SysCallStatus.COMPLETED,
            SysCallStatus.FAILED,
        )
