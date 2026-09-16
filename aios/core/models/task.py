"""
Task model.

A Task represents a unit of work assigned to an AIOS agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID, TaskID
from aios.core.validation import require_non_empty_string


class TaskStatus(str, Enum):
    """
    Lifecycle state of an AIOS task.
    """

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """
    Canonical representation of work assigned to an agent.
    """

    task_id: TaskID
    agent_id: AgentID
    description: str

    status: TaskStatus = TaskStatus.CREATED

    metadata: dict[str, object] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """
        Validate the Task contract.
        """

        if not isinstance(self.task_id, TaskID):
            raise ValidationError(
                "task_id must be a TaskID."
            )

        if not isinstance(self.agent_id, AgentID):
            raise ValidationError(
                "agent_id must be an AgentID."
            )

        require_non_empty_string(
            self.description,
            "description",
        )

        if not isinstance(self.status, TaskStatus):
            raise ValidationError(
                "status must be a TaskStatus."
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        if not isinstance(self.created_at, datetime):
            raise ValidationError(
                "created_at must be a datetime."
            )

        if not isinstance(self.updated_at, datetime):
            raise ValidationError(
                "updated_at must be a datetime."
            )