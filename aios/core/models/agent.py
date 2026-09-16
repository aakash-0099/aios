"""
Agent model.

An Agent represents an AI agent managed by AIOS.

Phase 1 responsibility:
    Define the canonical representation and validate it.

Actual agent execution belongs to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID
from aios.core.validation import require_non_empty_string


class AgentStatus(str, Enum):
    """
    Lifecycle state of an AIOS agent.
    """

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Agent:
    """
    Canonical representation of an AIOS-managed agent.
    """

    agent_id: AgentID
    name: str
    status: AgentStatus = AgentStatus.CREATED

    metadata: dict[str, object] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """
        Validate the Agent contract immediately after construction.
        """

        if not isinstance(self.agent_id, AgentID):
            raise ValidationError(
                "agent_id must be an AgentID."
            )

        require_non_empty_string(
            self.name,
            "name",
        )

        if not isinstance(self.status, AgentStatus):
            raise ValidationError(
                "status must be an AgentStatus."
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        if not isinstance(self.created_at, datetime):
            raise ValidationError(
                "created_at must be a datetime."
            )