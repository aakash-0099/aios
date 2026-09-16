"""
AIOS memory model.

This defines the shared representation of memory.

Actual memory storage and retrieval are implemented later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID, MemoryID
from aios.core.validation import require_non_empty_string


@dataclass
class Memory:
    """
    Representation of an AIOS memory entry.
    """

    memory_id: MemoryID
    agent_id: AgentID
    content: str

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """Validate the Memory contract."""

        if not isinstance(self.memory_id, MemoryID):
            raise ValidationError(
                "memory_id must be a MemoryID."
            )

        if not isinstance(self.agent_id, AgentID):
            raise ValidationError(
                "agent_id must be an AgentID."
            )

        require_non_empty_string(
            self.content,
            "content",
        )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        if not isinstance(self.created_at, datetime):
            raise ValidationError(
                "created_at must be a datetime."
            )