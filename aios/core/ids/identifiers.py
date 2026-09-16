"""
Shared AIOS identifiers.

Every major AIOS object gets its own identifier type.  Keeping
identifiers centralized prevents different modules from inventing
their own ID formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True)
class AgentID:
    """
    Unique identifier for an AIOS agent.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "AgentID":
        """Generate a new unique AgentID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class TaskID:
    """
    Unique identifier for an AIOS task.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "TaskID":
        """Generate a new unique TaskID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class RequestID:
    """
    Unique identifier for an AgentRequest.

    This identifier is also used for request/response correlation.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "RequestID":
        """Generate a new unique RequestID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ResourceID:
    """
    Unique identifier for an AIOS-managed resource.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "ResourceID":
        """Generate a new unique ResourceID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MemoryID:
    """
    Unique identifier for a memory object.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "MemoryID":
        """Generate a new unique MemoryID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ToolID:
    """
    Unique identifier for an AIOS-managed tool.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "ToolID":
        """Generate a new unique ToolID."""
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)