"""
AIOS resource model.

Resource provides a common representation for things managed by
the future AIOS resource-management subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import ResourceID
from aios.core.validation import require_non_empty_string


class ResourceType(str, Enum):
    """
    Initial resource categories.
    """

    MEMORY = "memory"
    STORAGE = "storage"
    TOOL = "tool"
    LLM = "llm"


@dataclass
class Resource:
    """
    Shared representation of an AIOS-managed resource.
    """

    resource_id: ResourceID
    resource_type: ResourceType
    name: str

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the Resource contract."""

        if not isinstance(self.resource_id, ResourceID):
            raise ValidationError(
                "resource_id must be a ResourceID."
            )

        if not isinstance(self.resource_type, ResourceType):
            raise ValidationError(
                "resource_type must be a ResourceType."
            )

        require_non_empty_string(
            self.name,
            "name",
        )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )