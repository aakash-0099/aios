"""
AIOS tool model.

This defines the shared identity of a tool.

Actual tool execution belongs to a later AIOS subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import ToolID
from aios.core.validation import require_non_empty_string


@dataclass
class Tool:
    """
    Representation of an AIOS-managed tool.
    """

    tool_id: ToolID

    name: str

    description: str

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the Tool contract."""

        if not isinstance(self.tool_id, ToolID):
            raise ValidationError(
                "tool_id must be a ToolID."
            )

        require_non_empty_string(
            self.name,
            "name",
        )

        require_non_empty_string(
            self.description,
            "description",
        )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )