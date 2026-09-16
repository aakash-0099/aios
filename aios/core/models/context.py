"""
AIOS execution context model.

Context is the common structure used to carry information into
an AIOS operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aios.core.exceptions import ValidationError


@dataclass(frozen=True)
class Context:
    """
    Shared execution context.

    Context construction is handled here.

    Context retrieval/assembly belongs to later phases.
    """

    system: str | None = None

    conversation: list[dict[str, Any]] = field(
        default_factory=list
    )

    memory: list[dict[str, Any]] = field(
        default_factory=list
    )

    working_state: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate context structure."""

        if self.system is not None and not isinstance(
            self.system,
            str,
        ):
            raise ValidationError(
                "system must be a string or None."
            )

        if not isinstance(self.conversation, list):
            raise ValidationError(
                "conversation must be a list."
            )

        if not all(
            isinstance(item, dict)
            for item in self.conversation
        ):
            raise ValidationError(
                "Every conversation item must be a dictionary."
            )

        if not isinstance(self.memory, list):
            raise ValidationError(
                "memory must be a list."
            )

        if not all(
            isinstance(item, dict)
            for item in self.memory
        ):
            raise ValidationError(
                "Every memory item must be a dictionary."
            )

        if not isinstance(self.working_state, dict):
            raise ValidationError(
                "working_state must be a dictionary."
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )