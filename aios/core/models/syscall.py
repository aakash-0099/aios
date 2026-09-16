"""
AIOS system-call definitions.

Phase 1 freezes the initial system-call vocabulary.

Execution of these calls belongs to the future AIOS kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aios.core.exceptions import ValidationError


class SystemCallType(str, Enum):
    """
    Initial AIOS system calls.

    These values form part of the shared AIOS contract.
    """

    LLM_CALL = "llm_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    STORAGE_READ = "storage_read"
    STORAGE_WRITE = "storage_write"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class SystemCall:
    """
    Description of a system call.

    This object describes an operation; it does not execute it.
    """

    call_type: SystemCallType

    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the system-call contract.
        """

        if not isinstance(self.call_type, SystemCallType):
            raise ValidationError(
                "call_type must be a SystemCallType."
            )

        if not isinstance(self.payload, dict):
            raise ValidationError(
                "payload must be a dictionary."
            )