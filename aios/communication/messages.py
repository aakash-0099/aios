"""Message types and correlation-aware wrappers for AIOS envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aios.core.exceptions import ValidationError
from aios.core.ids import RequestID
from aios.core.models import AgentRequest, AgentResponse


class MessageType(str, Enum):
    """Wire-level names for the supported kernel system calls."""

    LLM_CALL = "llm_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    STORAGE_READ = "storage_read"
    STORAGE_WRITE = "storage_write"
    TOOL_CALL = "tool_call"


def new_correlation_id() -> RequestID:
    """Create the request ID shared by a request and its response."""
    return RequestID.generate()


@dataclass(frozen=True)
class RequestMessage:
    """IPC payload wrapper around the canonical ``AgentRequest`` envelope."""

    request: AgentRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request, AgentRequest):
            raise ValidationError("request must be an AgentRequest.")

    @property
    def correlation_id(self) -> RequestID:
        """Return the envelope request ID used to match its response."""
        return self.request.request_id


@dataclass(frozen=True)
class ResponseMessage:
    """IPC payload wrapper around the canonical ``AgentResponse`` envelope."""

    response: AgentResponse

    def __post_init__(self) -> None:
        if not isinstance(self.response, AgentResponse):
            raise ValidationError("response must be an AgentResponse.")

    @property
    def correlation_id(self) -> RequestID:
        """Return the originating request ID used to match this response."""
        return self.response.request_id
