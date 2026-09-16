"""
Agent request model.

AgentRequest is the canonical request boundary between an agent
and the AIOS system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID, RequestID, TaskID
from aios.core.models.context import Context
from aios.core.models.syscall import SystemCall


@dataclass(frozen=True)
class AgentRequest:
    """
    Request submitted to AIOS.

    Correlation identifiers allow future components to trace:

        Agent -> Task -> Request -> Response
    """

    request_id: RequestID
    agent_id: AgentID
    task_id: TaskID
    syscall: SystemCall

    context: Context | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate the request contract.
        """

        if not isinstance(self.request_id, RequestID):
            raise ValidationError(
                "request_id must be a RequestID."
            )

        if not isinstance(self.agent_id, AgentID):
            raise ValidationError(
                "agent_id must be an AgentID."
            )

        if not isinstance(self.task_id, TaskID):
            raise ValidationError(
                "task_id must be a TaskID."
            )

        if not isinstance(self.syscall, SystemCall):
            raise ValidationError(
                "syscall must be a SystemCall."
            )

        if self.context is not None and not isinstance(
            self.context,
            Context,
        ):
            raise ValidationError(
                "context must be a Context or None."
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError(
                "metadata must be a dictionary."
            )