"""
Wire format for `AgentRequest`.

Nothing in Phase 2 previously needed to turn a request into bytes:
a request lived entirely on the stack, in the calling thread's
memory. A queue backend that leaves the process -- Redis, in
particular -- can only carry bytes/strings, so this module is the
one place that knows how to flatten an `AgentRequest` into a plain,
JSON-safe `dict` and rebuild it on the other side.

This is deliberately narrow: it only round-trips the fields the
dispatcher actually needs to route and execute a syscall (identity,
syscall type, payload, metadata). `Context` is not included --
Phase 2's mock handlers never read it, and giving it a wire format
is Phase 3/4's job once a real Context Manager exists to produce
and consume one.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID, RequestID, TaskID
from aios.core.models import AgentRequest, SystemCall, SystemCallType


def request_to_dict(request: AgentRequest) -> dict[str, Any]:
    """
    Flatten an `AgentRequest` into a JSON-safe dictionary.
    """

    return {
        "request_id": str(request.request_id),
        "agent_id": str(request.agent_id),
        "task_id": str(request.task_id),
        "call_type": request.syscall.call_type.value,
        "payload": request.syscall.payload,
        "metadata": request.metadata,
    }


def request_from_dict(data: dict[str, Any]) -> AgentRequest:
    """
    Rebuild an `AgentRequest` from `request_to_dict`'s output.

    Raises `ValidationError` if `data` is missing a required key or
    holds a syscall type outside the frozen Phase 1 vocabulary --
    this is the boundary check for anything arriving off a queue
    that another process (or a future version of this one) wrote.
    """

    try:
        return AgentRequest(
            request_id=RequestID(UUID(data["request_id"])),
            agent_id=AgentID(UUID(data["agent_id"])),
            task_id=TaskID(UUID(data["task_id"])),
            syscall=SystemCall(
                call_type=SystemCallType(data["call_type"]),
                payload=data.get("payload", {}),
            ),
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise ValidationError(
            f"Serialized request is missing field: {exc}."
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"Serialized request is malformed: {exc}."
        ) from exc
