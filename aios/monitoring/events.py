"""
Trace event schema for AIOS monitoring.

Every observable action in the system is described by one immutable
`TraceEvent`.  Event names are defined as constants in this single
module, grouped by the component that emits them, so new packages
add their own group here instead of inventing names inline.

Naming convention: ``<noun>.<past_tense_verb>``, namespaced by the
noun the event is about (``syscall.*``, ``response.*``).  The
``component`` field of the event carries the emitting package name
(e.g. ``"kernel"``), so the same noun namespace can be reused by
different components without collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class KernelEvents:
    """
    Event names emitted by `aios.kernel`.

    Add future packages' events as their own class in this module
    (e.g. ``class SchedulerEvents``), following the same
    ``<noun>.<verb>`` convention.
    """

    SYSCALL_CREATED = "syscall.created"
    SYSCALL_VALIDATED = "syscall.validated"
    SYSCALL_QUEUED = "syscall.queued"
    SYSCALL_STARTED = "syscall.started"
    SYSCALL_HANDLER_INVOKED = "syscall.handler_invoked"
    SYSCALL_COMPLETED = "syscall.completed"
    SYSCALL_FAILED = "syscall.failed"
    RESPONSE_RETURNED = "response.returned"


@dataclass(frozen=True)
class TraceEvent:
    """
    One immutable, JSON-serialisable observation.

    ``wall_time`` is epoch seconds (``time.time()``) for wall-clock
    correlation with logs; ``mono_time`` is ``time.monotonic()`` for
    gap-free duration maths.  ``correlation_id`` ties the events of
    one request together (the AIOS ``RequestID``); ``agent_id`` ties
    them to the requesting agent.  ``data`` holds event-specific
    key/value pairs and must contain JSON-serialisable values only
    (`aios.monitoring.tracer.emit` coerces anything else with
    ``repr()`` rather than failing).
    """

    wall_time: float
    mono_time: float
    correlation_id: str | None
    agent_id: str | None
    component: str
    event: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a plain-dict representation safe for ``json.dumps``.
        """

        return {
            "wall_time": self.wall_time,
            "mono_time": self.mono_time,
            "correlation_id": self.correlation_id,
            "agent_id": self.agent_id,
            "component": self.component,
            "event": self.event,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceEvent:
        """
        Rebuild a `TraceEvent` from `to_dict`'s output.
        """

        return cls(
            wall_time=float(payload["wall_time"]),
            mono_time=float(payload["mono_time"]),
            correlation_id=payload.get("correlation_id"),
            agent_id=payload.get("agent_id"),
            component=str(payload["component"]),
            event=str(payload["event"]),
            data=dict(payload.get("data", {})),
        )
