"""
Task status transition rules.

This module is the single source of truth for legal task status
transitions. It defines one immutable mapping, ``TRANSITIONS``, keyed
by every ``TaskStatus`` member, plus ``validate_transition`` as the
pure predicate that enforces it.

Transition table (terminal states map to an empty frozenset):

    created    -> queued, cancelled
    queued     -> running, cancelled
    running    -> completed, failed, cancelled
    completed  -> (terminal)
    failed     -> (terminal)
    cancelled  -> (terminal)

Decisions on ambiguous points:

- ``failed`` is terminal. No existing AIOS code models a retry
  transition, so every ``failed -> ...`` transition is illegal. If
  retries are introduced later, ``failed -> queued`` (or ``created``)
  can be added to the table.
- Same-state transitions (``X -> X``) are illegal. No existing code
  requires them, and rejecting them fails loudly on accidental
  no-op status updates.
- Cancellation is legal from every non-terminal state only.
"""

from __future__ import annotations

from aios.core.exceptions import InvalidStateTransitionError
from aios.core.models.task import TaskStatus

TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {TaskStatus.QUEUED, TaskStatus.CANCELLED}
    ),
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.CANCELLED}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

# Module-level coverage check: every TaskStatus member must appear as
# a key. Adding a new status without updating TRANSITIONS fails loudly
# here instead of silently making the new status unreachable.
_missing_statuses = set(TaskStatus) - set(TRANSITIONS)
if _missing_statuses:
    raise RuntimeError(
        "TRANSITIONS is missing TaskStatus members: "
        f"{', '.join(sorted(s.value for s in _missing_statuses))}."
    )
del _missing_statuses


def validate_transition(
    current_status: TaskStatus,
    new_status: TaskStatus,
) -> None:
    """
    Validate that a task may move from ``current_status`` to ``new_status``.

    Returns ``None`` when the transition is legal. Raises
    ``InvalidStateTransitionError`` (with both statuses in the message)
    when it is not.
    """

    if new_status not in TRANSITIONS[current_status]:
        raise InvalidStateTransitionError(
            "Invalid task status transition: "
            f"{current_status.value} -> {new_status.value}."
        )

    return None


def is_terminal(status: TaskStatus) -> bool:
    """
    Return True when ``status`` has no outgoing transitions.
    """

    return not TRANSITIONS[status]


def allowed_transitions(status: TaskStatus) -> frozenset[TaskStatus]:
    """
    Return the set of statuses reachable from ``status``.
    """

    return TRANSITIONS[status]
