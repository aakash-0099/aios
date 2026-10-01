"""
Tests for task status transition rules.
"""

import pytest

from aios.agents.task import (
    TRANSITIONS,
    allowed_transitions,
    is_terminal,
    validate_transition,
)
from aios.core.exceptions import InvalidStateTransitionError
from aios.core.models.task import TaskStatus

# Generated from the table itself so the tests cannot drift from it.
LEGAL_TRANSITIONS = [
    (current, new)
    for current in TaskStatus
    for new in TRANSITIONS[current]
]

ILLEGAL_TRANSITIONS = [
    (current, new)
    for current in TaskStatus
    for new in TaskStatus
    if new not in TRANSITIONS[current]
]

TERMINAL_STATUSES = [status for status in TaskStatus if is_terminal(status)]

NON_TERMINAL_STATUSES = [
    status for status in TaskStatus if not is_terminal(status)
]


@pytest.mark.parametrize(("current", "new"), LEGAL_TRANSITIONS)
def test_legal_transition_is_allowed(
    current: TaskStatus,
    new: TaskStatus,
) -> None:
    validate_transition(current, new)


@pytest.mark.parametrize(("current", "new"), ILLEGAL_TRANSITIONS)
def test_illegal_transition_raises(
    current: TaskStatus,
    new: TaskStatus,
) -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(current, new)


@pytest.mark.parametrize("status", NON_TERMINAL_STATUSES)
def test_cancellation_is_legal_from_non_terminal(status: TaskStatus) -> None:
    validate_transition(status, TaskStatus.CANCELLED)


@pytest.mark.parametrize("status", TERMINAL_STATUSES)
def test_cancellation_is_illegal_from_terminal(status: TaskStatus) -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(status, TaskStatus.CANCELLED)


def test_running_to_failed_is_legal() -> None:
    validate_transition(TaskStatus.RUNNING, TaskStatus.FAILED)


def test_failed_is_terminal() -> None:
    assert is_terminal(TaskStatus.FAILED)
    assert allowed_transitions(TaskStatus.FAILED) == frozenset()


@pytest.mark.parametrize(
    "new_status",
    [status for status in TaskStatus if status is not TaskStatus.FAILED],
)
def test_failed_cannot_transition_to_anything(new_status: TaskStatus) -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(TaskStatus.FAILED, new_status)


@pytest.mark.parametrize("status", TERMINAL_STATUSES)
def test_terminal_states_have_no_outgoing_transitions(
    status: TaskStatus,
) -> None:
    assert allowed_transitions(status) == frozenset()

    for new_status in TaskStatus:
        with pytest.raises(InvalidStateTransitionError):
            validate_transition(status, new_status)


def test_transition_table_covers_every_status() -> None:
    assert set(TRANSITIONS) == set(TaskStatus)


def test_transition_table_values_are_frozensets() -> None:
    for targets in TRANSITIONS.values():
        assert isinstance(targets, frozenset)


@pytest.mark.parametrize("status", list(TaskStatus))
def test_same_state_transition_is_illegal(status: TaskStatus) -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(status, status)


def test_error_message_contains_both_statuses() -> None:
    with pytest.raises(InvalidStateTransitionError) as excinfo:
        validate_transition(TaskStatus.COMPLETED, TaskStatus.RUNNING)

    message = str(excinfo.value)

    assert TaskStatus.COMPLETED.value in message
    assert TaskStatus.RUNNING.value in message
