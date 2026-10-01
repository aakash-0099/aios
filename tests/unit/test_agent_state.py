"""
Unit tests for agent lifecycle states and state machine transitions.
"""

import pytest

from aios.agents.agent_state import (
    AgentLifecycleState,
    AgentState,
    InvalidStateTransitionError,
    LEGAL_TRANSITIONS,
    is_valid_transition,
    normalize_state,
    validate_transition,
)
from aios.core.exceptions import AgentError, ValidationError


def test_agent_lifecycle_states_exist():
    """Verify all required lifecycle states are defined."""
    assert AgentLifecycleState.CREATED == "created"
    assert AgentLifecycleState.RUNNING == "running"
    assert AgentLifecycleState.PAUSED == "paused"
    assert AgentLifecycleState.TERMINATED == "terminated"
    assert AgentState is AgentLifecycleState


def test_legal_transitions_explicit_table():
    """Verify explicit transitions table entries."""
    assert AgentLifecycleState.RUNNING in LEGAL_TRANSITIONS[AgentLifecycleState.CREATED]
    assert AgentLifecycleState.TERMINATED in LEGAL_TRANSITIONS[AgentLifecycleState.CREATED]

    assert AgentLifecycleState.PAUSED in LEGAL_TRANSITIONS[AgentLifecycleState.RUNNING]
    assert AgentLifecycleState.TERMINATED in LEGAL_TRANSITIONS[AgentLifecycleState.RUNNING]

    assert AgentLifecycleState.RUNNING in LEGAL_TRANSITIONS[AgentLifecycleState.PAUSED]
    assert AgentLifecycleState.TERMINATED in LEGAL_TRANSITIONS[AgentLifecycleState.PAUSED]

    assert len(LEGAL_TRANSITIONS[AgentLifecycleState.TERMINATED]) == 0


@pytest.mark.parametrize(
    "current,target",
    [
        (AgentLifecycleState.CREATED, AgentLifecycleState.RUNNING),
        (AgentLifecycleState.CREATED, AgentLifecycleState.TERMINATED),
        (AgentLifecycleState.RUNNING, AgentLifecycleState.PAUSED),
        (AgentLifecycleState.RUNNING, AgentLifecycleState.TERMINATED),
        (AgentLifecycleState.PAUSED, AgentLifecycleState.RUNNING),
        (AgentLifecycleState.PAUSED, AgentLifecycleState.TERMINATED),
    ],
)
def test_legal_transitions_succeed(current, target):
    """Permitted state transitions should not raise."""
    validate_transition(current, target)
    assert is_valid_transition(current, target) is True


@pytest.mark.parametrize(
    "current,target",
    [
        (AgentLifecycleState.TERMINATED, AgentLifecycleState.RUNNING),
        (AgentLifecycleState.TERMINATED, AgentLifecycleState.PAUSED),
        (AgentLifecycleState.TERMINATED, AgentLifecycleState.CREATED),
        (AgentLifecycleState.TERMINATED, AgentLifecycleState.TERMINATED),
        (AgentLifecycleState.CREATED, AgentLifecycleState.PAUSED),
        (AgentLifecycleState.CREATED, AgentLifecycleState.CREATED),
        (AgentLifecycleState.RUNNING, AgentLifecycleState.CREATED),
        (AgentLifecycleState.RUNNING, AgentLifecycleState.RUNNING),
        (AgentLifecycleState.PAUSED, AgentLifecycleState.CREATED),
        (AgentLifecycleState.PAUSED, AgentLifecycleState.PAUSED),
    ],
)
def test_illegal_transitions_raise(current, target):
    """Illegal state transitions must raise InvalidStateTransitionError."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(current, target)
    assert is_valid_transition(current, target) is False


def test_invalid_state_transition_error_inherits_core_exceptions():
    """InvalidStateTransitionError must inherit from AgentError and ValidationError."""
    assert issubclass(InvalidStateTransitionError, AgentError)
    assert issubclass(InvalidStateTransitionError, ValidationError)


def test_normalize_state_accepts_strings_case_insensitively():
    """String state values should normalize correctly."""
    assert normalize_state("created") == AgentLifecycleState.CREATED
    assert normalize_state("CREATED") == AgentLifecycleState.CREATED
    assert normalize_state("running") == AgentLifecycleState.RUNNING
    assert normalize_state("Running ") == AgentLifecycleState.RUNNING
    assert normalize_state("paused") == AgentLifecycleState.PAUSED
    assert normalize_state("PAUSED") == AgentLifecycleState.PAUSED
    assert normalize_state("terminated") == AgentLifecycleState.TERMINATED
    assert normalize_state("TERMINATED") == AgentLifecycleState.TERMINATED


def test_normalize_state_rejects_invalid_strings():
    """Unknown state names must raise InvalidStateTransitionError."""
    with pytest.raises(InvalidStateTransitionError):
        normalize_state("unknown_state")

    with pytest.raises(InvalidStateTransitionError):
        normalize_state(123)  # type: ignore[arg-type]
