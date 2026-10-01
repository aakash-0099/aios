"""
Agent lifecycle state machine.

Defines the legal lifecycle states of an AIOS agent, the explicit transition
table between states, and transition validation functions.
"""

from __future__ import annotations

from enum import Enum

from aios.core.exceptions import AgentError, ValidationError


class AgentLifecycleState(str, Enum):
    """
    Lifecycle states of an agent managed by AgentManager.
    """

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"


# Alias for contract compatibility
AgentState = AgentLifecycleState


class InvalidStateTransitionError(AgentError, ValidationError):
    """
    Raised when an illegal agent lifecycle state transition is attempted.
    """


LEGAL_TRANSITIONS: dict[AgentLifecycleState, set[AgentLifecycleState]] = {
    AgentLifecycleState.CREATED: {
        AgentLifecycleState.RUNNING,
        AgentLifecycleState.TERMINATED,
    },
    AgentLifecycleState.RUNNING: {
        AgentLifecycleState.PAUSED,
        AgentLifecycleState.TERMINATED,
    },
    AgentLifecycleState.PAUSED: {
        AgentLifecycleState.RUNNING,
        AgentLifecycleState.TERMINATED,
    },
    AgentLifecycleState.TERMINATED: set(),
}


def normalize_state(state: AgentLifecycleState | str) -> AgentLifecycleState:
    """
    Normalize an input state into an AgentLifecycleState enum member.

    Raises:
        InvalidStateTransitionError: If the state is not a valid lifecycle state.
    """
    if isinstance(state, AgentLifecycleState):
        return state

    if isinstance(state, str):
        cleaned = state.strip().lower()
        for member in AgentLifecycleState:
            if member.value == cleaned or member.name.lower() == cleaned:
                return member
        raise InvalidStateTransitionError(
            f"Invalid agent lifecycle state: {state!r}. "
            f"Valid states are: {[s.value for s in AgentLifecycleState]}"
        )

    raise InvalidStateTransitionError(
        f"State must be an AgentLifecycleState or str, got {type(state).__name__}."
    )


def validate_transition(
    current_state: AgentLifecycleState | str,
    new_state: AgentLifecycleState | str,
) -> None:
    """
    Validate that transitioning from current_state to new_state is permitted.

    Args:
        current_state: Current lifecycle state of the agent.
        new_state: Desired target lifecycle state.

    Raises:
        InvalidStateTransitionError: If the transition is not permitted by the
            lifecycle state machine.
    """
    current = normalize_state(current_state)
    target = normalize_state(new_state)

    allowed_targets = LEGAL_TRANSITIONS.get(current, set())
    if target not in allowed_targets:
        raise InvalidStateTransitionError(
            f"Illegal state transition from {current.name} to {target.name}. "
            f"Allowed transitions from {current.name}: "
            f"{[s.name for s in allowed_targets] or 'None (terminal state)'}."
        )


def is_valid_transition(
    current_state: AgentLifecycleState | str,
    new_state: AgentLifecycleState | str,
) -> bool:
    """
    Check if a state transition is legal without raising an exception.
    """
    try:
        validate_transition(current_state, new_state)
        return True
    except InvalidStateTransitionError:
        return False
