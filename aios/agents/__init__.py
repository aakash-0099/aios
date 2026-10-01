"""
AIOS Agents package.

Contains lifecycle management, state machine validation, and agent tracking.
"""

from .agent import AgentRecord, ManagedAgent
from .agent_manager import (
    AgentManager,
    AgentNotFoundError,
    DuplicateAgentError,
    validate_agent_id,
)
from .agent_state import (
    LEGAL_TRANSITIONS,
    AgentLifecycleState,
    AgentState,
    InvalidStateTransitionError,
    is_valid_transition,
    normalize_state,
    validate_transition,
)

__all__ = [
    "AgentLifecycleState",
    "AgentManager",
    "AgentNotFoundError",
    "AgentRecord",
    "AgentState",
    "DuplicateAgentError",
    "InvalidStateTransitionError",
    "LEGAL_TRANSITIONS",
    "ManagedAgent",
    "is_valid_transition",
    "normalize_state",
    "validate_agent_id",
    "validate_transition",
]
