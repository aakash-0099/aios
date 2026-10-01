"""
Agent record model.

This module provides the lifecycle-tracking wrapper (AgentRecord / ManagedAgent)
for an AIOS agent.

CRITICAL ARCHITECTURAL RULE:
Do NOT redefine the Agent dataclass here. The canonical Agent model lives in
aios.core.models.Agent. This module holds the lifecycle-tracking record that
wraps aios.core.models.Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aios.agents.agent_state import (
    AgentLifecycleState,
    normalize_state,
    validate_transition,
)
from aios.core.exceptions import ValidationError
from aios.core.ids import AgentID
from aios.core.models import Agent


@dataclass
class AgentRecord:
    """
    Lifecycle-tracking record for an agent managed by AgentManager.

    Wraps an aios.core.models.Agent and tracks its lifecycle state
    and timestamps.
    """

    agent: Agent
    state: AgentLifecycleState = AgentLifecycleState.CREATED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """
        Validate the AgentRecord contract immediately after construction.
        """
        if not isinstance(self.agent, Agent):
            raise ValidationError(
                f"agent must be an instance of aios.core.models.Agent, "
                f"got {type(self.agent).__name__}."
            )

        self.state = normalize_state(self.state)

        if not isinstance(self.created_at, datetime):
            raise ValidationError("created_at must be a datetime.")

        if not isinstance(self.updated_at, datetime):
            raise ValidationError("updated_at must be a datetime.")

    @property
    def agent_id(self) -> AgentID:
        """
        Convenience accessor for the underlying Agent's AgentID.
        """
        return self.agent.agent_id

    @property
    def name(self) -> str:
        """
        Convenience accessor for the underlying Agent's name.
        """
        return self.agent.name

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Convenience accessor for the underlying Agent's metadata dictionary.
        """
        return self.agent.metadata

    @property
    def status(self) -> AgentLifecycleState:
        """
        Alias for state to provide compatibility with interface contracts.
        """
        return self.state

    @status.setter
    def status(self, new_state: AgentLifecycleState | str) -> None:
        self.state = normalize_state(new_state)

    def update_state(self, new_state: AgentLifecycleState | str) -> None:
        """
        Transition the agent to a new lifecycle state.

        Validates the transition via validate_transition() before updating
        state and the updated_at timestamp.
        """
        target = normalize_state(new_state)
        validate_transition(self.state, target)
        self.state = target
        self.updated_at = datetime.now(timezone.utc)


# ManagedAgent is provided as an alias for AgentRecord
ManagedAgent = AgentRecord
