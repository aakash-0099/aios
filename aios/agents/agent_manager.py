"""
Agent Manager.

Manages the lifecycle and registry of agents in AIOS.

Sits above the Kernel/Scheduler, tracking which agents exist and their
lifecycle state (created, running, paused, terminated).
Decoupled from task execution and kernel execution.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import UUID

from aios.agents.agent import AgentRecord
from aios.agents.agent_state import (
    AgentLifecycleState,
    normalize_state,
)
from aios.core.exceptions import AgentError, NotFoundError, ValidationError
from aios.core.ids import AgentID
from aios.core.models import Agent


class DuplicateAgentError(AgentError, ValidationError):
    """
    Raised when attempting to create/register an agent with an ID that
    is already present in the manager.
    """


class AgentNotFoundError(NotFoundError, AgentError):
    """
    Raised when an operation targets an agent ID that is not registered.
    """


def validate_agent_id(agent_id: Any) -> AgentID:
    """
    Validate that an identifier is a valid AgentID or safely coercible.

    Args:
        agent_id: An AgentID instance, a UUID instance, or a valid UUID string.

    Returns:
        The canonical AgentID instance.

    Raises:
        ValidationError: If the agent_id is None, an invalid type, empty,
            or cannot be parsed as a UUID.
    """
    if agent_id is None:
        raise ValidationError("agent_id cannot be None.")

    if isinstance(agent_id, AgentID):
        if not isinstance(agent_id.value, UUID):
            raise ValidationError(
                f"AgentID value must be a UUID, got {type(agent_id.value).__name__}."
            )
        return agent_id

    if isinstance(agent_id, UUID):
        return AgentID(agent_id)

    if isinstance(agent_id, str):
        cleaned = agent_id.strip()
        if not cleaned:
            raise ValidationError("agent_id cannot be empty.")
        try:
            return AgentID(UUID(cleaned))
        except (ValueError, AttributeError, TypeError) as err:
            raise ValidationError(
                f"agent_id string must be a valid UUID, got {agent_id!r}."
            ) from err

    raise ValidationError(
        f"agent_id must be an AgentID, UUID, or valid UUID string; "
        f"got {type(agent_id).__name__}."
    )


class AgentManager:
    """
    In-memory registry and lifecycle manager for AIOS agents.

    Provides operations to create, get, list, update lifecycle status,
    and remove agents, with strict ID validation and state machine
    transition enforcement.
    """

    def __init__(self) -> None:
        self._agents: dict[AgentID, AgentRecord] = {}

    def create(self, agent: Agent) -> AgentRecord:
        """
        Register a new agent and begin tracking its lifecycle.

        Args:
            agent: The canonical aios.core.models.Agent instance to manage.

        Returns:
            The created AgentRecord in CREATED state.

        Raises:
            ValidationError: If agent is not an Agent instance or has an invalid ID.
            DuplicateAgentError: If an agent with the same ID already exists.
        """
        if not isinstance(agent, Agent):
            raise ValidationError(
                f"agent must be an instance of aios.core.models.Agent, "
                f"got {type(agent).__name__}."
            )

        validated_id = validate_agent_id(agent.agent_id)

        if validated_id in self._agents:
            raise DuplicateAgentError(
                f"Agent with ID '{validated_id}' is already registered."
            )

        record = AgentRecord(agent=agent, state=AgentLifecycleState.CREATED)
        self._agents[validated_id] = record
        return record

    def get(self, agent_id: AgentID | str | UUID) -> AgentRecord:
        """
        Retrieve an agent's lifecycle record by ID.

        Args:
            agent_id: The ID of the agent to retrieve.

        Returns:
            The corresponding AgentRecord.

        Raises:
            ValidationError: If agent_id is invalid.
            AgentNotFoundError: If no agent with the given ID exists.
        """
        validated_id = validate_agent_id(agent_id)
        if validated_id not in self._agents:
            raise AgentNotFoundError(
                f"Agent with ID '{validated_id}' not found."
            )
        return self._agents[validated_id]

    def list(
        self,
        state: AgentLifecycleState | str | None = None,
    ) -> list[AgentRecord]:
        """
        List all managed agents, optionally filtered by lifecycle state.

        Args:
            state: Optional state filter.

        Returns:
            A list of AgentRecord instances.
        """
        if state is None:
            return list(self._agents.values())

        target_state = normalize_state(state)
        return [
            rec for rec in self._agents.values() if rec.state == target_state
        ]

    def update_status(
        self,
        agent_id: AgentID | str | UUID,
        new_state: AgentLifecycleState | str,
    ) -> AgentRecord:
        """
        Update an agent's lifecycle status.

        Routes through validate_transition to enforce valid state machine
        transitions.

        Args:
            agent_id: The ID of the agent to update.
            new_state: The desired target lifecycle state.

        Returns:
            The updated AgentRecord.

        Raises:
            ValidationError: If agent_id is invalid.
            AgentNotFoundError: If no agent with the given ID exists.
            InvalidStateTransitionError: If the transition is illegal.
        """
        validated_id = validate_agent_id(agent_id)
        if validated_id not in self._agents:
            raise AgentNotFoundError(
                f"Agent with ID '{validated_id}' not found."
            )

        record = self._agents[validated_id]
        record.update_state(new_state)
        return record

    def remove(self, agent_id: AgentID | str | UUID) -> None:
        """
        Remove an agent from management.

        Args:
            agent_id: The ID of the agent to remove.

        Raises:
            ValidationError: If agent_id is invalid.
            AgentNotFoundError: If no agent with the given ID exists.
        """
        validated_id = validate_agent_id(agent_id)
        if validated_id not in self._agents:
            raise AgentNotFoundError(
                f"Agent with ID '{validated_id}' not found."
            )

        del self._agents[validated_id]

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: Any) -> bool:
        try:
            validated_id = validate_agent_id(agent_id)
            return validated_id in self._agents
        except ValidationError:
            return False

    def __iter__(self) -> Iterator[AgentRecord]:
        return iter(self._agents.values())

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()
