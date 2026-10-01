"""
Unit tests for AgentRecord and AgentManager.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from aios.agents import (
    AgentLifecycleState,
    AgentManager,
    AgentNotFoundError,
    AgentRecord,
    DuplicateAgentError,
    InvalidStateTransitionError,
    ManagedAgent,
)
from aios.core.exceptions import (
    AgentError,
    NotFoundError,
    ValidationError,
)
from aios.core.ids import AgentID
from aios.core.models import Agent


def _make_agent(name: str = "test-agent") -> Agent:
    return Agent(
        agent_id=AgentID.generate(),
        name=name,
        metadata={"role": "assistant"},
    )


# ---------------------------------------------------------------------------
# AgentRecord tests
# ---------------------------------------------------------------------------


def test_agent_record_initialization_and_properties():
    """AgentRecord wraps an Agent and exposes convenience properties."""
    agent = _make_agent("agent-1")
    record = AgentRecord(agent=agent)

    assert record.agent is agent
    assert record.agent_id == agent.agent_id
    assert record.name == "agent-1"
    assert record.metadata == {"role": "assistant"}
    assert record.state == AgentLifecycleState.CREATED
    assert record.status == AgentLifecycleState.CREATED
    assert isinstance(record.created_at, datetime)
    assert isinstance(record.updated_at, datetime)


def test_managed_agent_alias():
    """ManagedAgent is an alias for AgentRecord."""
    assert ManagedAgent is AgentRecord


def test_agent_record_rejects_non_agent():
    """AgentRecord constructor requires an aios.core.models.Agent instance."""
    with pytest.raises(ValidationError):
        AgentRecord(agent="not-an-agent")  # type: ignore[arg-type]


def test_agent_record_update_state():
    """AgentRecord.update_state validates transition and updates timestamp."""
    agent = _make_agent()
    record = AgentRecord(agent=agent)
    initial_updated_at = record.updated_at

    record.update_state(AgentLifecycleState.RUNNING)
    assert record.state == AgentLifecycleState.RUNNING
    assert record.updated_at >= initial_updated_at

    with pytest.raises(InvalidStateTransitionError):
        record.update_state(AgentLifecycleState.CREATED)


# ---------------------------------------------------------------------------
# AgentManager CRUD happy paths
# ---------------------------------------------------------------------------


def test_agent_manager_create_happy_path():
    """Create adds an agent and returns a tracked AgentRecord."""
    mgr = AgentManager()
    agent = _make_agent("agent-alpha")

    record = mgr.create(agent)

    assert isinstance(record, AgentRecord)
    assert record.agent == agent
    assert record.state == AgentLifecycleState.CREATED
    assert record.agent_id == agent.agent_id
    assert len(mgr) == 1
    assert agent.agent_id in mgr


def test_agent_manager_get_happy_path():
    """Get retrieves the record using AgentID, UUID string, or UUID."""
    mgr = AgentManager()
    agent = _make_agent("agent-beta")
    created = mgr.create(agent)

    # By AgentID
    assert mgr.get(agent.agent_id) is created

    # By UUID string
    assert mgr.get(str(agent.agent_id)) is created

    # By UUID value
    assert mgr.get(agent.agent_id.value) is created


def test_agent_manager_list_happy_path():
    """List returns all records, and can filter by state."""
    mgr = AgentManager()
    assert mgr.list() == []

    agent1 = _make_agent("agent-1")
    agent2 = _make_agent("agent-2")
    r1 = mgr.create(agent1)
    r2 = mgr.create(agent2)

    all_agents = mgr.list()
    assert len(all_agents) == 2
    assert r1 in all_agents
    assert r2 in all_agents

    # Transition agent2 to RUNNING
    mgr.update_status(agent2.agent_id, AgentLifecycleState.RUNNING)

    created_only = mgr.list(state=AgentLifecycleState.CREATED)
    assert created_only == [r1]

    running_only = mgr.list(state=AgentLifecycleState.RUNNING)
    assert running_only == [r2]


def test_agent_manager_remove_happy_path():
    """Remove deletes an agent from the registry."""
    mgr = AgentManager()
    agent = _make_agent("agent-to-remove")
    mgr.create(agent)
    assert len(mgr) == 1

    mgr.remove(agent.agent_id)
    assert len(mgr) == 0
    assert agent.agent_id not in mgr

    with pytest.raises(AgentNotFoundError):
        mgr.get(agent.agent_id)


# ---------------------------------------------------------------------------
# Rejections and Error Paths
# ---------------------------------------------------------------------------


def test_duplicate_id_rejection():
    """Attempting to register the same agent or ID twice raises DuplicateAgentError."""
    mgr = AgentManager()
    agent = _make_agent("first")
    mgr.create(agent)

    # Same agent instance
    with pytest.raises(DuplicateAgentError):
        mgr.create(agent)

    # Different agent instance with identical agent_id
    duplicate_agent = Agent(
        agent_id=agent.agent_id,
        name="copy",
    )
    with pytest.raises(DuplicateAgentError):
        mgr.create(duplicate_agent)


def test_duplicate_agent_error_hierarchy():
    """DuplicateAgentError must inherit from AgentError and ValidationError."""
    assert issubclass(DuplicateAgentError, AgentError)
    assert issubclass(DuplicateAgentError, ValidationError)


def test_invalid_state_transition_rejection():
    """mgr.update_status raises InvalidStateTransitionError on illegal moves."""
    mgr = AgentManager()
    agent = _make_agent("agent-trans")
    mgr.create(agent)

    # Illegal transition: CREATED -> PAUSED
    with pytest.raises(InvalidStateTransitionError):
        mgr.update_status(agent.agent_id, AgentLifecycleState.PAUSED)

    # Move to RUNNING, then TERMINATED
    mgr.update_status(agent.agent_id, AgentLifecycleState.RUNNING)
    mgr.update_status(agent.agent_id, AgentLifecycleState.TERMINATED)

    # Illegal transition: TERMINATED -> RUNNING (specifically flagged in spec)
    with pytest.raises(InvalidStateTransitionError):
        mgr.update_status(agent.agent_id, AgentLifecycleState.RUNNING)

    # Illegal transition: TERMINATED -> PAUSED
    with pytest.raises(InvalidStateTransitionError):
        mgr.update_status(agent.agent_id, AgentLifecycleState.PAUSED)


def test_valid_state_transitions_lifecycle_path():
    """Step through valid transitions CREATED -> RUNNING -> PAUSED -> RUNNING -> TERMINATED."""
    mgr = AgentManager()
    agent = _make_agent("lifecycle-agent")
    record = mgr.create(agent)
    assert record.state == AgentLifecycleState.CREATED

    mgr.update_status(agent.agent_id, AgentLifecycleState.RUNNING)
    assert record.state == AgentLifecycleState.RUNNING

    mgr.update_status(agent.agent_id, AgentLifecycleState.PAUSED)
    assert record.state == AgentLifecycleState.PAUSED

    mgr.update_status(agent.agent_id, AgentLifecycleState.RUNNING)
    assert record.state == AgentLifecycleState.RUNNING

    mgr.update_status(agent.agent_id, AgentLifecycleState.TERMINATED)
    assert record.state == AgentLifecycleState.TERMINATED


def test_remove_non_existent_agent_raises_error():
    """Removing an unregistered agent ID must raise AgentNotFoundError / NotFoundError."""
    mgr = AgentManager()
    non_existent = AgentID.generate()

    with pytest.raises(AgentNotFoundError):
        mgr.remove(non_existent)

    with pytest.raises(NotFoundError):
        mgr.remove(str(uuid4()))


def test_get_non_existent_agent_raises_error():
    """Getting an unregistered agent ID must raise AgentNotFoundError."""
    mgr = AgentManager()
    with pytest.raises(AgentNotFoundError):
        mgr.get(AgentID.generate())


def test_update_status_non_existent_agent_raises_error():
    """Updating status of an unregistered agent ID must raise AgentNotFoundError."""
    mgr = AgentManager()
    with pytest.raises(AgentNotFoundError):
        mgr.update_status(AgentID.generate(), AgentLifecycleState.RUNNING)


def test_agent_not_found_error_hierarchy():
    """AgentNotFoundError must inherit from NotFoundError and AgentError."""
    assert issubclass(AgentNotFoundError, NotFoundError)
    assert issubclass(AgentNotFoundError, AgentError)


def test_operations_reject_invalid_agent_ids():
    """All operations validate agent ID format and raise ValidationError for malformed IDs."""
    mgr = AgentManager()

    invalid_ids = ["not-a-valid-uuid", "", "   ", 12345, None]

    for invalid_id in invalid_ids:
        with pytest.raises(ValidationError):
            mgr.get(invalid_id)  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            mgr.remove(invalid_id)  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            mgr.update_status(invalid_id, AgentLifecycleState.RUNNING)  # type: ignore[arg-type]


def test_create_rejects_non_agent():
    """mgr.create raises ValidationError if passed a non-Agent object."""
    mgr = AgentManager()
    with pytest.raises(ValidationError):
        mgr.create("invalid-agent")  # type: ignore[arg-type]
