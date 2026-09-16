"""
Tests for shared AIOS identifiers.
"""

from aios.core import (
    AgentID,
    MemoryID,
    RequestID,
    ResourceID,
    TaskID,
    ToolID,
)


def test_agent_ids_are_unique():
    """Each generated AgentID should be unique."""

    first = AgentID.generate()
    second = AgentID.generate()

    assert first != second


def test_all_identifier_types_can_be_generated():
    """Every Phase 1 identifier type should generate successfully."""

    assert isinstance(AgentID.generate(), AgentID)
    assert isinstance(TaskID.generate(), TaskID)
    assert isinstance(RequestID.generate(), RequestID)
    assert isinstance(ResourceID.generate(), ResourceID)
    assert isinstance(MemoryID.generate(), MemoryID)
    assert isinstance(ToolID.generate(), ToolID)


def test_identifier_string_representation():
    """Identifiers should have a stable string representation."""

    identifier = AgentID.generate()

    assert str(identifier)