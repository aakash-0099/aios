"""
Contract tests for AIOS Phase 1.

These tests are different from ordinary functionality tests.

Their purpose is to freeze the rules that every future AIOS
component must obey when creating shared objects.
"""

import pytest

from aios.core import (
    Agent,
    AgentID,
    AgentRequest,
    AgentResponse,
    Context,
    Memory,
    MemoryID,
    RequestID,
    RequestStatus,
    Resource,
    ResourceID,
    ResourceType,
    SystemCall,
    SystemCallType,
    Task,
    TaskID,
    Tool,
    ToolID,
)
from aios.core.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Agent contracts
# ---------------------------------------------------------------------------


def test_agent_requires_agent_id():
    """
    Agent must use the canonical AgentID type.
    """

    with pytest.raises(ValidationError):
        Agent(
            agent_id="not-an-agent-id",  # type: ignore[arg-type]
            name="test-agent",
        )


def test_agent_name_cannot_be_empty():
    """
    Agent names must contain meaningful text.
    """

    with pytest.raises(ValidationError):
        Agent(
            agent_id=AgentID.generate(),
            name="",
        )


def test_agent_name_cannot_be_whitespace():
    """
    Whitespace-only names are also invalid.
    """

    with pytest.raises(ValidationError):
        Agent(
            agent_id=AgentID.generate(),
            name="   ",
        )


# ---------------------------------------------------------------------------
# Task contracts
# ---------------------------------------------------------------------------


def test_task_requires_task_id():
    with pytest.raises(ValidationError):
        Task(
            task_id="invalid",  # type: ignore[arg-type]
            agent_id=AgentID.generate(),
            description="Do something",
        )


def test_task_requires_agent_id():
    with pytest.raises(ValidationError):
        Task(
            task_id=TaskID.generate(),
            agent_id="invalid",  # type: ignore[arg-type]
            description="Do something",
        )


def test_task_description_cannot_be_empty():
    with pytest.raises(ValidationError):
        Task(
            task_id=TaskID.generate(),
            agent_id=AgentID.generate(),
            description="",
        )


# ---------------------------------------------------------------------------
# SystemCall contracts
# ---------------------------------------------------------------------------


def test_system_call_requires_valid_type():
    with pytest.raises(ValidationError):
        SystemCall(
            call_type="invalid",  # type: ignore[arg-type]
        )


def test_system_call_payload_must_be_dictionary():
    with pytest.raises(ValidationError):
        SystemCall(
            call_type=SystemCallType.LLM_CALL,
            payload="invalid",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# AgentRequest contracts
# ---------------------------------------------------------------------------


def test_request_requires_request_id():
    with pytest.raises(ValidationError):
        AgentRequest(
            request_id="invalid",  # type: ignore[arg-type]
            agent_id=AgentID.generate(),
            task_id=TaskID.generate(),
            syscall=SystemCall(
                call_type=SystemCallType.LLM_CALL
            ),
        )


def test_request_requires_agent_id():
    with pytest.raises(ValidationError):
        AgentRequest(
            request_id=RequestID.generate(),
            agent_id="invalid",  # type: ignore[arg-type]
            task_id=TaskID.generate(),
            syscall=SystemCall(
                call_type=SystemCallType.LLM_CALL
            ),
        )


def test_request_requires_task_id():
    with pytest.raises(ValidationError):
        AgentRequest(
            request_id=RequestID.generate(),
            agent_id=AgentID.generate(),
            task_id="invalid",  # type: ignore[arg-type]
            syscall=SystemCall(
                call_type=SystemCallType.LLM_CALL
            ),
        )


def test_request_requires_system_call():
    with pytest.raises(ValidationError):
        AgentRequest(
            request_id=RequestID.generate(),
            agent_id=AgentID.generate(),
            task_id=TaskID.generate(),
            syscall="invalid",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# AgentResponse contracts
# ---------------------------------------------------------------------------


def test_success_response_requires_result():
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id=RequestID.generate(),
            status=RequestStatus.SUCCESS,
        )


def test_success_response_cannot_have_error():
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id=RequestID.generate(),
            status=RequestStatus.SUCCESS,
            result="success",
            error="something went wrong",
        )


def test_failed_response_requires_error():
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id=RequestID.generate(),
            status=RequestStatus.FAILED,
        )


def test_failed_response_cannot_have_result():
    with pytest.raises(ValidationError):
        AgentResponse(
            request_id=RequestID.generate(),
            status=RequestStatus.FAILED,
            result="unexpected result",
            error="failure",
        )


# ---------------------------------------------------------------------------
# Resource contracts
# ---------------------------------------------------------------------------


def test_resource_requires_valid_id():
    with pytest.raises(ValidationError):
        Resource(
            resource_id="invalid",  # type: ignore[arg-type]
            resource_type=ResourceType.MEMORY,
            name="memory",
        )


def test_resource_requires_valid_type():
    with pytest.raises(ValidationError):
        Resource(
            resource_id=ResourceID.generate(),
            resource_type="invalid",  # type: ignore[arg-type]
            name="memory",
        )


def test_resource_name_cannot_be_empty():
    with pytest.raises(ValidationError):
        Resource(
            resource_id=ResourceID.generate(),
            resource_type=ResourceType.MEMORY,
            name="",
        )


# ---------------------------------------------------------------------------
# Context contracts
# ---------------------------------------------------------------------------


def test_context_conversation_must_be_list():
    with pytest.raises(ValidationError):
        Context(
            conversation="invalid",  # type: ignore[arg-type]
        )


def test_context_conversation_items_must_be_dicts():
    with pytest.raises(ValidationError):
        Context(
            conversation=["invalid"],  # type: ignore[list-item]
        )


def test_context_memory_items_must_be_dicts():
    with pytest.raises(ValidationError):
        Context(
            memory=["invalid"],  # type: ignore[list-item]
        )


# ---------------------------------------------------------------------------
# Memory contracts
# ---------------------------------------------------------------------------


def test_memory_requires_memory_id():
    with pytest.raises(ValidationError):
        Memory(
            memory_id="invalid",  # type: ignore[arg-type]
            agent_id=AgentID.generate(),
            content="memory",
        )


def test_memory_requires_agent_id():
    with pytest.raises(ValidationError):
        Memory(
            memory_id=MemoryID.generate(),
            agent_id="invalid",  # type: ignore[arg-type]
            content="memory",
        )


def test_memory_content_cannot_be_empty():
    with pytest.raises(ValidationError):
        Memory(
            memory_id=MemoryID.generate(),
            agent_id=AgentID.generate(),
            content="",
        )


# ---------------------------------------------------------------------------
# Tool contracts
# ---------------------------------------------------------------------------


def test_tool_requires_tool_id():
    with pytest.raises(ValidationError):
        Tool(
            tool_id="invalid",  # type: ignore[arg-type]
            name="calculator",
            description="Calculator tool",
        )


def test_tool_name_cannot_be_empty():
    with pytest.raises(ValidationError):
        Tool(
            tool_id=ToolID.generate(),
            name="",
            description="Calculator tool",
        )


def test_tool_description_cannot_be_empty():
    with pytest.raises(ValidationError):
        Tool(
            tool_id=ToolID.generate(),
            name="calculator",
            description="",
        )