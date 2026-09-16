"""
Tests for the shared AIOS data models.
"""

from aios.core import (
    Agent,
    AgentID,
    AgentRequest,
    AgentResponse,
    AgentStatus,
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
    TaskStatus,
    Tool,
    ToolID,
)


def test_agent_can_be_created():
    agent = Agent(
        agent_id=AgentID.generate(),
        name="test-agent",
    )

    assert agent.name == "test-agent"
    assert agent.status == AgentStatus.CREATED


def test_task_references_agent():
    agent_id = AgentID.generate()

    task = Task(
        task_id=TaskID.generate(),
        agent_id=agent_id,
        description="Test task",
    )

    assert task.agent_id == agent_id
    assert task.status == TaskStatus.CREATED


def test_all_initial_system_calls_exist():
    expected = {
        "llm_call",
        "memory_read",
        "memory_write",
        "storage_read",
        "storage_write",
        "tool_call",
    }

    actual = {
        syscall.value
        for syscall in SystemCallType
    }

    assert actual == expected


def test_system_call_can_carry_payload():
    syscall = SystemCall(
        call_type=SystemCallType.LLM_CALL,
        payload={"prompt": "Hello"},
    )

    assert syscall.call_type == SystemCallType.LLM_CALL
    assert syscall.payload["prompt"] == "Hello"


def test_agent_request_contains_correlation_information():
    agent_id = AgentID.generate()
    task_id = TaskID.generate()
    request_id = RequestID.generate()

    request = AgentRequest(
        request_id=request_id,
        agent_id=agent_id,
        task_id=task_id,
        syscall=SystemCall(
            call_type=SystemCallType.MEMORY_READ,
        ),
    )

    assert request.request_id == request_id
    assert request.agent_id == agent_id
    assert request.task_id == task_id


def test_agent_response_correlates_with_request():
    request_id = RequestID.generate()

    response = AgentResponse(
        request_id=request_id,
        status=RequestStatus.SUCCESS,
        result={"value": "test"},
    )

    assert response.request_id == request_id
    assert response.status == RequestStatus.SUCCESS
    assert response.result["value"] == "test"


def test_resource_can_be_created():
    resource = Resource(
        resource_id=ResourceID.generate(),
        resource_type=ResourceType.MEMORY,
        name="agent-memory",
    )

    assert resource.resource_type == ResourceType.MEMORY


def test_context_can_be_created():
    context = Context(
        system="You are an AI agent.",
        working_state={"step": 1},
    )

    assert context.system == "You are an AI agent."
    assert context.working_state["step"] == 1


def test_memory_can_be_created():
    memory = Memory(
        memory_id=MemoryID.generate(),
        agent_id=AgentID.generate(),
        content="Important information.",
    )

    assert memory.content == "Important information."


def test_tool_can_be_created():
    tool = Tool(
        tool_id=ToolID.generate(),
        name="calculator",
        description="Performs calculations.",
    )

    assert tool.name == "calculator"