"""
Cross-model contract tests.

These tests verify relationships between the shared AIOS models,
rather than testing the models independently.
"""

from aios.core import (
    AgentID,
    AgentRequest,
    AgentResponse,
    RequestID,
    RequestStatus,
    SystemCall,
    SystemCallType,
    TaskID,
)


def test_request_and_response_share_correlation_id():
    """
    A response must be traceable back to its originating request.
    """

    request_id = RequestID.generate()

    request = AgentRequest(
        request_id=request_id,
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(
            call_type=SystemCallType.LLM_CALL,
            payload={"prompt": "Hello"},
        ),
    )

    response = AgentResponse(
        request_id=request.request_id,
        status=RequestStatus.SUCCESS,
        result="Hello",
    )

    assert response.request_id == request.request_id


def test_request_contains_agent_and_task_identity():
    """
    Every request must identify both its agent and task.
    """

    agent_id = AgentID.generate()
    task_id = TaskID.generate()

    request = AgentRequest(
        request_id=RequestID.generate(),
        agent_id=agent_id,
        task_id=task_id,
        syscall=SystemCall(
            call_type=SystemCallType.MEMORY_READ,
        ),
    )

    assert request.agent_id == agent_id
    assert request.task_id == task_id