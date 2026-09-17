"""
Integration tests: an agent-shaped flow through the Phase 2 kernel.

Unlike `tests/unit/test_kernel.py`, these tests exercise the kernel
the way an agent actually would: issuing a sequence of different
syscalls against one long-lived `Kernel` instance, and issuing
several requests concurrently to confirm each syscall's lifecycle
is genuinely independent of the others.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from aios.core import (
    AgentID,
    AgentRequest,
    RequestID,
    RequestStatus,
    SystemCall,
    SystemCallType,
    TaskID,
)
from aios.kernel import Kernel


def _request_for(
    agent_id: AgentID,
    call_type: SystemCallType,
    payload: dict[str, object] | None = None,
) -> AgentRequest:
    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=agent_id,
        task_id=TaskID.generate(),
        syscall=SystemCall(
            call_type=call_type,
            payload=payload or {},
        ),
    )


def test_agent_can_exercise_every_syscall_in_one_session():
    """
    A single agent issuing one request per syscall type, in turn,
    against one kernel, should get back six correlated successes.
    """

    kernel = Kernel()
    agent_id = AgentID.generate()

    requests = [
        _request_for(agent_id, call_type)
        for call_type in SystemCallType
    ]

    responses = [kernel.handle(request) for request in requests]

    for request, response in zip(requests, responses):
        assert response.request_id == request.request_id
        assert response.status == RequestStatus.SUCCESS


def test_concurrent_requests_do_not_interfere():
    """
    Several requests dispatched at once must each resolve to their
    own response, with no cross-talk between syscalls. This is the
    property Phase 2's thread-per-syscall design exists to provide,
    and the property Phase 3's isolation tests will later push much
    harder on across real agents.
    """

    kernel = Kernel()
    agent_id = AgentID.generate()

    requests = [
        _request_for(
            agent_id,
            SystemCallType.MEMORY_READ,
            payload={"resource_id": i},
        )
        for i in range(20)
    ]

    with ThreadPoolExecutor(max_workers=20) as pool:
        responses = list(pool.map(kernel.handle, requests))

    seen_request_ids = {response.request_id for response in responses}

    assert len(seen_request_ids) == len(requests)

    for request, response in zip(requests, responses):
        assert response.request_id == request.request_id
        assert response.status == RequestStatus.SUCCESS
        assert response.result["resource_id"] == (
            request.syscall.payload["resource_id"]
        )


def test_phase_3_can_replace_a_mock_without_touching_the_kernel():
    """
    Simulates the Phase 2 -> Phase 3 handoff: a real handler is
    registered in place of the mock, and the exact same kernel call
    site (`kernel.handle`) picks it up with no other change.
    """

    kernel = Kernel()
    agent_id = AgentID.generate()

    def _real_memory_manager(request: AgentRequest) -> dict[str, object]:
        return {
            "handler": "real_memory_manager",
            "resource_id": request.syscall.payload.get("resource_id"),
            "content": "hydrated from a real store",
        }

    kernel.register_handler(
        SystemCallType.MEMORY_READ,
        _real_memory_manager,
    )

    request = _request_for(
        agent_id,
        SystemCallType.MEMORY_READ,
        payload={"resource_id": 42},
    )
    response = kernel.handle(request)

    assert response.status == RequestStatus.SUCCESS
    assert response.result["handler"] == "real_memory_manager"
    assert response.result["content"] == "hydrated from a real store"
