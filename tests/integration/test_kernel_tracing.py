"""
Integration tests: the Kernel emits a complete, ordered trace.

With tracing captured, every request that flows through
`Kernel.handle` produces the full lifecycle -- created, queued,
validated, started, handler_invoked, completed/failed,
response_returned -- under its own correlation id, even when many
requests run concurrently.  With tracing unconfigured the
responses are byte-identical to the traced run.
"""

from __future__ import annotations

import threading

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
from aios.monitoring import (
    KernelEvents,
    capture,
    reset,
)

EXPECTED_SUCCESS_SEQUENCE = [
    KernelEvents.SYSCALL_CREATED,
    KernelEvents.SYSCALL_QUEUED,
    KernelEvents.SYSCALL_VALIDATED,
    KernelEvents.SYSCALL_STARTED,
    KernelEvents.SYSCALL_HANDLER_INVOKED,
    KernelEvents.SYSCALL_COMPLETED,
    KernelEvents.RESPONSE_RETURNED,
]

EXPECTED_FAILURE_SEQUENCE = [
    KernelEvents.SYSCALL_CREATED,
    KernelEvents.SYSCALL_QUEUED,
    KernelEvents.SYSCALL_VALIDATED,
    KernelEvents.SYSCALL_STARTED,
    KernelEvents.SYSCALL_HANDLER_INVOKED,
    KernelEvents.SYSCALL_FAILED,
    KernelEvents.RESPONSE_RETURNED,
]


def _build_request(
    call_type: SystemCallType = SystemCallType.LLM_CALL,
    payload: dict[str, object] | None = None,
) -> AgentRequest:
    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(
            call_type=call_type,
            payload=payload or {},
        ),
    )


def _events_for(sink, correlation_id: str):
    return [
        event
        for event in sink.events()
        if event.correlation_id == correlation_id
    ]


def test_single_syscall_produces_the_expected_ordered_sequence():
    kernel = Kernel()
    request = _build_request()

    with capture() as sink:
        response = kernel.handle(request, agent_name="agent-001")

    kernel.close()

    assert response.status == RequestStatus.SUCCESS
    events = _events_for(sink, str(request.request_id))
    assert [event.event for event in events] == EXPECTED_SUCCESS_SEQUENCE
    assert all(event.agent_id == str(request.agent_id) for event in events)
    assert all(event.component == "kernel" for event in events)


def test_twenty_concurrent_syscalls_each_produce_a_complete_sequence():
    requests = [_build_request() for _ in range(20)]
    kernel = Kernel()

    with capture() as sink:
        responses: list = []

        def _worker(request: AgentRequest, agent_index: int) -> None:
            responses.append(
                kernel.handle(request, agent_name=f"agent-{agent_index}")
            )

        threads = [
            threading.Thread(target=_worker, args=(request, index))
            for index, request in enumerate(requests)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    kernel.close()

    assert len(responses) == 20
    assert all(r.status == RequestStatus.SUCCESS for r in responses)

    by_correlation: dict[str, list] = {}
    for event in sink.events():
        by_correlation.setdefault(event.correlation_id, []).append(event)

    assert len(by_correlation) == 20
    for request in requests:
        correlation_id = str(request.request_id)
        sequence = [
            event.event for event in by_correlation[correlation_id]
        ]
        assert sequence == EXPECTED_SUCCESS_SEQUENCE
        # created must always precede completed, per correlation id
        assert sequence.index(KernelEvents.SYSCALL_CREATED) < sequence.index(
            KernelEvents.SYSCALL_COMPLETED
        )


def test_failing_handler_produces_a_failed_event():
    kernel = Kernel()

    def _raising_handler(request: AgentRequest) -> None:
        raise RuntimeError("resource unavailable")

    kernel.register_handler(SystemCallType.MEMORY_READ, _raising_handler)

    request = _build_request(call_type=SystemCallType.MEMORY_READ)

    with capture() as sink:
        response = kernel.handle(request, agent_name="agent-001")

    kernel.close()

    assert response.status == RequestStatus.FAILED
    assert response.error == "resource unavailable"

    events = _events_for(sink, str(request.request_id))
    assert [event.event for event in events] == EXPECTED_FAILURE_SEQUENCE

    failed_index = EXPECTED_FAILURE_SEQUENCE.index(KernelEvents.SYSCALL_FAILED)
    failed = events[failed_index]
    assert failed.data["exc_type"] == "RuntimeError"
    assert failed.data["error"] == "resource unavailable"


def test_unconfigured_tracing_leaves_responses_identical():
    requests = [
        _build_request(call_type=call_type)
        for call_type in (
            SystemCallType.LLM_CALL,
            SystemCallType.MEMORY_READ,
            SystemCallType.MEMORY_WRITE,
            SystemCallType.STORAGE_READ,
            SystemCallType.STORAGE_WRITE,
            SystemCallType.TOOL_CALL,
        )
    ]

    reset()
    untraced_kernel = Kernel()
    untraced_responses = [
        untraced_kernel.handle(request, agent_name="agent-x")
        for request in requests
    ]
    untraced_kernel.close()

    traced_kernel = Kernel()
    with capture() as sink:
        traced_responses = [
            traced_kernel.handle(request, agent_name="agent-x")
            for request in requests
        ]
    traced_kernel.close()

    def _shape(responses: list) -> list:
        return [
            (response.status, response.error, response.result)
            for response in responses
        ]

    assert _shape(traced_responses) == _shape(untraced_responses)
    assert all(
        response.status == RequestStatus.SUCCESS
        for response in traced_responses
    )
    assert len(sink.events()) == len(requests) * len(EXPECTED_SUCCESS_SEQUENCE)
