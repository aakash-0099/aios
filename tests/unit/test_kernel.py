"""
Unit tests for the Phase 2 kernel.

These tests cover, in isolation:
    - `SysCall` lifecycle transitions (success and failure).
    - `HandlerRegistry` resolution and its failure mode.
    - `SysCallEnvelope` / request serialization round-tripping.
    - `InMemoryQueueBackend`, the default queue transport.
    - `Dispatcher`: worker-pool concurrency bound, deferred handler
      resolution, and shutdown.
    - `KernelRuntime`, now a thin wrapper around a `Dispatcher`.
    - `Kernel.handle` end to end, including timeout and the
      "unsupported syscall never crashes the caller" contract.
"""

from __future__ import annotations

import threading
import time

import pytest

from aios.core import (
    AgentID,
    AgentRequest,
    RequestID,
    RequestStatus,
    SystemCall,
    SystemCallType,
    TaskID,
    ValidationError,
)
from aios.core.exceptions import SystemCallError
from aios.kernel import (
    Dispatcher,
    HandlerRegistry,
    InMemoryQueueBackend,
    Kernel,
    KernelRuntime,
    SysCall,
    SysCallEnvelope,
    request_from_dict,
    request_to_dict,
)
from aios.kernel.lifecycle import SysCallStatus


def _build_request(
    call_type: SystemCallType = SystemCallType.LLM_CALL,
    payload: dict[str, object] | None = None,
) -> AgentRequest:
    """
    Build a minimal, valid `AgentRequest` for a given syscall type.
    """

    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(
            call_type=call_type,
            payload=payload or {},
        ),
    )


# ---------------------------------------------------------------
# SysCall lifecycle
# ---------------------------------------------------------------


def test_syscall_completes_and_records_response():
    request = _build_request()

    syscall = SysCall(
        agent_name="agent-001",
        request=request,
        handler=lambda req: {"ok": True},
    )

    syscall.start()
    finished = syscall.wait_for_completion(timeout=5)

    assert finished is True
    assert syscall.error is None
    assert syscall.response == {"ok": True}
    assert syscall.lifecycle.status == SysCallStatus.COMPLETED
    assert syscall.lifecycle.duration_seconds is not None
    assert syscall.lifecycle.duration_seconds >= 0
    assert syscall.succeeded is True


def test_syscall_captures_handler_exception_as_error():
    request = _build_request()

    def _raising_handler(req: AgentRequest) -> None:
        raise RuntimeError("handler exploded")

    syscall = SysCall(
        agent_name="agent-001",
        request=request,
        handler=_raising_handler,
    )

    syscall.start()
    finished = syscall.wait_for_completion(timeout=5)

    assert finished is True
    assert syscall.response is None
    assert syscall.error == "handler exploded"
    assert syscall.lifecycle.status == SysCallStatus.FAILED
    assert syscall.succeeded is False


def test_each_syscall_gets_a_unique_pid():
    request = _build_request()

    first = SysCall(agent_name="agent-001", request=request, handler=lambda req: None)
    second = SysCall(agent_name="agent-001", request=request, handler=lambda req: None)

    assert first.pid != second.pid


def test_syscall_bind_handler_rejects_double_bind():
    request = _build_request()
    syscall = SysCall(agent_name="agent-001", request=request, handler=None)

    syscall.bind_handler(lambda req: None)

    with pytest.raises(RuntimeError):
        syscall.bind_handler(lambda req: None)


def test_syscall_without_a_handler_fails_deterministically():
    request = _build_request()
    syscall = SysCall(agent_name="agent-001", request=request, handler=None)

    syscall.start()
    finished = syscall.wait_for_completion(timeout=5)

    assert finished is True
    assert syscall.succeeded is False
    assert "No handler is bound" in syscall.error


# ---------------------------------------------------------------
# HandlerRegistry
# ---------------------------------------------------------------


def test_handler_registry_resolves_registered_handler():
    registry = HandlerRegistry()
    handler = lambda req: "result"  # noqa: E731

    registry.register(SystemCallType.LLM_CALL, handler)

    assert registry.resolve(SystemCallType.LLM_CALL) is handler
    assert registry.is_registered(SystemCallType.LLM_CALL) is True


def test_handler_registry_raises_for_unknown_syscall():
    registry = HandlerRegistry()

    with pytest.raises(SystemCallError):
        registry.resolve(SystemCallType.TOOL_CALL)


def test_default_registry_covers_all_six_syscalls():
    kernel = Kernel()

    for call_type in SystemCallType:
        assert kernel.handler_registry.is_registered(call_type)

    kernel.close()


# ---------------------------------------------------------------
# Request serialization (the wire format a queue backend carries)
# ---------------------------------------------------------------


def test_request_round_trips_through_dict():
    original = _build_request(
        call_type=SystemCallType.MEMORY_READ,
        payload={"resource_id": 123},
    )

    rebuilt = request_from_dict(request_to_dict(original))

    assert rebuilt.request_id == original.request_id
    assert rebuilt.agent_id == original.agent_id
    assert rebuilt.task_id == original.task_id
    assert rebuilt.syscall.call_type == original.syscall.call_type
    assert rebuilt.syscall.payload == original.syscall.payload


def test_request_from_dict_rejects_missing_field():
    with pytest.raises(ValidationError):
        request_from_dict({"request_id": "not-even-a-uuid"})


def test_request_from_dict_rejects_unknown_syscall_type():
    data = request_to_dict(_build_request())
    data["call_type"] = "not_a_real_syscall"

    with pytest.raises(ValidationError):
        request_from_dict(data)


# ---------------------------------------------------------------
# InMemoryQueueBackend
# ---------------------------------------------------------------


def test_in_memory_queue_backend_put_then_get():
    backend = InMemoryQueueBackend()
    envelope = SysCallEnvelope(pid=1, agent_name="agent-001", request={})

    backend.put(envelope)

    assert backend.size() == 1
    assert backend.get(timeout=1) == envelope
    assert backend.size() == 0


def test_in_memory_queue_backend_get_times_out_with_none():
    backend = InMemoryQueueBackend()

    assert backend.get(timeout=0.05) is None


# ---------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------


def test_dispatcher_resolves_handler_after_dequeue_not_at_submit():
    """
    Registering a handler *after* submit, but before the worker
    gets to it, must still work -- proof that resolution happens
    on the worker side, not eagerly inside submit().
    """

    registry = HandlerRegistry()
    dispatcher = Dispatcher(registry=registry, worker_count=1)

    try:
        request = _build_request()
        # Nothing registered yet: submit must not raise.
        syscall = dispatcher.submit(request, agent_name="agent-001")

        registry.register(SystemCallType.LLM_CALL, lambda req: "late-bound")

        finished = syscall.wait_for_completion(timeout=5)

        assert finished is True
        assert syscall.succeeded is True
        assert syscall.response == "late-bound"
    finally:
        dispatcher.close()


def test_dispatcher_reports_unsupported_syscall_without_raising():
    dispatcher = Dispatcher(registry=HandlerRegistry(), worker_count=1)

    try:
        request = _build_request()
        syscall = dispatcher.submit(request, agent_name="agent-001")  # must not raise

        finished = syscall.wait_for_completion(timeout=5)

        assert finished is True
        assert syscall.succeeded is False
        assert "No handler registered" in syscall.error
    finally:
        dispatcher.close()


def test_dispatcher_bounds_concurrency_to_worker_count():
    registry = HandlerRegistry()
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def _slow_handler(req: AgentRequest) -> str:
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.15)
        with lock:
            concurrent -= 1
        return "done"

    registry.register(SystemCallType.LLM_CALL, _slow_handler)
    dispatcher = Dispatcher(registry=registry, worker_count=2)

    try:
        syscalls = [dispatcher.submit(_build_request(), "agent-001") for _ in range(6)]

        for syscall in syscalls:
            assert syscall.wait_for_completion(timeout=5) is True

        assert all(syscall.succeeded for syscall in syscalls)
        assert max_concurrent == 2
    finally:
        dispatcher.close()


def test_dispatcher_close_stops_all_workers():
    dispatcher = Dispatcher(worker_count=3)
    workers = list(dispatcher._workers)  # noqa: SLF001

    dispatcher.close()

    assert all(not worker.is_alive() for worker in workers)


# ---------------------------------------------------------------
# KernelRuntime
# ---------------------------------------------------------------


def test_runtime_tracks_active_syscall_until_awaited():
    runtime = KernelRuntime()
    release = time.monotonic() + 0.1

    def _slow_handler(req: AgentRequest) -> str:
        time.sleep(max(release - time.monotonic(), 0))
        return "done"

    runtime.registry.register(SystemCallType.LLM_CALL, _slow_handler)

    request = _build_request()
    syscall = runtime.submit(request)

    assert runtime.active_count() == 1
    assert runtime.is_active(syscall.pid) is True

    finished = runtime.wait_for(syscall, timeout=5)

    assert finished is True
    assert runtime.active_count() == 0
    assert runtime.is_active(syscall.pid) is False

    runtime.close()


def test_runtime_submit_does_not_raise_for_unsupported_syscall():
    """
    Handler resolution now happens on the dispatcher's worker
    thread, after the syscall is queued -- so `submit()` itself
    always succeeds. The failure surfaces on the syscall once it
    has run through the dispatcher, not synchronously here.
    """

    runtime = KernelRuntime(registry=HandlerRegistry())

    try:
        request = _build_request()
        syscall = runtime.submit(request)  # must not raise

        finished = runtime.wait_for(syscall, timeout=5)

        assert finished is True
        assert syscall.succeeded is False
        assert "No handler registered" in syscall.error
    finally:
        runtime.close()


def test_runtime_rejects_both_dispatcher_and_registry():
    with pytest.raises(ValueError):
        KernelRuntime(registry=HandlerRegistry(), dispatcher=Dispatcher())


def test_runtime_can_be_given_a_pre_built_dispatcher_with_custom_worker_count():
    dispatcher = Dispatcher(worker_count=1)
    runtime = KernelRuntime(dispatcher=dispatcher)

    assert runtime.dispatcher is dispatcher

    runtime.close()


# ---------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------


def test_kernel_rejects_non_agent_request():
    kernel = Kernel()

    with pytest.raises(ValidationError):
        kernel.handle({"not": "a request"})  # type: ignore[arg-type]

    kernel.close()


@pytest.mark.parametrize("call_type", list(SystemCallType))
def test_kernel_handles_every_default_mock_syscall(call_type):
    kernel = Kernel()
    request = _build_request(call_type=call_type)

    response = kernel.handle(request)

    assert response.request_id == request.request_id
    assert response.status == RequestStatus.SUCCESS
    assert response.error is None
    assert response.result is not None

    kernel.close()


def test_kernel_returns_failed_response_for_unsupported_syscall():
    kernel = Kernel(runtime=KernelRuntime(registry=HandlerRegistry()))
    request = _build_request()

    response = kernel.handle(request)

    assert response.status == RequestStatus.FAILED
    assert response.result is None
    assert "No handler registered" in response.error

    kernel.close()


def test_kernel_returns_failed_response_when_handler_raises():
    kernel = Kernel()

    def _raising_handler(req: AgentRequest) -> None:
        raise RuntimeError("resource unavailable")

    kernel.register_handler(SystemCallType.MEMORY_READ, _raising_handler)

    request = _build_request(call_type=SystemCallType.MEMORY_READ)
    response = kernel.handle(request)

    assert response.status == RequestStatus.FAILED
    assert response.error == "resource unavailable"

    kernel.close()


def test_kernel_returns_failed_response_on_timeout():
    kernel = Kernel(timeout=0.05)

    def _slow_handler(req: AgentRequest) -> str:
        time.sleep(1)
        return "too late"

    kernel.register_handler(SystemCallType.LLM_CALL, _slow_handler)

    request = _build_request(call_type=SystemCallType.LLM_CALL)
    response = kernel.handle(request)

    assert response.status == RequestStatus.FAILED
    assert "timed out" in response.error

    kernel.close()


def test_kernel_register_handler_overrides_default_mock():
    kernel = Kernel()

    kernel.register_handler(
        SystemCallType.TOOL_CALL,
        lambda req: {"handler": "real_tool_manager"},
    )

    request = _build_request(call_type=SystemCallType.TOOL_CALL)
    response = kernel.handle(request)

    assert response.status == RequestStatus.SUCCESS
    assert response.result == {"handler": "real_tool_manager"}

    kernel.close()


def test_kernel_close_stops_dispatcher_workers():
    kernel = Kernel()
    workers = list(kernel._runtime.dispatcher._workers)  # noqa: SLF001

    kernel.close()

    assert all(not worker.is_alive() for worker in workers)
