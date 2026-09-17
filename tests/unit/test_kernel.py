"""
Unit tests for the Phase 2 kernel.

These tests cover, in isolation:
    - `SysCall` lifecycle transitions (success and failure).
    - `HandlerRegistry` resolution and its failure mode.
    - `KernelRuntime` submission/tracking/waiting.
    - `Kernel.handle` end to end, including timeout and the
      "unsupported syscall never crashes the caller" contract.
"""

from __future__ import annotations

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
from aios.kernel import HandlerRegistry, Kernel, KernelRuntime, SysCall
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

    first = SysCall(
        agent_name="agent-001",
        request=request,
        handler=lambda req: None,
    )
    second = SysCall(
        agent_name="agent-001",
        request=request,
        handler=lambda req: None,
    )

    assert first.pid != second.pid


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


# ---------------------------------------------------------------
# KernelRuntime
# ---------------------------------------------------------------


def test_runtime_tracks_active_syscall_until_awaited():
    runtime = KernelRuntime()
    release = time.monotonic() + 0.1

    def _slow_handler(req: AgentRequest) -> str:
        time.sleep(max(release - time.monotonic(), 0))
        return "done"

    registry = runtime.registry
    registry.register(SystemCallType.LLM_CALL, _slow_handler)

    request = _build_request()
    syscall = runtime.submit(request)

    assert runtime.active_count() == 1
    assert runtime.is_active(syscall.pid) is True

    finished = runtime.wait_for(syscall, timeout=5)

    assert finished is True
    assert runtime.active_count() == 0
    assert runtime.is_active(syscall.pid) is False


def test_runtime_submit_raises_for_unsupported_syscall():
    runtime = KernelRuntime(registry=HandlerRegistry())
    request = _build_request()

    with pytest.raises(SystemCallError):
        runtime.submit(request)


# ---------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------


def test_kernel_rejects_non_agent_request():
    kernel = Kernel()

    with pytest.raises(ValidationError):
        kernel.handle({"not": "a request"})  # type: ignore[arg-type]


@pytest.mark.parametrize("call_type", list(SystemCallType))
def test_kernel_handles_every_default_mock_syscall(call_type):
    kernel = Kernel()
    request = _build_request(call_type=call_type)

    response = kernel.handle(request)

    assert response.request_id == request.request_id
    assert response.status == RequestStatus.SUCCESS
    assert response.error is None
    assert response.result is not None


def test_kernel_returns_failed_response_for_unsupported_syscall():
    kernel = Kernel(runtime=KernelRuntime(registry=HandlerRegistry()))
    request = _build_request()

    response = kernel.handle(request)

    assert response.status == RequestStatus.FAILED
    assert response.result is None
    assert "No handler registered" in response.error


def test_kernel_returns_failed_response_when_handler_raises():
    kernel = Kernel()

    def _raising_handler(req: AgentRequest) -> None:
        raise RuntimeError("resource unavailable")

    kernel.register_handler(SystemCallType.MEMORY_READ, _raising_handler)

    request = _build_request(call_type=SystemCallType.MEMORY_READ)
    response = kernel.handle(request)

    assert response.status == RequestStatus.FAILED
    assert response.error == "resource unavailable"


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
