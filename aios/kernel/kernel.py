"""
AIOS kernel dispatcher.

`Kernel` is the one public entry point Phase 2 establishes. An
agent (or, later, the SDK on its behalf) builds an `AgentRequest`
and calls `Kernel.handle`. Nothing above the kernel ever touches a
`SysCall`, a handler, or a resource directly.

The kernel deliberately does not know how any syscall is carried
out. That is the whole point of freezing this contract now: Phase 3
can replace every mock handler with a real Memory Manager, Storage
Manager, Tool Manager, and LLM Core, and this file does not change.
"""

from __future__ import annotations

from aios.core.exceptions import AIOSException, ValidationError
from aios.core.models import (
    AgentRequest,
    AgentResponse,
    RequestStatus,
    SystemCallType,
)
from aios.kernel.dispatcher import HandlerRegistry
from aios.kernel.runtime import KernelRuntime
from aios.kernel.syscall import SysCallHandler

#: Default time the kernel will wait for a syscall before treating
#: it as failed. `None` disables the timeout entirely.
DEFAULT_SYSCALL_TIMEOUT_SECONDS: float = 30.0


class Kernel:
    """
    Central dispatcher: validate, submit, wait, respond.

    `handle` follows one fixed sequence for every syscall type:

        1. Validate the request.
        2. Submit it as a `SysCall` (resolves + starts a handler
           thread).
        3. Wait for that syscall to finish.
        4. Return a normalized `AgentResponse`.

    A malformed request (not an `AgentRequest`) is a caller bug and
    raises `ValidationError` immediately. Everything else that can
    go wrong at or after dispatch -- an unsupported syscall, a
    handler that raises, a syscall that times out -- is reported as
    a `FAILED` `AgentResponse` rather than an exception, the same
    way a real syscall reports an error code instead of crashing
    its caller.
    """

    def __init__(
        self,
        runtime: KernelRuntime | None = None,
        timeout: float | None = DEFAULT_SYSCALL_TIMEOUT_SECONDS,
    ) -> None:
        self._runtime = runtime or KernelRuntime()
        self._timeout = timeout

    def handle(
        self,
        request: AgentRequest,
        agent_name: str | None = None,
    ) -> AgentResponse:
        """
        Dispatch `request` and return its `AgentResponse`.
        """

        if not isinstance(request, AgentRequest):
            raise ValidationError(
                "request must be an AgentRequest."
            )

        try:
            syscall = self._runtime.submit(
                request,
                agent_name=agent_name or str(request.agent_id),
            )
        except AIOSException as exc:
            return AgentResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error=str(exc),
            )

        completed = self._runtime.wait_for(
            syscall,
            timeout=self._timeout,
        )

        if not completed:
            return AgentResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error=(
                    f"Syscall '{request.syscall.call_type.value}' "
                    f"timed out after {self._timeout} seconds."
                ),
            )

        if syscall.error is not None:
            return AgentResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error=syscall.error,
            )

        return AgentResponse(
            request_id=request.request_id,
            status=RequestStatus.SUCCESS,
            result=syscall.response,
        )

    def register_handler(
        self,
        call_type: SystemCallType,
        handler: SysCallHandler,
    ) -> None:
        """
        Bind a handler for `call_type`, replacing any existing one.

        This is the seam Phase 3 uses to swap a mock handler for a
        real Memory/Storage/Tool/LLM implementation without any
        change to the kernel itself.
        """

        self._runtime.register_handler(call_type, handler)

    @property
    def handler_registry(self) -> HandlerRegistry:
        """
        Expose the underlying registry, mainly so tests can assert
        which syscalls are currently supported.
        """

        return self._runtime.registry
