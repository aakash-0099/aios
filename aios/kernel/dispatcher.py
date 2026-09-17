"""
AIOS Phase 2 dispatcher.

The dispatcher is responsible for:

1. Maintaining the syscall -> handler registry.
2. Creating SysCall objects.
3. Placing SysCalls into a queue.
4. Having worker threads pick queued SysCalls.
5. Starting the SysCall.
6. Waiting for that SysCall to finish before the worker
   takes another queued syscall.

This is intentionally a small Phase 2 dispatcher.

It is NOT the real Phase 5 Scheduler.

Phase 5 will replace/extend this mechanism with resource-specific
queues and FIFO/Round-Robin scheduling strategies.
"""

from __future__ import annotations

import time
from queue import Queue
from threading import Event, Lock, Thread

from aios.core.exceptions import SystemCallError
from aios.core.models import AgentRequest, SystemCallType
from aios.kernel.syscall import SysCall, SysCallHandler


class HandlerRegistry:
    """
    Maps a SystemCallType to the handler that executes it.

    The Kernel never knows how a syscall is implemented.
    """

    def __init__(self) -> None:
        self._handlers: dict[
            SystemCallType,
            SysCallHandler,
        ] = {}

        self._lock = Lock()

    def register(
        self,
        call_type: SystemCallType,
        handler: SysCallHandler,
    ) -> None:
        """
        Register or replace a handler.
        """

        if not isinstance(call_type, SystemCallType):
            raise SystemCallError(
                "call_type must be a SystemCallType."
            )

        if not callable(handler):
            raise SystemCallError(
                "handler must be callable."
            )

        with self._lock:
            self._handlers[call_type] = handler

    def resolve(
        self,
        call_type: SystemCallType,
    ) -> SysCallHandler:
        """
        Resolve the handler for a syscall type.
        """

        with self._lock:
            try:
                return self._handlers[call_type]

            except KeyError as exc:
                raise SystemCallError(
                    f"No handler registered for syscall "
                    f"'{call_type.value}'."
                ) from exc

    def is_registered(
        self,
        call_type: SystemCallType,
    ) -> bool:
        """
        Return whether a handler is registered.
        """

        with self._lock:
            return call_type in self._handlers


# ---------------------------------------------------------------------
# Mock handlers
# ---------------------------------------------------------------------


MOCK_HANDLER_DELAY_SECONDS = 0.05


def _mock_llm_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for LLM_CALL until Phase 3.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    prompt = request.syscall.payload.get("prompt")

    return {
        "handler": "mock_llm",
        "completion": f"[mock completion for: {prompt!r}]",
    }


def _mock_memory_read_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for MEMORY_READ until the Memory Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    resource_id = request.syscall.payload.get(
        "resource_id"
    )

    return {
        "handler": "mock_memory_read",
        "resource_id": resource_id,
        "content": None,
    }


def _mock_memory_write_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for MEMORY_WRITE until the Memory Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    return {
        "handler": "mock_memory_write",
        "acknowledged": True,
    }


def _mock_storage_read_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for STORAGE_READ until the Storage Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    resource_id = request.syscall.payload.get(
        "resource_id"
    )

    return {
        "handler": "mock_storage_read",
        "resource_id": resource_id,
        "content": None,
    }


def _mock_storage_write_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for STORAGE_WRITE until the Storage Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    return {
        "handler": "mock_storage_write",
        "acknowledged": True,
    }


def _mock_tool_call_handler(
    request: AgentRequest,
) -> dict[str, object]:
    """
    Stand-in for TOOL_CALL until the Tool Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)

    tool_name = request.syscall.payload.get(
        "tool_name"
    )

    return {
        "handler": "mock_tool_call",
        "tool_name": tool_name,
        "output": None,
    }


def build_default_registry() -> HandlerRegistry:
    """
    Build the default Phase 2 mock handler registry.
    """

    registry = HandlerRegistry()

    registry.register(
        SystemCallType.LLM_CALL,
        _mock_llm_handler,
    )

    registry.register(
        SystemCallType.MEMORY_READ,
        _mock_memory_read_handler,
    )

    registry.register(
        SystemCallType.MEMORY_WRITE,
        _mock_memory_write_handler,
    )

    registry.register(
        SystemCallType.STORAGE_READ,
        _mock_storage_read_handler,
    )

    registry.register(
        SystemCallType.STORAGE_WRITE,
        _mock_storage_write_handler,
    )

    registry.register(
        SystemCallType.TOOL_CALL,
        _mock_tool_call_handler,
    )

    return registry


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


class Dispatcher:
    """
    Phase 2 syscall dispatcher.

    It owns:

        request queue
        handler registry
        dispatcher worker threads

    It deliberately does NOT implement FIFO/RR scheduling policies.
    Those belong to Phase 5.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        worker_count: int = 4,
    ) -> None:

        if worker_count <= 0:
            raise ValueError(
                "worker_count must be greater than zero."
            )

        self._registry = (
            registry or build_default_registry()
        )

        self._queue: Queue[SysCall | None] = Queue()

        self._worker_count = worker_count

        self._workers: list[Thread] = []

        self._stop_event = Event()

        self._started = False

        self._lock = Lock()

        self.start()

    def start(self) -> None:
        """
        Start dispatcher worker threads.
        """

        with self._lock:

            if self._started:
                return

            self._started = True

            for index in range(self._worker_count):

                worker = Thread(
                    target=self._worker_loop,
                    name=f"aios-dispatcher-{index + 1}",
                    daemon=True,
                )

                self._workers.append(worker)

                worker.start()

    def submit(
        self,
        request: AgentRequest,
        agent_name: str,
    ) -> SysCall:
        """
        Create and enqueue a SysCall.

        Handler resolution happens before the syscall is queued.
        This preserves the existing behavior where an unsupported
        syscall fails immediately during submission.
        """

        handler = self._registry.resolve(
            request.syscall.call_type
        )

        syscall = SysCall(
            agent_name=agent_name,
            request=request,
            handler=handler,
        )

        self._queue.put(syscall)

        return syscall

    def _worker_loop(self) -> None:
        """
        Dispatcher worker loop.

        Each worker:

            1. gets a syscall from the queue
            2. starts the SysCall thread
            3. waits for that SysCall to finish
            4. takes the next syscall

        Multiple workers allow multiple syscalls to execute
        concurrently.
        """

        while not self._stop_event.is_set():

            syscall = self._queue.get()

            try:

                if syscall is None:
                    return

                syscall.start()

                # The worker owns one execution slot until
                # this syscall completes.
                syscall.join()

            finally:

                self._queue.task_done()

    def register_handler(
        self,
        call_type: SystemCallType,
        handler: SysCallHandler,
    ) -> None:
        """
        Register or replace a syscall handler.
        """

        self._registry.register(
            call_type,
            handler,
        )

    @property
    def registry(self) -> HandlerRegistry:
        """
        Return the handler registry.
        """

        return self._registry

    def queue_size(self) -> int:
        """
        Number of syscalls waiting in the dispatcher queue.
        """

        return self._queue.qsize()

    def close(self) -> None:
        """
        Stop dispatcher workers gracefully.

        Already queued syscalls are allowed to finish before
        workers consume the shutdown sentinels.
        """

        with self._lock:

            if not self._started:
                return

            self._stop_event.set()

            for _ in self._workers:
                self._queue.put(None)

            self._started = False

        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()