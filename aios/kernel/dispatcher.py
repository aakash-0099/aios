"""
AIOS Phase 2 dispatcher.

The dispatcher is responsible for:

1. Maintaining the syscall -> handler registry.
2. Creating SysCall objects and tracking them by pid while they
   are in flight.
3. Placing a lightweight, serializable envelope for each syscall
   onto a queue backend (in-memory today, optionally Redis).
4. Having worker threads pick queued envelopes, resolve the real
   SysCall by pid, resolve its handler, start it, and wait for it
   to finish before taking another.

Handler resolution happens on the worker side, after an envelope
comes off the queue -- not at submit time. This is what makes the
queue backend swappable for something like Redis: the thing that
travels through the queue is plain, serializable data (pid, agent
name, flattened request), never a bound Python callable or a live
Thread object, neither of which Redis (or any external transport)
could carry.

This is intentionally a small Phase 2 dispatcher. It is NOT the
real Phase 5 Scheduler. Phase 5 will replace/extend this mechanism
with resource-specific queues and FIFO/Round-Robin scheduling
strategies.
"""

from __future__ import annotations

import time
from threading import Event, Lock, Thread

from aios.core.exceptions import SystemCallError
from aios.core.models import AgentRequest, SystemCallType
from aios.kernel.queue_backend import (
    InMemoryQueueBackend,
    SysCallEnvelope,
    SysCallQueueBackend,
)
from aios.kernel.serialization import request_to_dict
from aios.kernel.syscall import SysCall, SysCallHandler

#: How often a worker re-checks its stop event while the queue
#: backend has nothing for it. Bounds shutdown latency.
WORKER_POLL_SECONDS = 0.2


class HandlerRegistry:
    """
    Maps a SystemCallType to the handler that executes it.

    The Kernel never knows how a syscall is implemented.
    """

    def __init__(self) -> None:
        self._handlers: dict[SystemCallType, SysCallHandler] = {}
        self._lock = Lock()

    def register(self, call_type: SystemCallType, handler: SysCallHandler) -> None:
        """
        Register or replace a handler.
        """

        if not isinstance(call_type, SystemCallType):
            raise SystemCallError("call_type must be a SystemCallType.")

        if not callable(handler):
            raise SystemCallError("handler must be callable.")

        with self._lock:
            self._handlers[call_type] = handler

    def resolve(self, call_type: SystemCallType) -> SysCallHandler:
        """
        Resolve the handler for a syscall type.
        """

        with self._lock:
            try:
                return self._handlers[call_type]
            except KeyError as exc:
                raise SystemCallError(
                    f"No handler registered for syscall '{call_type.value}'."
                ) from exc

    def is_registered(self, call_type: SystemCallType) -> bool:
        """
        Return whether a handler is registered.
        """

        with self._lock:
            return call_type in self._handlers


# ---------------------------------------------------------------------
# Mock handlers
# ---------------------------------------------------------------------


MOCK_HANDLER_DELAY_SECONDS = 0.05


def _mock_llm_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for LLM_CALL until Phase 3.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    prompt = request.syscall.payload.get("prompt")
    return {
        "handler": "mock_llm",
        "completion": f"[mock completion for: {prompt!r}]",
    }


def _mock_memory_read_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for MEMORY_READ until the Memory Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    resource_id = request.syscall.payload.get("resource_id")
    return {
        "handler": "mock_memory_read",
        "resource_id": resource_id,
        "content": None,
    }


def _mock_memory_write_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for MEMORY_WRITE until the Memory Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    return {"handler": "mock_memory_write", "acknowledged": True}


def _mock_storage_read_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for STORAGE_READ until the Storage Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    resource_id = request.syscall.payload.get("resource_id")
    return {
        "handler": "mock_storage_read",
        "resource_id": resource_id,
        "content": None,
    }


def _mock_storage_write_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for STORAGE_WRITE until the Storage Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    return {"handler": "mock_storage_write", "acknowledged": True}


def _mock_tool_call_handler(request: AgentRequest) -> dict[str, object]:
    """
    Stand-in for TOOL_CALL until the Tool Manager exists.
    """

    time.sleep(MOCK_HANDLER_DELAY_SECONDS)
    tool_name = request.syscall.payload.get("tool_name")
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
    registry.register(SystemCallType.LLM_CALL, _mock_llm_handler)
    registry.register(SystemCallType.MEMORY_READ, _mock_memory_read_handler)
    registry.register(SystemCallType.MEMORY_WRITE, _mock_memory_write_handler)
    registry.register(SystemCallType.STORAGE_READ, _mock_storage_read_handler)
    registry.register(SystemCallType.STORAGE_WRITE, _mock_storage_write_handler)
    registry.register(SystemCallType.TOOL_CALL, _mock_tool_call_handler)
    return registry


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


class Dispatcher:
    """
    Phase 2 syscall dispatcher.

    It owns:

        the syscall queue backend
        the handler registry
        the table of syscalls currently in flight, by pid
        the dispatcher worker threads

    It deliberately does NOT implement FIFO/RR scheduling policies.
    Those belong to Phase 5.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        worker_count: int = 4,
        queue_backend: SysCallQueueBackend | None = None,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than zero.")

        self._registry = registry or build_default_registry()
        self._queue_backend = queue_backend or InMemoryQueueBackend()
        self._pending: dict[int, SysCall] = {}
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

    def submit(self, request: AgentRequest, agent_name: str) -> SysCall:
        """
        Create a SysCall, track it by pid, and enqueue its envelope.

        The SysCall starts out unbound -- no handler yet. Only a
        plain, serializable envelope (`pid`, `agent_name`, and the
        request flattened to a dict) goes on the queue. A worker
        resolves the actual handler after dequeuing, which is what
        lets the queue itself become Redis without ever asking a
        queue backend to carry a live callable or Thread object.
        """

        syscall = SysCall(agent_name=agent_name, request=request, handler=None)

        with self._lock:
            self._pending[syscall.pid] = syscall

        envelope = SysCallEnvelope(
            pid=syscall.pid,
            agent_name=agent_name,
            request=request_to_dict(request),
        )
        self._queue_backend.put(envelope)

        return syscall

    def _worker_loop(self) -> None:
        """
        Dispatcher worker loop.

        Each worker:

            1. gets an envelope from the queue backend
            2. looks up the live SysCall it refers to
            3. resolves and binds its handler
            4. starts the SysCall thread
            5. waits for that SysCall to finish
            6. takes the next envelope

        Multiple workers allow multiple syscalls to execute
        concurrently, bounded by `worker_count`.
        """

        while not self._stop_event.is_set():
            envelope = self._queue_backend.get(timeout=WORKER_POLL_SECONDS)

            if envelope is None:
                continue

            self._execute(envelope)

    def _execute(self, envelope: SysCallEnvelope) -> None:
        with self._lock:
            syscall = self._pending.pop(envelope.pid, None)

        if syscall is None:
            # Nothing local is waiting on this pid -- most likely
            # an envelope left over from a previous process using
            # the same external queue. Nothing to report it to, so
            # it is silently dropped rather than executed blind.
            return

        try:
            handler = self._registry.resolve(syscall.request.syscall.call_type)
        except SystemCallError as exc:
            syscall.lifecycle.mark_started()
            syscall.error = str(exc)
            syscall.lifecycle.mark_failed()
            syscall.event.set()
            return

        syscall.bind_handler(handler)
        syscall.start()
        syscall.join()

    def register_handler(self, call_type: SystemCallType, handler: SysCallHandler) -> None:
        """
        Register or replace a syscall handler.
        """

        self._registry.register(call_type, handler)

    @property
    def registry(self) -> HandlerRegistry:
        """
        Return the handler registry.
        """

        return self._registry

    def queue_size(self) -> int:
        """
        Number of envelopes waiting in the dispatcher's queue backend.
        """

        return self._queue_backend.size()

    def close(self) -> None:
        """
        Stop dispatcher workers gracefully and close the queue backend.

        Already-dequeued syscalls are allowed to finish; workers
        notice the stop event within `WORKER_POLL_SECONDS` and exit.
        """

        with self._lock:
            if not self._started:
                return

            self._stop_event.set()
            self._started = False

        for worker in self._workers:
            worker.join(timeout=WORKER_POLL_SECONDS * 2 + 1.0)

        self._workers.clear()
        self._queue_backend.close()
