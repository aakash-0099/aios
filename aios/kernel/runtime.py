"""
System-call submission and tracking.

`KernelRuntime` is the thin layer the `Kernel` talks to. It no
longer resolves handlers or starts threads itself -- that work
belongs to `Dispatcher`, which owns the syscall queue and the
worker pool that actually executes them. `KernelRuntime` exists to:

    1. Own (or accept) a `Dispatcher`.
    2. Forward `submit` to it.
    3. Track which syscalls are in flight, for introspection.
    4. Provide `wait_for`, mirroring the old thread-per-syscall
       API so `Kernel` did not have to change.

This is also the seam a future Redis-backed queue plugs into:
`Kernel` and `KernelRuntime` only ever depend on `Dispatcher`'s
public interface (`submit`, `register_handler`, `registry`,
`close`), never on how the queue itself is implemented.
"""

from __future__ import annotations

from threading import Lock

from aios.core.models import AgentRequest, SystemCallType
from aios.kernel.dispatcher import Dispatcher, HandlerRegistry
from aios.kernel.syscall import SysCall, SysCallHandler

#: Default agent name used until a real Agent Manager supplies one.
_UNKNOWN_AGENT = "unknown-agent"

#: Default number of dispatcher workers when none is specified.
DEFAULT_WORKER_COUNT = 4


class KernelRuntime:
    """
    Owns a `Dispatcher` and tracks syscalls submitted through it.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        dispatcher: Dispatcher | None = None,
        worker_count: int = DEFAULT_WORKER_COUNT,
    ) -> None:
        if dispatcher is not None and registry is not None:
            raise ValueError(
                "Pass either a dispatcher or a registry, not both: "
                "a dispatcher already owns its own registry."
            )

        self._dispatcher = dispatcher or Dispatcher(
            registry=registry,
            worker_count=worker_count,
        )

        self._active: dict[int, SysCall] = {}
        self._lock = Lock()

    def register_handler(
        self,
        call_type: SystemCallType,
        handler: SysCallHandler,
    ) -> None:
        """
        Bind a handler for `call_type`, replacing any existing one.

        This is the mechanism Phase 3 uses to swap a mock handler
        for a real Memory/Storage/Tool/LLM implementation without
        the kernel changing at all.
        """

        self._dispatcher.register_handler(call_type, handler)

    def submit(
        self,
        request: AgentRequest,
        agent_name: str = _UNKNOWN_AGENT,
    ) -> SysCall:
        """
        Enqueue `request` on the dispatcher and track it as active.

        Raises `SystemCallError` (via the dispatcher's registry
        resolution) if no handler is registered for the request's
        syscall type. The syscall is only tracked once resolution
        succeeds and it has actually been queued.
        """

        syscall = self._dispatcher.submit(request, agent_name)

        with self._lock:
            self._active[syscall.pid] = syscall

        return syscall

    def wait_for(
        self,
        syscall: SysCall,
        timeout: float | None = None,
    ) -> bool:
        """
        Block until `syscall` finishes or `timeout` elapses.

        Returns `True` if it finished in time. Either way, once the
        wait is over the syscall is dropped from the active table.
        A timed-out syscall is left for its dispatcher worker to
        finish in its own time; it is simply no longer tracked here.
        """

        finished = syscall.wait_for_completion(timeout)

        with self._lock:
            self._active.pop(syscall.pid, None)

        return finished

    def active_count(self) -> int:
        """
        Number of syscalls currently submitted and not yet awaited.
        """

        with self._lock:
            return len(self._active)

    def is_active(self, pid: int) -> bool:
        """
        Whether a syscall with the given pid is still tracked.
        """

        with self._lock:
            return pid in self._active

    @property
    def registry(self) -> HandlerRegistry:
        """
        The handler registry backing this runtime's dispatcher.
        """

        return self._dispatcher.registry

    @property
    def dispatcher(self) -> Dispatcher:
        """
        The underlying dispatcher, mainly for tests and diagnostics
        (e.g. inspecting `queue_size()`).
        """

        return self._dispatcher

    def close(self) -> None:
        """
        Shut down the underlying dispatcher's worker threads.
        """

        self._dispatcher.close()
