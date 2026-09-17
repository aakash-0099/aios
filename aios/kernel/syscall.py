"""
System-call execution unit.

Phase 2 binds each dispatched AgentRequest to its own thread.
A SysCall represents one independent unit of work. The dispatcher
decides when it should start, while SysCall owns execution state,
completion signaling, response/error state, and lifecycle tracking.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

from aios.core.models import AgentRequest
from aios.kernel.lifecycle import SysCallLifecycle


# Monotonically increasing syscall identifiers.
# Unique for the lifetime of this Python process.
_pid_sequence = itertools.count(1)


SysCallHandler = Callable[[AgentRequest], Any]


class SysCall(Thread):
    """
    A single system call executing on its own thread.

    The dispatcher creates and queues the SysCall.
    A dispatcher worker starts the SysCall when it is ready.

    The SysCall:
        - executes the bound handler
        - records lifecycle information
        - stores the response
        - stores any handler error
        - signals completion through Event
    """

    def __init__(
        self,
        agent_name: str,
        request: AgentRequest,
        handler: SysCallHandler | None = None,
    ) -> None:
        super().__init__(daemon=True)

        self.agent_name = agent_name
        self.request = request
        self.handler = handler

        self.event: Event = Event()

        self.pid: int = next(_pid_sequence)

        self.lifecycle: SysCallLifecycle = SysCallLifecycle()

        self.response: Any = None
        self.error: str | None = None

    def bind_handler(
        self,
        handler: SysCallHandler,
    ) -> None:
        """
        Bind the handler that will execute this syscall.

        This is normally done by the dispatcher before the syscall
        is started.
        """

        if self.handler is not None:
            raise RuntimeError(
                "A handler is already bound to this syscall."
            )

        self.handler = handler

    def run(self) -> None:
        """
        Execute the bound handler.

        This method runs on the SysCall's own thread.
        Exceptions are captured so that the completion Event is
        always signaled.
        """

        self.lifecycle.mark_started()

        try:
            if self.handler is None:
                raise RuntimeError(
                    "No handler is bound to this syscall."
                )

            self.response = self.handler(self.request)

        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.lifecycle.mark_failed()

        else:
            self.lifecycle.mark_completed()

        finally:
            # The Kernel may be waiting on this event.
            self.event.set()

    def wait_for_completion(
        self,
        timeout: float | None = None,
    ) -> bool:
        """
        Block until the syscall finishes or timeout expires.

        Returns:
            True  -> syscall completed
            False -> timeout occurred
        """

        return self.event.wait(timeout)

    @property
    def succeeded(self) -> bool:
        """
        Whether the syscall completed successfully.
        """

        return (
            self.lifecycle.is_finished
            and self.error is None
        )