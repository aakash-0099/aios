"""
Isolated, asynchronous, in-memory IPC transport.

This module is deliberately self-contained. It has never heard of
a `Kernel`, a `Dispatcher`, a syscall, or a `Scheduler` -- it only
knows `RequestID`, borrowed from `aios.core.ids` purely as an
opaque correlation key. That isolation is the point: `IPCChannel`
is a generic, reusable async mailbox that a future kernel, a future
scheduler, or a pair of agents can build request/response messaging
on top of, without this module ever depending on any of them.

Two operations are the whole surface:

    send(message)     -- place a message onto the channel
    receive(...)       -- take a message off the channel, either the
                           next one in FIFO order, or a specific one
                           by correlation id

Everything else here -- timeouts, closing, correlation bookkeeping,
the `request()` convenience -- exists to make those two operations
safe to use concurrently from many asyncio tasks at once.

Nothing here talks to a network, a file, or another process. It is
"IPC" in the sense of "a channel two independent pieces of code use
to pass messages," not in the operating-system sense of crossing a
process boundary. If a real IPC transport were ever needed, only
`send`/`receive`'s implementation would need to change; nothing
that depends on `IPCChannel`'s interface would.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Coroutine, Generic, TypeVar, cast

from aios.core.exceptions import AIOSException
from aios.core.ids import RequestID

T = TypeVar("T")


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class IPCError(AIOSException):
    """
    Base class for every error this module raises.
    """


class IPCTimeoutError(IPCError):
    """
    Raised when a `receive` (or `receive_correlated`, or `request`)
    call does not complete within its timeout.
    """


class IPCChannelClosedError(IPCError):
    """
    Raised by `send`, and by any `receive` call that is waiting (or
    starts waiting) on a channel that has been closed.
    """


# ---------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IPCMessage(Generic[T]):
    """
    A single message travelling through an `IPCChannel`.

    `correlation_id` is what makes request/response possible over
    an otherwise unordered, shared channel: a caller sends a message
    tagged with a fresh id, then asks `receive_correlated` for a
    message carrying that same id -- however many unrelated messages
    other callers send in between.

    `payload` is intentionally untyped from this module's point of
    view (`Generic[T]`): `IPCChannel` never inspects it, so it can
    carry a plain dict in a test, or later an `AgentRequest` /
    `AgentResponse`, without `ipc.py` changing either way.
    """

    correlation_id: RequestID
    payload: T
    sender: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(cls, payload: T, sender: str | None = None) -> "IPCMessage[T]":
        """
        Build a message with a freshly generated correlation id.

        Convenience for the common case: a caller does not have an
        existing id to correlate against yet, it is originating the
        conversation.
        """

        return cls(
            correlation_id=RequestID.generate(),
            payload=payload,
            sender=sender,
        )


# ---------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------



class IPCChannel(Generic[T]):
    """
    An isolated, in-memory, asyncio-native message channel.

    A channel is a single shared mailbox: any number of asyncio
    tasks may `send` onto it, and any number may `receive` from it.
    Messages are FIFO by default. `receive_correlated` instead waits
    for one specific message (by correlation id), leaving everything
    else on the channel undisturbed for other callers -- this is
    what lets a request/response exchange share a channel with
    unrelated traffic.

    A channel exists entirely in this process' memory and is closed
    exactly once. After closing:
        - `send` always raises `IPCChannelClosedError`.
        - `receive` / `receive_correlated` still return anything
          already queued or already matched, but once nothing is
          left to return, they raise `IPCChannelClosedError` too,
          including to a call already waiting when `close()` runs.

    Not thread-safe: an `IPCChannel` is meant to be used from
    asyncio tasks on a single event loop, matching everything else
    in this module.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[IPCMessage[T]] = asyncio.Queue()
        self._waiters: dict[RequestID, asyncio.Future[IPCMessage[T]]] = {}
        self._closed_event = asyncio.Event()

    # -- send -----------------------------------------------------

    async def send(self, message: IPCMessage[T]) -> None:
        """
        Place `message` onto the channel.

        If a task is already blocked in `receive_correlated` for
        this message's correlation id, the message is handed to it
        directly instead of going through the general queue -- so a
        correlated waiter is never starved by unrelated traffic sent
        afterwards, and does not need to scan the whole queue to
        find its reply.

        Raises `IPCChannelClosedError` if the channel is closed.

        Synchronous below the first `await`: the waiter lookup and
        hand-off happen without ever yielding to the event loop, so
        no lock is needed to keep them atomic with respect to other
        tasks on the same loop.
        """

        if self._closed_event.is_set():
            raise IPCChannelClosedError("Cannot send on a closed channel.")

        waiter = self._waiters.pop(message.correlation_id, None)

        if waiter is not None and not waiter.done():
            waiter.set_result(message)
            return

        await self._queue.put(message)

    # -- receive ----------------------------------------------------

    async def receive(self, timeout: float | None = None) -> IPCMessage[T]:
        """
        Take the next message off the channel, in FIFO order.

        `timeout` is in seconds; `None` (the default) waits
        indefinitely. Raises `IPCTimeoutError` if `timeout` elapses
        first, or `IPCChannelClosedError` if the channel is (or
        becomes) closed with nothing left to deliver.
        """

        return await self._race_against_close(self._queue.get(), timeout)

    async def receive_correlated(
        self,
        correlation_id: RequestID,
        timeout: float | None = None,
        *,
        _exclude: "IPCMessage[T] | None" = None,
    ) -> IPCMessage[T]:
        """
        Wait for a message whose `correlation_id` matches, regardless
        of send order or what else arrives on the channel meanwhile.

        A matching message already sitting in the general queue
        (sent before anyone asked to correlate on it) is returned
        immediately; everything else in the queue is left exactly as
        it was, in its original order. Otherwise this registers a
        one-shot waiter that `send` will resolve directly the moment
        a matching message arrives.

        `_exclude` skips one specific message instance during the
        queue scan even if its id matches. This is for `request()`:
        without it, a caller waiting on the same id it just sent a
        message with can end up matching its own outbound message
        instead of the reply, if the reply hasn't arrived yet and
        its own message is still sitting, unconsumed, in the queue.
        """

        match = self._pop_matching_from_queue(correlation_id, exclude=_exclude)
        if match is not None:
            return match

        if self._closed_event.is_set():
            raise IPCChannelClosedError("Cannot receive on a closed channel.")

        future: asyncio.Future[IPCMessage[T]] = (
            asyncio.get_running_loop().create_future()
        )
        self._waiters[correlation_id] = future

        try:
            return await self._race_against_close(future, timeout)
        finally:
            self._waiters.pop(correlation_id, None)

    # -- request/response convenience --------------------------------

    async def request(
        self,
        payload: T,
        sender: str | None = None,
        timeout: float | None = None,
    ) -> IPCMessage[T]:
        """
        Send `payload` as a fresh message and wait for its reply.

        Equivalent to building an `IPCMessage.new(payload, sender)`,
        sending it, and calling `receive_correlated` on the id it
        was given -- the shape a kernel or an agent would actually
        use this channel for: send a request, wait for the response
        carrying the same correlation id, whatever else is on the
        channel in between.
        """

        message = IPCMessage.new(payload=payload, sender=sender)
        await self.send(message)
        return await self.receive_correlated(
            message.correlation_id,
            timeout=timeout,
            _exclude=message,
        )

    # -- lifecycle ----------------------------------------------------

    async def close(self) -> None:
        """
        Close the channel.

        Idempotent: closing an already-closed channel does nothing.
        Every task currently blocked in `receive` / `receive_correlated`
        is woken immediately with `IPCChannelClosedError`, rather than
        left hanging until its timeout.
        """

        if self._closed_event.is_set():
            return

        self._closed_event.set()

        waiters = list(self._waiters.values())
        self._waiters.clear()

        for future in waiters:
            if not future.done():
                future.set_exception(
                    IPCChannelClosedError("Channel was closed while waiting.")
                )

    async def __aenter__(self) -> "IPCChannel[T]":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    @property
    def is_closed(self) -> bool:
        """
        Whether `close()` has been called.
        """

        return self._closed_event.is_set()

    def pending_count(self) -> int:
        """
        Number of messages currently sitting in the general queue.

        Does not count messages already claimed by a correlated
        waiter (those never touch the general queue).
        """

        return self._queue.qsize()

    # -- internals ----------------------------------------------------

    def _pop_matching_from_queue(
        self,
        correlation_id: RequestID,
        *,
        exclude: "IPCMessage[T] | None" = None,
    ) -> IPCMessage[T] | None:
        """
        Remove and return the first queued message matching
        `correlation_id`, if any, restoring the order of everything
        else that was in the queue. `exclude`, if given, is skipped
        by identity even if it matches -- see `receive_correlated`.
        """

        drained: list[IPCMessage[T]] = []
        match: IPCMessage[T] | None = None

        while not self._queue.empty():
            item = self._queue.get_nowait()
            if (
                match is None
                and item.correlation_id == correlation_id
                and item is not exclude
            ):
                match = item
            else:
                drained.append(item)

        for item in drained:
            self._queue.put_nowait(item)

        return match

    async def _race_against_close(
        self,
        awaitable: (
            "asyncio.Future[IPCMessage[T]] "
            "| Coroutine[object, object, IPCMessage[T]]"
        ),
        timeout: float | None,
    ) -> IPCMessage[T]:
        """
        Await `awaitable`, but stop early -- with the right error --
        if `timeout` elapses or the channel is closed first.

        This is the single mechanism behind both `receive`'s and
        `receive_correlated`'s timeout/closed handling: race the
        real wait against `self._closed_event`, honor `timeout`
        across both, and clean up whichever side did not win.
        """

        target = (
            awaitable
            if isinstance(awaitable, asyncio.Future)
            else asyncio.ensure_future(awaitable)
        )
        # `Event.wait()` resolves to `bool`, not `IPCMessage[T]`. We
        # never call `.result()` on this future -- only check set
        # membership and `.cancel()` it -- so the cast is safe: it
        # only tells the type checker to treat it like the other
        # side of `asyncio.wait()`'s set, which requires one shared
        # future type, not a claim about what it actually resolves to.
        closed_wait = cast(
            "asyncio.Future[IPCMessage[T]]",
            asyncio.ensure_future(self._closed_event.wait()),
        )

        done, pending = await asyncio.wait(
            {target, closed_wait},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if target in done:
            return target.result()

        if closed_wait in done:
            raise IPCChannelClosedError("Channel was closed while waiting.")

        raise IPCTimeoutError(f"No message received within {timeout} seconds.")


