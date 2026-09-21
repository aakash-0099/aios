"""
Pluggable transport for queued syscalls.

`Dispatcher` does not know or care what carries a syscall from
`submit()` to a worker thread. It only knows `SysCallQueueBackend`:
put an envelope in, get one out, ask how many are waiting. Today
that's an in-process `queue.Queue`. It can become Redis -- or
anything else -- without `Dispatcher`, `Kernel`, or `KernelRuntime`
changing, the same way `HandlerRegistry` lets Phase 3 swap a mock
handler for a real one without the kernel changing.

Design note -- what Redis does and does not buy you here:
    `RedisQueueBackend` makes the queue itself external: durable
    across a process restart, inspectable from outside the app,
    and shareable if you ever run more than one dispatcher. What it
    does *not* do, by itself, is let a syscall be *executed* by a
    different process than the one that submitted it. `Dispatcher`
    still resolves handlers and runs the syscall thread in the same
    process that called `submit()`, and the caller's `SysCall.event`
    -- an in-memory `threading.Event` -- cannot be observed from
    another process. Genuinely distributed workers need a different
    result-delivery mechanism (e.g. a Redis response channel keyed
    by request id) layered on top of this; that is future work, not
    something this backend claims to solve.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Any


@dataclass(frozen=True)
class SysCallEnvelope:
    """
    The wire-level unit a queue backend actually carries.

    `pid` lets the worker that dequeues this envelope find the
    live `SysCall` object waiting in the dispatcher's pending
    table; `request` is that syscall's request, already flattened
    by `aios.kernel.serialization.request_to_dict`.
    """

    pid: int
    agent_name: str
    request: dict[str, Any]


class SysCallQueueBackend(ABC):
    """
    Minimal contract a syscall queue transport must satisfy.
    """

    @abstractmethod
    def put(self, envelope: SysCallEnvelope) -> None:
        """
        Enqueue an envelope.
        """

    @abstractmethod
    def get(self, timeout: float) -> SysCallEnvelope | None:
        """
        Dequeue the next envelope, or `None` if `timeout` elapses.

        A `None` return always means "nothing arrived in time," so
        that a worker can use it purely to re-check its stop event
        and loop again -- it is never used as a shutdown signal.
        """

    def size(self) -> int:
        """
        Best-effort count of envelopes currently waiting.
        """

        return 0

    def close(self) -> None:
        """
        Release any resources held by this backend (connections,
        threads). The default implementation has nothing to do.
        """


class InMemoryQueueBackend(SysCallQueueBackend):
    """
    In-process FIFO queue. Phase 2's original, still the default.
    """

    def __init__(self) -> None:
        self._queue: Queue[SysCallEnvelope] = Queue()

    def put(self, envelope: SysCallEnvelope) -> None:
        self._queue.put(envelope)

    def get(self, timeout: float) -> SysCallEnvelope | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def size(self) -> int:
        return self._queue.qsize()


class RedisQueueBackend(SysCallQueueBackend):
    """
    Redis-backed FIFO queue, using a single Redis list as the
    channel: `LPUSH` to enqueue, `BRPOP` to dequeue.

    Requires the `redis` package and a reachable Redis server. The
    connection is not established until construction, so a bad
    `redis_url` fails fast with the underlying `redis` exception
    rather than surfacing later as a mysterious dispatcher hang.
    """

    def __init__(
        self,
        redis_url: str,
        queue_key: str = "aios:syscalls",
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without the dependency
            raise ImportError(
                "RedisQueueBackend requires the 'redis' package. "
                "Install it with: pip install redis"
            ) from exc

        self._queue_key = queue_key
        self._client = redis.Redis.from_url(redis_url)
        # Fail fast on a bad URL/unreachable server instead of
        # deferring the error to the first put()/get().
        self._client.ping()

    def put(self, envelope: SysCallEnvelope) -> None:
        self._client.lpush(self._queue_key, json.dumps(asdict(envelope)))

    def get(self, timeout: float) -> SysCallEnvelope | None:
        # BRPOP's timeout is whole seconds on older Redis servers;
        # round up so a fractional poll interval still blocks
        # rather than returning immediately.
        redis_timeout = max(1, int(timeout + 0.999))
        result = self._client.brpop(self._queue_key, timeout=redis_timeout)

        if result is None:
            return None

        _key, raw_value = result
        data = json.loads(raw_value)
        return SysCallEnvelope(
            pid=data["pid"],
            agent_name=data["agent_name"],
            request=data["request"],
        )

    def size(self) -> int:
        return self._client.llen(self._queue_key)

    def close(self) -> None:
        self._client.close()
