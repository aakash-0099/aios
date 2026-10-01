"""
Tests for the Redis-backed syscall queue.

These are skipped automatically if the `redis` package is not
installed, or if no Redis server is reachable at `REDIS_URL` (or
`redis://localhost:6379/0` by default) -- Redis is an optional
backend, not a requirement to run the rest of the test suite.

Each test uses its own queue key so parallel test runs (or a
lingering queue from a previous run) cannot bleed into each other.
"""

from __future__ import annotations

import os
import uuid

import pytest

redis = pytest.importorskip("redis")

from aios.core.models import AgentRequest, SystemCall, SystemCallType  # noqa: E402
from aios.core.ids import AgentID, RequestID, TaskID  # noqa: E402
from aios.kernel.dispatcher import Dispatcher, HandlerRegistry  # noqa: E402
from aios.kernel.queue_backend import RedisQueueBackend, SysCallEnvelope  # noqa: E402

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _redis_is_reachable() -> bool:
    try:
        client = redis.Redis.from_url(REDIS_URL)
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001 - any connection failure means "skip"
        return False


pytestmark = pytest.mark.skipif(
    not _redis_is_reachable(),
    reason=f"No Redis server reachable at {REDIS_URL!r}; skipping Redis-backed queue tests.",
)


def _unique_key() -> str:
    return f"aios:test:{uuid.uuid4()}"


def _build_request(
    call_type: SystemCallType = SystemCallType.LLM_CALL,
    payload: dict[str, object] | None = None,
) -> AgentRequest:
    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(call_type=call_type, payload=payload or {}),
    )


def test_redis_backend_fails_fast_on_bad_url():
    with pytest.raises(Exception):
        RedisQueueBackend(redis_url="redis://localhost:1/0")


def test_redis_backend_put_then_get_round_trips():
    backend = RedisQueueBackend(redis_url=REDIS_URL, queue_key=_unique_key())

    try:
        envelope = SysCallEnvelope(pid=1, agent_name="agent-001", request={"call_type": "llm_call"})
        backend.put(envelope)

        assert backend.size() == 1
        assert backend.get(timeout=1) == envelope
        assert backend.size() == 0
    finally:
        backend.close()


def test_redis_backend_get_times_out_with_none():
    backend = RedisQueueBackend(redis_url=REDIS_URL, queue_key=_unique_key())

    try:
        assert backend.get(timeout=1) is None
    finally:
        backend.close()


def test_redis_backend_is_fifo():
    backend = RedisQueueBackend(redis_url=REDIS_URL, queue_key=_unique_key())

    try:
        first = SysCallEnvelope(pid=1, agent_name="a", request={})
        second = SysCallEnvelope(pid=2, agent_name="a", request={})

        backend.put(first)
        backend.put(second)

        assert backend.get(timeout=1) == first
        assert backend.get(timeout=1) == second
    finally:
        backend.close()


def test_dispatcher_wired_to_redis_executes_syscalls_correctly():
    """
    End-to-end: a real Dispatcher, with a real Redis list as its
    queue, resolving handlers after dequeue and running them.
    """

    registry = HandlerRegistry()
    registry.register(
        SystemCallType.LLM_CALL,
        lambda req: {"echo": req.syscall.payload.get("n")},
    )

    backend = RedisQueueBackend(redis_url=REDIS_URL, queue_key=_unique_key())
    dispatcher = Dispatcher(registry=registry, worker_count=3, queue_backend=backend)

    try:
        syscalls = [
            dispatcher.submit(_build_request(payload={"n": i}), "agent-x")
            for i in range(10)
        ]

        for syscall in syscalls:
            assert syscall.wait_for_completion(timeout=5) is True

        assert all(syscall.succeeded for syscall in syscalls)
        assert [syscall.response["echo"] for syscall in syscalls] == list(range(10))
        assert dispatcher.queue_size() == 0
    finally:
        dispatcher.close()


def test_dispatcher_wired_to_redis_reports_unsupported_syscall():
    backend = RedisQueueBackend(redis_url=REDIS_URL, queue_key=_unique_key())
    dispatcher = Dispatcher(registry=HandlerRegistry(), worker_count=1, queue_backend=backend)

    try:
        syscall = dispatcher.submit(_build_request(), "agent-x")

        assert syscall.wait_for_completion(timeout=5) is True
        assert syscall.succeeded is False
        assert "No handler registered" in syscall.error
    finally:
        dispatcher.close()
