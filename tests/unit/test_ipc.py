"""
Unit tests for the isolated async IPC transport.

Every payload here is a fake, made up purely for these tests
(plain strings and dicts) -- nothing in `aios.communication.ipc`
depends on `AgentRequest`/`AgentResponse`, the kernel, or the
scheduler, and these tests are written to prove that, not to
exercise Phase 1/2 domain objects.
"""

from __future__ import annotations

import asyncio

import pytest

from aios.communication.ipc import (
    IPCChannel,
    IPCChannelClosedError,
    IPCMessage,
    IPCTimeoutError,
)
from aios.core.ids import RequestID


# ---------------------------------------------------------------
# IPCMessage
# ---------------------------------------------------------------


def test_message_new_generates_a_fresh_correlation_id():
    first = IPCMessage.new(payload="fake-payload-1")
    second = IPCMessage.new(payload="fake-payload-2")

    assert isinstance(first.correlation_id, RequestID)
    assert first.correlation_id != second.correlation_id


def test_message_carries_an_arbitrary_fake_payload():
    message = IPCMessage.new(payload={"fake": "payload", "n": 1}, sender="agent-fake")

    assert message.payload == {"fake": "payload", "n": 1}
    assert message.sender == "agent-fake"


# ---------------------------------------------------------------
# Basic send / receive (FIFO)
# ---------------------------------------------------------------


async def test_receive_returns_messages_in_fifo_order():
    channel: IPCChannel[str] = IPCChannel()

    await channel.send(IPCMessage.new(payload="first-fake-message"))
    await channel.send(IPCMessage.new(payload="second-fake-message"))

    first = await channel.receive(timeout=1)
    second = await channel.receive(timeout=1)

    assert first.payload == "first-fake-message"
    assert second.payload == "second-fake-message"


async def test_pending_count_reflects_queued_messages():
    channel: IPCChannel[str] = IPCChannel()

    assert channel.pending_count() == 0

    await channel.send(IPCMessage.new(payload="fake"))
    assert channel.pending_count() == 1

    await channel.receive(timeout=1)
    assert channel.pending_count() == 0


async def test_receive_times_out_on_an_empty_channel():
    channel: IPCChannel[str] = IPCChannel()

    with pytest.raises(IPCTimeoutError):
        await channel.receive(timeout=0.05)


# ---------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------


async def test_receive_correlated_finds_a_message_already_queued():
    channel: IPCChannel[str] = IPCChannel()
    target_id = RequestID.generate()

    await channel.send(IPCMessage(correlation_id=RequestID.generate(), payload="unrelated-fake"))
    await channel.send(IPCMessage(correlation_id=target_id, payload="the-fake-reply"))

    matched = await channel.receive_correlated(target_id, timeout=1)

    assert matched.payload == "the-fake-reply"
    # The unrelated message already queued is left in place, in order.
    remaining = await channel.receive(timeout=1)
    assert remaining.payload == "unrelated-fake"


async def test_receive_correlated_waits_for_a_message_sent_later():
    channel: IPCChannel[str] = IPCChannel()
    target_id = RequestID.generate()

    async def _reply_soon() -> None:
        await asyncio.sleep(0.05)
        await channel.send(IPCMessage(correlation_id=target_id, payload="fake-reply-later"))

    asyncio.create_task(_reply_soon())

    matched = await channel.receive_correlated(target_id, timeout=1)

    assert matched.payload == "fake-reply-later"


async def test_receive_correlated_ignores_non_matching_traffic():
    channel: IPCChannel[str] = IPCChannel()
    target_id = RequestID.generate()

    async def _send_noise_then_reply() -> None:
        for i in range(3):
            await channel.send(IPCMessage.new(payload=f"fake-noise-{i}"))
            await asyncio.sleep(0.01)
        await channel.send(IPCMessage(correlation_id=target_id, payload="the-real-fake-reply"))

    asyncio.create_task(_send_noise_then_reply())

    matched = await channel.receive_correlated(target_id, timeout=1)

    assert matched.payload == "the-real-fake-reply"


async def test_multiple_correlated_waiters_each_get_their_own_reply():
    channel: IPCChannel[str] = IPCChannel()
    id_a = RequestID.generate()
    id_b = RequestID.generate()

    waiter_a = asyncio.create_task(channel.receive_correlated(id_a, timeout=1))
    waiter_b = asyncio.create_task(channel.receive_correlated(id_b, timeout=1))
    await asyncio.sleep(0.01)  # let both waiters register

    # Send B's reply first, to prove routing is by id, not send order.
    await channel.send(IPCMessage(correlation_id=id_b, payload="fake-reply-b"))
    await channel.send(IPCMessage(correlation_id=id_a, payload="fake-reply-a"))

    result_a = await waiter_a
    result_b = await waiter_b

    assert result_a.payload == "fake-reply-a"
    assert result_b.payload == "fake-reply-b"


async def test_receive_correlated_times_out_without_a_match():
    channel: IPCChannel[str] = IPCChannel()

    with pytest.raises(IPCTimeoutError):
        await channel.receive_correlated(RequestID.generate(), timeout=0.05)


# ---------------------------------------------------------------
# request() convenience
# ---------------------------------------------------------------


async def test_request_round_trips_through_a_fake_server_task():
    channel: IPCChannel[dict] = IPCChannel()

    async def _fake_server() -> None:
        incoming = await channel.receive(timeout=1)
        reply_payload = {"echo": incoming.payload["op"]}
        await channel.send(IPCMessage(correlation_id=incoming.correlation_id, payload=reply_payload))

    asyncio.create_task(_fake_server())

    response = await channel.request(payload={"op": "fake-ping"}, sender="agent-fake", timeout=1)

    assert response.payload == {"echo": "fake-ping"}


# ---------------------------------------------------------------
# Closed channel handling
# ---------------------------------------------------------------


async def test_send_raises_after_close():
    channel: IPCChannel[str] = IPCChannel()
    await channel.close()

    with pytest.raises(IPCChannelClosedError):
        await channel.send(IPCMessage.new(payload="too-late-fake-message"))


async def test_receive_raises_once_closed_and_drained():
    channel: IPCChannel[str] = IPCChannel()
    await channel.send(IPCMessage.new(payload="fake-leftover"))
    await channel.close()

    # Whatever was already queued before close is still delivered...
    leftover = await channel.receive(timeout=1)
    assert leftover.payload == "fake-leftover"

    # ...but once it's drained, the channel reports closed.
    with pytest.raises(IPCChannelClosedError):
        await channel.receive(timeout=1)


async def test_receive_correlated_raises_once_closed_without_a_match():
    channel: IPCChannel[str] = IPCChannel()
    await channel.close()

    with pytest.raises(IPCChannelClosedError):
        await channel.receive_correlated(RequestID.generate(), timeout=1)


async def test_close_wakes_a_pending_receive_immediately():
    channel: IPCChannel[str] = IPCChannel()

    waiter = asyncio.create_task(channel.receive(timeout=5))
    await asyncio.sleep(0.01)  # let it start waiting

    await channel.close()

    with pytest.raises(IPCChannelClosedError):
        await waiter


async def test_close_wakes_a_pending_correlated_receive_immediately():
    channel: IPCChannel[str] = IPCChannel()

    waiter = asyncio.create_task(channel.receive_correlated(RequestID.generate(), timeout=5))
    await asyncio.sleep(0.01)

    await channel.close()

    with pytest.raises(IPCChannelClosedError):
        await waiter


async def test_close_is_idempotent():
    channel: IPCChannel[str] = IPCChannel()

    await channel.close()
    await channel.close()  # must not raise

    assert channel.is_closed is True


async def test_channel_as_async_context_manager_closes_on_exit():
    async with IPCChannel() as channel:
        await channel.send(IPCMessage.new(payload="fake"))
        assert channel.is_closed is False

    assert channel.is_closed is True


# ---------------------------------------------------------------
# Isolation: this module must not depend on the kernel or scheduler
# ---------------------------------------------------------------


def test_ipc_module_does_not_import_kernel_or_scheduler():
    import inspect

    from aios.communication import ipc as ipc_module

    source = inspect.getsource(ipc_module)

    assert "aios.kernel" not in source
    assert "aios.scheduler" not in source
    assert "aios import kernel" not in source
    assert "aios import scheduler" not in source
