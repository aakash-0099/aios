"""
Unit tests for the aios.monitoring tracing foundation.

Covers the TraceEvent schema, the no-op default, emit's
never-raise guarantee, sink thread-safety, capture()'s restore
semantics, and the exact output of both render views.
"""

from __future__ import annotations

import json
import threading

import pytest

from aios.monitoring import (
    InMemorySink,
    JsonlSink,
    NullSink,
    TraceEvent,
    capture,
    configure,
    dropped_count,
    emit,
    get_sink,
    reset,
)
from aios.monitoring.render import load_events, render_lanes, render_timeline


@pytest.fixture(autouse=True)
def _clean_tracer():
    """
    Every test starts and ends with the default no-op sink.
    """

    reset()
    yield
    reset()


def _event(
    mono_time: float,
    event: str,
    correlation_id: str | None = "c1",
    agent_id: str | None = "a1",
    data: dict[str, object] | None = None,
) -> TraceEvent:
    return TraceEvent(
        wall_time=mono_time,
        mono_time=mono_time,
        correlation_id=correlation_id,
        agent_id=agent_id,
        component="kernel",
        event=event,
        data=data or {},
    )


# ---------------------------------------------------------------
# TraceEvent schema
# ---------------------------------------------------------------


def test_trace_event_round_trip():
    event = TraceEvent(
        wall_time=1.5,
        mono_time=2.5,
        correlation_id="c1",
        agent_id="a1",
        component="kernel",
        event="syscall.created",
        data={"call_type": "llm_call", "pid": 3},
    )

    assert TraceEvent.from_dict(event.to_dict()) == event


def test_trace_event_to_dict_is_json_serialisable():
    event = _event(100.0, "syscall.created", data={"pid": 3})

    assert json.loads(json.dumps(event.to_dict())) == event.to_dict()


def test_trace_event_is_immutable():
    event = _event(100.0, "syscall.created")

    with pytest.raises(AttributeError):
        event.event = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------
# emit / configure / reset
# ---------------------------------------------------------------


def test_emit_with_default_null_sink_is_a_no_op():
    emit("kernel", "syscall.created", correlation_id="c1", pid=1)

    assert isinstance(get_sink(), NullSink)
    assert dropped_count() == 0


def test_emit_never_raises_when_sink_fails():
    class _BadSink:
        def write(self, event: TraceEvent) -> None:
            raise RuntimeError("boom")

    configure(_BadSink())

    emit("kernel", "syscall.created")
    emit("kernel", "syscall.created")

    assert dropped_count() == 2


def test_emit_converts_non_serialisable_data_with_repr():
    marker = object()

    with capture() as sink:
        emit("kernel", "test.event", correlation_id="c1", bad=marker, ok=5)

    event = sink.events()[0]
    assert event.data["ok"] == 5
    assert event.data["bad"] == repr(marker)
    assert isinstance(event.data["bad"], str)


def test_reset_restores_null_sink_and_zeroes_counter():
    class _BadSink:
        def write(self, event: TraceEvent) -> None:
            raise RuntimeError("boom")

    configure(_BadSink())
    emit("kernel", "syscall.created")
    reset()

    assert isinstance(get_sink(), NullSink)
    assert dropped_count() == 0


# ---------------------------------------------------------------
# capture()
# ---------------------------------------------------------------


def test_capture_restores_the_previous_sink():
    sentinel = NullSink()
    configure(sentinel)

    with capture() as sink:
        assert get_sink() is sink
        emit("kernel", "syscall.created", correlation_id="c1")

    assert get_sink() is sentinel
    assert len(sink.events()) == 1


def test_capture_restores_the_previous_sink_on_exception():
    sentinel = NullSink()
    configure(sentinel)

    with pytest.raises(RuntimeError, match="inner"):
        with capture():
            raise RuntimeError("inner")

    assert get_sink() is sentinel


# ---------------------------------------------------------------
# InMemorySink thread-safety
# ---------------------------------------------------------------


def test_in_memory_sink_is_thread_safe():
    sink = InMemorySink()
    threads_per = 20
    events_per_thread = 50

    def _write() -> None:
        for index in range(events_per_thread):
            sink.write(
                _event(float(index), "syscall.created", correlation_id=f"c{index}")
            )

    threads = [threading.Thread(target=_write) for _ in range(threads_per)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(sink.events()) == threads_per * events_per_thread


# ---------------------------------------------------------------
# JsonlSink
# ---------------------------------------------------------------


def test_jsonl_sink_round_trip(tmp_path):
    path = tmp_path / "nested" / "dir" / "trace.jsonl"
    events = [
        _event(100.0, "syscall.created", data={"pid": 1}),
        _event(100.05, "syscall.completed"),
    ]

    sink = JsonlSink(path)
    for event in events:
        sink.write(event)
    sink.close()

    assert load_events(path) == events


def test_jsonl_sink_concurrent_writes_are_valid_json_per_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = JsonlSink(path)
    threads_per = 20
    events_per_thread = 25

    def _write(offset: int) -> None:
        for index in range(events_per_thread):
            sink.write(
                _event(
                    float(offset * events_per_thread + index),
                    "syscall.created",
                    correlation_id=f"c{offset}",
                )
            )

    threads = [
        threading.Thread(target=_write, args=(offset,))
        for offset in range(threads_per)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sink.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == threads_per * events_per_thread
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == threads_per * events_per_thread
    assert all(isinstance(record, dict) for record in parsed)


def test_jsonl_sink_close_is_idempotent(tmp_path):
    sink = JsonlSink(tmp_path / "trace.jsonl")
    sink.write(_event(1.0, "syscall.created"))

    sink.close()
    sink.close()


# ---------------------------------------------------------------
# render views
# ---------------------------------------------------------------


def test_render_timeline_exact_output():
    events = [
        _event(100.0, "syscall.created", data={"call_type": "llm_call"}),
        _event(100.025, "syscall.completed"),
    ]

    expected = "\n".join(
        [
            "correlation=c1 agent=a1 events=2 duration=25.0 ms",
            "  +    0.0 ms  (+   0.0 ms)  [kernel]  syscall.created"
            "  call_type=llm_call",
            "  +   25.0 ms  (+  25.0 ms)  [kernel]  syscall.completed",
        ]
    )

    assert render_timeline(events) == expected


def test_render_timeline_groups_by_correlation_id():
    events = [
        _event(100.0, "syscall.created", correlation_id="c1"),
        _event(100.01, "syscall.created", correlation_id="c2"),
        _event(100.02, "syscall.completed", correlation_id="c1"),
    ]

    output = render_timeline(events)

    assert output.index("correlation=c1") < output.index("correlation=c2")
    c1_block = output.split("correlation=c2")[0]
    assert c1_block.count("syscall.") == 2


def test_render_lanes_exact_output():
    events = [
        _event(100.0, "syscall.created", correlation_id="c1", agent_id="a1",
               data={"call_type": "llm_call"}),
        _event(100.01, "syscall.completed", correlation_id="c2", agent_id="a2"),
    ]

    expected = "\n".join(
        [
            "agent=a1 events=1",
            "  +    0.0 ms  [kernel]  syscall.created  cid=c1  call_type=llm_call",
            "",
            "agent=a2 events=1",
            "  +   10.0 ms  [kernel]  syscall.completed  cid=c2",
        ]
    )

    assert render_lanes(events) == expected


def test_render_views_handle_empty_input():
    assert render_timeline([]) == "(no events)"
    assert render_lanes([]) == "(no events)"
