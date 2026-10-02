"""
Module-level tracing API.

The tracer is process-global and thread-safe.  With no sink
configured it keeps a `NullSink`, so an unconfigured system
behaves exactly as if tracing did not exist: `emit` builds one
small object and calls a no-op.  `emit` never raises -- a failing
sink is swallowed and counted in `dropped_count()`.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator

from .events import TraceEvent
from .sinks import InMemorySink, NullSink, Sink

_SAFE_SCALARS = (str, int, float, bool, type(None))

_sink: Sink = NullSink()
_sink_lock = Lock()
_dropped = 0
_dropped_lock = Lock()


def _swap_sink(sink: Sink) -> Sink:
    """
    Install ``sink`` as the current sink, returning the previous one.
    """

    global _sink
    with _sink_lock:
        previous = _sink
        _sink = sink
        return previous


def configure(sink: Sink) -> None:
    """
    Install ``sink`` as the process-global trace sink.
    """

    _swap_sink(sink)


def get_sink() -> Sink:
    """
    Return the currently installed sink.
    """

    with _sink_lock:
        return _sink


def reset() -> None:
    """
    Restore the default no-op sink and zero the dropped counter.
    """

    global _dropped
    _swap_sink(NullSink())
    with _dropped_lock:
        _dropped = 0


def dropped_count() -> int:
    """
    Number of events lost because the active sink raised.
    """

    with _dropped_lock:
        return _dropped


def _json_safe(value: Any) -> Any:
    """
    Coerce non-JSON-serialisable values with ``repr()`` so a stray
    object in ``data`` can never make ``emit`` fail.
    """

    if isinstance(value, _SAFE_SCALARS):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def emit(
    component: str,
    event: str,
    correlation_id: str | None = None,
    agent_id: str | None = None,
    **data: Any,
) -> None:
    """
    Build a `TraceEvent` and hand it to the current sink.

    Never raises: a sink that throws is swallowed and counted in
    `dropped_count()`.  Values in ``data`` that are not
    JSON-serialisable are replaced by their ``repr()``.
    """

    global _dropped
    safe_data = {key: _json_safe(value) for key, value in data.items()}
    trace_event = TraceEvent(
        wall_time=time.time(),
        mono_time=time.monotonic(),
        correlation_id=correlation_id,
        agent_id=agent_id,
        component=component,
        event=event,
        data=safe_data,
    )
    sink = get_sink()
    try:
        sink.write(trace_event)
    except Exception:  # noqa: BLE001 - tracing must never break the app
        with _dropped_lock:
            _dropped += 1


@contextmanager
def capture() -> Iterator[InMemorySink]:
    """
    Temporarily install an `InMemorySink` and yield it.

    The previous sink is restored afterwards, even if the wrapped
    code raises.  Intended for tests and demos.
    """

    sink = InMemorySink()
    previous = _swap_sink(sink)
    try:
        yield sink
    finally:
        _swap_sink(previous)
