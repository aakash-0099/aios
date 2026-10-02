"""
AIOS monitoring: tracing foundation.

The default sink is a no-op, so an unconfigured system behaves
exactly as before.  Configure a sink to collect events::

    from aios.monitoring import JsonlSink, configure

    configure(JsonlSink("traces/demo.jsonl"))

Or capture events in memory (tests, demos)::

    from aios.monitoring import capture

    with capture() as sink:
        ...  # run the system
        for event in sink.events():
            ...

Render a trace file::

    python -m aios.monitoring.render traces/demo.jsonl
"""

from .events import KernelEvents, TraceEvent
from .sinks import InMemorySink, JsonlSink, NullSink, Sink
from .tracer import (
    capture,
    configure,
    dropped_count,
    emit,
    get_sink,
    reset,
)

__all__ = [
    "InMemorySink",
    "JsonlSink",
    "KernelEvents",
    "NullSink",
    "Sink",
    "TraceEvent",
    "capture",
    "configure",
    "dropped_count",
    "emit",
    "get_sink",
    "load_events",
    "render_lanes",
    "render_timeline",
    "reset",
]

_RENDER_FUNCTIONS = ("load_events", "render_lanes", "render_timeline")


def __getattr__(name: str) -> object:
    # Imported lazily so `python -m aios.monitoring.render` does not
    # find aios.monitoring.render already in sys.modules.
    if name in _RENDER_FUNCTIONS:
        from . import render

        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
