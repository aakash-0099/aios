"""
Trace sinks: where `TraceEvent` objects go.

A sink is anything with a ``write(event)`` method.  The default
sink is `NullSink`, so an unconfigured system traces nothing and
pays nothing.  `InMemorySink` serves tests and demos; `JsonlSink`
appends one JSON object per line to a file, creating parent
directories as needed and flushing every write so a trace survives
a crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Protocol

from .events import TraceEvent


class Sink(Protocol):
    """
    Minimal contract a trace sink must satisfy.
    """

    def write(self, event: TraceEvent) -> None:
        """
        Consume one event.  Implementations must never raise --
        `aios.monitoring.tracer.emit` guards this, but a sink that
        cannot fail keeps the guarantee even for direct callers.
        """


class NullSink:
    """
    Default sink: does nothing, costs nothing.
    """

    def write(self, event: TraceEvent) -> None:
        return None


class InMemorySink:
    """
    Thread-safe in-memory sink, mainly for tests and demos.
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def write(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[TraceEvent]:
        """
        Return a copy of everything written so far.
        """

        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """
        Drop all stored events.
        """

        with self._lock:
            self._events.clear()


class JsonlSink:
    """
    Append-only JSONL file sink.

    One JSON object per line, flushed after every write.  Parent
    directories are created on construction.  `close()` is
    idempotent, and writes after `close()` are silently dropped --
    a sink must never break the application it observes.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._closed = False
        self._fh = open(self._path, "a", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        line = json.dumps(event.to_dict())
        with self._lock:
            if self._closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """
        Close the underlying file.  Safe to call more than once.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._fh.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
