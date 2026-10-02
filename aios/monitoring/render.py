"""
Plain-text renderings of a JSONL trace.

Both views are pure functions returning strings -- no printing
inside, so tests can pin exact output and callers decide where the
text goes.  `python -m aios.monitoring.render <file>` prints both
views for any trace file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .events import TraceEvent


def load_events(path: str | Path) -> list[TraceEvent]:
    """
    Read a JSONL trace file back into `TraceEvent` objects.

    Blank lines are skipped; every other line must be one JSON
    object as written by `aios.monitoring.sinks.JsonlSink`.
    """

    events: list[TraceEvent] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(TraceEvent.from_dict(json.loads(line)))
    return events


def _fmt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _fmt_opt(value: str | None) -> str:
    return value if value is not None else "none"


def render_timeline(events: list[TraceEvent]) -> str:
    """
    Render events grouped by correlation id.

    Each group shows the agent, the total duration, and every event
    as ``+<ms from first event> ms`` with the gap since the previous
    event in parentheses, so queueing delays stand out.  Groups and
    events are ordered by ``mono_time``.
    """

    if not events:
        return "(no events)"

    groups: dict[str | None, list[TraceEvent]] = {}
    for event in events:
        groups.setdefault(event.correlation_id, []).append(event)

    ordered: list[tuple[str | None, list[TraceEvent]]] = []
    for correlation_id, group in groups.items():
        group.sort(key=lambda event: event.mono_time)
        ordered.append((correlation_id, group))
    ordered.sort(key=lambda item: (item[1][0].mono_time, item[0] or ""))

    blocks: list[str] = []
    for correlation_id, group in ordered:
        first = group[0].mono_time
        duration_ms = (group[-1].mono_time - first) * 1000.0
        lines = [
            f"correlation={_fmt_opt(correlation_id)}"
            f" agent={_fmt_opt(group[0].agent_id)}"
            f" events={len(group)}"
            f" duration={duration_ms:.1f} ms"
        ]
        previous_ms = 0.0
        for event in group:
            ms = (event.mono_time - first) * 1000.0
            gap = ms - previous_ms
            previous_ms = ms
            pairs = " ".join(
                f"{key}={_fmt_value(value)}"
                for key, value in event.data.items()
            )
            suffix = f"  {pairs}" if pairs else ""
            lines.append(
                f"  +{ms:>7.1f} ms  (+{gap:>6.1f} ms)"
                f"  [{event.component}]  {event.event}{suffix}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_lanes(events: list[TraceEvent]) -> str:
    """
    Render one lane per agent, all against a shared time origin.

    Because every lane uses the same ``t0``, interleaving across
    concurrent agents is visible at a glance.  Each event carries
    the first 8 characters of its correlation id so a single
    request can be followed across lanes.
    """

    if not events:
        return "(no events)"

    t0 = min(event.mono_time for event in events)
    lanes: dict[str | None, list[TraceEvent]] = {}
    for event in events:
        lanes.setdefault(event.agent_id, []).append(event)

    blocks: list[str] = []
    for agent_id in sorted(lanes, key=lambda agent: agent or ""):
        group = sorted(lanes[agent_id], key=lambda event: event.mono_time)
        lines = [f"agent={_fmt_opt(agent_id)} events={len(group)}"]
        for event in group:
            ms = (event.mono_time - t0) * 1000.0
            short_cid = (
                event.correlation_id[:8]
                if event.correlation_id is not None
                else "none"
            )
            pairs = " ".join(
                f"{key}={_fmt_value(value)}"
                for key, value in event.data.items()
            )
            suffix = f"  {pairs}" if pairs else ""
            lines.append(
                f"  +{ms:>7.1f} ms  [{event.component}]"
                f"  {event.event}  cid={short_cid}{suffix}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    """
    CLI: print the timeline and lanes views for a trace file.
    """

    parser = argparse.ArgumentParser(
        description="Render an AIOS JSONL trace file as text."
    )
    parser.add_argument("path", help="path to a JSONL trace file")
    args = parser.parse_args(argv)

    events = load_events(args.path)
    print(render_timeline(events))
    print()
    print(render_lanes(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
