"""
Demo: run a concurrent multi-agent workload through the Kernel
with tracing enabled, then render the resulting trace.

Each of --agents agents submits --runs syscalls of different types
concurrently, from its own thread.  Events are appended to a JSONL
trace file; the timeline and lanes views are printed at the end.

Usage::

    python scripts/demo_trace.py --runs 3 --agents 3
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# Allow running straight from the repo: python scripts/demo_trace.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aios.core import (
    AgentID,
    AgentRequest,
    RequestID,
    RequestStatus,
    SystemCall,
    SystemCallType,
    TaskID,
)
from aios.kernel import Kernel
from aios.monitoring import (
    JsonlSink,
    configure,
    dropped_count,
    load_events,
    render_lanes,
    render_timeline,
    reset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_PATH = REPO_ROOT / "traces" / "demo.jsonl"

CALL_TYPES = [
    SystemCallType.LLM_CALL,
    SystemCallType.MEMORY_READ,
    SystemCallType.TOOL_CALL,
    SystemCallType.STORAGE_WRITE,
]


def _build_request(index: int) -> AgentRequest:
    """
    Build one demo request, cycling through the syscall types.
    """

    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(
            call_type=CALL_TYPES[index % len(CALL_TYPES)],
            payload={"prompt": f"demo prompt {index}"},
        ),
    )


def _run_scenario(agents: int, runs: int) -> None:
    """
    Run `agents` concurrent workers, each submitting `runs` syscalls.
    """

    kernel = Kernel()
    try:
        def _worker(agent_index: int) -> None:
            for run_index in range(runs):
                response = kernel.handle(
                    _build_request(agent_index * runs + run_index),
                    agent_name=f"agent-{agent_index}",
                )
                if response.status is not RequestStatus.SUCCESS:
                    print(
                        f"agent-{agent_index} run {run_index} failed:"
                        f" {response.error}",
                        file=sys.stderr,
                    )

        threads = [
            threading.Thread(target=_worker, args=(agent_index,))
            for agent_index in range(agents)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        kernel.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trace a concurrent multi-agent workload through the Kernel."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="syscalls each agent submits (default: 3)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="number of concurrent agents (default: 3)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE_PATH,
        help="JSONL trace file to write (default: traces/demo.jsonl)",
    )
    args = parser.parse_args(argv)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    configure(JsonlSink(args.trace))
    try:
        _run_scenario(args.agents, args.runs)
    finally:
        reset()

    events = load_events(args.trace)
    print(f"trace file: {args.trace}")
    print(
        f"{len(events)} events, {dropped_count()} dropped"
    )
    print()
    print("TIMELINE")
    print(render_timeline(events))
    print()
    print("LANES")
    print(render_lanes(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
