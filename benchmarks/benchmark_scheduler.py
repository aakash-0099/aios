"""
AIOS Scheduler Benchmark: FIFO vs. Round Robin (RR).

Reproduces Table 4 from the AIOS paper (Mei et al., 2024):
"Evaluation of Different Scheduling Strategies with Multiple Concurrent Agents"

Measures:
- Total Execution Time
- Average Waiting Time
- P90 (90th percentile) Waiting Time
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aios.core import (
    AgentID,
    AgentRequest,
    RequestID,
    ResourceType,
    SystemCall,
    SystemCallType,
    TaskID,
)
from aios.scheduler import (
    ResourceScheduler,
    Scheduler,
    SchedulingStrategy,
)


@dataclass
class AgentTaskProfile:
    agent_id: AgentID
    task_id: TaskID
    request_id: RequestID
    name: str
    total_tokens: int  # Amount of work needed to complete


@dataclass
class BenchmarkResult:
    agent_count: int
    strategy: str
    total_execution_time: float
    avg_waiting_time: float
    p90_waiting_time: float
    total_preemptions: int
    completed_requests: int


def generate_workload(agent_count: int, seed: int = 42) -> list[AgentTaskProfile]:
    """
    Generate a realistic heterogeneous workload across N agents.
    Mix of short queries (e.g. 4-8 tokens), medium queries (12-20 tokens),
    and long analytical queries (30-60 tokens).
    """
    rng = random.Random(seed)
    profiles = []
    agent_types = [
        ("ShortQA_Agent", 6),
        ("MediumAnalysis_Agent", 16),
        ("LongCodeGen_Agent", 45),
        ("ToolUser_Agent", 8),
        ("Translator_Agent", 14),
    ]

    for i in range(agent_count):
        agent_type, base_tokens = agent_types[i % len(agent_types)]
        jitter = rng.randint(-3, 6)
        total_tokens = max(3, base_tokens + jitter)
        profiles.append(
            AgentTaskProfile(
                agent_id=AgentID.generate(),
                task_id=TaskID.generate(),
                request_id=RequestID.generate(),
                name=f"{agent_type}_{i+1}",
                total_tokens=total_tokens,
            )
        )
    return profiles


def run_benchmark(
    profiles: list[AgentTaskProfile],
    strategy: SchedulingStrategy,
    quantum: int = 5,
    token_latency_ms: float = 2.0,  # 2ms per token simulation
) -> BenchmarkResult:
    """
    Run simulated scheduling for the given workload.
    """
    # Initialize scheduler
    scheduler = ResourceScheduler(
        resource_type=ResourceType.LLM,
        strategy=strategy,
        quantum=quantum,
    )

    remaining_work: dict[RequestID, int] = {}
    queued_time: dict[RequestID, float] = {}
    first_service_time: dict[RequestID, float] = {}
    completed_time: dict[RequestID, float] = {}
    preemptions: dict[RequestID, int] = {}

    current_clock = 0.0

    # Submit all agent requests at t = 0
    for p in profiles:
        req = AgentRequest(
            request_id=p.request_id,
            agent_id=p.agent_id,
            task_id=p.task_id,
            syscall=SystemCall(
                call_type=SystemCallType.LLM_CALL,
                payload={"prompt": f"Task for {p.name}"},
            ),
            metadata={"total_tokens": p.total_tokens},
        )
        remaining_work[p.request_id] = p.total_tokens
        queued_time[p.request_id] = current_clock
        preemptions[p.request_id] = 0
        scheduler.submit(req)

    # Dispatch loop
    while not scheduler.is_empty():
        req = scheduler.next_request()
        if req is None:
            break

        req_id = req.request_id
        # Record first time this request got serviced
        if req_id not in first_service_time:
            first_service_time[req_id] = current_clock

        tokens_left = remaining_work[req_id]

        if strategy == SchedulingStrategy.FIFO:
            # FIFO runs task to completion
            execution_duration = (tokens_left * token_latency_ms) / 1000.0
            current_clock += execution_duration
            remaining_work[req_id] = 0
            completed_time[req_id] = current_clock
            scheduler.complete(req_id)

        elif strategy == SchedulingStrategy.ROUND_ROBIN:
            # RR runs task for min(quantum, tokens_left)
            tokens_to_run = min(quantum, tokens_left)
            execution_duration = (tokens_to_run * token_latency_ms) / 1000.0
            current_clock += execution_duration
            remaining_work[req_id] -= tokens_to_run

            if remaining_work[req_id] > 0:
                preemptions[req_id] += 1
                scheduler.preempt(
                    req_id,
                    partial_context={"tokens_done": tokens_to_run},
                )
            else:
                completed_time[req_id] = current_clock
                scheduler.complete(req_id)

    # Calculate wait times
    wait_times = [
        first_service_time[p.request_id] - queued_time[p.request_id]
        for p in profiles
    ]
    wait_times.sort()

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

    # 90th percentile wait time
    if wait_times:
        p90_idx = int(math.ceil(0.90 * len(wait_times))) - 1
        p90_wait = wait_times[max(0, min(p90_idx, len(wait_times) - 1))]
    else:
        p90_wait = 0.0

    return BenchmarkResult(
        agent_count=len(profiles),
        strategy="FIFO" if strategy == SchedulingStrategy.FIFO else "Round Robin (RR)",
        total_execution_time=current_clock,
        avg_waiting_time=avg_wait,
        p90_waiting_time=p90_wait,
        total_preemptions=sum(preemptions.values()),
        completed_requests=len(completed_time),
    )


def print_table_4(results: list[BenchmarkResult]) -> None:
    """
    Format and print Table 4 matching the AIOS paper style.
    """
    print("\n" + "=" * 84)
    print(" AIOS SCHEDULER BENCHMARK: REPRODUCING TABLE 4 (Mei et al., 2024)")
    print("=" * 84)
    header = (
        f"{'Agents (N)':<12} | {'Strategy':<18} | {'Total Time (s)':<14} | "
        f"{'Avg Wait (s)':<13} | {'P90 Wait (s)':<13}"
    )
    print(header)
    print("-" * 84)

    for r in results:
        row = (
            f"{r.agent_count:<12} | {r.strategy:<18} | {r.total_execution_time:<14.4f} | "
            f"{r.avg_waiting_time:<13.4f} | {r.p90_waiting_time:<13.4f}"
        )
        print(row)
        if "Round Robin" in r.strategy:
            print("-" * 84)

    print("=" * 84)
    print("Observation:")
    print("- Round Robin (RR) drastically reduces average and P90 wait time by preventing")
    print("  long tasks from causing head-of-line blocking for short agent tasks.")
    print("=" * 84 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOS Scheduler Benchmark")
    parser.add_argument(
        "--agents",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="List of agent counts to benchmark (e.g. 5 10 20)",
    )
    parser.add_argument(
        "--quantum",
        type=int,
        default=5,
        help="Round Robin token quantum (default: 5)",
    )
    args = parser.parse_args()

    results: list[BenchmarkResult] = []

    for n in args.agents:
        profiles = generate_workload(n)

        # Run FIFO
        fifo_res = run_benchmark(profiles, SchedulingStrategy.FIFO, quantum=args.quantum)
        results.append(fifo_res)

        # Run Round Robin
        rr_res = run_benchmark(profiles, SchedulingStrategy.ROUND_ROBIN, quantum=args.quantum)
        results.append(rr_res)

    print_table_4(results)


if __name__ == "__main__":
    main()
