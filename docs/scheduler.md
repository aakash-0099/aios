# AIOS Scheduler — Architecture & Integration Guide

## Module Map

```
aios/scheduler/
├── __init__.py              # Package exports
├── resource_pool.py         # ResourcePool: capacity ledger (NEW)
├── resource_scheduler.py    # ResourceScheduler: queue ordering + preemption
├── scheduler.py             # Scheduler: central dispatch across resource types
├── priority.py              # PriorityTaskQueue / PriorityRequestQueue (aging)
├── fifo.py                  # FIFOTaskQueue
└── queue.py                 # TaskQueue protocol
```

## Separation of Concerns

### Before (review finding)
`ResourceScheduler` owned both capacity counters *and* queue ordering.
`Scheduler` never called `can_admit`, so admission was dead code on the
central dispatch path.

### After (this refactor)

| Concern | Owner | File |
|---------|-------|------|
| **Capacity accounting** — total/available/reserved, invariant enforcement, reserve/release | `ResourcePool` | `resource_pool.py` |
| **Queue management** — FIFO/RR/Priority ordering, blocked tracking, preemption, metrics | `ResourceScheduler` | `resource_scheduler.py` |
| **Central dispatch** — route by syscall type, fair arbitration, admit/release on central path | `Scheduler` | `scheduler.py` |

`ResourceScheduler` holds a `ResourcePool` instance (`self.pool`) and
thin-wraps its methods (`can_admit`, `reserve`, `release`, `snapshot`) for
callers who don't want to reach in.

## Key APIs

### ResourcePool (capacity only)

```python
pool = ResourcePool(total_capacity={"slots": 8, "tokens_per_min": 100_000})
decision = pool.can_admit(task_or_request_or_dict)  # -> AdmissionDecision
res_id   = pool.reserve(task)                        # -> ReservationID
pool.release(res_id)                                 # by ReservationID
pool.release_by_task(task.task_id)                    # by TaskID (convenience)
snap     = pool.snapshot()                            # -> ResourceState
```

### Scheduler (central)

```python
scheduler = Scheduler(
    strategy=SchedulingStrategy.ROUND_ROBIN,
    quantum=5,
    resource_capacities={
        ResourceType.LLM:     {"slots": 4},
        ResourceType.MEMORY:  {"mb": 512},
        ResourceType.STORAGE: {"iops": 1000},
        ResourceType.TOOL:    {"concurrent": 8},
    },
)

# Admission check (NEW — addresses review finding)
decision = scheduler.admit(request)          # -> AdmissionDecision
pool     = scheduler.get_pool(ResourceType.LLM)  # direct pool access

# Submit / dispatch
scheduler.submit(request, priority=0)
next_req = scheduler.next()                  # fair round-robin across types
next_req = scheduler.next(ResourceType.LLM)  # from one type only

# Lifecycle
scheduler.complete(request.request_id)
scheduler.release_by_task(task.task_id)       # NEW — task-keyed release
scheduler.close()
```

## Invariant

`ResourcePool` enforces on **every mutation**:

```
available[k] + reserved[k] == total[k]   for all k
```

Assertion fires via `_assert_invariants()` after every `reserve()` and
`release()`.

## Kernel Integration (TODO — Phase 5+)

The Kernel's `dispatcher.py` currently uses a `worker_count` concurrency
model and does not call the scheduler at all.  The integration path:

1. **On agent request arrival** → `scheduler.admit(request)`
   - `ACCEPTED` → `scheduler.submit(request)`, proceed to dispatch
   - `WAITING`  → park in a wait queue, re-check on `release`
   - `REJECTED` → return error to agent

2. **On dispatch tick** → `request = scheduler.next()`

3. **On completion** → `scheduler.complete(request_id)` then
   `scheduler.release_by_task(task_id)` to free capacity

4. **On preemption (Phase 6)** → `scheduler.preempt(request_id, ctx)`

This keeps the Kernel as the orchestrator and the Scheduler as a pure
scheduling/admission engine, matching the AIOS paper's layered architecture.

## Running Tests

```bash
# scheduler tests only (34 tests)
python -m pytest tests/unit/test_scheduler.py -v

# full suite
python -m pytest --tb=short
```
