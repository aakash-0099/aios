"""
Unit tests for the AIOS Scheduler subsystem.

Covers:
- Empty queue behavior
- All-blocked queue behavior
- Starvation prevention (low-priority task not skipped forever)
- FIFO vs Round Robin produce measurably different outcomes
- Central queues per resource type and syscall routing
- Round Robin preemption and context snapshotting
- Capacity accounting, admission decisions, and invariant preservation
- Scheduler lifecycle and closing
- ResourcePool standalone tests (separate concern from queue management)
- Scheduler.admit() on the central path
- Task-id-keyed release via release_by_task()
"""

from __future__ import annotations

import pytest

from aios.core import (
    AgentID,
    AgentRequest,
    RequestID,
    ResourceError,
    ResourceType,
    SchedulerError,
    SystemCall,
    SystemCallType,
    Task,
    TaskID,
    ValidationError,
)
from aios.scheduler import (
    AdmissionDecision,
    DecisionType,
    FIFOTaskQueue,
    PriorityRequestQueue,
    PriorityTaskQueue,
    ReservationID,
    ResourcePool,
    ResourceScheduler,
    ResourceState,
    Scheduler,
    SchedulingStrategy,
)


def _make_request(
    call_type: SystemCallType = SystemCallType.LLM_CALL,
    payload: dict | None = None,
    metadata: dict | None = None,
) -> AgentRequest:
    return AgentRequest(
        request_id=RequestID.generate(),
        agent_id=AgentID.generate(),
        task_id=TaskID.generate(),
        syscall=SystemCall(call_type=call_type, payload=payload or {}),
        metadata=metadata or {},
    )


def _make_task(
    metadata: dict | None = None,
    description: str = "test task",
) -> Task:
    return Task(
        task_id=TaskID.generate(),
        agent_id=AgentID.generate(),
        description=description,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# 1. Empty Queue Tests
# ---------------------------------------------------------------------------


def test_empty_scheduler_next_returns_none():
    scheduler = Scheduler(strategy=SchedulingStrategy.FIFO)
    assert scheduler.is_empty()
    assert scheduler.size() == 0
    assert scheduler.next() is None
    scheduler.close()


def test_empty_resource_scheduler_next_returns_none():
    res_sched = ResourceScheduler(resource_type=ResourceType.LLM)
    assert res_sched.is_empty()
    assert res_sched.size() == 0
    assert res_sched.next_request() is None


# ---------------------------------------------------------------------------
# 2. All-Blocked Queue Tests
# ---------------------------------------------------------------------------


def test_all_blocked_queue_returns_none_without_hanging():
    scheduler = Scheduler(strategy=SchedulingStrategy.FIFO)
    req1 = _make_request()
    req2 = _make_request()

    scheduler.submit(req1)
    scheduler.submit(req2)
    assert scheduler.size() == 2

    # Block both requests
    assert scheduler.block(req1.request_id, reason="waiting for IO")
    assert scheduler.block(req2.request_id, reason="waiting for lock")

    # Queue has 2 items, but all are blocked -> next() returns None
    assert scheduler.next() is None

    # Check resource scheduler status
    llm_sched = scheduler.get_resource_scheduler(ResourceType.LLM)
    assert llm_sched.is_all_blocked()

    # Unblocking one allows it to be scheduled
    assert scheduler.unblock(req1.request_id)
    assert not llm_sched.is_all_blocked()
    next_req = scheduler.next()
    assert next_req is not None
    assert next_req.request_id == req1.request_id

    scheduler.close()


# ---------------------------------------------------------------------------
# 3. Starvation Prevention Tests
# ---------------------------------------------------------------------------


def test_starvation_prevention_priority_queue():
    """
    A low-priority task (priority=5) must NOT be skipped forever when high-priority
    tasks (priority=0) are continuously enqueued.
    With aging_threshold=2, waiting tasks gain age and their effective priority
    boosts until the low-priority task executes.
    """
    queue = PriorityTaskQueue(aging_threshold=2)
    agent_id = AgentID.generate()

    low_task = Task(
        task_id=TaskID.generate(),
        agent_id=agent_id,
        description="Low priority batch work",
    )
    # Enqueue low-priority task (priority=5)
    queue.enqueue(low_task, priority=5)

    dequeued_ids: list[TaskID] = []

    # Continuously enqueue and dequeue high-priority tasks
    for i in range(15):
        high_task = Task(
            task_id=TaskID.generate(),
            agent_id=agent_id,
            description=f"High priority urgent work {i}",
        )
        queue.enqueue(high_task, priority=0)
        task = queue.dequeue()
        dequeued_ids.append(task.task_id)
        if task.task_id == low_task.task_id:
            break

    # Assert that the low-priority task was dequeued despite the continuous stream
    assert low_task.task_id in dequeued_ids
    # It should be dequeued within a finite number of steps (not skipped forever)
    assert len(dequeued_ids) <= 12


def test_starvation_prevention_request_queue():
    """Verify anti-starvation in PriorityRequestQueue."""
    req_queue = PriorityRequestQueue(aging_threshold=2)
    low_req = _make_request()
    req_queue.enqueue(low_req, priority=4)

    dequeued_ids = []
    for _ in range(12):
        high_req = _make_request()
        req_queue.enqueue(high_req, priority=0)
        r = req_queue.dequeue()
        dequeued_ids.append(r.request_id)
        if r.request_id == low_req.request_id:
            break

    assert low_req.request_id in dequeued_ids


# ---------------------------------------------------------------------------
# 4. FIFO vs Round Robin Measurably Different Outcomes Tests
# ---------------------------------------------------------------------------


def test_fifo_vs_rr_measurably_different_outcomes():
    """
    Simulate a workload: 1 long task (needs 10 steps/tokens) followed by
    4 short tasks (each needs 2 steps/tokens).

    Under FIFO:
        The long task runs completely (10 ticks) before any short task gets to run.
        Wait times for short tasks will be high (Head-of-Line blocking).

    Under Round Robin (quantum=2):
        Long task runs 2 ticks, then gets preempted.
        Short tasks run and finish quickly.
        Wait times and turnaround times for short tasks are dramatically lower.
    """
    # 1. Simulate FIFO
    fifo_sched = ResourceScheduler(
        resource_type=ResourceType.LLM,
        strategy=SchedulingStrategy.FIFO,
    )
    long_req = _make_request(metadata={"work": 10})
    short_reqs = [_make_request(metadata={"work": 2}) for _ in range(4)]

    fifo_sched.submit(long_req)
    for r in short_reqs:
        fifo_sched.submit(r)

    # In FIFO, long_req is popped first and runs until done
    current_tick = 0
    fifo_finish_times = {}

    while not fifo_sched.is_empty():
        req = fifo_sched.next_request()
        if req is None:
            break
        work = req.metadata["work"]
        current_tick += work
        fifo_finish_times[req.request_id] = current_tick
        fifo_sched.complete(req.request_id)

    fifo_short_finish_times = [
        fifo_finish_times[r.request_id] for r in short_reqs
    ]
    fifo_avg_short_finish = sum(fifo_short_finish_times) / len(fifo_short_finish_times)

    # 2. Simulate Round Robin with quantum = 2
    rr_sched = ResourceScheduler(
        resource_type=ResourceType.LLM,
        strategy=SchedulingStrategy.ROUND_ROBIN,
        quantum=2,
    )
    long_req_rr = _make_request(metadata={"work": 10, "remaining": 10})
    short_reqs_rr = [
        _make_request(metadata={"work": 2, "remaining": 2}) for _ in range(4)
    ]

    rr_sched.submit(long_req_rr)
    for r in short_reqs_rr:
        rr_sched.submit(r)

    rr_tick = 0
    rr_finish_times = {}

    while not rr_sched.is_empty():
        req = rr_sched.next_request()
        if req is None:
            break
        rem = req.metadata["remaining"]
        slice_amount = min(2, rem)
        rr_tick += slice_amount
        rem -= slice_amount
        req.metadata["remaining"] = rem

        if rem > 0:
            # Preempt and re-queue at tail
            rr_sched.preempt(req.request_id, partial_context=f"progress: {rem}")
        else:
            rr_finish_times[req.request_id] = rr_tick
            rr_sched.complete(req.request_id)

    rr_short_finish_times = [
        rr_finish_times[r.request_id] for r in short_reqs_rr
    ]
    rr_avg_short_finish = sum(rr_short_finish_times) / len(rr_short_finish_times)

    # In FIFO: Long task finishes at 10. Short tasks finish at 12, 14, 16, 18. Avg = 15.0
    # In RR: Long task runs 2 ticks, then short tasks finish at 4, 6, 8, 10! Avg = 7.0
    assert rr_avg_short_finish < fifo_avg_short_finish
    assert max(rr_short_finish_times) < max(fifo_short_finish_times)


# ---------------------------------------------------------------------------
# 5. Central Resource Queues and Routing
# ---------------------------------------------------------------------------


def test_central_resource_routing():
    scheduler = Scheduler(strategy=SchedulingStrategy.FIFO)

    req_llm = _make_request(call_type=SystemCallType.LLM_CALL)
    req_mem = _make_request(call_type=SystemCallType.MEMORY_READ)
    req_sto = _make_request(call_type=SystemCallType.STORAGE_WRITE)
    req_tool = _make_request(call_type=SystemCallType.TOOL_CALL)

    scheduler.submit(req_llm)
    scheduler.submit(req_mem)
    scheduler.submit(req_sto)
    scheduler.submit(req_tool)

    assert scheduler.size(ResourceType.LLM) == 1
    assert scheduler.size(ResourceType.MEMORY) == 1
    assert scheduler.size(ResourceType.STORAGE) == 1
    assert scheduler.size(ResourceType.TOOL) == 1
    assert scheduler.size() == 4

    # Direct query by resource type
    assert scheduler.next(ResourceType.LLM) == req_llm
    assert scheduler.next(ResourceType.TOOL) == req_tool
    assert scheduler.next(ResourceType.MEMORY) == req_mem
    assert scheduler.next(ResourceType.STORAGE) == req_sto

    assert scheduler.is_empty()
    scheduler.close()


def test_fair_arbitration_across_resource_types():
    scheduler = Scheduler(strategy=SchedulingStrategy.FIFO)

    r_llm = _make_request(call_type=SystemCallType.LLM_CALL)
    r_tool = _make_request(call_type=SystemCallType.TOOL_CALL)

    scheduler.submit(r_llm)
    scheduler.submit(r_tool)

    # Calling next(None) arbitrates fairly across active resource queues
    res1 = scheduler.next()
    res2 = scheduler.next()
    res3 = scheduler.next()

    assert {res1.request_id, res2.request_id} == {r_llm.request_id, r_tool.request_id}
    assert res3 is None
    scheduler.close()


# ---------------------------------------------------------------------------
# 6. Capacity Accounting, Invariants, and Admission
# ---------------------------------------------------------------------------


def test_admission_accepted_waiting_rejected():
    res_sched = ResourceScheduler(
        resource_type=ResourceType.LLM,
        total_capacity={"slots": 10.0, "memory": 100.0},
    )

    # 1. Fits within available -> ACCEPTED
    task1 = _make_task(metadata={"resources": {"slots": 6.0, "memory": 50.0}})
    dec1 = res_sched.can_admit(task1)
    assert dec1.decision == DecisionType.ACCEPTED

    # Reserve it
    res_id1 = res_sched.reserve(task1)
    snap1 = res_sched.snapshot()
    assert snap1.available["slots"] == 4.0
    assert snap1.reserved["slots"] == 6.0
    assert snap1.available["slots"] + snap1.reserved["slots"] == 10.0

    # 2. Exceeds available, but <= total -> WAITING
    task2 = _make_task(metadata={"resources": {"slots": 5.0, "memory": 20.0}})
    dec2 = res_sched.can_admit(task2)
    assert dec2.decision == DecisionType.WAITING

    # 3. Exceeds total capacity -> REJECTED
    task3 = _make_task(metadata={"resources": {"slots": 15.0}})
    dec3 = res_sched.can_admit(task3)
    assert dec3.decision == DecisionType.REJECTED

    # Release task 1 -> restores capacity
    res_sched.release(res_id1)
    snap2 = res_sched.snapshot()
    assert snap2.available["slots"] == 10.0
    assert snap2.reserved["slots"] == 0.0

    # Now task 2 can be admitted
    assert res_sched.can_admit(task2).decision == DecisionType.ACCEPTED


def test_double_release_and_invalid_reservation_raises():
    res_sched = ResourceScheduler(
        resource_type=ResourceType.MEMORY,
        total_capacity={"units": 50.0},
    )
    task = _make_task(metadata={"resources": {"units": 10.0}})
    res_id = res_sched.reserve(task)
    res_sched.release(res_id)

    # Double release raises ResourceError
    with pytest.raises(ResourceError):
        res_sched.release(res_id)

    # Unknown reservation raises ResourceError
    with pytest.raises(ResourceError):
        res_sched.release(ReservationID.generate())


# ---------------------------------------------------------------------------
# 7. Scheduler Lifecycle and Closure
# ---------------------------------------------------------------------------


def test_scheduler_close_behavior():
    scheduler = Scheduler(strategy=SchedulingStrategy.ROUND_ROBIN)
    scheduler.close()
    assert scheduler.is_closed

    with pytest.raises(SchedulerError):
        scheduler.submit(_make_request())

    with pytest.raises(SchedulerError):
        scheduler.next()


# ===========================================================================
# 8. ResourcePool Standalone Tests (separation of concerns)
# ===========================================================================


class TestResourcePool:
    """Tests for ResourcePool as an independent capacity ledger."""

    def test_import_resource_pool_directly(self):
        """Spec requires: from aios.scheduler import ResourcePool."""
        from aios.scheduler import ResourcePool as RP
        assert RP is not None

    def test_scalar_capacity(self):
        pool = ResourcePool(total_capacity=50.0)
        snap = pool.snapshot()
        assert snap.total == {"units": 50.0}
        assert snap.available == {"units": 50.0}
        assert snap.reserved == {"units": 0.0}

    def test_dict_capacity(self):
        pool = ResourcePool(total_capacity={"gpu": 4.0, "ram_gb": 64.0})
        snap = pool.snapshot()
        assert snap.total == {"gpu": 4.0, "ram_gb": 64.0}

    def test_reserve_release_cycle(self):
        pool = ResourcePool(total_capacity={"slots": 10.0})
        task = _make_task(metadata={"resources": {"slots": 3.0}})

        res_id = pool.reserve(task)
        snap = pool.snapshot()
        assert snap.available["slots"] == 7.0
        assert snap.reserved["slots"] == 3.0
        assert snap.active_reservations == 1

        pool.release(res_id)
        snap2 = pool.snapshot()
        assert snap2.available["slots"] == 10.0
        assert snap2.reserved["slots"] == 0.0
        assert snap2.active_reservations == 0

    def test_can_admit_accepted_waiting_rejected(self):
        pool = ResourcePool(total_capacity={"slots": 5.0})

        # ACCEPTED
        dec1 = pool.can_admit({"slots": 3.0})
        assert dec1.decision == DecisionType.ACCEPTED

        # Reserve to reduce available
        pool.reserve({"slots": 3.0})

        # WAITING (3 > 2 available, but 3 <= 5 total)
        dec2 = pool.can_admit({"slots": 3.0})
        assert dec2.decision == DecisionType.WAITING

        # REJECTED (exceeds total)
        dec3 = pool.can_admit({"slots": 6.0})
        assert dec3.decision == DecisionType.REJECTED

    def test_invariant_always_holds(self):
        pool = ResourcePool(total_capacity={"a": 10.0, "b": 20.0})
        r1 = pool.reserve({"a": 5.0, "b": 10.0})
        r2 = pool.reserve({"a": 3.0, "b": 5.0})

        snap = pool.snapshot()
        for key in snap.total:
            assert round(snap.available[key] + snap.reserved[key], 6) == round(
                snap.total[key], 6
            )

        pool.release(r1)
        pool.release(r2)

        snap2 = pool.snapshot()
        assert snap2.available == snap2.total
        assert all(v == 0.0 for v in snap2.reserved.values())

    def test_double_release_raises(self):
        pool = ResourcePool(total_capacity=100.0)
        rid = pool.reserve({"units": 10.0})
        pool.release(rid)
        with pytest.raises(ResourceError):
            pool.release(rid)

    def test_release_by_task(self):
        """release_by_task() lets callers free capacity without tracking ReservationID."""
        pool = ResourcePool(total_capacity={"slots": 10.0})
        task = _make_task(metadata={"resources": {"slots": 4.0}})

        pool.reserve(task)
        snap = pool.snapshot()
        assert snap.available["slots"] == 6.0

        pool.release_by_task(task.task_id)
        snap2 = pool.snapshot()
        assert snap2.available["slots"] == 10.0

    def test_release_by_task_unknown_raises(self):
        pool = ResourcePool(total_capacity=100.0)
        with pytest.raises(ResourceError):
            pool.release_by_task(TaskID.generate())


# ===========================================================================
# 9. Scheduler.admit() on the Central Path
# ===========================================================================


class TestSchedulerAdmit:
    """Verify admit() exists on Scheduler and delegates correctly."""

    def test_admit_accepted(self):
        scheduler = Scheduler(
            strategy=SchedulingStrategy.FIFO,
            resource_capacities={ResourceType.LLM: {"slots": 10.0}},
        )
        req = _make_request(metadata={"resources": {"slots": 3.0}})
        decision = scheduler.admit(req)
        assert decision.decision == DecisionType.ACCEPTED
        scheduler.close()

    def test_admit_rejected(self):
        scheduler = Scheduler(
            strategy=SchedulingStrategy.FIFO,
            resource_capacities={ResourceType.LLM: {"slots": 5.0}},
        )
        req = _make_request(metadata={"resources": {"slots": 99.0}})
        decision = scheduler.admit(req)
        assert decision.decision == DecisionType.REJECTED
        scheduler.close()

    def test_admit_waiting(self):
        scheduler = Scheduler(
            strategy=SchedulingStrategy.FIFO,
            resource_capacities={ResourceType.LLM: {"slots": 10.0}},
        )
        # Reserve most of the capacity via pool
        pool = scheduler.get_pool(ResourceType.LLM)
        pool.reserve({"slots": 8.0})

        req = _make_request(metadata={"resources": {"slots": 5.0}})
        decision = scheduler.admit(req)
        assert decision.decision == DecisionType.WAITING
        scheduler.close()

    def test_admit_on_closed_scheduler_raises(self):
        scheduler = Scheduler()
        scheduler.close()
        with pytest.raises(SchedulerError):
            scheduler.admit(_make_request())

    def test_admit_with_explicit_resource_type(self):
        scheduler = Scheduler(
            resource_capacities={ResourceType.STORAGE: {"disk_gb": 50.0}},
        )
        task = _make_task(metadata={"resources": {"disk_gb": 10.0}})
        decision = scheduler.admit(task, resource_type=ResourceType.STORAGE)
        assert decision.decision == DecisionType.ACCEPTED
        scheduler.close()


# ===========================================================================
# 10. Scheduler.release_by_task() on the Central Path
# ===========================================================================


class TestSchedulerReleaseByTask:
    """Verify task-id-keyed release works through the central Scheduler."""

    def test_reserve_and_release_by_task(self):
        scheduler = Scheduler(
            resource_capacities={ResourceType.LLM: {"slots": 10.0}},
        )
        task = _make_task(metadata={"resources": {"slots": 5.0}})
        pool = scheduler.get_pool(ResourceType.LLM)

        pool.reserve(task)
        snap = pool.snapshot()
        assert snap.available["slots"] == 5.0

        scheduler.release_by_task(task.task_id, resource_type=ResourceType.LLM)
        snap2 = pool.snapshot()
        assert snap2.available["slots"] == 10.0

    def test_release_by_task_unknown_raises(self):
        scheduler = Scheduler()
        with pytest.raises(ResourceError):
            scheduler.release_by_task(TaskID.generate())


# ===========================================================================
# 11. Import Surface Verification
# ===========================================================================


class TestImportSurface:
    """
    Verify that every symbol the spec requires can be imported from the
    expected module path.
    """

    def test_import_resource_pool(self):
        from aios.scheduler import ResourcePool
        assert ResourcePool is not None

    def test_import_resource_pool_from_module(self):
        from aios.scheduler.resource_pool import ResourcePool
        assert ResourcePool is not None

    def test_import_decision_types(self):
        from aios.scheduler import AdmissionDecision, DecisionType
        assert DecisionType.ACCEPTED is not None
        assert AdmissionDecision is not None

    def test_import_reservation_id(self):
        from aios.scheduler import ReservationID
        rid = ReservationID.generate()
        assert rid.value is not None

    def test_scheduler_has_admit(self):
        assert hasattr(Scheduler, "admit")

    def test_scheduler_has_release_by_task(self):
        assert hasattr(Scheduler, "release_by_task")

    def test_scheduler_has_get_pool(self):
        assert hasattr(Scheduler, "get_pool")
