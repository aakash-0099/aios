"""
Resource-specific queue management and scheduling decisions.

This module provides:
1. `ResourceScheduler`: Manages resource-specific queues, supporting FIFO,
   Round Robin with preemption/interrupts (hook for Phase 6 Context Manager),
   and Priority scheduling with anti-starvation.
2. Capacity admission decisions (ACCEPTED, WAITING, REJECTED) ensuring the
   fundamental invariant: available + reserved == total.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from aios.core.exceptions import ResourceError, ValidationError
from aios.core.ids import RequestID, TaskID
from aios.core.models.context import Context
from aios.core.models.request import AgentRequest
from aios.core.models.resource import Resource, ResourceType
from aios.core.models.task import Task
from aios.scheduler.priority import PriorityRequestQueue


class SchedulingStrategy(str, Enum):
    """Supported scheduling strategies."""
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


class DecisionType(str, Enum):
    """
    Admission decision types.

    ACCEPTED: Fits within current available capacity.
    WAITING: Exceeds current available capacity, but does not exceed total
             capacity (can be satisfied later when resources are released).
    REJECTED: Exceeds total capacity (can never be satisfied).
    """
    ACCEPTED = "accepted"
    WAITING = "waiting"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReservationID:
    """Unique identifier for a resource reservation."""
    value: UUID

    @classmethod
    def generate(cls) -> ReservationID:
        return cls(uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class AdmissionDecision:
    """Decision returned by can_admit."""
    decision: DecisionType
    reason: str
    target_id: RequestID | TaskID | None = None
    required_resources: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceState:
    """Immutable snapshot of resource allocations."""
    total: dict[str, float]
    available: dict[str, float]
    reserved: dict[str, float]
    active_reservations: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ResourceScheduler:
    """
    Resource-specific queue manager and admission scheduler.

    Handles:
    - Enqueueing and dequeueing according to FIFO, Round Robin, or Priority.
    - Round Robin preemption: tasks exceeding their time slice or token budget
      can be interrupted, snapshotted, and requeued at the tail of the queue.
    - Capacity accounting: tracks total, available, and reserved capacities.
    - Enforces invariant: available + reserved == total at all times.
    """

    def __init__(
        self,
        resource_type: ResourceType | str = ResourceType.LLM,
        strategy: SchedulingStrategy | str = SchedulingStrategy.FIFO,
        total_capacity: dict[str, float] | float = 100.0,
        quantum: float | int | None = None,
        context_switch_hook: Callable[[AgentRequest, Any], None] | None = None,
        aging_threshold: int = 5,
    ) -> None:
        """
        Initialize the resource scheduler.

        Args:
            resource_type: Resource category (LLM, MEMORY, STORAGE, TOOL).
            strategy: Scheduling policy ("fifo", "round_robin", "priority").
            total_capacity: Total capacity mapping (e.g. {"slots": 10.0}) or scalar.
            quantum: Time slice or token limit for Round Robin preemption.
            context_switch_hook: Callback invoked when a request is preempted.
            aging_threshold: Dequeue cycles before boosting priority in Priority strategy.
        """
        if isinstance(resource_type, str):
            try:
                self.resource_type = ResourceType(resource_type)
            except ValueError:
                self.resource_type = ResourceType.LLM
        else:
            self.resource_type = resource_type

        if isinstance(strategy, str):
            self.strategy = SchedulingStrategy(strategy.lower())
        else:
            self.strategy = strategy

        self.quantum = quantum
        self.context_switch_hook = context_switch_hook
        self.aging_threshold = aging_threshold

        # Capacity management
        self._total: dict[str, float] = {}
        if isinstance(total_capacity, (int, float)):
            self._total["units"] = float(total_capacity)
        elif isinstance(total_capacity, dict):
            self._total = {k: float(v) for k, v in total_capacity.items()}
        else:
            raise ValidationError("total_capacity must be a float or a dictionary.")

        self._available: dict[str, float] = dict(self._total)
        self._reserved: dict[str, float] = {k: 0.0 for k in self._total}
        self._reservations: dict[ReservationID, dict[str, float]] = {}

        # Queues & internal state
        self._fifo_queue: deque[AgentRequest] = deque()
        self._rr_queue: deque[AgentRequest] = deque()
        self._priority_queue = PriorityRequestQueue(aging_threshold=self.aging_threshold)

        # In-flight and blocked tracking
        self._current_request: AgentRequest | None = None
        self._blocked_requests: dict[RequestID, tuple[AgentRequest, str]] = {}
        self._all_requests: dict[RequestID, AgentRequest] = {}

        # Metrics
        self._queued_at: dict[RequestID, datetime] = {}
        self._started_at: dict[RequestID, datetime] = {}
        self._completed_at: dict[RequestID, datetime] = {}
        self._preemption_counts: dict[RequestID, int] = {}
        self._total_wait_time: float = 0.0
        self._completed_count: int = 0

        self._assert_invariants()

    def _assert_invariants(self) -> None:
        """Enforce: available + reserved == total for all resource dimensions."""
        for key, total_val in self._total.items():
            avail = self._available.get(key, 0.0)
            res = self._reserved.get(key, 0.0)
            if round(avail + res, 6) != round(total_val, 6):
                raise ResourceError(
                    f"Invariant violated for resource '{key}': "
                    f"available ({avail}) + reserved ({res}) != total ({total_val})"
                )

    # -------------------------------------------------------------------------
    # Capacity Accounting & Admission (Phase 1 Member 7 Contract)
    # -------------------------------------------------------------------------

    def _extract_demands(self, item: Task | AgentRequest | dict[str, float]) -> dict[str, float]:
        """Extract resource requirements dictionary from task, request, or dict."""
        if isinstance(item, dict):
            return {k: float(v) for k, v in item.items()}

        if isinstance(item, Task):
            if "resources" in item.metadata and isinstance(item.metadata["resources"], dict):
                return {k: float(v) for k, v in item.metadata["resources"].items()}
            return {"units": 1.0}

        if isinstance(item, AgentRequest):
            if "resources" in item.metadata and isinstance(item.metadata["resources"], dict):
                return {k: float(v) for k, v in item.metadata["resources"].items()}
            return {"units": 1.0}

        return {"units": 1.0}

    def can_admit(self, item: Task | AgentRequest | dict[str, float]) -> AdmissionDecision:
        """
        Check if an item can be admitted now, later, or never.

        Returns:
            AdmissionDecision with ACCEPTED, WAITING, or REJECTED.
        """
        demands = self._extract_demands(item)
        target_id = getattr(item, "request_id", None) or getattr(item, "task_id", None)

        # 1. Check if demands exceed TOTAL capacity (REJECTED)
        for key, req_val in demands.items():
            tot = self._total.get(key)
            if tot is None or req_val > tot:
                return AdmissionDecision(
                    decision=DecisionType.REJECTED,
                    reason=(
                        f"Requested {key}={req_val} exceeds total system capacity "
                        f"{tot if tot is not None else 0.0}."
                    ),
                    target_id=target_id,
                    required_resources=demands,
                )

        # 2. Check if demands exceed AVAILABLE capacity (WAITING)
        for key, req_val in demands.items():
            avail = self._available.get(key, 0.0)
            if req_val > avail:
                return AdmissionDecision(
                    decision=DecisionType.WAITING,
                    reason=(
                        f"Requested {key}={req_val} exceeds currently available capacity "
                        f"{avail} (total capacity is {self._total.get(key)})."
                    ),
                    target_id=target_id,
                    required_resources=demands,
                )

        # 3. Fits inside available capacity (ACCEPTED)
        return AdmissionDecision(
            decision=DecisionType.ACCEPTED,
            reason="Required resources are currently available.",
            target_id=target_id,
            required_resources=demands,
        )

    def reserve(self, item: Task | AgentRequest | dict[str, float]) -> ReservationID:
        """
        Reserve capacity for an admitted item.

        Raises:
            ResourceError if resources cannot be admitted.
        """
        decision = self.can_admit(item)
        if decision.decision != DecisionType.ACCEPTED:
            raise ResourceError(f"Cannot reserve: {decision.reason}")

        demands = decision.required_resources
        reservation_id = ReservationID.generate()

        for key, req_val in demands.items():
            self._available[key] -= req_val
            self._reserved[key] += req_val

        self._reservations[reservation_id] = demands
        self._assert_invariants()
        return reservation_id

    def release(self, reservation_id: ReservationID) -> None:
        """
        Release previously reserved capacity.

        Raises:
            ResourceError if reservation_id is unknown or already released.
        """
        if not isinstance(reservation_id, ReservationID) or reservation_id not in self._reservations:
            raise ResourceError(f"Unknown or already released reservation: {reservation_id}")

        demands = self._reservations.pop(reservation_id)
        for key, req_val in demands.items():
            self._available[key] += req_val
            self._reserved[key] -= req_val

        self._assert_invariants()

    def snapshot(self) -> ResourceState:
        """Return an immutable snapshot of resource capacity."""
        return ResourceState(
            total=dict(self._total),
            available=dict(self._available),
            reserved=dict(self._reserved),
            active_reservations=len(self._reservations),
        )

    # -------------------------------------------------------------------------
    # Queue Management & Preemption (Phase 2 & Phase 5 Contract)
    # -------------------------------------------------------------------------

    def submit(self, request: AgentRequest, priority: int = 0) -> None:
        """
        Add a request to the resource queue.
        """
        now = datetime.now(timezone.utc)
        self._all_requests[request.request_id] = request
        self._queued_at[request.request_id] = now
        self._preemption_counts.setdefault(request.request_id, 0)

        if self.strategy == SchedulingStrategy.FIFO:
            self._fifo_queue.append(request)
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            self._rr_queue.append(request)
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self._priority_queue.enqueue(request, priority=priority)

    def next_request(self) -> AgentRequest | None:
        """
        Fetch the next eligible request according to the active strategy.
        Skips blocked requests. Returns None if empty or all blocked.
        """
        now = datetime.now(timezone.utc)

        if self.strategy == SchedulingStrategy.FIFO:
            # Pop next unblocked
            temp = deque()
            selected = None
            while self._fifo_queue:
                req = self._fifo_queue.popleft()
                if req.request_id in self._blocked_requests:
                    temp.append(req)
                else:
                    selected = req
                    break
            # Restore skipped blocked requests
            while temp:
                self._fifo_queue.appendleft(temp.pop())

        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            temp = deque()
            selected = None
            while self._rr_queue:
                req = self._rr_queue.popleft()
                if req.request_id in self._blocked_requests:
                    temp.append(req)
                else:
                    selected = req
                    break
            # Restore skipped blocked requests
            while temp:
                self._rr_queue.appendleft(temp.pop())

        elif self.strategy == SchedulingStrategy.PRIORITY:
            if self._priority_queue.is_empty():
                selected = None
            else:
                # Priority queue pop with unblocked filtering
                temp_reqs = []
                selected = None
                while not self._priority_queue.is_empty():
                    req = self._priority_queue.dequeue()
                    if req.request_id in self._blocked_requests:
                        temp_reqs.append(req)
                    else:
                        selected = req
                        break
                # Re-enqueue blocked ones
                for r in temp_reqs:
                    self._priority_queue.enqueue(r)

        if selected is not None:
            self._current_request = selected
            if selected.request_id not in self._started_at:
                self._started_at[selected.request_id] = now
                wait_sec = (now - self._queued_at[selected.request_id]).total_seconds()
                self._total_wait_time += max(0.0, wait_sec)

        return selected

    def preempt(
        self,
        request_id: RequestID,
        partial_context: Any = None,
    ) -> AgentRequest | None:
        """
        Interrupt an in-flight request, snapshot its context, and requeue it.

        This implements Round Robin preemption (Phase 5 / Phase 6 integration).
        """
        target = self._all_requests.get(request_id)
        if target is None:
            return None

        # Increment preemption count
        self._preemption_counts[request_id] = (
            self._preemption_counts.get(request_id, 0) + 1
        )

        # Update context if partial context provided
        updated_request = target
        if partial_context is not None:
            if isinstance(partial_context, Context):
                new_ctx = partial_context
            elif isinstance(target.context, Context):
                # Merge into existing context's working_state
                state = dict(target.context.working_state)
                state["preempted_partial_output"] = partial_context
                new_ctx = Context(
                    system=target.context.system,
                    conversation=list(target.context.conversation),
                    working_state=state,
                    memory=list(target.context.memory),
                )
            else:
                new_ctx = Context(
                    system="System",
                    working_state={"preempted_partial_output": partial_context},
                )

            # Create updated request representation
            metadata = dict(target.metadata)
            metadata["preemptions"] = self._preemption_counts[request_id]
            updated_request = AgentRequest(
                request_id=target.request_id,
                agent_id=target.agent_id,
                task_id=target.task_id,
                syscall=target.syscall,
                context=new_ctx,
                metadata=metadata,
            )
            self._all_requests[request_id] = updated_request

        # Invoke hook if registered
        if self.context_switch_hook is not None:
            self.context_switch_hook(updated_request, partial_context)

        # Place at tail of round-robin queue (or priority/fifo)
        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            self._rr_queue.append(updated_request)
        elif self.strategy == SchedulingStrategy.FIFO:
            self._fifo_queue.append(updated_request)
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self._priority_queue.enqueue(updated_request)

        if self._current_request and self._current_request.request_id == request_id:
            self._current_request = None

        return updated_request

    def complete(self, request_id: RequestID) -> bool:
        """Mark a request as finished and record completion metrics."""
        now = datetime.now(timezone.utc)
        if request_id in self._all_requests:
            self._completed_at[request_id] = now
            self._completed_count += 1
            if self._current_request and self._current_request.request_id == request_id:
                self._current_request = None
            return True
        return False

    def block(self, request_id: RequestID, reason: str = "") -> bool:
        """Mark a request as blocked (waiting on I/O, event, or resource)."""
        req = self._all_requests.get(request_id)
        if req is not None:
            self._blocked_requests[request_id] = (req, reason)
            return True
        return False

    def unblock(self, request_id: RequestID) -> bool:
        """Unblock a request, allowing it to be scheduled again."""
        if request_id in self._blocked_requests:
            del self._blocked_requests[request_id]
            return True
        return False

    def is_all_blocked(self) -> bool:
        """Return True if queue has items and every single one is blocked."""
        sz = self.size()
        if sz == 0:
            return False
        return len(self._blocked_requests) >= sz

    def size(self) -> int:
        """Return total number of pending/queued requests."""
        if self.strategy == SchedulingStrategy.FIFO:
            return len(self._fifo_queue)
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            return len(self._rr_queue)
        elif self.strategy == SchedulingStrategy.PRIORITY:
            return self._priority_queue.size()
        return 0

    def is_empty(self) -> bool:
        return self.size() == 0

    def get_metrics(self) -> dict[str, Any]:
        """Return performance metrics."""
        avg_wait = (
            self._total_wait_time / self._completed_count
            if self._completed_count > 0
            else 0.0
        )
        return {
            "completed_count": self._completed_count,
            "pending_count": self.size(),
            "average_wait_seconds": avg_wait,
            "total_preemptions": sum(self._preemption_counts.values()),
        }
