"""
Resource-specific queue management.

``ResourceScheduler`` handles queue ordering (FIFO / Round Robin / Priority)
and preemption. It **delegates** all capacity accounting to a ``ResourcePool``
instance, keeping the two concerns cleanly separated.

Separation rationale (addresses review finding):
    resource_pool.py  — pure capacity ledger: can_admit, reserve, release,
                        snapshot, invariant assertion.  No queues.
    resource_scheduler.py (this file) — queue ordering, blocked-request
                        tracking, preemption, metrics.  No capacity counters.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from aios.core.exceptions import ResourceError
from aios.core.ids import RequestID, TaskID
from aios.core.models.context import Context
from aios.core.models.request import AgentRequest
from aios.core.models.resource import ResourceType
from aios.core.models.task import Task
from aios.scheduler.priority import PriorityRequestQueue
from aios.scheduler.resource_pool import (
    AdmissionDecision,
    DecisionType,
    ReservationID,
    ResourcePool,
    ResourceState,
)


class SchedulingStrategy:
    """Supported scheduling strategies (plain string constants)."""

    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


class ResourceScheduler:
    """
    Resource-specific queue manager.

    Handles:
    - Enqueueing / dequeueing according to FIFO, Round Robin, or Priority.
    - Round Robin preemption with context-switch hook (Phase 6 integration).
    - Block / unblock tracking.

    Capacity accounting is fully delegated to ``self.pool`` (a ``ResourcePool``).
    """

    def __init__(
        self,
        resource_type: ResourceType | str = ResourceType.LLM,
        strategy: str = SchedulingStrategy.FIFO,
        total_capacity: dict[str, float] | float = 100.0,
        quantum: float | int | None = None,
        context_switch_hook: Callable[[AgentRequest, Any], None] | None = None,
        aging_threshold: int = 5,
    ) -> None:
        if isinstance(resource_type, str):
            try:
                self.resource_type = ResourceType(resource_type)
            except ValueError:
                self.resource_type = ResourceType.LLM
        else:
            self.resource_type = resource_type

        self.strategy = strategy
        self.quantum = quantum
        self.context_switch_hook = context_switch_hook
        self.aging_threshold = aging_threshold

        # Capacity accounting — delegated to ResourcePool
        self.pool = ResourcePool(total_capacity=total_capacity)

        # Queues
        self._fifo_queue: deque[AgentRequest] = deque()
        self._rr_queue: deque[AgentRequest] = deque()
        self._priority_queue = PriorityRequestQueue(
            aging_threshold=self.aging_threshold,
        )

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

    # ------------------------------------------------------------------
    # Capacity delegation (thin wrappers so callers can still call these
    # on ResourceScheduler without knowing about ResourcePool)
    # ------------------------------------------------------------------

    def can_admit(
        self, item: Task | AgentRequest | dict[str, float]
    ) -> AdmissionDecision:
        """Delegate to ``self.pool.can_admit``."""
        return self.pool.can_admit(item)

    def reserve(
        self, item: Task | AgentRequest | dict[str, float]
    ) -> ReservationID:
        """Delegate to ``self.pool.reserve``."""
        return self.pool.reserve(item)

    def release(self, reservation_id: ReservationID) -> None:
        """Delegate to ``self.pool.release``."""
        self.pool.release(reservation_id)

    def release_by_task(self, task_id: TaskID) -> None:
        """Delegate to ``self.pool.release_by_task``."""
        self.pool.release_by_task(task_id)

    def snapshot(self) -> ResourceState:
        """Delegate to ``self.pool.snapshot``."""
        return self.pool.snapshot()

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def submit(self, request: AgentRequest, priority: int = 0) -> None:
        """Add a request to the resource queue."""
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
        Fetch the next eligible request.  Skips blocked requests.
        Returns None if empty or all blocked.
        """
        now = datetime.now(timezone.utc)
        selected: AgentRequest | None = None

        if self.strategy == SchedulingStrategy.FIFO:
            selected = self._pop_unblocked_fifo()

        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            selected = self._pop_unblocked_rr()

        elif self.strategy == SchedulingStrategy.PRIORITY:
            selected = self._pop_unblocked_priority()

        if selected is not None:
            self._current_request = selected
            if selected.request_id not in self._started_at:
                self._started_at[selected.request_id] = now
                wait_sec = (
                    now - self._queued_at[selected.request_id]
                ).total_seconds()
                self._total_wait_time += max(0.0, wait_sec)

        return selected

    # ------ internal queue helpers ------------------------------------

    def _pop_unblocked_fifo(self) -> AgentRequest | None:
        temp: deque[AgentRequest] = deque()
        selected = None
        while self._fifo_queue:
            req = self._fifo_queue.popleft()
            if req.request_id in self._blocked_requests:
                temp.append(req)
            else:
                selected = req
                break
        while temp:
            self._fifo_queue.appendleft(temp.pop())
        return selected

    def _pop_unblocked_rr(self) -> AgentRequest | None:
        temp: deque[AgentRequest] = deque()
        selected = None
        while self._rr_queue:
            req = self._rr_queue.popleft()
            if req.request_id in self._blocked_requests:
                temp.append(req)
            else:
                selected = req
                break
        while temp:
            self._rr_queue.appendleft(temp.pop())
        return selected

    def _pop_unblocked_priority(self) -> AgentRequest | None:
        if self._priority_queue.is_empty():
            return None
        temp_reqs: list[AgentRequest] = []
        selected = None
        while not self._priority_queue.is_empty():
            req = self._priority_queue.dequeue()
            if req.request_id in self._blocked_requests:
                temp_reqs.append(req)
            else:
                selected = req
                break
        for r in temp_reqs:
            self._priority_queue.enqueue(r)
        return selected

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def preempt(
        self,
        request_id: RequestID,
        partial_context: Any = None,
    ) -> AgentRequest | None:
        """
        Interrupt an in-flight request, snapshot its context, requeue it.
        """
        target = self._all_requests.get(request_id)
        if target is None:
            return None

        self._preemption_counts[request_id] = (
            self._preemption_counts.get(request_id, 0) + 1
        )

        updated_request = target
        if partial_context is not None:
            if isinstance(partial_context, Context):
                new_ctx = partial_context
            elif isinstance(target.context, Context):
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
                    working_state={
                        "preempted_partial_output": partial_context,
                    },
                )

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

        if self.context_switch_hook is not None:
            self.context_switch_hook(updated_request, partial_context)

        # Re-queue at tail
        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            self._rr_queue.append(updated_request)
        elif self.strategy == SchedulingStrategy.FIFO:
            self._fifo_queue.append(updated_request)
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self._priority_queue.enqueue(updated_request)

        if (
            self._current_request
            and self._current_request.request_id == request_id
        ):
            self._current_request = None

        return updated_request

    # ------------------------------------------------------------------
    # Completion / block / unblock
    # ------------------------------------------------------------------

    def complete(self, request_id: RequestID) -> bool:
        """Mark a request as finished."""
        now = datetime.now(timezone.utc)
        if request_id in self._all_requests:
            self._completed_at[request_id] = now
            self._completed_count += 1
            if (
                self._current_request
                and self._current_request.request_id == request_id
            ):
                self._current_request = None
            return True
        return False

    def block(self, request_id: RequestID, reason: str = "") -> bool:
        req = self._all_requests.get(request_id)
        if req is not None:
            self._blocked_requests[request_id] = (req, reason)
            return True
        return False

    def unblock(self, request_id: RequestID) -> bool:
        if request_id in self._blocked_requests:
            del self._blocked_requests[request_id]
            return True
        return False

    def is_all_blocked(self) -> bool:
        sz = self.size()
        if sz == 0:
            return False
        return len(self._blocked_requests) >= sz

    def size(self) -> int:
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
