"""
Central AIOS Scheduler.

The Scheduler coordinates central request queues for all four AIOS resource
types (LLM, Memory, Storage, Tool) under one unified interface with swappable
strategies (FIFO, Round Robin with preemption, or Priority with anti-starvation).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

from aios.core.exceptions import SchedulerError, ValidationError
from aios.core.ids import RequestID
from aios.core.models.request import AgentRequest
from aios.core.models.resource import ResourceType
from aios.core.models.syscall import SystemCallType
from aios.scheduler.resource_scheduler import (
    AdmissionDecision,
    DecisionType,
    ReservationID,
    ResourceScheduler,
    ResourceState,
    SchedulingStrategy,
)


def map_syscall_to_resource_type(call_type: SystemCallType) -> ResourceType:
    """Map an AIOS SystemCallType to its corresponding ResourceType."""
    if call_type == SystemCallType.LLM_CALL:
        return ResourceType.LLM
    elif call_type in (SystemCallType.MEMORY_READ, SystemCallType.MEMORY_WRITE):
        return ResourceType.MEMORY
    elif call_type in (SystemCallType.STORAGE_READ, SystemCallType.STORAGE_WRITE):
        return ResourceType.STORAGE
    elif call_type == SystemCallType.TOOL_CALL:
        return ResourceType.TOOL
    return ResourceType.LLM


class Scheduler:
    """
    Central AIOS Resource Scheduler.

    Features:
    - Central queues per resource type (LLM, Memory, Storage, Tool).
    - Swappable FIFO / Round Robin / Priority strategies behind one interface.
    - Public API: submit(request), next() -> Request, close().
    - Preemption / interrupt handling for in-flight requests.
    """

    def __init__(
        self,
        strategy: SchedulingStrategy | str = SchedulingStrategy.FIFO,
        quantum: float | int | None = None,
        context_switch_hook: Callable[[AgentRequest, Any], None] | None = None,
        resource_capacities: dict[ResourceType, float] | None = None,
        aging_threshold: int = 5,
    ) -> None:
        """
        Initialize the central scheduler.

        Args:
            strategy: Default scheduling policy ("fifo", "round_robin", "priority").
            quantum: Time slice or token limit for Round Robin.
            context_switch_hook: Phase 6 context save/restore callback.
            resource_capacities: Initial capacities per resource type.
            aging_threshold: Aging threshold for priority starvation prevention.
        """
        if isinstance(strategy, str):
            self.strategy = SchedulingStrategy(strategy.lower())
        else:
            self.strategy = strategy

        self.quantum = quantum
        self.context_switch_hook = context_switch_hook
        self.aging_threshold = aging_threshold
        self._closed: bool = False

        # Central queues per resource type
        capacities = resource_capacities or {}
        self._schedulers: dict[ResourceType, ResourceScheduler] = {
            res_type: ResourceScheduler(
                resource_type=res_type,
                strategy=self.strategy,
                total_capacity=capacities.get(res_type, 100.0),
                quantum=self.quantum,
                context_switch_hook=self.context_switch_hook,
                aging_threshold=self.aging_threshold,
            )
            for res_type in ResourceType
        }

        # Track which resource type owns which request
        self._request_resources: dict[RequestID, ResourceType] = {}
        # Cycle order for arbitrating between resource types when next() is called with None
        self._resource_order = list(ResourceType)
        self._round_robin_idx: int = 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    def get_resource_scheduler(self, resource_type: ResourceType) -> ResourceScheduler:
        """Access the underlying ResourceScheduler for a specific resource type."""
        return self._schedulers[resource_type]

    def submit(self, request: AgentRequest, priority: int = 0) -> None:
        """
        Submit an AgentRequest to the appropriate resource queue.

        The system call in request determines the resource category:
            LLM_CALL       -> LLM queue
            MEMORY_*       -> Memory queue
            STORAGE_*      -> Storage queue
            TOOL_CALL      -> Tool queue
        """
        if self._closed:
            raise SchedulerError("Cannot submit request: scheduler is closed.")

        if not isinstance(request, AgentRequest):
            raise ValidationError("request must be an AgentRequest instance.")

        res_type = map_syscall_to_resource_type(request.syscall.call_type)
        self._request_resources[request.request_id] = res_type
        self._schedulers[res_type].submit(request, priority=priority)

    def next(
        self,
        resource_type: ResourceType | str | None = None,
    ) -> AgentRequest | None:
        """
        Fetch the next request to be dispatched.

        Args:
            resource_type: If provided, fetches from that specific resource queue.
                If None, arbitrates across all resource queues in a fair round-robin order.

        Returns:
            The next AgentRequest to run, or None if the queue is empty or all-blocked.
        """
        if self._closed:
            raise SchedulerError("Cannot fetch next request: scheduler is closed.")

        if resource_type is not None:
            if isinstance(resource_type, str):
                try:
                    res_type = ResourceType(resource_type.lower())
                except ValueError:
                    res_type = ResourceType.LLM
            else:
                res_type = resource_type

            scheduler = self._schedulers.get(res_type)
            if scheduler is None:
                return None
            return scheduler.next_request()

        # When resource_type is None: Fair arbitration across resource types
        n = len(self._resource_order)
        for i in range(n):
            idx = (self._round_robin_idx + i) % n
            res_type = self._resource_order[idx]
            scheduler = self._schedulers[res_type]
            if not scheduler.is_empty():
                req = scheduler.next_request()
                if req is not None:
                    self._round_robin_idx = (idx + 1) % n
                    return req

        return None

    def preempt(
        self,
        request_id: RequestID,
        partial_context: Any = None,
    ) -> AgentRequest | None:
        """
        Interrupt an in-flight request, snapshot its context, and requeue it.
        """
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return None
        return self._schedulers[res_type].preempt(request_id, partial_context=partial_context)

    def complete(self, request_id: RequestID) -> bool:
        """Mark a request as completed across the scheduler."""
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].complete(request_id)

    def block(self, request_id: RequestID, reason: str = "") -> bool:
        """Mark a request as blocked."""
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].block(request_id, reason=reason)

    def unblock(self, request_id: RequestID) -> bool:
        """Unblock a request."""
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].unblock(request_id)

    def size(self, resource_type: ResourceType | None = None) -> int:
        """Return number of pending requests in one queue or across all queues."""
        if resource_type is not None:
            return self._schedulers[resource_type].size()
        return sum(s.size() for s in self._schedulers.values())

    def is_empty(self, resource_type: ResourceType | None = None) -> bool:
        """Return True if empty."""
        return self.size(resource_type=resource_type) == 0

    def get_metrics(self) -> dict[str, Any]:
        """Aggregate performance and latency metrics across all resource types."""
        return {
            res_type.value: scheduler.get_metrics()
            for res_type, scheduler in self._schedulers.items()
        }

    def close(self) -> None:
        """Close the scheduler and release queued resources."""
        self._closed = True
        self._request_resources.clear()
