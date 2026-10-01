"""
Central AIOS Scheduler.

Coordinates central request queues for all four AIOS resource types
(LLM, Memory, Storage, Tool) under one unified interface with swappable
strategies (FIFO, Round Robin with preemption, or Priority with
anti-starvation).

Addresses review finding: ``admit()`` is now a first-class method on
``Scheduler``.  It delegates to the per-resource ``ResourcePool.can_admit``
so that admission is reachable from the central dispatch path (not only
via direct ``ResourceScheduler`` access).
"""

from __future__ import annotations

from typing import Any, Callable

from aios.core.exceptions import SchedulerError, ValidationError
from aios.core.ids import RequestID, TaskID
from aios.core.models.request import AgentRequest
from aios.core.models.resource import ResourceType
from aios.core.models.syscall import SystemCallType
from aios.core.models.task import Task
from aios.scheduler.resource_pool import (
    AdmissionDecision,
    DecisionType,
    ReservationID,
    ResourcePool,
    ResourceState,
)
from aios.scheduler.resource_scheduler import (
    ResourceScheduler,
    SchedulingStrategy,
)


def map_syscall_to_resource_type(call_type: SystemCallType) -> ResourceType:
    """Map an AIOS SystemCallType to its corresponding ResourceType."""
    if call_type == SystemCallType.LLM_CALL:
        return ResourceType.LLM
    elif call_type in (SystemCallType.MEMORY_READ, SystemCallType.MEMORY_WRITE):
        return ResourceType.MEMORY
    elif call_type in (
        SystemCallType.STORAGE_READ,
        SystemCallType.STORAGE_WRITE,
    ):
        return ResourceType.STORAGE
    elif call_type == SystemCallType.TOOL_CALL:
        return ResourceType.TOOL
    return ResourceType.LLM


class Scheduler:
    """
    Central AIOS Resource Scheduler.

    Public API:
        submit(request)                  — enqueue to the correct resource queue
        admit(request_or_task)           — capacity admission check (delegates to ResourcePool)
        next(resource_type=None)         — fetch next eligible request
        preempt(request_id, ctx)         — interrupt and requeue
        complete(request_id)             — mark done
        release(reservation_id)          — free reserved capacity by ReservationID
        release_by_task(task_id)          — free reserved capacity by TaskID
        close()                          — shut down
    """

    def __init__(
        self,
        strategy: str = SchedulingStrategy.FIFO,
        quantum: float | int | None = None,
        context_switch_hook: Callable[[AgentRequest, Any], None] | None = None,
        resource_capacities: dict[ResourceType, float | dict[str, float]]
        | None = None,
        aging_threshold: int = 5,
    ) -> None:
        self.strategy = strategy
        self.quantum = quantum
        self.context_switch_hook = context_switch_hook
        self.aging_threshold = aging_threshold
        self._closed: bool = False

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

        self._request_resources: dict[RequestID, ResourceType] = {}
        self._resource_order = list(ResourceType)
        self._round_robin_idx: int = 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    def get_resource_scheduler(
        self, resource_type: ResourceType
    ) -> ResourceScheduler:
        """Access the underlying per-resource scheduler."""
        return self._schedulers[resource_type]

    def get_pool(self, resource_type: ResourceType) -> ResourcePool:
        """Access the underlying ResourcePool for a resource type."""
        return self._schedulers[resource_type].pool

    # ------------------------------------------------------------------
    # Admission (addresses review: "admit() absent on Scheduler")
    # ------------------------------------------------------------------

    def admit(
        self,
        item: AgentRequest | Task | dict[str, float],
        resource_type: ResourceType | None = None,
    ) -> AdmissionDecision:
        """
        Check whether *item* can be admitted given current capacity.

        Determines the resource type automatically from the request's
        syscall when *resource_type* is None and *item* is an AgentRequest.

        Returns:
            AdmissionDecision with ACCEPTED / WAITING / REJECTED.
        """
        if self._closed:
            raise SchedulerError("Cannot admit: scheduler is closed.")

        res_type = self._resolve_resource_type(item, resource_type)
        return self._schedulers[res_type].can_admit(item)

    # ------------------------------------------------------------------
    # Submit / Next
    # ------------------------------------------------------------------

    def submit(self, request: AgentRequest, priority: int = 0) -> None:
        """
        Submit an AgentRequest to the appropriate resource queue.

        The syscall type determines the resource category:
            LLM_CALL       -> LLM queue
            MEMORY_*       -> Memory queue
            STORAGE_*      -> Storage queue
            TOOL_CALL      -> Tool queue
        """
        if self._closed:
            raise SchedulerError(
                "Cannot submit request: scheduler is closed."
            )

        if not isinstance(request, AgentRequest):
            raise ValidationError(
                "request must be an AgentRequest instance."
            )

        res_type = map_syscall_to_resource_type(request.syscall.call_type)
        self._request_resources[request.request_id] = res_type
        self._schedulers[res_type].submit(request, priority=priority)

    def next(
        self,
        resource_type: ResourceType | str | None = None,
    ) -> AgentRequest | None:
        """
        Fetch the next request to be dispatched.

        If *resource_type* is given, fetches from that queue only.
        If None, arbitrates across all resource queues in fair round-robin.
        """
        if self._closed:
            raise SchedulerError(
                "Cannot fetch next request: scheduler is closed."
            )

        if resource_type is not None:
            res_type = self._coerce_resource_type(resource_type)
            scheduler = self._schedulers.get(res_type)
            if scheduler is None:
                return None
            return scheduler.next_request()

        # Fair arbitration across resource types
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

    # ------------------------------------------------------------------
    # Preemption / completion / block / unblock
    # ------------------------------------------------------------------

    def preempt(
        self,
        request_id: RequestID,
        partial_context: Any = None,
    ) -> AgentRequest | None:
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return None
        return self._schedulers[res_type].preempt(
            request_id, partial_context=partial_context
        )

    def complete(self, request_id: RequestID) -> bool:
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].complete(request_id)

    def block(self, request_id: RequestID, reason: str = "") -> bool:
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].block(request_id, reason=reason)

    def unblock(self, request_id: RequestID) -> bool:
        res_type = self._request_resources.get(request_id)
        if res_type is None:
            return False
        return self._schedulers[res_type].unblock(request_id)

    # ------------------------------------------------------------------
    # Capacity release (addresses review: "release on the central path")
    # ------------------------------------------------------------------

    def release(self, reservation_id: ReservationID) -> None:
        """Release reserved capacity by ReservationID (tries all pools)."""
        for scheduler in self._schedulers.values():
            try:
                scheduler.release(reservation_id)
                return
            except Exception:
                continue
        from aios.core.exceptions import ResourceError

        raise ResourceError(
            f"ReservationID {reservation_id} not found in any pool."
        )

    def release_by_task(
        self,
        task_id: TaskID,
        resource_type: ResourceType | None = None,
    ) -> None:
        """Release reserved capacity by TaskID."""
        if resource_type is not None:
            self._schedulers[resource_type].release_by_task(task_id)
            return
        for scheduler in self._schedulers.values():
            try:
                scheduler.release_by_task(task_id)
                return
            except Exception:
                continue
        from aios.core.exceptions import ResourceError

        raise ResourceError(
            f"No reservation found for task {task_id} in any pool."
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def size(self, resource_type: ResourceType | None = None) -> int:
        if resource_type is not None:
            return self._schedulers[resource_type].size()
        return sum(s.size() for s in self._schedulers.values())

    def is_empty(self, resource_type: ResourceType | None = None) -> bool:
        return self.size(resource_type=resource_type) == 0

    def get_metrics(self) -> dict[str, Any]:
        return {
            res_type.value: scheduler.get_metrics()
            for res_type, scheduler in self._schedulers.items()
        }

    def close(self) -> None:
        self._closed = True
        self._request_resources.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_resource_type(
        resource_type: ResourceType | str,
    ) -> ResourceType:
        if isinstance(resource_type, str):
            try:
                return ResourceType(resource_type.lower())
            except ValueError:
                return ResourceType.LLM
        return resource_type

    def _resolve_resource_type(
        self,
        item: AgentRequest | Task | dict[str, float],
        explicit: ResourceType | None,
    ) -> ResourceType:
        """
        Determine resource type: use explicit if given, else infer from
        item's syscall, else default to LLM.
        """
        if explicit is not None:
            return explicit
        if isinstance(item, AgentRequest):
            return map_syscall_to_resource_type(item.syscall.call_type)
        return ResourceType.LLM
