"""
AIOS Scheduler package.

Provides:
- Scheduler: Central scheduler with queues per resource type.
- ResourceScheduler: Resource-specific queue manager with preemption.
- ResourcePool: Standalone capacity accounting (invariant: available + reserved == total).
- FIFOTaskQueue: Strict FIFO task queue.
- PriorityTaskQueue: Priority-ordered task queue with starvation prevention.
- PriorityRequestQueue: Priority-ordered request queue.
- SchedulingStrategy: Supported scheduling policies (FIFO, ROUND_ROBIN, PRIORITY).
- AdmissionDecision, DecisionType, ReservationID, ResourceState.
"""

from aios.scheduler.fifo import FIFOTaskQueue
from aios.scheduler.priority import PriorityRequestQueue, PriorityTaskQueue
from aios.scheduler.queue import TaskQueue
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
from aios.scheduler.scheduler import Scheduler, map_syscall_to_resource_type

__all__ = [
    "AdmissionDecision",
    "DecisionType",
    "FIFOTaskQueue",
    "PriorityRequestQueue",
    "PriorityTaskQueue",
    "ReservationID",
    "ResourcePool",
    "ResourceScheduler",
    "ResourceState",
    "Scheduler",
    "SchedulingStrategy",
    "TaskQueue",
    "map_syscall_to_resource_type",
]
