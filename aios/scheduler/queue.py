"""Shared queue interface for the AIOS scheduler.

`TaskQueue` is the contract every task queue conforms to. Both
`FIFOTaskQueue` and `PriorityTaskQueue` explicitly subclass it, so a
signature drift between the interface and an implementation surfaces at
import/instantiation time rather than at call time.

Signatures below mirror the actual implementations: `peek` raises
`QueueEmptyError` on an empty queue and `remove` returns a bool. The
optional `priority` argument is ignored by FIFO ordering and is the
knob the priority queue sorts on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aios.core.ids import TaskID
from aios.core.models import Task


@runtime_checkable
class TaskQueue(Protocol):
    """Enqueue, dequeue, peek, size, contains, remove for `Task` objects."""

    def enqueue(self, task: Task, priority: int = 0) -> None:
        """Add `task`, transitioning it to QUEUED. Raises on duplicate ID."""
        ...

    def dequeue(self) -> Task:
        """Remove and return the next task, transitioned to RUNNING."""
        ...

    def peek(self) -> Task:
        """Return the next task without mutating the queue."""
        ...

    def size(self) -> int:
        """Return the number of queued tasks."""
        ...

    def contains(self, task_id: TaskID) -> bool:
        """Return True if `task_id` is currently queued."""
        ...

    def remove(self, task_id: TaskID) -> bool:
        """Cancel and drop `task_id`. Returns True if it was queued."""
        ...
