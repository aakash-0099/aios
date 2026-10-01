"""
Priority task and request queues with starvation prevention.

Tasks and requests are ordered by priority (lower number = higher priority),
then by insertion order within the same priority level.

Starvation Prevention:
    When configured with an ``aging_threshold > 0``, items waiting in the
    queue accumulate age whenever another item is dequeued. For every
    ``aging_threshold`` cycles of waiting, an item's effective priority is
    boosted (decreased by 1), guaranteeing that low-priority items will
    eventually reach the front of the queue and be executed.

State transitions enforced for tasks:
    enqueue  : CREATED -> QUEUED
    dequeue  : QUEUED  -> RUNNING
    remove   : QUEUED  -> CANCELLED
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from aios.agents.task import validate_transition
from aios.core.exceptions import DuplicateTaskError, QueueEmptyError
from aios.core.ids import RequestID, TaskID
from aios.core.models.request import AgentRequest
from aios.core.models.task import Task, TaskStatus


class PriorityTaskQueue:
    """
    Priority-ordered implementation of the TaskQueue interface with anti-starvation.

    Lower ``priority`` numbers are dequeued first. Tasks with equal
    priority are dequeued in FIFO order. Duplicate task IDs are
    rejected. All status transitions are validated against the
    canonical task state machine.
    """

    def __init__(self, aging_threshold: int = 0) -> None:
        """
        Initialize the queue.

        Args:
            aging_threshold: If > 0, waiting tasks gain age on every dequeue.
                Every aging_threshold ticks boosts effective priority by 1,
                preventing starvation of low-priority tasks.
        """
        # List of [base_priority, sequence, task, age]
        self._entries: list[list[Any]] = []
        self._ids: set[TaskID] = set()
        self._sequence = itertools.count()
        self._aging_threshold = max(0, aging_threshold)

    @property
    def aging_threshold(self) -> int:
        return self._aging_threshold

    def enqueue(self, task: Task, priority: int = 0) -> None:
        """
        Add ``task`` to the queue with the given ``priority``.

        Lower numbers are higher priority. Transitions the task from
        CREATED to QUEUED. Raises ``DuplicateTaskError`` if the task
        is already queued. Raises ``InvalidStateTransitionError`` if
        the task is not in a state that allows enqueueing.
        """
        if task.task_id in self._ids:
            raise DuplicateTaskError(
                f"Task {task.task_id} is already queued."
            )

        validate_transition(task.status, TaskStatus.QUEUED)
        queued_task = replace(
            task,
            status=TaskStatus.QUEUED,
            updated_at=datetime.now(timezone.utc),
        )

        seq = next(self._sequence)
        # Entry format: [base_priority, seq, task, age]
        self._entries.append([priority, seq, queued_task, 0])
        self._ids.add(queued_task.task_id)

    def _effective_priority(self, entry: list[Any]) -> int:
        base_priority, _, _, age = entry
        if self._aging_threshold > 0:
            boost = age // self._aging_threshold
            return max(0, base_priority - boost)
        return base_priority

    def dequeue(self) -> Task:
        """
        Remove and return the highest-priority task.

        Transitions the task from QUEUED to RUNNING. Raises
        ``QueueEmptyError`` if the queue is empty.
        """
        if not self._entries:
            raise QueueEmptyError("Cannot dequeue: queue is empty.")

        # Find best entry according to (effective_priority, sequence)
        best_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._effective_priority(self._entries[i]), self._entries[i][1]),
        )
        _, _, task, _ = self._entries.pop(best_idx)
        self._ids.discard(task.task_id)

        # Apply aging to remaining entries if aging is enabled
        if self._aging_threshold > 0:
            for entry in self._entries:
                entry[3] += 1

        validate_transition(task.status, TaskStatus.RUNNING)
        running_task = replace(
            task,
            status=TaskStatus.RUNNING,
            updated_at=datetime.now(timezone.utc),
        )

        return running_task

    def peek(self) -> Task:
        """
        Return the highest-priority task without removing it.

        Raises ``QueueEmptyError`` if the queue is empty.
        """
        if not self._entries:
            raise QueueEmptyError("Cannot peek: queue is empty.")

        best_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._effective_priority(self._entries[i]), self._entries[i][1]),
        )
        return self._entries[best_idx][2]

    def size(self) -> int:
        """Return the number of tasks currently in the queue."""
        return len(self._entries)

    def contains(self, task_id: TaskID) -> bool:
        """Return True if a task with ``task_id`` is in the queue."""
        return task_id in self._ids

    def remove(self, task_id: TaskID) -> bool:
        """
        Remove the task with ``task_id`` from the queue.

        Transitions the task from QUEUED to CANCELLED. Returns True
        if the task was found and removed, False if it was not in
        the queue.
        """
        if task_id not in self._ids:
            return False

        for index, (_, _, queued_task, _) in enumerate(self._entries):
            if queued_task.task_id == task_id:
                validate_transition(queued_task.status, TaskStatus.CANCELLED)
                self._entries.pop(index)
                self._ids.discard(task_id)
                return True

        return False


class PriorityRequestQueue:
    """
    Priority-ordered queue for AgentRequest objects with anti-starvation.

    Lower ``priority`` numbers are dequeued first. Requests with equal
    priority are dequeued in FIFO order.
    """

    def __init__(self, aging_threshold: int = 0) -> None:
        self._entries: list[list[Any]] = []  # [base_priority, seq, request, age]
        self._ids: set[RequestID] = set()
        self._sequence = itertools.count()
        self._aging_threshold = max(0, aging_threshold)

    @property
    def aging_threshold(self) -> int:
        return self._aging_threshold

    def enqueue(self, request: AgentRequest, priority: int = 0) -> None:
        """Enqueue an AgentRequest with an optional priority."""
        seq = next(self._sequence)
        self._entries.append([priority, seq, request, 0])
        self._ids.add(request.request_id)

    def _effective_priority(self, entry: list[Any]) -> int:
        base_priority, _, _, age = entry
        if self._aging_threshold > 0:
            boost = age // self._aging_threshold
            return max(0, base_priority - boost)
        return base_priority

    def dequeue(self) -> AgentRequest:
        """Dequeue the highest-priority request."""
        if not self._entries:
            raise QueueEmptyError("Cannot dequeue: request queue is empty.")

        best_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._effective_priority(self._entries[i]), self._entries[i][1]),
        )
        _, _, request, _ = self._entries.pop(best_idx)
        self._ids.discard(request.request_id)

        if self._aging_threshold > 0:
            for entry in self._entries:
                entry[3] += 1

        return request

    def peek(self) -> AgentRequest:
        """Peek at the highest-priority request."""
        if not self._entries:
            raise QueueEmptyError("Cannot peek: request queue is empty.")

        best_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._effective_priority(self._entries[i]), self._entries[i][1]),
        )
        return self._entries[best_idx][2]

    def size(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def contains(self, request_id: RequestID) -> bool:
        return request_id in self._ids

    def remove(self, request_id: RequestID) -> bool:
        if request_id not in self._ids:
            return False

        for index, (_, _, req, _) in enumerate(self._entries):
            if req.request_id == request_id:
                self._entries.pop(index)
                self._ids.discard(request_id)
                return True
        return False
