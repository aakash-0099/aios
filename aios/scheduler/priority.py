"""
Priority task queue.

Tasks are ordered by priority (lower number = higher priority),
then by insertion order within the same priority level.

Decision on the missing-priority-field question: ``aios.core.models.Task``
has no priority field, and modifying the frozen Phase 1 core model is
out of scope for this package. Priority is therefore tracked externally
-- the caller supplies a ``priority`` integer at enqueue time, and this
queue stores it alongside the task. This keeps the core model stable
while still giving the scheduler a priority-ordered queue.

State transitions enforced here (same as FIFO):

    enqueue  : CREATED -> QUEUED
    dequeue  : QUEUED  -> RUNNING
    remove   : QUEUED  -> CANCELLED
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import datetime, timezone

from aios.agents.task import validate_transition
from aios.core.exceptions import DuplicateTaskError, QueueEmptyError
from aios.core.ids import TaskID
from aios.core.models.task import Task, TaskStatus


class PriorityTaskQueue:
    """
    Priority-ordered implementation of the TaskQueue interface.

    Lower ``priority`` numbers are dequeued first. Tasks with equal
    priority are dequeued in FIFO order. Duplicate task IDs are
    rejected. All status transitions are validated against the
    canonical task state machine.
    """

    def __init__(self) -> None:
        # List of (priority, sequence, task) tuples, kept sorted.
        self._entries: list[tuple[int, int, Task]] = []
        self._ids: set[TaskID] = set()
        self._sequence = itertools.count()

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
        self._entries.append((priority, seq, queued_task))
        self._entries.sort(key=lambda e: (e[0], e[1]))
        self._ids.add(queued_task.task_id)

    def dequeue(self) -> Task:
        """
        Remove and return the highest-priority task.

        Transitions the task from QUEUED to RUNNING. Raises
        ``QueueEmptyError`` if the queue is empty.
        """
        if not self._entries:
            raise QueueEmptyError("Cannot dequeue: queue is empty.")

        _, _, task = self._entries.pop(0)
        self._ids.discard(task.task_id)

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

        return self._entries[0][2]

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

        for index, (_, _, queued_task) in enumerate(self._entries):
            if queued_task.task_id == task_id:
                validate_transition(queued_task.status, TaskStatus.CANCELLED)
                cancelled_task = replace(
                    queued_task,
                    status=TaskStatus.CANCELLED,
                    updated_at=datetime.now(timezone.utc),
                )
                self._entries.pop(index)
                self._ids.discard(task_id)
                return True

        return False
