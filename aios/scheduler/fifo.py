"""
FIFO task queue.

Strict first-in-first-out ordering. Every enqueue, dequeue, and
remove operation validates the task's status transition through
``aios.agents.task.validate_transition`` -- the single source of truth
for legal task state machines.

State transitions enforced here:

    enqueue  : CREATED -> QUEUED
    dequeue  : QUEUED  -> RUNNING
    remove   : QUEUED  -> CANCELLED
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone

from aios.agents.task import validate_transition
from aios.core.exceptions import DuplicateTaskError, QueueEmptyError
from aios.core.ids import TaskID
from aios.core.models.task import Task, TaskStatus


class FIFOTaskQueue:
    """
    Strict FIFO implementation of the TaskQueue interface.

    Tasks are returned in the exact order they were enqueued.
    Duplicate task IDs are rejected. All status transitions are
    validated against the canonical task state machine.
    """

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()
        self._ids: set[TaskID] = set()

    def enqueue(self, task: Task) -> None:
        """
        Add ``task`` to the back of the queue.

        Transitions the task from CREATED to QUEUED. Raises
        ``DuplicateTaskError`` if the task is already queued.
        Raises ``InvalidStateTransitionError`` if the task is not
        in a state that allows enqueueing.
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

        self._queue.append(queued_task)
        self._ids.add(queued_task.task_id)

    def dequeue(self) -> Task:
        """
        Remove and return the task at the front of the queue.

        Transitions the task from QUEUED to RUNNING. Raises
        ``QueueEmptyError`` if the queue is empty.
        """
        if not self._queue:
            raise QueueEmptyError("Cannot dequeue: queue is empty.")

        task = self._queue.popleft()
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
        Return the task at the front of the queue without removing it.

        Raises ``QueueEmptyError`` if the queue is empty.
        """
        if not self._queue:
            raise QueueEmptyError("Cannot peek: queue is empty.")

        return self._queue[0]

    def size(self) -> int:
        """Return the number of tasks currently in the queue."""
        return len(self._queue)

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

        for queued_task in self._queue:
            if queued_task.task_id == task_id:
                validate_transition(queued_task.status, TaskStatus.CANCELLED)
                cancelled_task = replace(
                    queued_task,
                    status=TaskStatus.CANCELLED,
                    updated_at=datetime.now(timezone.utc),
                )
                self._queue.remove(queued_task)
                self._ids.discard(task_id)
                return True

        return False
