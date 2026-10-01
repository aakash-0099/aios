"""
FIFO task queue.

follows strictly first in first out rule
"""

from __future__ import annotations

from collections import deque

from aios.core.exceptions import DuplicateTaskError, QueueEmptyError
from aios.core.ids import TaskID
from aios.core.models import Task


class FIFOTaskQueue:
    """
    Strict FIFO implementation of the TaskQueue interface.
    """

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()
        self._ids: set[TaskID] = set()

    def enqueue(self, task: Task) -> None:
        if task.task_id in self._ids:
            raise DuplicateTaskError(
                f"Task {task.task_id} is already queued."  #"if the problem is already present in throws duplicate entry error"
            )

        self._queue.append(task)
        self._ids.add(task.task_id)

    def dequeue(self) -> Task:
        if not self._queue:
            raise QueueEmptyError("Cannot dequeue: queue is empty.")

        task = self._queue.popleft()
        self._ids.discard(task.task_id)
        return task

    def peek(self) -> Task:
        if not self._queue:
            raise QueueEmptyError("Cannot peek: queue is empty.")

        return self._queue[0]

    def size(self) -> int:
        return len(self._queue)

    def contains(self, task_id: TaskID) -> bool:
        return task_id in self._ids

    def remove(self, task_id: TaskID) -> bool:
        if task_id not in self._ids:
            return False

        for queued_task in self._queue:
            if queued_task.task_id == task_id:
                self._queue.remove(queued_task)
                break

        self._ids.discard(task_id)
        return True