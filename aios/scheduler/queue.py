from __future__ import annotations
from typing import Protocol, runtime_checkable
from aios.core.ids import TaskID
from aios.core.models import Task

@runtime_checkable
class TaskQueue(Protocol):

    def enqueue(self, task:Task) -> None:
        ...

    def dequeue(self)-> Task:
        ...

    def peek(self) -> Task:
        ...

    def size(self) -> int:
        ...

    def contains(self, task_id: TaskID) -> bool:
        ...
 
    def remove(self, task_id: TaskID) -> bool:
        ...