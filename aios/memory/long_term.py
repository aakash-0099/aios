
from __future__ import annotations

from .memory import BaseMemory
from .store import InMemoryStore, MemoryStore


class LongTermMemory(BaseMemory):

    def __init__(self, store: MemoryStore | None = None) -> None:
        """
        Initialize long-term memory with a storage backend.

        Args:
            store: Optional MemoryStore implementation. If not provided,
                   creates a new InMemoryStore instance.

        Note:
            Currently defaults to InMemoryStore. Future implementations
            will support persistent backends (SQLite, vector stores)
            while maintaining the same interface.
        """
        if store is None:
            store = InMemoryStore()
        super().__init__(store)
