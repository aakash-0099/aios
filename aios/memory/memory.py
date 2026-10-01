"""
Base memory interface for AIOS memory subsystem.

This module defines the BaseMemory class that provides common memory
operations across both working memory and long-term memory.

The interface uses the shared aios.core.models.Memory model and delegates
storage to a MemoryStore backend.
"""

from __future__ import annotations

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory

from .store import MemoryStore


class BaseMemory:
    """
    Base class providing common memory operations.

    This class encapsulates the core memory operations (add, get, search,
    update, delete) and enforces agent isolation through the underlying store.

    Both WorkingMemory and LongTermMemory inherit from this base to provide
    consistent behavior with different lifecycle semantics.
    """

    def __init__(self, store: MemoryStore) -> None:
        """
        Initialize memory with a storage backend.

        Args:
            store: The MemoryStore implementation to use for persistence
        """
        self._store = store

    def add(self, memory: Memory) -> None:
        """
        Add a new memory to storage.

        Args:
            memory: The Memory object to store

        Raises:
            ValidationError: if the memory object is invalid
        """
        self._store.add(memory)

    def get(self, memory_id: MemoryID, agent_id: AgentID) -> Memory:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: The unique identifier of the memory
            agent_id: The agent that owns the memory

        Returns:
            The requested Memory object

        Raises:
            NotFoundError: if memory does not exist or belongs to another agent
        """
        return self._store.get(memory_id, agent_id)

    def search(
        self,
        agent_id: AgentID,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        """
        Search memories for a specific agent.

        Args:
            agent_id: The agent whose memories to search
            query: Optional search string (searches memory content)
            limit: Optional maximum number of results to return

        Returns:
            List of matching Memory objects (may be empty)
        """
        return self._store.search(agent_id, query=query, limit=limit)

    def update(self, memory: Memory) -> None:
        """
        Update an existing memory.

        The memory_id and agent_id cannot be changed.
        Updates content and metadata while preserving created_at.

        Args:
            memory: The Memory object with updated data

        Raises:
            NotFoundError: if memory does not exist
            ValidationError: if the updated memory is invalid
        """
        self._store.update(memory)

    def delete(self, memory_id: MemoryID, agent_id: AgentID) -> None:
        """
        Delete a specific memory.

        Args:
            memory_id: The unique identifier of the memory to delete
            agent_id: The agent that owns the memory

        Raises:
            NotFoundError: if memory does not exist or belongs to another agent
        """
        self._store.delete(memory_id, agent_id)
