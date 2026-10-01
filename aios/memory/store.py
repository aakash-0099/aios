

from __future__ import annotations

from typing import Protocol

from aios.core.exceptions import NotFoundError
from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory


class MemoryStore(Protocol):
    """
    Protocol defining the memory storage interface.

    All memory storage backends must implement this interface to ensure
    consistent behavior across different storage implementations.

    Agent isolation is enforced by all implementations:
    - Memories are always scoped by agent_id
    - Cross-agent memory access is not permitted
    """

    def add(self, memory: Memory) -> None:
        """
        Store a new memory.

        Args:
            memory: The Memory object to store

        Raises:
            ValidationError: if memory is invalid
        """
        ...

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
        ...

    def search(
        self,
        agent_id: AgentID,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        """
        Search memories for a specific agent.

        Search implementation:
        - If query is None or empty: return all memories for the agent
        - If query is provided: case-insensitive substring match on content
        - Results are ordered by creation time (newest first)
        - Agent isolation is strictly enforced

        Args:
            agent_id: The agent whose memories to search
            query: Optional search string (searches memory content)
            limit: Optional maximum number of results to return

        Returns:
            List of matching Memory objects (may be empty)
        """
        ...

    def update(self, memory: Memory) -> None:
        """
        Update an existing memory.

        The memory_id and agent_id cannot be changed.
        Updates content, metadata, and preserves created_at.

        Args:
            memory: The Memory object with updated data

        Raises:
            NotFoundError: if memory does not exist
        """
        ...

    def delete(self, memory_id: MemoryID, agent_id: AgentID) -> None:
        """
        Delete a specific memory.

        Args:
            memory_id: The unique identifier of the memory to delete
            agent_id: The agent that owns the memory

        Raises:
            NotFoundError: if memory does not exist or belongs to another agent
        """
        ...

    def clear(self, agent_id: AgentID) -> None:
        """
        Delete all memories for a specific agent.

        This is primarily used by working memory to clear temporary state.

        Args:
            agent_id: The agent whose memories should be cleared
        """
        ...


class InMemoryStore:
    """
    In-memory implementation of MemoryStore.

    Uses a dictionary keyed by (agent_id, memory_id) for O(1) lookups
    while maintaining strict agent isolation.

    This is the initial backend implementation. It does not persist
    data across application restarts.

    Thread safety: Not thread-safe. Synchronization must be handled
    at a higher layer if needed.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        # Key: (agent_id, memory_id) -> Value: Memory
        self._store: dict[tuple[AgentID, MemoryID], Memory] = {}

    def add(self, memory: Memory) -> None:
        """
        Store a new memory.

        Note: This implementation allows duplicate memory_ids for different
        agents but not for the same agent. Duplicate memory_id for the same
        agent will overwrite silently (consistent with dict behavior).
        """
        key = (memory.agent_id, memory.memory_id)
        self._store[key] = memory

    def get(self, memory_id: MemoryID, agent_id: AgentID) -> Memory:
        """Retrieve a specific memory by ID."""
        key = (agent_id, memory_id)

        if key not in self._store:
            raise NotFoundError(
                f"Memory {memory_id} not found for agent {agent_id}."
            )

        return self._store[key]

    def search(
        self,
        agent_id: AgentID,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        """
        Search memories for a specific agent.

        Implementation:
        - Filters all memories by agent_id
        - If query provided: case-insensitive substring match on content
        - Orders by created_at descending (newest first)
        - Applies limit if specified
        """
        # Get all memories for this agent
        agent_memories = [
            memory
            for (aid, _), memory in self._store.items()
            if aid == agent_id
        ]

        # Apply query filter if provided
        if query and query.strip():
            query_lower = query.lower()
            agent_memories = [
                memory
                for memory in agent_memories
                if query_lower in memory.content.lower()
            ]

        # Sort by creation time (newest first)
        agent_memories.sort(key=lambda m: m.created_at, reverse=True)

        # Apply limit if specified
        if limit is not None and limit > 0:
            agent_memories = agent_memories[:limit]

        return agent_memories

    def update(self, memory: Memory) -> None:
        """
        Update an existing memory.

        The updated memory must exist and must belong to the correct agent.
        """
        key = (memory.agent_id, memory.memory_id)

        if key not in self._store:
            raise NotFoundError(
                f"Memory {memory.memory_id} not found for agent {memory.agent_id}."
            )

        self._store[key] = memory

    def delete(self, memory_id: MemoryID, agent_id: AgentID) -> None:
        """Delete a specific memory."""
        key = (agent_id, memory_id)

        if key not in self._store:
            raise NotFoundError(
                f"Memory {memory_id} not found for agent {agent_id}."
            )

        del self._store[key]

    def clear(self, agent_id: AgentID) -> None:
        """Delete all memories for a specific agent."""
        # Find all keys for this agent
        keys_to_delete = [
            key
            for key in self._store.keys()
            if key[0] == agent_id
        ]

        # Delete them
        for key in keys_to_delete:
            del self._store[key]
