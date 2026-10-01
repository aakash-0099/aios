"""
Working memory implementation for AIOS.

Working memory represents temporary, active memory that can be cleared.
It is used for short-term context and intermediate reasoning that does
not need to persist long-term.

Working memory is independent from long-term memory and uses the same
Memory model and storage abstractions.
"""

from __future__ import annotations

from typing import Callable

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory

from .memory import BaseMemory
from .store import InMemoryStore, MemoryStore


class WorkingMemory(BaseMemory):
    """
    Working memory for temporary, active agent memory with bounded capacity.

    Working memory provides:
    - All standard memory operations (add, get, search, update, delete)
    - Per-agent capacity management with deterministic eviction
    - clear() operation to delete all temporary memories for an agent
    - Independent storage from long-term memory

    Capacity and Eviction:
    - Each agent has independent capacity tracking
    - When capacity is reached, oldest memory (by created_at) is evicted
    - Evicted memories are sent to storage via eviction callback
    - Eviction is deterministic and testable

    Use cases:
    - Temporary reasoning steps
    - Active conversation context
    - Intermediate results that should be discarded after task completion

    Working memory does not automatically expire or clear itself.
    The caller must explicitly invoke clear() when appropriate.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        capacity_per_agent: int = 100,
        eviction_callback: Callable[[Memory, AgentID], None] | None = None,
    ) -> None:
        """
        Initialize working memory with a storage backend.

        Args:
            store: Optional MemoryStore implementation. If not provided,
                   creates a new InMemoryStore instance.
            capacity_per_agent: Maximum number of memories per agent.
                               When exceeded, oldest memory is evicted.
                               Must be >= 1. Default is 100.
            eviction_callback: Optional callback invoked when a memory is
                              evicted. Receives (memory, agent_id).
                              Used to send evicted memories to storage.

        Raises:
            ValueError: if capacity_per_agent < 1
        """
        if capacity_per_agent < 1:
            raise ValueError("capacity_per_agent must be at least 1")

        if store is None:
            store = InMemoryStore()
        super().__init__(store)

        self._capacity_per_agent = capacity_per_agent
        self._eviction_callback = eviction_callback

    def add(self, memory: Memory) -> None:
        """
        Add a new memory to working storage.

        If the agent has reached capacity, the oldest memory (by created_at)
        is evicted and sent to storage via the eviction callback before
        adding the new memory.

        Args:
            memory: The Memory object to store

        Raises:
            ValidationError: if the memory object is invalid
        """
        agent_id = memory.agent_id

        # Check current capacity for this agent
        current_memories = self._store.search(agent_id)

        # If at capacity, evict oldest memory
        if len(current_memories) >= self._capacity_per_agent:
            self._evict_oldest(agent_id, current_memories)

        # Add the new memory
        self._store.add(memory)

    def _evict_oldest(
        self, agent_id: AgentID, current_memories: list[Memory]
    ) -> None:
        """
        Evict the oldest memory for an agent.

        Eviction policy: oldest memory by created_at is selected.

        Args:
            agent_id: The agent whose memory should be evicted
            current_memories: Current memories for the agent (must be non-empty)
        """
        if not current_memories:
            return

        # Find oldest memory (deterministic by created_at)
        oldest_memory = min(current_memories, key=lambda m: m.created_at)

        # Remove from working memory
        self._store.delete(oldest_memory.memory_id, agent_id)

        # Send to storage via callback if provided
        if self._eviction_callback is not None:
            self._eviction_callback(oldest_memory, agent_id)

    def clear(self, agent_id: AgentID) -> None:
        """
        Clear all working memories for a specific agent.

        This removes all temporary memories associated with the agent.
        Use this when:
        - Task is complete and intermediate state should be discarded
        - Agent context should be reset
        - Memory usage should be reduced

        Args:
            agent_id: The agent whose working memories should be cleared

        Note:
            This only affects working memory. Long-term memories are not
            affected by this operation. Cleared memories are NOT sent to
            storage (unlike evicted memories).
        """
        self._store.clear(agent_id)
