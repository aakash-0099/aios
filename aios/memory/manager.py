"""
Memory manager coordinating working and long-term memory.

The MemoryManager provides a unified facade for accessing both working
memory and long-term memory, including integration with storage for
memory eviction.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory
from aios.storage.manager import StorageManager

from .long_term import LongTermMemory
from .store import InMemoryStore, MemoryStore
from .working_memory import WorkingMemory


class MemoryManager:
    """
    Facade coordinating working and long-term memory.

    The MemoryManager provides:
    - Unified access point to both memory types
    - Simple initialization with sensible defaults
    - Independent storage backends for working vs long-term memory
    - Direct integration with StorageManager for eviction
    - Clean interface for other AIOS components

    Working Memory Eviction:
    When working memory reaches capacity, the oldest memory is evicted
    and sent DIRECTLY to StorageManager via a normal Python method call.
    This does NOT go through the Kernel or AgentRequest system.

    Architecture:
        WorkingMemory (capacity reached)
              ↓
        MemoryManager._handle_eviction()
              ↓
        StorageManager.store_memory()  ← Direct Python call
              ↓
        Storage subsystem

    Usage:
        from aios.storage import InMemoryStorageManager
        
        storage = InMemoryStorageManager()
        manager = MemoryManager(
            storage_manager=storage,
            capacity_per_agent=10
        )
        
        # Add to working memory
        manager.working.add(temp_memory)
        
        # When capacity is exceeded, memory is evicted to storage
        # via direct call to storage.store_memory()
        
        # Add to long-term memory
        manager.long_term.add(important_memory)
        
        # Clear working memory
        manager.working.clear(agent_id)

    The manager uses separate storage backends by default to ensure
    working memory and long-term memory are completely independent.
    """

    def __init__(
        self,
        working_store: MemoryStore | None = None,
        long_term_store: MemoryStore | None = None,
        storage_manager: StorageManager | None = None,
        capacity_per_agent: int = 100,
    ) -> None:
        """
        Initialize the memory manager with storage backends.

        Args:
            working_store: Optional MemoryStore for working memory.
                          If not provided, creates a new InMemoryStore.
            long_term_store: Optional MemoryStore for long-term memory.
                            If not provided, creates a new InMemoryStore.
            storage_manager: Optional StorageManager for evicted memories.
                            If provided, evicted working memories are sent
                            to storage via DIRECT method call (not Kernel).
            capacity_per_agent: Maximum working memory per agent. When
                               exceeded, oldest memory is evicted.
                               Default is 100.

        Note:
            By default, working and long-term memory use separate
            InMemoryStore instances to maintain independence. This
            ensures clearing working memory never affects long-term
            memory, and vice versa.
        """
        self._storage_manager = storage_manager

        # Create working memory with eviction callback
        self.working = WorkingMemory(
            store=working_store,
            capacity_per_agent=capacity_per_agent,
            eviction_callback=self._handle_eviction if storage_manager else None,
        )

        self.long_term = LongTermMemory(store=long_term_store)

    def add(self, agent_id: AgentID, memory: Memory) -> None:
        """Add a memory to the agent's bounded working memory."""
        if memory.agent_id != agent_id:
            raise ValueError("memory.agent_id must match agent_id")
        self.working.add(memory)

    def get(self, agent_id: AgentID, memory_id: MemoryID) -> Memory:
        """Retrieve one of the agent's working memories."""
        return self.working.get(memory_id, agent_id)

    def search(self, agent_id: AgentID, query: str | None = None) -> list[Memory]:
        """Search the agent's working memories, newest first."""
        return self.working.search(agent_id, query=query)

    def update(
        self, agent_id: AgentID, memory_id: MemoryID, **changes: Any
    ) -> Memory:
        """Update content or metadata without changing memory identity."""
        memory = self.working.get(memory_id, agent_id)
        allowed_changes = {"content", "metadata"}
        invalid_changes = changes.keys() - allowed_changes
        if invalid_changes:
            names = ", ".join(sorted(invalid_changes))
            raise ValueError(f"unsupported memory update fields: {names}")

        updated = replace(memory, **changes)
        self.working.update(updated)
        return updated

    def delete(self, agent_id: AgentID, memory_id: MemoryID) -> None:
        """Delete one of the agent's working memories."""
        self.working.delete(memory_id, agent_id)

    def _handle_eviction(self, memory: Memory, agent_id: AgentID) -> None:
        """
        Handle eviction of a memory from working memory.

        This is a DIRECT Python method call to StorageManager.
        It does NOT construct an AgentRequest or go through the Kernel.

        Architecture:
            WorkingMemory → MemoryManager._handle_eviction() 
                         → StorageManager.store_memory()

        Args:
            memory: The evicted Memory object
            agent_id: The agent that owns the memory
        """
        if self._storage_manager is not None:
            # Direct method call to storage (NOT through Kernel)
            self._storage_manager.store_memory(memory, agent_id)

    @classmethod
    def create_default(cls, storage_manager: StorageManager | None = None) -> MemoryManager:
        """
        Create a MemoryManager with default in-memory storage.

        This is a convenience factory method that explicitly creates
        separate InMemoryStore instances for working and long-term memory.

        Args:
            storage_manager: Optional StorageManager for evicted memories

        Returns:
            A new MemoryManager with independent in-memory storage
        """
        return cls(
            working_store=InMemoryStore(),
            long_term_store=InMemoryStore(),
            storage_manager=storage_manager,
        )
