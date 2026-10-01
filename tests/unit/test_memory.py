"""
Comprehensive tests for the AIOS memory subsystem.

Tests cover:
- CRUD operations (add, get, update, delete)
- Search functionality with various queries
- Agent isolation (memories cannot leak between agents)
- Working memory with clear() functionality
- Long-term memory operations
- MemoryManager coordination
- Backend independence (no SQLite, ChromaDB, LLMs required)
- Phase 2: Bounded capacity, eviction, storage integration
"""

from datetime import datetime, timezone

import pytest

from aios.core.exceptions import NotFoundError, ValidationError
from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory
from aios.memory import (
    InMemoryStore,
    LongTermMemory,
    MemoryManager,
    WorkingMemory,
)


# Test Fixtures


@pytest.fixture
def agent_a() -> AgentID:
    """Agent A for isolation testing."""
    return AgentID.generate()


@pytest.fixture
def agent_b() -> AgentID:
    """Agent B for isolation testing."""
    return AgentID.generate()


@pytest.fixture
def memory_store() -> InMemoryStore:
    """Fresh in-memory store for each test."""
    return InMemoryStore()


@pytest.fixture
def sample_memory(agent_a: AgentID) -> Memory:
    """Sample memory for testing."""
    return Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_a,
        content="Test memory content",
        metadata={"source": "test"},
    )


# Test Double for Storage


class FakeStorageManager:
    """
    Test double for StorageManager.

    This fake implementation tracks evicted memories without requiring
    the real storage subsystem. Used to verify:
    - Eviction triggers correctly
    - Correct memories are evicted
    - Storage receives exact Memory objects
    - Direct method call (not Kernel/AgentRequest)
    """

    def __init__(self) -> None:
        """Initialize empty storage tracking."""
        self.stored_memories: list[tuple[Memory, AgentID]] = []
        self.call_count = 0

    def store_memory(self, memory: Memory, agent_id: AgentID) -> None:
        """
        Store an evicted memory (fake implementation).

        Args:
            memory: The Memory object to store
            agent_id: The agent that owns the memory
        """
        self.stored_memories.append((memory, agent_id))
        self.call_count += 1

    def get_stored_for_agent(self, agent_id: AgentID) -> list[Memory]:
        """Get all stored memories for a specific agent."""
        return [
            memory
            for memory, aid in self.stored_memories
            if aid == agent_id
        ]

    def clear(self) -> None:
        """Clear all stored memories."""
        self.stored_memories.clear()
        self.call_count = 0


# InMemoryStore Tests


class TestInMemoryStore:
    """Tests for the InMemoryStore backend."""

    def test_add_and_get_memory(
        self, memory_store: InMemoryStore, sample_memory: Memory
    ):
        """Test adding and retrieving a memory."""
        memory_store.add(sample_memory)

        retrieved = memory_store.get(
            sample_memory.memory_id, sample_memory.agent_id
        )

        assert retrieved.memory_id == sample_memory.memory_id
        assert retrieved.agent_id == sample_memory.agent_id
        assert retrieved.content == sample_memory.content
        assert retrieved.metadata == sample_memory.metadata

    def test_get_nonexistent_memory_raises_not_found(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test that getting a nonexistent memory raises NotFoundError."""
        with pytest.raises(NotFoundError):
            memory_store.get(MemoryID.generate(), agent_a)

    def test_update_memory(
        self, memory_store: InMemoryStore, sample_memory: Memory
    ):
        """Test updating an existing memory."""
        memory_store.add(sample_memory)

        # Update the memory
        updated_memory = Memory(
            memory_id=sample_memory.memory_id,
            agent_id=sample_memory.agent_id,
            content="Updated content",
            metadata={"source": "test", "updated": True},
            created_at=sample_memory.created_at,
        )
        memory_store.update(updated_memory)

        # Retrieve and verify
        retrieved = memory_store.get(
            sample_memory.memory_id, sample_memory.agent_id
        )
        assert retrieved.content == "Updated content"
        assert retrieved.metadata["updated"] is True

    def test_update_nonexistent_memory_raises_not_found(
        self, memory_store: InMemoryStore, sample_memory: Memory
    ):
        """Test that updating a nonexistent memory raises NotFoundError."""
        with pytest.raises(NotFoundError):
            memory_store.update(sample_memory)

    def test_delete_memory(
        self, memory_store: InMemoryStore, sample_memory: Memory
    ):
        """Test deleting a memory."""
        memory_store.add(sample_memory)

        memory_store.delete(sample_memory.memory_id, sample_memory.agent_id)

        # Verify it's gone
        with pytest.raises(NotFoundError):
            memory_store.get(sample_memory.memory_id, sample_memory.agent_id)

    def test_delete_nonexistent_memory_raises_not_found(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test that deleting a nonexistent memory raises NotFoundError."""
        with pytest.raises(NotFoundError):
            memory_store.delete(MemoryID.generate(), agent_a)

    def test_search_all_memories_for_agent(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test searching all memories for an agent."""
        memories = [
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content=f"Memory {i}",
            )
            for i in range(3)
        ]

        for memory in memories:
            memory_store.add(memory)

        results = memory_store.search(agent_a)

        assert len(results) == 3
        assert all(m.agent_id == agent_a for m in results)

    def test_search_with_query(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test searching memories with a query string."""
        memory_store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="The quick brown fox",
            )
        )
        memory_store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="The lazy dog",
            )
        )
        memory_store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Something completely different",
            )
        )

        # Search for "fox"
        results = memory_store.search(agent_a, query="fox")
        assert len(results) == 1
        assert "fox" in results[0].content

        # Search for "the" (case-insensitive)
        results = memory_store.search(agent_a, query="the")
        assert len(results) == 2

    def test_search_with_no_results(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test searching with a query that matches nothing."""
        memory_store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Some content",
            )
        )

        results = memory_store.search(agent_a, query="nonexistent")
        assert len(results) == 0

    def test_search_with_limit(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test searching with a result limit."""
        for i in range(5):
            memory_store.add(
                Memory(
                    memory_id=MemoryID.generate(),
                    agent_id=agent_a,
                    content=f"Memory {i}",
                )
            )

        results = memory_store.search(agent_a, limit=2)
        assert len(results) == 2

    def test_search_orders_by_created_at_desc(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test that search results are ordered newest first."""
        # Add memories with specific timestamps
        old_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Old memory",
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        new_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="New memory",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        memory_store.add(old_memory)
        memory_store.add(new_memory)

        results = memory_store.search(agent_a)

        assert len(results) == 2
        assert results[0].content == "New memory"
        assert results[1].content == "Old memory"

    def test_clear_removes_all_agent_memories(
        self, memory_store: InMemoryStore, agent_a: AgentID
    ):
        """Test clearing all memories for an agent."""
        for i in range(3):
            memory_store.add(
                Memory(
                    memory_id=MemoryID.generate(),
                    agent_id=agent_a,
                    content=f"Memory {i}",
                )
            )

        memory_store.clear(agent_a)

        results = memory_store.search(agent_a)
        assert len(results) == 0


# Agent Isolation Tests


class TestAgentIsolation:
    """Tests verifying strict agent isolation."""

    def test_get_memory_from_different_agent_fails(
        self, memory_store: InMemoryStore, agent_a: AgentID, agent_b: AgentID
    ):
        """Test that Agent B cannot retrieve Agent A's memory."""
        memory_a = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A's memory",
        )
        memory_store.add(memory_a)

        # Agent B tries to get Agent A's memory
        with pytest.raises(NotFoundError):
            memory_store.get(memory_a.memory_id, agent_b)

    def test_search_only_returns_agent_memories(
        self, memory_store: InMemoryStore, agent_a: AgentID, agent_b: AgentID
    ):
        """Test that search only returns memories for the specified agent."""
        # Add memories for both agents
        memory_a1 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A memory 1",
        )
        memory_a2 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A memory 2",
        )
        memory_b1 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_b,
            content="Agent B memory 1",
        )

        memory_store.add(memory_a1)
        memory_store.add(memory_a2)
        memory_store.add(memory_b1)

        # Search Agent A's memories
        results_a = memory_store.search(agent_a)
        assert len(results_a) == 2
        assert all(m.agent_id == agent_a for m in results_a)

        # Search Agent B's memories
        results_b = memory_store.search(agent_b)
        assert len(results_b) == 1
        assert all(m.agent_id == agent_b for m in results_b)

    def test_delete_memory_from_different_agent_fails(
        self, memory_store: InMemoryStore, agent_a: AgentID, agent_b: AgentID
    ):
        """Test that Agent B cannot delete Agent A's memory."""
        memory_a = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A's memory",
        )
        memory_store.add(memory_a)

        # Agent B tries to delete Agent A's memory
        with pytest.raises(NotFoundError):
            memory_store.delete(memory_a.memory_id, agent_b)

        # Verify memory still exists for Agent A
        retrieved = memory_store.get(memory_a.memory_id, agent_a)
        assert retrieved.memory_id == memory_a.memory_id

    def test_clear_only_affects_specified_agent(
        self, memory_store: InMemoryStore, agent_a: AgentID, agent_b: AgentID
    ):
        """Test that clearing Agent A's memories doesn't affect Agent B."""
        memory_a = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A's memory",
        )
        memory_b = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_b,
            content="Agent B's memory",
        )

        memory_store.add(memory_a)
        memory_store.add(memory_b)

        # Clear Agent A's memories
        memory_store.clear(agent_a)

        # Verify Agent A has no memories
        assert len(memory_store.search(agent_a)) == 0

        # Verify Agent B's memory is unaffected
        assert len(memory_store.search(agent_b)) == 1


# WorkingMemory Tests (Phase 1)


class TestWorkingMemoryBasic:
    """Tests for basic WorkingMemory functionality without capacity."""

    def test_working_memory_supports_all_operations(self, agent_a: AgentID):
        """Test that working memory supports add, get, search, update, delete."""
        working = WorkingMemory(capacity_per_agent=100)

        # Add
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Working memory test",
        )
        working.add(memory)

        # Get
        retrieved = working.get(memory.memory_id, agent_a)
        assert retrieved.content == "Working memory test"

        # Search
        results = working.search(agent_a)
        assert len(results) == 1

        # Update
        updated = Memory(
            memory_id=memory.memory_id,
            agent_id=agent_a,
            content="Updated working memory",
            created_at=memory.created_at,
        )
        working.update(updated)
        retrieved = working.get(memory.memory_id, agent_a)
        assert retrieved.content == "Updated working memory"

        # Delete
        working.delete(memory.memory_id, agent_a)
        with pytest.raises(NotFoundError):
            working.get(memory.memory_id, agent_a)

    def test_working_memory_clear(self, agent_a: AgentID):
        """Test clearing working memory."""
        working = WorkingMemory(capacity_per_agent=100)

        # Add multiple memories
        for i in range(3):
            working.add(
                Memory(
                    memory_id=MemoryID.generate(),
                    agent_id=agent_a,
                    content=f"Memory {i}",
                )
            )

        # Verify they exist
        assert len(working.search(agent_a)) == 3

        # Clear
        working.clear(agent_a)

        # Verify they're gone
        assert len(working.search(agent_a)) == 0


# Phase 2: Capacity and Eviction Tests


class TestWorkingMemoryCapacity:
    """Tests for Phase 2 bounded working memory with eviction."""

    def test_capacity_validation(self):
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError):
            WorkingMemory(capacity_per_agent=0)

        with pytest.raises(ValueError):
            WorkingMemory(capacity_per_agent=-1)

        # Valid capacity should work
        working = WorkingMemory(capacity_per_agent=1)
        assert working is not None

    def test_capacity_enforced_per_agent(self, agent_a: AgentID):
        """Test that capacity is enforced when adding memories."""
        fake_storage = FakeStorageManager()
        working = WorkingMemory(
            capacity_per_agent=3,
            eviction_callback=fake_storage.store_memory,
        )

        # Add up to capacity
        for i in range(3):
            working.add(
                Memory(
                    memory_id=MemoryID.generate(),
                    agent_id=agent_a,
                    content=f"Memory {i}",
                )
            )

        # All 3 should be in working memory
        assert len(working.search(agent_a)) == 3
        assert fake_storage.call_count == 0

        # Adding 4th should evict oldest
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 3",
            )
        )

        # Still 3 in working memory, 1 evicted
        assert len(working.search(agent_a)) == 3
        assert fake_storage.call_count == 1

    def test_oldest_memory_evicted_first(self, agent_a: AgentID):
        """Test that oldest memory (by created_at) is evicted."""
        fake_storage = FakeStorageManager()
        working = WorkingMemory(
            capacity_per_agent=2,
            eviction_callback=fake_storage.store_memory,
        )

        # Add memories with explicit timestamps
        old_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Old memory",
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        new_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="New memory",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        working.add(old_memory)
        working.add(new_memory)

        # Both should be in working memory
        assert len(working.search(agent_a)) == 2

        # Add another memory
        newest_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Newest memory",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        working.add(newest_memory)

        # Old memory should be evicted
        assert len(working.search(agent_a)) == 2
        assert fake_storage.call_count == 1

        evicted = fake_storage.stored_memories[0][0]
        assert evicted.content == "Old memory"

        # New and newest should remain
        remaining = working.search(agent_a)
        remaining_contents = {m.content for m in remaining}
        assert "New memory" in remaining_contents
        assert "Newest memory" in remaining_contents
        assert "Old memory" not in remaining_contents

    def test_capacity_per_agent_isolation(
        self, agent_a: AgentID, agent_b: AgentID
    ):
        """Test that capacity is tracked independently per agent."""
        fake_storage = FakeStorageManager()
        working = WorkingMemory(
            capacity_per_agent=2,
            eviction_callback=fake_storage.store_memory,
        )

        # Agent A: add up to capacity
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="A1",
            )
        )
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="A2",
            )
        )

        # Agent B: add up to capacity
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_b,
                content="B1",
            )
        )
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_b,
                content="B2",
            )
        )

        # Both agents at capacity, no evictions yet
        assert len(working.search(agent_a)) == 2
        assert len(working.search(agent_b)) == 2
        assert fake_storage.call_count == 0

        # Agent A adds another - should evict A's oldest
        working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="A3",
            )
        )

        # Agent A evicted 1, Agent B unchanged
        assert len(working.search(agent_a)) == 2
        assert len(working.search(agent_b)) == 2
        assert fake_storage.call_count == 1

        evicted = fake_storage.stored_memories[0][0]
        assert evicted.agent_id == agent_a

        # Verify Agent B's memories unchanged
        b_memories = working.search(agent_b)
        b_contents = {m.content for m in b_memories}
        assert "B1" in b_contents
        assert "B2" in b_contents

    def test_eviction_without_callback_works(self, agent_a: AgentID):
        """Test that eviction works even without storage callback."""
        working = WorkingMemory(
            capacity_per_agent=2,
            eviction_callback=None,  # No callback
        )

        for i in range(3):
            working.add(
                Memory(
                    memory_id=MemoryID.generate(),
                    agent_id=agent_a,
                    content=f"Memory {i}",
                )
            )

        # Should still maintain capacity
        assert len(working.search(agent_a)) == 2


# Phase 2: Storage Integration Tests


class TestStorageIntegration:
    """Tests verifying direct storage integration (not Kernel)."""

    def test_evicted_memory_sent_to_storage(self, agent_a: AgentID):
        """Test that evicted memory is sent to storage."""
        fake_storage = FakeStorageManager()
        working = WorkingMemory(
            capacity_per_agent=2,
            eviction_callback=fake_storage.store_memory,
        )

        mem1 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Memory 1",
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        mem2 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Memory 2",
            created_at=datetime(2023, 2, 1, tzinfo=timezone.utc),
        )
        mem3 = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Memory 3",
            created_at=datetime(2023, 3, 1, tzinfo=timezone.utc),
        )

        working.add(mem1)
        working.add(mem2)
        working.add(mem3)  # Should evict mem1

        # Verify storage received the evicted memory
        assert fake_storage.call_count == 1
        evicted_memory, evicted_agent = fake_storage.stored_memories[0]

        assert evicted_memory.memory_id == mem1.memory_id
        assert evicted_memory.content == "Memory 1"
        assert evicted_agent == agent_a

    def test_manager_integrates_with_storage(self, agent_a: AgentID):
        """Test MemoryManager integrates working memory with storage."""
        fake_storage = FakeStorageManager()

        manager = MemoryManager(
            storage_manager=fake_storage,
            capacity_per_agent=2,
        )

        # Add memories up to capacity
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 1",
                created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )
        )
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 2",
                created_at=datetime(2023, 2, 1, tzinfo=timezone.utc),
            )
        )

        assert fake_storage.call_count == 0

        # Adding 3rd should trigger eviction to storage
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 3",
                created_at=datetime(2023, 3, 1, tzinfo=timezone.utc),
            )
        )

        # Verify storage was called directly (not through Kernel)
        assert fake_storage.call_count == 1
        evicted_memory, _ = fake_storage.stored_memories[0]
        assert evicted_memory.content == "Memory 1"

    def test_storage_integration_is_direct_call_not_kernel(
        self, agent_a: AgentID
    ):
        """
        Test that eviction calls storage directly, NOT through Kernel.

        This test verifies the architecture requirement that eviction
        flows as: WorkingMemory → MemoryManager → StorageManager
        without constructing AgentRequest or involving Kernel.
        """
        fake_storage = FakeStorageManager()

        manager = MemoryManager(
            storage_manager=fake_storage,
            capacity_per_agent=1,
        )

        # Add 2 memories to trigger eviction
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 1",
            )
        )
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Memory 2",
            )
        )

        # Verify:
        # 1. Storage was called
        assert fake_storage.call_count == 1

        # 2. Storage received Memory object (not AgentRequest)
        evicted_memory, evicted_agent = fake_storage.stored_memories[0]
        assert isinstance(evicted_memory, Memory)
        assert isinstance(evicted_agent, AgentID)

        # 3. No Kernel involvement (would be detected by different signature)
        # The fact that we're calling store_memory(Memory, AgentID)
        # proves it's a direct call, not a Kernel syscall


# LongTermMemory Tests


class TestLongTermMemory:
    """Tests for LongTermMemory functionality."""

    def test_long_term_memory_supports_all_operations(self, agent_a: AgentID):
        """Test that long-term memory supports add, get, search, update, delete."""
        long_term = LongTermMemory()

        # Add
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Long-term memory test",
        )
        long_term.add(memory)

        # Get
        retrieved = long_term.get(memory.memory_id, agent_a)
        assert retrieved.content == "Long-term memory test"

        # Search
        results = long_term.search(agent_a)
        assert len(results) == 1

        # Update
        updated = Memory(
            memory_id=memory.memory_id,
            agent_id=agent_a,
            content="Updated long-term memory",
            created_at=memory.created_at,
        )
        long_term.update(updated)
        retrieved = long_term.get(memory.memory_id, agent_a)
        assert retrieved.content == "Updated long-term memory"

        # Delete
        long_term.delete(memory.memory_id, agent_a)
        with pytest.raises(NotFoundError):
            long_term.get(memory.memory_id, agent_a)

    def test_long_term_memory_uses_default_store(self):
        """Test that LongTermMemory creates its own store by default."""
        long_term1 = LongTermMemory()
        long_term2 = LongTermMemory()

        agent = AgentID.generate()
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent,
            content="Test",
        )

        # Add to long_term1
        long_term1.add(memory)

        # Should not appear in long_term2 (different store)
        assert len(long_term2.search(agent)) == 0


# MemoryManager Tests


class TestMemoryManager:
    """Tests for MemoryManager coordination."""

    def test_manager_provides_both_memory_types(self):
        """Test that manager exposes both working and long-term memory."""
        manager = MemoryManager()

        assert isinstance(manager.working, WorkingMemory)
        assert isinstance(manager.long_term, LongTermMemory)

    def test_manager_facade_crud_is_agent_scoped(
        self, agent_a: AgentID, agent_b: AgentID
    ):
        manager = MemoryManager()
        memory_a = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A memory",
        )
        memory_b = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_b,
            content="Agent B memory",
        )

        manager.add(agent_a, memory_a)
        manager.add(agent_b, memory_b)

        assert manager.get(agent_a, memory_a.memory_id) is memory_a
        assert manager.search(agent_a, "agent a") == [memory_a]
        assert manager.search(agent_a, "agent b") == []
        with pytest.raises(NotFoundError):
            manager.get(agent_b, memory_a.memory_id)

        updated = manager.update(
            agent_a, memory_a.memory_id, content="Updated Agent A memory"
        )
        assert updated.memory_id == memory_a.memory_id
        assert updated.agent_id == agent_a
        assert updated.created_at == memory_a.created_at
        assert manager.get(agent_a, memory_a.memory_id).content == (
            "Updated Agent A memory"
        )

        manager.delete(agent_a, memory_a.memory_id)
        with pytest.raises(NotFoundError):
            manager.get(agent_a, memory_a.memory_id)
        assert manager.get(agent_b, memory_b.memory_id) is memory_b

    def test_manager_facade_rejects_wrong_agent_on_add(
        self, agent_a: AgentID, agent_b: AgentID
    ):
        manager = MemoryManager()
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Agent A memory",
        )

        with pytest.raises(ValueError, match="memory.agent_id"):
            manager.add(agent_b, memory)

    def test_manager_working_and_long_term_are_independent(
        self, agent_a: AgentID
    ):
        """Test that working and long-term memory are completely independent."""
        manager = MemoryManager()

        working_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Working memory",
        )
        long_term_memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Long-term memory",
        )

        # Add to different memory types
        manager.working.add(working_memory)
        manager.long_term.add(long_term_memory)

        # Verify independence
        working_results = manager.working.search(agent_a)
        long_term_results = manager.long_term.search(agent_a)

        assert len(working_results) == 1
        assert len(long_term_results) == 1
        assert working_results[0].content == "Working memory"
        assert long_term_results[0].content == "Long-term memory"

    def test_clearing_working_does_not_affect_long_term(
        self, agent_a: AgentID
    ):
        """Test that clearing working memory doesn't affect long-term."""
        manager = MemoryManager()

        # Add memories to both
        manager.working.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Working",
            )
        )
        manager.long_term.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="Long-term",
            )
        )

        # Clear working memory
        manager.working.clear(agent_a)

        # Verify working is empty
        assert len(manager.working.search(agent_a)) == 0

        # Verify long-term is unaffected
        assert len(manager.long_term.search(agent_a)) == 1

    def test_create_default_factory_method(self):
        """Test the create_default factory method."""
        fake_storage = FakeStorageManager()
        manager = MemoryManager.create_default(storage_manager=fake_storage)

        assert isinstance(manager.working, WorkingMemory)
        assert isinstance(manager.long_term, LongTermMemory)


# Backend Independence Tests


class TestBackendIndependence:
    """Tests verifying no external dependencies."""

    def test_memory_subsystem_works_without_sqlite(self, agent_a: AgentID):
        """Test that memory subsystem works without SQLite."""
        manager = MemoryManager()

        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Test content",
        )

        manager.working.add(memory)
        results = manager.working.search(agent_a)

        assert len(results) == 1

    def test_memory_subsystem_works_without_vector_db(self, agent_a: AgentID):
        """Test that memory subsystem works without vector databases."""
        manager = MemoryManager()

        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Test content for search",
        )

        manager.long_term.add(memory)
        results = manager.long_term.search(agent_a, query="search")

        assert len(results) == 1

    def test_search_is_deterministic(self, agent_a: AgentID):
        """Test that search behavior is deterministic."""
        store = InMemoryStore()

        store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="The quick brown fox",
            )
        )
        store.add(
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="The lazy dog",
            )
        )

        # Search multiple times
        results1 = store.search(agent_a, query="the")
        results2 = store.search(agent_a, query="the")

        # Results should be identical
        assert len(results1) == len(results2)
        assert [m.content for m in results1] == [m.content for m in results2]


# Memory Model Validation Tests


class TestMemoryValidation:
    """Tests for Memory model validation."""

    def test_memory_requires_valid_fields(self, agent_a: AgentID):
        """Test that Memory validates its fields."""
        # Valid memory
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Valid content",
        )
        assert memory.content == "Valid content"

        # Empty content should raise ValidationError
        with pytest.raises(ValidationError):
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="",
            )

        # Whitespace-only content should raise ValidationError
        with pytest.raises(ValidationError):
            Memory(
                memory_id=MemoryID.generate(),
                agent_id=agent_a,
                content="   ",
            )

    def test_memory_preserves_created_at(self, agent_a: AgentID):
        """Test that created_at is preserved during updates."""
        store = InMemoryStore()

        original_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        memory = Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_a,
            content="Original",
            created_at=original_time,
        )

        store.add(memory)

        # Update with same created_at
        updated = Memory(
            memory_id=memory.memory_id,
            agent_id=agent_a,
            content="Updated",
            created_at=original_time,
        )
        store.update(updated)

        retrieved = store.get(memory.memory_id, agent_a)
        assert retrieved.created_at == original_time
