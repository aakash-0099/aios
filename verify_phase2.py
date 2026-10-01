"""
Phase 2 verification script for memory subsystem.

This script verifies:
1. Capacity enforcement per agent
2. Deterministic eviction (oldest first)
3. Direct storage integration (not Kernel)
4. Agent isolation for capacity
5. Memory objects returned (Context Builder compatible)
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory
from aios.memory import MemoryManager, WorkingMemory
from aios.storage import InMemoryStorageManager


class FakeStorage:
    """Test double for verifying storage calls."""
    
    def __init__(self):
        self.stored = []
        self.call_count = 0
    
    def store_memory(self, memory, agent_id):
        self.stored.append((memory, agent_id))
        self.call_count += 1


def test_capacity_enforcement():
    """Test that capacity is enforced per agent."""
    print("Testing capacity enforcement...")
    
    fake_storage = FakeStorage()
    working = WorkingMemory(
        capacity_per_agent=3,
        eviction_callback=fake_storage.store_memory
    )
    
    agent_id = AgentID.generate()
    
    # Add 3 memories (at capacity)
    for i in range(3):
        working.add(Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_id,
            content=f"Memory {i}"
        ))
    
    assert len(working.search(agent_id)) == 3
    assert fake_storage.call_count == 0
    print("✓ Capacity respected (no eviction yet)")
    
    # Add 4th memory (should trigger eviction)
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Memory 3"
    ))
    
    assert len(working.search(agent_id)) == 3
    assert fake_storage.call_count == 1
    print("✓ Eviction triggered at capacity")


def test_oldest_evicted_first():
    """Test that oldest memory is evicted first."""
    print("\nTesting oldest-first eviction...")
    
    fake_storage = FakeStorage()
    working = WorkingMemory(
        capacity_per_agent=2,
        eviction_callback=fake_storage.store_memory
    )
    
    agent_id = AgentID.generate()
    
    old_mem = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Old memory",
        created_at=datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    new_mem = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="New memory",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    newest_mem = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Newest memory",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    
    working.add(old_mem)
    working.add(new_mem)
    working.add(newest_mem)  # Should evict old_mem
    
    assert fake_storage.call_count == 1
    evicted = fake_storage.stored[0][0]
    assert evicted.content == "Old memory"
    print("✓ Oldest memory evicted first")
    
    # Verify new and newest remain
    remaining = working.search(agent_id)
    contents = {m.content for m in remaining}
    assert "New memory" in contents
    assert "Newest memory" in contents
    assert "Old memory" not in contents
    print("✓ Newer memories remain in working memory")


def test_per_agent_capacity_isolation():
    """Test that capacity is tracked independently per agent."""
    print("\nTesting per-agent capacity isolation...")
    
    fake_storage = FakeStorage()
    working = WorkingMemory(
        capacity_per_agent=2,
        eviction_callback=fake_storage.store_memory
    )
    
    agent_a = AgentID.generate()
    agent_b = AgentID.generate()
    
    # Agent A: fill to capacity
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_a,
        content="A1"
    ))
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_a,
        content="A2"
    ))
    
    # Agent B: fill to capacity
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_b,
        content="B1"
    ))
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_b,
        content="B2"
    ))
    
    assert len(working.search(agent_a)) == 2
    assert len(working.search(agent_b)) == 2
    assert fake_storage.call_count == 0
    print("✓ Both agents at capacity independently")
    
    # Agent A adds another (should evict from A only)
    working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_a,
        content="A3"
    ))
    
    assert len(working.search(agent_a)) == 2
    assert len(working.search(agent_b)) == 2  # B unchanged
    assert fake_storage.call_count == 1
    
    evicted = fake_storage.stored[0][0]
    assert evicted.agent_id == agent_a
    print("✓ Agent A eviction doesn't affect Agent B")


def test_storage_integration_direct_call():
    """Test that storage is called directly (not through Kernel)."""
    print("\nTesting direct storage integration...")
    
    fake_storage = FakeStorage()
    manager = MemoryManager(
        storage_manager=fake_storage,
        capacity_per_agent=1
    )
    
    agent_id = AgentID.generate()
    
    # Add 2 memories to trigger eviction
    mem1 = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Memory 1"
    )
    mem2 = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Memory 2"
    )
    
    manager.working.add(mem1)
    manager.working.add(mem2)  # Should evict mem1
    
    # Verify storage was called
    assert fake_storage.call_count == 1
    print("✓ Storage callback invoked")
    
    # Verify correct signature (Memory, AgentID)
    evicted_memory, evicted_agent = fake_storage.stored[0]
    assert isinstance(evicted_memory, Memory)
    assert isinstance(evicted_agent, AgentID)
    print("✓ Storage receives Memory object (not AgentRequest)")
    
    # Verify it's the correct memory
    assert evicted_memory.memory_id == mem1.memory_id
    print("✓ Correct memory sent to storage")
    
    print("✓ VERIFIED: Direct call to storage (no Kernel)")


def test_context_builder_compatibility():
    """Test that Memory objects are returned for Context Builder."""
    print("\nTesting Context Builder compatibility...")
    
    manager = MemoryManager()
    agent_id = AgentID.generate()
    
    memory = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Test memory"
    )
    
    # Add to working memory
    manager.working.add(memory)
    
    # Get returns Memory
    retrieved = manager.working.get(memory.memory_id, agent_id)
    assert isinstance(retrieved, Memory)
    assert retrieved.content == "Test memory"
    print("✓ get() returns Memory object")
    
    # Search returns list[Memory]
    results = manager.working.search(agent_id)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], Memory)
    print("✓ search() returns list[Memory]")
    
    # Same for long-term
    manager.long_term.add(memory)
    results = manager.long_term.search(agent_id)
    assert isinstance(results, list)
    assert all(isinstance(m, Memory) for m in results)
    print("✓ Long-term memory also returns Memory objects")
    
    print("✓ VERIFIED: Context Builder compatible (returns Memory objects)")


def test_real_storage_manager():
    """Test with real InMemoryStorageManager."""
    print("\nTesting with InMemoryStorageManager...")
    
    storage = InMemoryStorageManager()
    manager = MemoryManager(
        storage_manager=storage,
        capacity_per_agent=2
    )
    
    agent_id = AgentID.generate()
    
    # Add 3 memories (should evict 1)
    for i in range(3):
        manager.working.add(Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_id,
            content=f"Memory {i}",
            created_at=datetime(2023, 1, i+1, tzinfo=timezone.utc)
        ))
    
    # Check working memory has 2
    assert len(manager.working.search(agent_id)) == 2
    print("✓ Working memory at capacity")
    
    # Check storage has 1
    stored = storage.get_stored_memories(agent_id)
    assert len(stored) == 1
    assert stored[0].content == "Memory 0"
    print("✓ Evicted memory in storage")
    print("✓ Real storage integration works")


def main():
    """Run all Phase 2 verification tests."""
    print("=" * 60)
    print("Phase 2 Memory Subsystem Verification")
    print("=" * 60)
    
    try:
        test_capacity_enforcement()
        test_oldest_evicted_first()
        test_per_agent_capacity_isolation()
        test_storage_integration_direct_call()
        test_context_builder_compatibility()
        test_real_storage_manager()
        
        print("\n" + "=" * 60)
        print("✓ ALL PHASE 2 TESTS PASSED!")
        print("=" * 60)
        print("\nKey Verifications:")
        print("  ✓ Capacity enforced per agent")
        print("  ✓ Oldest memory evicted first (deterministic)")
        print("  ✓ Direct storage call (NOT through Kernel)")
        print("  ✓ Agent isolation maintained")
        print("  ✓ Returns Memory objects (Context Builder compatible)")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
