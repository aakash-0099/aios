"""
Quick verification script for memory subsystem.

This script performs basic smoke tests to verify the implementation works.
"""

import sys

# Add parent directory to path for imports
sys.path.insert(0, ".")

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory
from aios.memory import MemoryManager, WorkingMemory, LongTermMemory, InMemoryStore


def test_basic_operations():
    """Test basic memory operations."""
    print("Testing basic operations...")
    
    agent_id = AgentID.generate()
    memory_id = MemoryID.generate()
    
    memory = Memory(
        memory_id=memory_id,
        agent_id=agent_id,
        content="Test memory content",
        metadata={"test": True}
    )
    
    # Test InMemoryStore
    store = InMemoryStore()
    store.add(memory)
    retrieved = store.get(memory_id, agent_id)
    assert retrieved.content == "Test memory content"
    print("✓ InMemoryStore add/get works")
    
    # Test search
    results = store.search(agent_id)
    assert len(results) == 1
    print("✓ Search works")
    
    # Test update
    updated_memory = Memory(
        memory_id=memory_id,
        agent_id=agent_id,
        content="Updated content",
        created_at=memory.created_at
    )
    store.update(updated_memory)
    retrieved = store.get(memory_id, agent_id)
    assert retrieved.content == "Updated content"
    print("✓ Update works")
    
    # Test delete
    store.delete(memory_id, agent_id)
    try:
        store.get(memory_id, agent_id)
        assert False, "Should have raised NotFoundError"
    except Exception as e:
        assert "not found" in str(e).lower()
    print("✓ Delete works")


def test_agent_isolation():
    """Test agent isolation."""
    print("\nTesting agent isolation...")
    
    store = InMemoryStore()
    agent_a = AgentID.generate()
    agent_b = AgentID.generate()
    
    memory_a = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_a,
        content="Agent A memory"
    )
    memory_b = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_b,
        content="Agent B memory"
    )
    
    store.add(memory_a)
    store.add(memory_b)
    
    # Agent A should only see their memory
    results_a = store.search(agent_a)
    assert len(results_a) == 1
    assert results_a[0].content == "Agent A memory"
    print("✓ Agent A isolated")
    
    # Agent B should only see their memory
    results_b = store.search(agent_b)
    assert len(results_b) == 1
    assert results_b[0].content == "Agent B memory"
    print("✓ Agent B isolated")


def test_working_memory():
    """Test working memory with clear()."""
    print("\nTesting working memory...")
    
    working = WorkingMemory()
    agent_id = AgentID.generate()
    
    for i in range(3):
        working.add(Memory(
            memory_id=MemoryID.generate(),
            agent_id=agent_id,
            content=f"Working memory {i}"
        ))
    
    results = working.search(agent_id)
    assert len(results) == 3
    print("✓ Working memory add works")
    
    working.clear(agent_id)
    results = working.search(agent_id)
    assert len(results) == 0
    print("✓ Working memory clear works")


def test_long_term_memory():
    """Test long-term memory."""
    print("\nTesting long-term memory...")
    
    long_term = LongTermMemory()
    agent_id = AgentID.generate()
    
    memory = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Long-term memory"
    )
    
    long_term.add(memory)
    results = long_term.search(agent_id)
    assert len(results) == 1
    print("✓ Long-term memory works")


def test_memory_manager():
    """Test memory manager."""
    print("\nTesting memory manager...")
    
    manager = MemoryManager()
    agent_id = AgentID.generate()
    
    # Add to working
    manager.working.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Working"
    ))
    
    # Add to long-term
    manager.long_term.add(Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Long-term"
    ))
    
    # Verify independence
    working_results = manager.working.search(agent_id)
    long_term_results = manager.long_term.search(agent_id)
    
    assert len(working_results) == 1
    assert len(long_term_results) == 1
    print("✓ Memory manager coordinates both memory types")
    
    # Clear working, verify long-term unaffected
    manager.working.clear(agent_id)
    assert len(manager.working.search(agent_id)) == 0
    assert len(manager.long_term.search(agent_id)) == 1
    print("✓ Working and long-term are independent")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Memory Subsystem Verification")
    print("=" * 60)
    
    try:
        test_basic_operations()
        test_agent_isolation()
        test_working_memory()
        test_long_term_memory()
        test_memory_manager()
        
        print("\n" + "=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
