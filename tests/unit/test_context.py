"""
Unit tests for context assembly.

The builder does not talk to the memory manager. Callers pass
``Memory`` instances in, and ``build()`` returns a validated
``Context`` snapshot.
"""

from __future__ import annotations

import pytest

from aios.context import ContextView, as_context_memory
from aios.context.builder import ContextBuilder
from aios.core import (
    AgentID,
    Context,
    Memory,
    MemoryID,
    ValidationError,
)


def _memory(content: str = "Important information.") -> Memory:
    return Memory(
        memory_id=MemoryID.generate(),
        agent_id=AgentID.generate(),
        content=content,
        metadata={"source": "test"},
    )


def test_build_with_all_fields_populated():
    memory = _memory("Remember this.")
    builder = ContextBuilder()

    builder.add_system_instruction("You are an AI agent.")
    builder.add_system_instruction("Be concise.")
    builder.add_message("user", "Hello")
    builder.add_message("assistant", "Hi")
    builder.add_memory_entries([memory])
    builder.set_working_state({"step": 1})
    builder.set_metadata({"trace": "req-1"})

    context = builder.build()

    assert isinstance(context, Context)
    assert context.system == "You are an AI agent.\nBe concise."
    assert context.conversation == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert context.memory == [memory]
    assert context.working_state == {"step": 1}
    assert context.metadata == {"trace": "req-1"}


def test_build_with_empty_fields():
    context = ContextBuilder().build()

    assert context.system is None
    assert context.conversation == []
    assert context.memory == []
    assert context.working_state == {}
    assert context.metadata == {}


def test_memory_instance_round_trips():
    memory = _memory("Round trip.")
    builder = ContextBuilder()
    builder.add_memory_entries([memory])

    context = builder.build()

    assert len(context.memory) == 1
    stored = context.memory[0]
    assert stored is memory
    assert isinstance(stored, Memory)
    assert stored.memory_id == memory.memory_id
    assert stored.agent_id == memory.agent_id
    assert stored.content == memory.content
    assert stored.metadata == memory.metadata
    assert stored.created_at == memory.created_at
    assert as_context_memory(memory) is memory


def test_builder_mutation_does_not_change_built_context():
    memory = _memory("Original.")
    later = _memory("Later.")
    state = {"step": 1, "notes": {"topic": "weather"}}
    builder = ContextBuilder()

    builder.add_system_instruction("Be helpful.")
    builder.add_message("user", "Hello")
    builder.add_memory_entries([memory])
    builder.set_working_state(state)
    builder.set_metadata({"version": 1})

    context = builder.build()

    builder.add_system_instruction("Be brief.")
    builder.add_message("assistant", "Hi")
    builder.add_memory_entries([later])
    builder.set_working_state({"step": 2})
    builder.set_metadata({"version": 2})
    state["step"] = 99
    state["notes"]["topic"] = "changed"

    assert context.system == "Be helpful."
    assert context.conversation == [
        {"role": "user", "content": "Hello"},
    ]
    assert context.memory == [memory]
    assert context.working_state == {
        "step": 1,
        "notes": {"topic": "weather"},
    }
    assert context.metadata == {"version": 1}

    rebuilt = builder.build()
    assert rebuilt.system == "Be helpful.\nBe brief."
    assert len(rebuilt.conversation) == 2
    assert rebuilt.memory == [memory, later]
    assert rebuilt.working_state == {"step": 2}
    assert context.working_state["notes"]["topic"] == "weather"


def test_caller_memory_list_is_not_shared():
    memory = _memory()
    entries = [memory]
    builder = ContextBuilder()
    builder.add_memory_entries(entries)

    context = builder.build()
    entries.clear()
    entries.append(_memory("other"))

    assert context.memory == [memory]


def test_context_view_snapshots_are_read_only():
    memory = _memory("Seen.")
    builder = ContextBuilder()
    builder.add_system_instruction("Be helpful.")
    builder.add_message("user", "Hello")
    builder.add_memory_entries([memory])
    builder.set_working_state({"step": 1})
    builder.set_metadata({"trace": "req-1"})

    context = builder.build()
    view = ContextView(context)

    snapshot = view.memory[0]

    assert view.context is context
    assert view.system == "Be helpful."
    assert view.conversation[0]["content"] == "Hello"
    assert isinstance(snapshot, Memory)
    assert snapshot.content == "Seen."
    assert snapshot is not memory
    assert view.working_state["step"] == 1
    assert view.metadata["trace"] == "req-1"

    with pytest.raises(TypeError):
        view.working_state["step"] = 2  # type: ignore[index]

    with pytest.raises(TypeError):
        view.conversation[0]["role"] = "system"  # type: ignore[index]

    snapshot.content = "Changed snapshot."

    assert context.working_state["step"] == 1
    assert context.conversation[0]["role"] == "user"
    assert context.memory[0].content == "Seen."


def test_builder_rejects_invalid_inputs():
    builder = ContextBuilder()

    with pytest.raises(ValidationError):
        builder.add_system_instruction("   ")

    with pytest.raises(ValidationError):
        builder.add_message("", "Hello")

    with pytest.raises(ValidationError):
        builder.add_memory_entries(["not-a-memory"])  # type: ignore[list-item]

    with pytest.raises(ValidationError):
        builder.set_working_state(["not-a-dict"])  # type: ignore[arg-type]
