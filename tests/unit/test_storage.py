"""Unit tests for aios.storage package."""

import json
from pathlib import Path
import pytest

from aios.core.ids import AgentID, MemoryID
from aios.core.models import Memory
from aios.storage.artifact import (
    Artifact,
    ArtifactExistsError,
    ArtifactNotFound,
    InvalidNamespaceError,
    PathTraversalError,
    StorageError,
)
from aios.storage.filesystem import atomic_write, safe_resolve
from aios.storage.manager import StorageManager
from aios.storage.namespace import resolve_namespace_dir, validate_namespace


def test_storage_end_to_end(tmp_path: Path) -> None:
    """Full lifecycle test for StorageManager in unit tests."""
    manager = StorageManager(tmp_path)
    content = b"Binary \x00\xFF data"
    meta = {"type": "test_data", "version": 1}

    # Create
    art = manager.create(
        namespace="main-ns",
        name="item.bin",
        content=content,
        metadata=meta,
    )
    assert art.name == "item.bin"
    assert art.size == len(content)
    assert art.metadata == meta

    # Read
    assert manager.read("main-ns", "item.bin") == content
    assert manager.exists("main-ns", "item.bin")

    # List
    items = manager.list("main-ns")
    assert len(items) == 1
    assert items[0].name == "item.bin"

    # Delete
    manager.delete_at("main-ns", "item.bin")
    assert not manager.exists("main-ns", "item.bin")


def test_store_memory_persists_agent_scoped_json(tmp_path: Path) -> None:
    manager = StorageManager(tmp_path)
    agent_id = AgentID.generate()
    memory = Memory(
        memory_id=MemoryID.generate(),
        agent_id=agent_id,
        content="Persisted memory",
        metadata={"source": "test"},
    )

    artifact_id = manager.store_memory(memory, agent_id)
    artifact = manager.retrieve(artifact_id)
    payload = json.loads(manager.read_artifact(artifact_id))

    assert artifact.namespace == f"agent-{agent_id}"
    assert artifact.metadata == {
        "kind": "memory",
        "memory_id": str(memory.memory_id),
    }
    assert payload == {
        "memory_id": str(memory.memory_id),
        "agent_id": str(agent_id),
        "content": memory.content,
        "metadata": memory.metadata,
        "created_at": memory.created_at.isoformat(),
    }
