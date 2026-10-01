"""Unit tests for aios.storage package."""

from pathlib import Path
import pytest

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
    manager.delete("main-ns", "item.bin")
    assert not manager.exists("main-ns", "item.bin")
