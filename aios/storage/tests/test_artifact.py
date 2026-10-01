"""Unit tests for Artifact dataclass and Storage exception hierarchy."""

from dataclasses import FrozenInstanceError
import pytest

from aios.storage.artifact import (
    Artifact,
    ArtifactExistsError,
    ArtifactNotFound,
    InvalidNamespaceError,
    PathTraversalError,
    StorageError,
)


def test_exception_hierarchy() -> None:
    """Verify all custom exceptions inherit from StorageError."""
    assert issubclass(ArtifactNotFound, StorageError)
    assert issubclass(PathTraversalError, StorageError)
    assert issubclass(InvalidNamespaceError, StorageError)
    assert issubclass(ArtifactExistsError, StorageError)
    assert issubclass(StorageError, Exception)

    err = ArtifactNotFound("not found")
    assert isinstance(err, StorageError)
    assert str(err) == "not found"


def test_artifact_dataclass_creation() -> None:
    """Verify artifact creation and attributes."""
    art = Artifact(
        id="ns1:doc.txt",
        namespace="ns1",
        name="doc.txt",
        size=42,
        created_at=123456789.0,
        content_type="text/plain",
        metadata={"author": "alice"},
    )
    assert art.id == "ns1:doc.txt"
    assert art.namespace == "ns1"
    assert art.name == "doc.txt"
    assert art.size == 42
    assert art.created_at == 123456789.0
    assert art.content_type == "text/plain"
    assert art.metadata == {"author": "alice"}


def test_artifact_frozen_immutability() -> None:
    """Verify artifact is frozen and immutable."""
    art = Artifact(
        id="ns1:doc.txt",
        namespace="ns1",
        name="doc.txt",
        size=10,
        created_at=100.0,
    )
    with pytest.raises(FrozenInstanceError):
        art.name = "changed.txt"  # type: ignore[misc]
