"""Artifact data model and storage exceptions for AIOS storage."""

from dataclasses import dataclass, field
from typing import Any


class StorageError(Exception):
    """Base exception for all storage-related errors."""


class ArtifactNotFound(StorageError):
    """Raised when an artifact cannot be found in the specified namespace."""


class PathTraversalError(StorageError):
    """Raised when a resolved path escapes its namespace root."""


class InvalidNamespaceError(StorageError):
    """Raised when a namespace name does not meet naming rules."""


class ArtifactExistsError(StorageError):
    """Raised when an artifact already exists and overwrite is False."""


@dataclass(frozen=True)
class Artifact:
    """Represents a stored artifact with content metadata.

    Attributes:
        id: Unique identifier for the artifact (e.g., 'namespace:name').
        namespace: The namespace isolating this artifact.
        name: Name / relative path of the artifact within its namespace.
        content_type: MIME type or format of the artifact content.
        size: Content size in bytes.
        created_at: Unix timestamp (seconds) when artifact was created.
        metadata: Arbitrary user/system metadata dictionary.
    """

    id: str
    namespace: str
    name: str
    size: int
    created_at: float
    content_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)
