"""AIOS Storage package for artifacts, namespaces, and isolated safe storage."""

from .artifact import (
    Artifact,
    ArtifactExistsError,
    ArtifactNotFound,
    InvalidNamespaceError,
    PathTraversalError,
    StorageError,
)
from .filesystem import atomic_write, safe_resolve
from .manager import StorageManager
from .namespace import resolve_namespace_dir, validate_namespace

__all__ = [
    "Artifact",
    "ArtifactExistsError",
    "ArtifactNotFound",
    "InvalidNamespaceError",
    "PathTraversalError",
    "StorageError",
    "StorageManager",
    "atomic_write",
    "resolve_namespace_dir",
    "safe_resolve",
    "validate_namespace",
]
