"""AIOS Storage package for artifacts, namespaces, and isolated safe storage."""

from .artifact import (
    Artifact,
    ArtifactExistsError,
    ArtifactID,
    ArtifactNotFound,
    InvalidArtifactName,
    InvalidNamespaceError,
    PathTraversalError,
    StorageError,
)
from .filesystem import RESERVED_COMPONENTS, atomic_write, internal_resolve, safe_resolve
from .manager import StorageManager
from .namespace import resolve_namespace_dir, validate_namespace

__all__ = [
    "Artifact",
    "ArtifactExistsError",
    "ArtifactID",
    "ArtifactNotFound",
    "InvalidArtifactName",
    "InvalidNamespaceError",
    "PathTraversalError",
    "RESERVED_COMPONENTS",
    "StorageError",
    "StorageManager",
    "atomic_write",
    "internal_resolve",
    "resolve_namespace_dir",
    "safe_resolve",
    "validate_namespace",
]
