"""Low-level filesystem operations with path traversal safety and atomic writes."""

import os
from pathlib import Path, PurePosixPath
import tempfile

from .artifact import InvalidArtifactName, PathTraversalError

RESERVED_COMPONENTS = frozenset({".meta"})
"""Path components reserved for internal bookkeeping.

Artifact names containing any of these are rejected so callers cannot write
into the metadata sidecar directory and corrupt or hide artifacts.
"""


def _reject_reserved(name: str) -> None:
    """Raise if any component of a user-supplied path is reserved.

    Args:
        name: The user-supplied relative path.

    Raises:
        InvalidArtifactName: If a path component is reserved.
    """
    parts = PurePosixPath(name.replace("\\", "/")).parts
    for part in parts:
        if part in RESERVED_COMPONENTS:
            raise InvalidArtifactName(
                f"Path component '{part}' in '{name}' is reserved for internal use"
            )


def safe_resolve(root: Path, name: str) -> Path:
    """Safely resolve a relative path against a namespace root.

    Resolves root first to handle symlinks (e.g. /tmp on macOS). Rejects
    absolute paths up front, then joins and resolves the candidate path,
    verifying it remains strictly within the namespace root.

    Args:
        root: The namespace root directory.
        name: The relative file path to resolve.

    Returns:
        The safely resolved Path.

    Raises:
        PathTraversalError: If the path is absolute, points to root, or escapes root.
        InvalidArtifactName: If a path component is reserved for internal use.
    """
    if not isinstance(name, str):
        raise PathTraversalError(f"Expected string path, got {type(name).__name__}")

    _reject_reserved(name)
    return _resolve_checked(root, name)


def internal_resolve(root: Path, name: str) -> Path:
    """Resolve a path for internal bookkeeping use only.

    Identical to safe_resolve but permits reserved components, so the manager
    can address its own metadata directory. Never call this with a
    user-supplied path.

    Args:
        root: The namespace root directory.
        name: The relative path to resolve.

    Returns:
        The safely resolved Path.

    Raises:
        PathTraversalError: If the path is absolute, points to root, or escapes root.
    """
    return _resolve_checked(root, name)


def _resolve_checked(root: Path, name: str) -> Path:
    """Apply traversal checks to a relative path and resolve it under root.

    Args:
        root: The namespace root directory.
        name: The relative path to resolve.

    Returns:
        The safely resolved Path.

    Raises:
        PathTraversalError: If the path is absolute, points to root, or escapes root.
    """
    if not isinstance(name, str):
        raise PathTraversalError(f"Expected string path, got {type(name).__name__}")

    raw_path = Path(name)
    if raw_path.is_absolute() or raw_path.drive or name.startswith(("/", "\\")):
        raise PathTraversalError(f"Absolute path '{name}' is not permitted")

    resolved_root = Path(root).resolve()
    candidate = (resolved_root / raw_path).resolve()

    if not candidate.is_relative_to(resolved_root):
        raise PathTraversalError(
            f"Path '{name}' resolves to '{candidate}', which escapes root '{resolved_root}'"
        )

    if candidate == resolved_root:
        raise PathTraversalError(
            f"Path '{name}' cannot resolve to the root directory itself"
        )

    return candidate


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically write binary data to a file.

    Creates parent directories as needed, writes to a temporary file in the
    same directory, flushes and syncs to disk, and uses os.replace() to atomically
    swap into the final destination path. Cleans up temp file on failure.

    Args:
        path: The final target file path.
        data: The bytes to write.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")

    target_path = Path(path)
    parent_dir = target_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_file_path = tempfile.mkstemp(dir=parent_dir, prefix=".tmp_")
    tmp_path = Path(tmp_file_path)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
