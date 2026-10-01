"""High-level StorageManager interface for artifacts and namespaces."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

from .artifact import (
    Artifact,
    ArtifactExistsError,
    ArtifactNotFound,
)
from .filesystem import atomic_write, safe_resolve
from .namespace import resolve_namespace_dir, validate_namespace


class StorageManager:
    """Manages creation, retrieval, listing, and deletion of artifacts isolated by namespaces."""

    def __init__(self, base_dir: Path | str) -> None:
        """Initialize the StorageManager with a base storage root directory."""
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        namespace: str,
        name: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
        content_type: str = "application/octet-stream",
        overwrite: bool = False,
    ) -> Artifact:
        """Create a new artifact in the specified namespace.

        Args:
            namespace: The isolating namespace.
            name: Artifact relative name/path.
            content: Raw binary content to store.
            metadata: Optional user metadata dict.
            content_type: MIME content type.
            overwrite: Whether to overwrite existing artifact.

        Returns:
            The created Artifact metadata object.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
            PathTraversalError: If name attempts path traversal outside namespace.
            ArtifactExistsError: If artifact already exists and overwrite is False.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=True)
        content_path = safe_resolve(ns_root, name)

        if content_path.exists() and not overwrite:
            raise ArtifactExistsError(
                f"Artifact '{name}' already exists in namespace '{namespace}'"
            )

        meta_path = safe_resolve(ns_root, f".meta/{name}.json")

        artifact = Artifact(
            id=f"{namespace}:{name}",
            namespace=namespace,
            name=name,
            content_type=content_type,
            size=len(content),
            created_at=time.time(),
            metadata=dict(metadata) if metadata is not None else {},
        )

        # Write content atomically
        atomic_write(content_path, content)

        # Write metadata sidecar atomically
        meta_bytes = json.dumps(asdict(artifact), ensure_ascii=False, indent=2).encode("utf-8")
        atomic_write(meta_path, meta_bytes)

        return artifact

    def read(self, namespace: str, name: str) -> bytes:
        """Read the raw content bytes of an artifact.

        Args:
            namespace: The isolating namespace.
            name: Artifact relative name/path.

        Returns:
            The raw binary content bytes.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
            PathTraversalError: If name attempts path traversal outside namespace.
            ArtifactNotFound: If the artifact does not exist.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=False)
        if not ns_root.exists():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        content_path = safe_resolve(ns_root, name)
        if not content_path.is_file():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        return content_path.read_bytes()

    def get_artifact(self, namespace: str, name: str) -> Artifact:
        """Retrieve the Artifact metadata for an artifact.

        Args:
            namespace: The isolating namespace.
            name: Artifact relative name/path.

        Returns:
            Artifact metadata instance.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
            PathTraversalError: If name attempts path traversal outside namespace.
            ArtifactNotFound: If the artifact does not exist.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=False)
        if not ns_root.exists():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        content_path = safe_resolve(ns_root, name)
        if not content_path.is_file():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        meta_path = safe_resolve(ns_root, f".meta/{name}.json")
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                return Artifact(**data)
            except Exception:
                pass

        stat = content_path.stat()
        return Artifact(
            id=f"{namespace}:{name}",
            namespace=namespace,
            name=name,
            size=stat.st_size,
            created_at=stat.st_mtime,
            content_type="application/octet-stream",
            metadata={},
        )

    def exists(self, namespace: str, name: str) -> bool:
        """Check if an artifact exists in the specified namespace.

        Args:
            namespace: The isolating namespace.
            name: Artifact relative name/path.

        Returns:
            True if artifact exists and is a file, False otherwise.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
            PathTraversalError: If name attempts path traversal outside namespace.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=False)
        if not ns_root.exists():
            return False

        content_path = safe_resolve(ns_root, name)
        return content_path.is_file()

    def list(self, namespace: str) -> list[Artifact]:
        """List all artifacts in the specified namespace.

        Skips `.meta/` and temporary files.

        Args:
            namespace: The isolating namespace.

        Returns:
            List of Artifact metadata objects.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=False)
        if not ns_root.exists():
            return []

        artifacts: list[Artifact] = []
        meta_root = (ns_root / ".meta").resolve()

        for root, dirs, files in os.walk(ns_root):
            root_path = Path(root).resolve()
            # Skip traversing inside .meta directory
            if root_path == meta_root or meta_root in root_path.parents:
                continue

            # Remove .meta in-place so os.walk doesn't descend into it
            dirs[:] = [d for d in dirs if d != ".meta"]

            for file_name in files:
                if file_name.startswith(".tmp_"):
                    continue
                file_path = (root_path / file_name).resolve()
                if not file_path.is_file():
                    continue

                rel_name = file_path.relative_to(ns_root).as_posix()
                try:
                    artifact = self.get_artifact(namespace, rel_name)
                    artifacts.append(artifact)
                except Exception:
                    stat = file_path.stat()
                    artifacts.append(
                        Artifact(
                            id=f"{namespace}:{rel_name}",
                            namespace=namespace,
                            name=rel_name,
                            size=stat.st_size,
                            created_at=stat.st_mtime,
                            content_type="application/octet-stream",
                            metadata={},
                        )
                    )

        artifacts.sort(key=lambda a: a.name)
        return artifacts

    def delete(self, namespace: str, name: str) -> None:
        """Delete an artifact and its metadata sidecar.

        Args:
            namespace: The isolating namespace.
            name: Artifact relative name/path.

        Raises:
            InvalidNamespaceError: If namespace name is invalid.
            PathTraversalError: If name attempts path traversal outside namespace.
            ArtifactNotFound: If artifact does not exist.
        """
        validate_namespace(namespace)
        ns_root = resolve_namespace_dir(self.base_dir, namespace, create=False)
        if not ns_root.exists():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        content_path = safe_resolve(ns_root, name)
        if not content_path.is_file():
            raise ArtifactNotFound(
                f"Artifact '{name}' not found in namespace '{namespace}'"
            )

        content_path.unlink()

        try:
            meta_path = safe_resolve(ns_root, f".meta/{name}.json")
            if meta_path.is_file():
                meta_path.unlink()
        except Exception:
            pass
