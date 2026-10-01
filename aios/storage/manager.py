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
    ArtifactID,
    ArtifactNotFound,
    InvalidArtifactName,
)
from .filesystem import atomic_write, internal_resolve, safe_resolve
from .namespace import resolve_namespace_dir, validate_namespace


class StorageManager:
    """Manages creation, retrieval, listing, and deletion of artifacts isolated by namespaces.

    The public contract is store/retrieve/delete over an opaque ArtifactID.
    Namespace-and-name calls are the equivalent internal surface and remain
    available for callers that need to enumerate or address artifacts directly.
    """

    def __init__(self, root_path: Path | str) -> None:
        """Initialize the StorageManager with a base storage root directory.

        Args:
            root_path: Directory under which all namespace directories live.
                Created if it does not exist.
        """
        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """Alias for root_path retained for backwards compatibility."""
        return self.root_path

    # --- Public contract -------------------------------------------------

    def store(
        self,
        namespace: str,
        key: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactID:
        """Store an artifact and return its opaque ID.

        Args:
            namespace: The isolating namespace.
            key: Artifact key within the namespace.
            data: Raw bytes to store.
            metadata: Optional user metadata dict.

        Returns:
            An ArtifactID to pass to retrieve() or delete().

        Raises:
            InvalidNamespaceError: If the namespace is invalid.
            InvalidArtifactName: If the key uses a reserved path component.
            PathTraversalError: If the key attempts path traversal.
            ArtifactExistsError: If the key already exists in the namespace.
        """
        return self.create(namespace, key, data, metadata=metadata).id

    def retrieve(self, artifact_id: ArtifactID) -> Artifact:
        """Retrieve artifact metadata by ID.

        Args:
            artifact_id: An ID previously returned by store().

        Returns:
            The Artifact metadata instance.

        Raises:
            ArtifactNotFound: If the artifact does not exist.
            InvalidArtifactName: If the ID refers to a reserved path component.
            PathTraversalError: If the ID encodes a traversing path.
        """
        namespace, name = self._parse_artifact_id(artifact_id)
        return self.get_artifact(namespace, name)

    def read_artifact(self, artifact_id: ArtifactID) -> bytes:
        """Retrieve raw artifact content by ID.

        Args:
            artifact_id: An ID previously returned by store().

        Returns:
            The raw bytes.

        Raises:
            ArtifactNotFound: If the artifact does not exist.
        """
        namespace, name = self._parse_artifact_id(artifact_id)
        return self.read(namespace, name)

    def delete(self, artifact_id: ArtifactID) -> None:
        """Delete an artifact by ID.

        Args:
            artifact_id: An ID previously returned by store().

        Raises:
            ArtifactNotFound: If the artifact does not exist.
        """
        namespace, name = self._parse_artifact_id(artifact_id)
        self.delete_at(namespace, name)

    @staticmethod
    def _parse_artifact_id(artifact_id: ArtifactID) -> tuple[str, str]:
        """Split an ArtifactID back into its namespace and name.

        Args:
            artifact_id: The opaque ID to split.

        Returns:
            A (namespace, name) tuple.

        Raises:
            ArtifactNotFound: If the ID is not a well-formed artifact ID.
        """
        if not isinstance(artifact_id, str) or ":" not in artifact_id:
            raise ArtifactNotFound(f"Malformed artifact id: {artifact_id!r}")
        namespace, _, name = artifact_id.partition(":")
        return namespace, name

    # --- Namespace + name surface ---------------------------------------

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

        meta_path = internal_resolve(ns_root, f".meta/{name}.json")

        artifact = Artifact(
            id=f"{namespace}:{name}",
            namespace=namespace,
            name=name,
            content_type=content_type,
            size=len(content),
            created_at=time.time(),
            metadata=dict(metadata) if metadata is not None else {},
        )

        # Serialize metadata first so a non-serialisable value fails before
        # anything is written to disk.
        meta_bytes = json.dumps(asdict(artifact), ensure_ascii=False, indent=2).encode("utf-8")

        # Write the sidecar first: a leftover sidecar is inert (retrieval
        # requires the content file), whereas leftover content without
        # metadata would be served with fabricated defaults.
        atomic_write(meta_path, meta_bytes)

        try:
            atomic_write(content_path, content)
        except Exception:
            try:
                if meta_path.is_file():
                    meta_path.unlink()
            except OSError:
                pass
            raise

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

        meta_path = internal_resolve(ns_root, f".meta/{name}.json")
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

    def delete_at(self, namespace: str, name: str) -> None:
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
            meta_path = internal_resolve(ns_root, f".meta/{name}.json")
            if meta_path.is_file():
                meta_path.unlink()
        except Exception:
            pass
