"""Acceptance and integration tests for StorageManager."""

import ast
from pathlib import Path
import sys
import pytest

import aios.storage
from aios.storage.artifact import (
    ArtifactExistsError,
    ArtifactNotFound,
    InvalidNamespaceError,
    PathTraversalError,
)
from aios.storage.manager import StorageManager


def test_create_and_read_non_utf8_bytes(tmp_path: Path) -> None:
    """Create then read returns identical bytes using arbitrary non-UTF8 binary data."""
    manager = StorageManager(tmp_path)
    # Non-UTF8 byte sequence with null bytes and invalid UTF-8 codepoints
    non_utf8_data = bytes([0x80, 0xFF, 0x00, 0xC0, 0xC1, 0xF5, 0xFF, 0x7F, 0x01])

    artifact = manager.create(
        namespace="default",
        name="blob.bin",
        content=non_utf8_data,
        content_type="application/octet-stream",
    )

    assert artifact.namespace == "default"
    assert artifact.name == "blob.bin"
    assert artifact.size == len(non_utf8_data)

    read_data = manager.read("default", "blob.bin")
    assert read_data == non_utf8_data


def test_namespace_isolation_no_collision(tmp_path: Path) -> None:
    """The same artifact name in two different namespaces does not collide."""
    manager = StorageManager(tmp_path)
    content_a = b"tenant A artifact data"
    content_b = b"tenant B artifact data"

    manager.create("tenant-a", "report.txt", content_a)
    manager.create("tenant-b", "report.txt", content_b)

    assert manager.read("tenant-a", "report.txt") == content_a
    assert manager.read("tenant-b", "report.txt") == content_b


@pytest.mark.parametrize(
    "traversal_name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "a/../../b",
        "./../x",
    ],
)
def test_traversal_attempts_rejected_in_manager(
    tmp_path: Path, traversal_name: str
) -> None:
    """Ensure traversal attempts in manager methods raise PathTraversalError."""
    manager = StorageManager(tmp_path)

    with pytest.raises(PathTraversalError):
        manager.create("ns", traversal_name, b"data")

    with pytest.raises(PathTraversalError):
        manager.read("ns", traversal_name)

    with pytest.raises(PathTraversalError):
        manager.exists("ns", traversal_name)

    with pytest.raises(PathTraversalError):
        manager.delete("ns", traversal_name)


def test_symlink_escape_rejected_in_manager(tmp_path: Path) -> None:
    """Symlink escape is rejected by manager."""
    if sys.platform.startswith("win"):
        pytest.skip("Symlink escape test skipped on Windows")

    manager = StorageManager(tmp_path)
    ns_root = tmp_path / "ns"
    ns_root.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_bytes(b"forbidden")

    symlink_file = ns_root / "leak_link"
    try:
        symlink_file.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment")

    with pytest.raises(PathTraversalError):
        manager.read("ns", "leak_link")


def test_reading_missing_artifact_raises_artifact_not_found(tmp_path: Path) -> None:
    """Reading a missing artifact raises ArtifactNotFound, never returns None."""
    manager = StorageManager(tmp_path)

    with pytest.raises(ArtifactNotFound):
        manager.read("default", "missing.txt")


def test_metadata_survives_cycle(tmp_path: Path) -> None:
    """Metadata (including nested dicts, unicode, numbers) survives write/read cycle."""
    manager = StorageManager(tmp_path)
    metadata = {
        "title": "Document \u2728 \u4e16\u754c",
        "nested": {
            "tags": ["ai", "storage", "phase-1"],
            "score": 99.5,
            "is_active": True,
        },
        "author": "Antigravity \U0001F680",
    }

    created_art = manager.create(
        namespace="metadata-ns",
        name="doc.json",
        content=b'{"hello": "world"}',
        metadata=metadata,
        content_type="application/json",
    )

    assert created_art.metadata == metadata

    fetched_art = manager.get_artifact("metadata-ns", "doc.json")
    assert fetched_art.metadata == metadata
    assert fetched_art.content_type == "application/json"
    assert fetched_art.size == len(b'{"hello": "world"}')


def test_list_scoped_to_namespace_and_skips_meta(tmp_path: Path) -> None:
    """list() is scoped to one namespace and does not leak entries from other namespaces or .meta/."""
    manager = StorageManager(tmp_path)

    manager.create("ns1", "file1.txt", b"1", metadata={"key": "val1"})
    manager.create("ns1", "nested/file2.txt", b"2", metadata={"key": "val2"})
    manager.create("ns2", "other.txt", b"other")

    ns1_list = manager.list("ns1")
    assert len(ns1_list) == 2
    names = [a.name for a in ns1_list]
    assert "file1.txt" in names
    assert "nested/file2.txt" in names
    assert "other.txt" not in names

    # Verify no .meta entries in names
    for item in ns1_list:
        assert not item.name.startswith(".meta")
        assert ".meta" not in item.name


def test_invalid_namespace_names_raise_error(tmp_path: Path) -> None:
    """Invalid namespace names raise InvalidNamespaceError in manager."""
    manager = StorageManager(tmp_path)

    with pytest.raises(InvalidNamespaceError):
        manager.create("", "file.txt", b"content")

    with pytest.raises(InvalidNamespaceError):
        manager.read("invalid/name", "file.txt")

    with pytest.raises(InvalidNamespaceError):
        manager.exists("..", "file.txt")

    with pytest.raises(InvalidNamespaceError):
        manager.list("-bad-lead")

    with pytest.raises(InvalidNamespaceError):
        manager.delete(".hidden", "file.txt")


def test_delete_missing_artifact_raises_not_found(tmp_path: Path) -> None:
    """delete() on a missing artifact raises ArtifactNotFound."""
    manager = StorageManager(tmp_path)

    with pytest.raises(ArtifactNotFound):
        manager.delete("default", "non_existent.txt")


def test_create_existing_artifact_overwrite_flag(tmp_path: Path) -> None:
    """create() raises ArtifactExistsError on existing artifact unless overwrite=True."""
    manager = StorageManager(tmp_path)
    manager.create("ns", "file.txt", b"v1")

    with pytest.raises(ArtifactExistsError):
        manager.create("ns", "file.txt", b"v2", overwrite=False)

    assert manager.read("ns", "file.txt") == b"v1"

    manager.create("ns", "file.txt", b"v2", overwrite=True)
    assert manager.read("ns", "file.txt") == b"v2"


def test_exists_and_delete(tmp_path: Path) -> None:
    """Test exists check and delete removes file and metadata."""
    manager = StorageManager(tmp_path)
    assert not manager.exists("ns", "file.txt")

    manager.create("ns", "file.txt", b"content", metadata={"v": 1})
    assert manager.exists("ns", "file.txt")

    manager.delete("ns", "file.txt")
    assert not manager.exists("ns", "file.txt")
    with pytest.raises(ArtifactNotFound):
        manager.read("ns", "file.txt")


def test_static_analysis_package_independence() -> None:
    """Static analysis: ensure aios/storage never imports aios.agent, aios.tools, aios.kernel, etc."""
    pkg_dir = Path(aios.storage.__file__).parent
    forbidden_prefixes = (
        "aios.agent",
        "aios.agents",
        "aios.tools",
        "aios.kernel",
        "aios.core.models",
        "aios.memory",
        "aios.llm",
        "aios.scheduler",
        "aios.security",
        "aios.communication",
        "aios.context",
        "aios.monitoring",
        "aios.sdk",
    )

    py_files = list(pkg_dir.glob("*.py"))
    assert len(py_files) >= 4, f"Expected storage source files, found: {py_files}"

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_prefixes:
                        assert not alias.name.startswith(forbidden), (
                            f"Forbidden import '{alias.name}' found in {py_file.name}:{node.lineno}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in forbidden_prefixes:
                        assert not node.module.startswith(forbidden), (
                            f"Forbidden from-import '{node.module}' found in {py_file.name}:{node.lineno}"
                        )
