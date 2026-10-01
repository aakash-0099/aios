"""Unit tests for low-level filesystem safety and atomic writes."""

import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

from aios.storage.artifact import InvalidArtifactName, PathTraversalError
from aios.storage.filesystem import atomic_write, internal_resolve, safe_resolve


@pytest.mark.parametrize(
    "traversal_name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "a/../../b",
        "./../x",
        "/absolute/path",
        "\\windows\\system32",
        "nested/../../outside",
        "..",
        ".",
    ],
)
def test_safe_resolve_rejects_traversal(tmp_path: Path, traversal_name: str) -> None:
    """Parametrized test rejecting directory traversal and absolute paths."""
    root = tmp_path / "ns_root"
    root.mkdir()

    with pytest.raises(PathTraversalError):
        safe_resolve(root, traversal_name)


def test_safe_resolve_symlink_escape(tmp_path: Path) -> None:
    """Test symlink pointing outside namespace root is rejected."""
    if sys.platform.startswith("win"):
        pytest.skip("Symlink escape test skipped on Windows")

    root = tmp_path / "ns_root"
    root.mkdir()

    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("secret")

    symlink_file = root / "symlink_escape"
    try:
        symlink_file.symlink_to(outside_target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment")

    with pytest.raises(PathTraversalError):
        safe_resolve(root, "symlink_escape")


@pytest.mark.parametrize(
    "reserved_name",
    [".meta", ".meta/a.txt", "a/.meta/b.txt", ".meta/../.meta/x"],
)
def test_safe_resolve_rejects_reserved_components(
    tmp_path: Path, reserved_name: str
) -> None:
    """Reserved internal path components are rejected for user-supplied names."""
    root = tmp_path / "ns_root"
    root.mkdir()

    with pytest.raises(InvalidArtifactName):
        safe_resolve(root, reserved_name)


def test_internal_resolve_allows_reserved_components(tmp_path: Path) -> None:
    """internal_resolve permits .meta but still blocks traversal."""
    root = tmp_path / "ns_root"
    root.mkdir()

    resolved = internal_resolve(root, ".meta/a.txt.json")
    assert resolved == (root / ".meta" / "a.txt.json").resolve()

    with pytest.raises(PathTraversalError):
        internal_resolve(root, ".meta/../../escape")


def test_safe_resolve_valid_paths(tmp_path: Path) -> None:
    """Test resolving valid relative paths inside root."""
    root = tmp_path / "ns_root"
    root.mkdir()

    resolved = safe_resolve(root, "artifact.bin")
    assert resolved == (root / "artifact.bin").resolve()

    nested = safe_resolve(root, "sub/dir/artifact.bin")
    assert nested == (root / "sub/dir/artifact.bin").resolve()


def test_atomic_write_creates_file_and_parents(tmp_path: Path) -> None:
    """Test atomic write creates file and missing parent directories."""
    target = tmp_path / "nested" / "dir" / "file.bin"
    raw_data = bytes([0x00, 0xFF, 0xFE, 0x80, 0x7F, 0x01, 0xAA])

    atomic_write(target, raw_data)
    assert target.is_file()
    assert target.read_bytes() == raw_data


def test_atomic_write_failure_cleans_up_temp_file(tmp_path: Path) -> None:
    """Verify simulated mid-write failure leaves no stray temp files."""
    target_dir = tmp_path / "target_dir"
    target_dir.mkdir()
    target_file = target_dir / "target.bin"

    # Simulate failure during os.replace
    with patch("os.replace", side_effect=OSError("Disk write simulated failure")):
        with pytest.raises(OSError, match="Disk write simulated failure"):
            atomic_write(target_file, b"sample content")

    # Assert no stray temp files remain in target_dir
    files_in_dir = list(target_dir.iterdir())
    assert len(files_in_dir) == 0, f"Found stray files: {files_in_dir}"
