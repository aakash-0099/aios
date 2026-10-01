"""Unit tests for namespace validation and directory resolution."""

from pathlib import Path
import pytest

from aios.storage.artifact import InvalidNamespaceError
from aios.storage.namespace import resolve_namespace_dir, validate_namespace


@pytest.mark.parametrize(
    "valid_name",
    [
        "default",
        "user123",
        "tenant-a",
        "ns.test",
        "ns_1",
        "A",
        "Z123-abc.def_ghi",
        "a" * 64,
    ],
)
def test_valid_namespaces(valid_name: str) -> None:
    """Test valid namespace strings pass validation."""
    assert validate_namespace(valid_name) == valid_name


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        ".",
        "..",
        "/",
        "\\",
        "../escape",
        "ns/sub",
        "ns\\sub",
        "-starts-with-dash",
        "_starts_with_underscore",
        ".starts.with.dot",
        "has spaces",
        "has$symbol",
        "a" * 65,
        None,
        123,
    ],
)
def test_invalid_namespaces(invalid_name: object) -> None:
    """Test invalid namespace strings raise InvalidNamespaceError."""
    with pytest.raises(InvalidNamespaceError):
        validate_namespace(invalid_name)  # type: ignore[arg-type]


def test_resolve_namespace_dir_create(tmp_path: Path) -> None:
    """Test namespace resolution creates directory when requested."""
    ns_dir = resolve_namespace_dir(tmp_path, "ns-1", create=True)
    assert ns_dir.is_dir()
    assert ns_dir.parent.resolve() == tmp_path.resolve()
    assert ns_dir.name == "ns-1"


def test_resolve_namespace_dir_no_create(tmp_path: Path) -> None:
    """Test namespace resolution does not create directory when create=False."""
    ns_dir = resolve_namespace_dir(tmp_path, "ns-2", create=False)
    assert not ns_dir.exists()
    assert ns_dir.parent.resolve() == tmp_path.resolve()
