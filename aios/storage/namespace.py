"""Namespace validation and path resolution for AIOS storage."""

from pathlib import Path
import re

from .artifact import InvalidNamespaceError

NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_namespace(namespace: str) -> str:
    """Validate a namespace name against the allowed pattern.

    Allowed: Starts with alphanumeric, followed by up to 63 alphanumeric,
    dot, underscore, or hyphen characters.
    Rejects empty strings, '.', '..', path separators, and special characters.

    Args:
        namespace: The namespace string to validate.

    Returns:
        The validated namespace string.

    Raises:
        InvalidNamespaceError: If the namespace is invalid.
    """
    if not isinstance(namespace, str) or not NAMESPACE_PATTERN.fullmatch(namespace):
        raise InvalidNamespaceError(
            f"Invalid namespace '{namespace}'. Namespace must match pattern ^[A-Za-z0-9][A-Za-z0-9._-]{{0,63}}$"
        )
    return namespace


def resolve_namespace_dir(
    base_dir: Path | str,
    namespace: str,
    create: bool = True,
) -> Path:
    """Resolve and optionally create the directory for a given namespace.

    Args:
        base_dir: The base root directory for all namespaces.
        namespace: The namespace name.
        create: Whether to create the directory if it does not exist.

    Returns:
        Resolved Path to the namespace directory.

    Raises:
        InvalidNamespaceError: If namespace name is invalid.
    """
    validate_namespace(namespace)
    base_path = Path(base_dir).resolve()
    ns_dir = (base_path / namespace).resolve()
    if create:
        ns_dir.mkdir(parents=True, exist_ok=True)
    return ns_dir
