"""
Shared validation helpers for AIOS core models.

The purpose of this module is to keep common validation rules in
one place instead of duplicating the same checks throughout the
AIOS data models.

These helpers raise AIOS ValidationError so every part of the
system gets a consistent exception type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aios.core.exceptions import ValidationError


def require_non_empty_string(
    value: str,
    field_name: str,
) -> None:
    """
    Require a string field to contain meaningful text.

    This is used for fields such as:
        - Agent.name
        - Task.description
        - Tool.name
        - Tool.description
        - Memory.content
    """

    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string."
        )

    if not value.strip():
        raise ValidationError(
            f"{field_name} cannot be empty."
        )


def require_mapping(
    value: Any,
    field_name: str,
) -> None:
    """
    Require a value to behave like a mapping.

    This protects dictionary-like metadata and payload fields.
    """

    if not isinstance(value, Mapping):
        raise ValidationError(
            f"{field_name} must be a mapping."
        )


def require_not_none(
    value: Any,
    field_name: str,
) -> None:
    """
    Require a field to be explicitly provided.
    """

    if value is None:
        raise ValidationError(
            f"{field_name} cannot be None."
        )