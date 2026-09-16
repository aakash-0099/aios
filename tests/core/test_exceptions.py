"""
Tests for the shared AIOS exception hierarchy.
"""

from aios.core import (
    AgentError,
    AIOSException,
    ConfigurationError,
    ExecutionError,
    NotFoundError,
    PermissionError,
    ResourceError,
    SystemCallError,
    TaskError,
    ValidationError,
)


def test_all_core_exceptions_inherit_from_aios_exception():
    """Every AIOS exception must share the same root."""

    exception_types = [
        ValidationError,
        ConfigurationError,
        NotFoundError,
        PermissionError,
        ResourceError,
        SystemCallError,
        AgentError,
        TaskError,
        ExecutionError,
    ]

    for exception_type in exception_types:
        assert issubclass(exception_type, AIOSException)