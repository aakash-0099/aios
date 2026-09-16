"""
AIOS shared exception hierarchy.

All AIOS components should use this hierarchy instead of creating
independent exception trees.

This allows callers and the kernel to handle errors consistently.
"""


class AIOSException(Exception):
    """
    Base exception for all AIOS-specific errors.
    """


class ValidationError(AIOSException):
    """
    Raised when an AIOS object or operation is invalid.
    """


class ConfigurationError(AIOSException):
    """
    Raised when AIOS configuration is invalid or incomplete.
    """


class NotFoundError(AIOSException):
    """
    Raised when a requested AIOS entity cannot be found.
    """


class PermissionError(AIOSException):
    """
    Raised when an operation is not permitted.
    """


class ResourceError(AIOSException):
    """
    Raised when an AIOS-managed resource encounters an error.
    """


class SystemCallError(AIOSException):
    """
    Raised when a system call cannot be processed.
    """


class AgentError(AIOSException):
    """
    Raised for agent-related errors.
    """


class TaskError(AIOSException):
    """
    Raised for task-related errors.
    """


class ExecutionError(AIOSException):
    """
    Raised when an AIOS operation fails during execution.
    """