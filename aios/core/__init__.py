"""
AIOS Core.

This package contains the stable shared contracts used throughout
the AIOS implementation.

Higher-level components must depend on core, but core must not
depend on higher-level components.
"""

from .config import AIOSSettings
from .exceptions import (
    AgentError,
    AIOSException,
    ConfigurationError,
    DuplicateTaskError,
    ExecutionError,
    NotFoundError,
    PermissionError,
    QueueEmptyError,
    ResourceError,
    SchedulerError,
    SystemCallError,
    TaskError,
    ValidationError,
)
from .ids import (
    AgentID,
    MemoryID,
    RequestID,
    ResourceID,
    TaskID,
    ToolID,
)
from .models import (
    Agent,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    Context,
    Memory,
    RequestStatus,
    Resource,
    ResourceType,
    SystemCall,
    SystemCallType,
    Task,
    TaskStatus,
    Tool,
)
from .validation import (
    require_mapping,
    require_non_empty_string,
    require_not_none,
)

__all__ = [
    "Agent",
    "AgentError",
    "AgentID",
    "AgentRequest",
    "AgentResponse",
    "AgentStatus",
    "AIOSException",
    "AIOSSettings",
    "ConfigurationError",
    "Context",
    "DuplicateTaskError",
    "ExecutionError",
    "Memory",
    "MemoryID",
    "NotFoundError",
    "PermissionError",
    "QueueEmptyError",
    "RequestID",
    "RequestStatus",
    "Resource",
    "ResourceError",
    "ResourceID",
    "ResourceType",
    "SchedulerError",
    "SystemCall",
    "SystemCallError",
    "SystemCallType",
    "Task",
    "TaskError",
    "TaskID",
    "TaskStatus",
    "Tool",
    "ToolID",
    "ValidationError",
    "require_mapping",
    "require_non_empty_string",
    "require_not_none",
]