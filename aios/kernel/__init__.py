"""
AIOS kernel.

Phase 2 public API: everything an agent (or, later, the SDK) needs
to dispatch a system call and get a response back.

Higher-level components should import from this package rather
than reaching into `aios.kernel.kernel`, `aios.kernel.runtime`,
`aios.kernel.syscall`, `aios.kernel.dispatcher`,
`aios.kernel.queue_backend`, `aios.kernel.serialization`, or
`aios.kernel.lifecycle` directly.
"""

from .dispatcher import Dispatcher, HandlerRegistry, build_default_registry
from .kernel import DEFAULT_SYSCALL_TIMEOUT_SECONDS, Kernel
from .lifecycle import SysCallLifecycle, SysCallStatus
from .queue_backend import (
    InMemoryQueueBackend,
    RedisQueueBackend,
    SysCallEnvelope,
    SysCallQueueBackend,
)
from .runtime import KernelRuntime
from .serialization import request_from_dict, request_to_dict
from .syscall import SysCall, SysCallHandler

__all__ = [
    "DEFAULT_SYSCALL_TIMEOUT_SECONDS",
    "Dispatcher",
    "HandlerRegistry",
    "InMemoryQueueBackend",
    "Kernel",
    "KernelRuntime",
    "RedisQueueBackend",
    "SysCall",
    "SysCallEnvelope",
    "SysCallHandler",
    "SysCallLifecycle",
    "SysCallQueueBackend",
    "SysCallStatus",
    "build_default_registry",
    "request_from_dict",
    "request_to_dict",
]
