"""
AIOS kernel.

Phase 2 public API: everything an agent (or, later, the SDK) needs
to dispatch a system call and get a response back.

Higher-level components should import from this package rather
than reaching into `aios.kernel.kernel`, `aios.kernel.runtime`,
`aios.kernel.syscall`, `aios.kernel.dispatcher`, or
`aios.kernel.lifecycle` directly.
"""

from .dispatcher import HandlerRegistry, build_default_registry
from .kernel import DEFAULT_SYSCALL_TIMEOUT_SECONDS, Kernel
from .lifecycle import SysCallLifecycle, SysCallStatus
from .runtime import KernelRuntime
from .syscall import SysCall, SysCallHandler

__all__ = [
    "DEFAULT_SYSCALL_TIMEOUT_SECONDS",
    "HandlerRegistry",
    "Kernel",
    "KernelRuntime",
    "SysCall",
    "SysCallHandler",
    "SysCallLifecycle",
    "SysCallStatus",
    "build_default_registry",
]
