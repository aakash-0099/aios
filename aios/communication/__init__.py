"""
AIOS inter-component communication.

`ipc.py` provides an isolated, asynchronous, in-memory transport
(`IPCChannel`) with no dependency on the kernel or scheduler. It is
a reusable building block those components -- or agents talking to
each other -- can adopt later; nothing here assumes who uses it.
"""

from .ipc import (
    IPCChannel,
    IPCChannelClosedError,
    IPCError,
    IPCMessage,
    IPCTimeoutError,
)

__all__ = [
    "IPCChannel",
    "IPCChannelClosedError",
    "IPCError",
    "IPCMessage",
    "IPCTimeoutError",
]
