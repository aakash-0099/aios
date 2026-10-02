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
from .messages import (
    MessageType,
    RequestMessage,
    ResponseMessage,
    new_correlation_id,
)
from .protocol import (
    request_from_dict,
    request_to_dict,
    response_from_dict,
    response_to_dict,
    validate_request,
    validate_response,
)

__all__ = [
    "IPCChannel",
    "IPCChannelClosedError",
    "IPCError",
    "IPCMessage",
    "IPCTimeoutError",
    "MessageType",
    "RequestMessage",
    "ResponseMessage",
    "new_correlation_id",
    "request_from_dict",
    "request_to_dict",
    "response_from_dict",
    "response_to_dict",
    "validate_request",
    "validate_response",
]
