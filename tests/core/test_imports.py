"""
Foundation import test.

This ensures that all shared objects can be imported without
requiring any future AIOS subsystem.
"""

from aios.core import (
    Agent,
    AgentRequest,
    AgentResponse,
    Context,
    Memory,
    Resource,
    SystemCall,
    Task,
    Tool,
)


def test_core_public_api():
    """
    Verify that the public core API is available.
    """

    assert Agent is not None
    assert Task is not None
    assert SystemCall is not None
    assert AgentRequest is not None
    assert AgentResponse is not None
    assert Resource is not None
    assert Context is not None
    assert Memory is not None
    assert Tool is not None