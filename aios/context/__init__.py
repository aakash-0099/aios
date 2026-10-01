"""
AIOS context assembly.

Builds the ``Context`` handed to the LLM layer from instructions,
conversation, memory entries, and working state. ``Context`` itself
remains the shared model in ``aios.core``.
"""

from .builder import ContextBuilder
from .context import ContextView, as_context_memory

__all__ = [
    "ContextBuilder",
    "ContextView",
    "as_context_memory",
]
