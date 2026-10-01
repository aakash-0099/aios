"""
AIOS context assembly.

Builds the ``Context`` handed to the LLM layer from instructions,
conversation, memory entries, and working state. ``Context`` itself
remains the shared model in ``aios.core``.
"""

from .builder import ContextBuilder
from .compressor import CompressionStrategy, ContextCompressor
from .context import ContextView, as_context_memory
from .manager import ContextManager
from .switcher import ContextSwitcher

__all__ = [
    "ContextBuilder",
    "CompressionStrategy",
    "ContextCompressor",
    "ContextManager",
    "ContextSwitcher",
    "ContextView",
    "as_context_memory",
]
