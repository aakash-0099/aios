"""In-memory management of manually supplied named contexts."""

from __future__ import annotations

import copy

from aios.context.compressor import CompressionStrategy, ContextCompressor
from aios.context.switcher import ContextSwitcher
from aios.core import Context, NotFoundError, ValidationError
from aios.core.validation import require_non_empty_string

DEFAULT_MAX_CONTEXT_SIZE = 4096


class ContextManager:
    """Store, retrieve, compress, and switch between named contexts in memory."""

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_CONTEXT_SIZE,
        compressor: ContextCompressor | None = None,
        strategy: CompressionStrategy = "summarize",
    ) -> None:
        if type(max_size) is not int or max_size <= 0:
            raise ValidationError("max_size must be a positive integer.")
        if compressor is not None and not isinstance(compressor, ContextCompressor):
            raise ValidationError("compressor must be a ContextCompressor.")
        if strategy not in ("summarize", "remove"):
            raise ValidationError("strategy must be 'summarize' or 'remove'.")
        self._contexts: dict[str, Context] = {}
        self.max_size = max_size
        self.strategy = strategy
        self.compressor = compressor or ContextCompressor()
        self.switcher = ContextSwitcher(self.get)

    @property
    def active_name(self) -> str | None:
        return self.switcher.active_name

    @property
    def active_context(self) -> Context:
        """Return an isolated snapshot of the currently selected context."""
        return self.current()

    def store(self, name: str, context: Context, *, overwrite: bool = False) -> None:
        """Store an isolated snapshot under ``name``."""
        require_non_empty_string(name, "context name")
        _require_context(context)
        if name in self._contexts and not overwrite:
            raise ValidationError(f"Context '{name}' already exists.")
        self._contexts[name] = copy.deepcopy(context)

    add = store

    def get(self, name: str) -> Context:
        """Return a snapshot, keeping the stored value isolated from mutation."""
        require_non_empty_string(name, "context name")
        try:
            return copy.deepcopy(self._contexts[name])
        except KeyError as exc:
            raise NotFoundError(f"Context '{name}' was not found.") from exc

    def list_names(self) -> list[str]:
        return list(self._contexts)

    def remove(self, name: str) -> Context:
        require_non_empty_string(name, "context name")
        try:
            removed = self._contexts.pop(name)
        except KeyError as exc:
            raise NotFoundError(f"Context '{name}' was not found.") from exc
        if self.active_name == name:
            self.switcher.clear()
        return copy.deepcopy(removed)

    def switch(self, name: str) -> Context:
        return self.switcher.switch(name)

    switch_to = switch

    def current(self) -> Context:
        return self.switcher.current()

    get_current = current

    def estimate_size(self, context_or_name: Context | str) -> int:
        context = (
            self.get(context_or_name)
            if isinstance(context_or_name, str)
            else context_or_name
        )
        _require_context(context)
        return self.compressor.estimate_size(context)

    def compress(
        self,
        name: str,
        max_size: int,
        strategy: CompressionStrategy = "summarize",
    ) -> Context:
        """Compress a stored context, replace it, and return a snapshot."""
        compressed = self.compressor.compress(self.get(name), max_size, strategy)
        self._contexts[name] = copy.deepcopy(compressed)
        return copy.deepcopy(compressed)

    def compress_if_needed(
        self,
        name: str,
        max_size: int | None = None,
        strategy: CompressionStrategy | None = None,
    ) -> Context:
        """Compress a stored context only when it exceeds its size threshold.

        ``max_size`` and ``strategy`` may override the manager defaults for one
        call. Whether compressed or not, the returned value is an isolated
        snapshot and the named context remains stored in the manager.
        """
        threshold = self.max_size if max_size is None else max_size
        selected_strategy = self.strategy if strategy is None else strategy

        if type(threshold) is not int or threshold <= 0:
            raise ValidationError("max_size must be a positive integer.")
        if selected_strategy not in ("summarize", "remove"):
            raise ValidationError("strategy must be 'summarize' or 'remove'.")

        context = self.get(name)
        if self.compressor.estimate_size(context) <= threshold:
            return context
        return self.compress(name, threshold, selected_strategy)

    def clear(self) -> None:
        self._contexts.clear()
        self.switcher.clear()

    def __len__(self) -> int:
        return len(self._contexts)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._contexts


def _require_context(context: Context) -> None:
    if not isinstance(context, Context):
        raise ValidationError("context must be a Context.")
