"""Active-context switching for named, caller-supplied contexts."""

from __future__ import annotations

from collections.abc import Callable

from aios.core import Context, NotFoundError, ValidationError
from aios.core.validation import require_non_empty_string

ContextLookup = Callable[[str], Context]


class ContextSwitcher:
    """Track which named context is active, resolving it through a lookup."""

    def __init__(self, lookup: ContextLookup) -> None:
        if not callable(lookup):
            raise ValidationError("lookup must be callable.")
        self._lookup = lookup
        self._active_name: str | None = None

    @property
    def active_name(self) -> str | None:
        return self._active_name

    @property
    def current_name(self) -> str | None:
        return self._active_name

    def switch(self, name: str) -> Context:
        """Make an existing named context active and return its snapshot."""
        require_non_empty_string(name, "context name")
        try:
            context = self._lookup(name)
        except NotFoundError:
            raise
        if not isinstance(context, Context):
            raise ValidationError("context lookup must return a Context.")
        self._active_name = name
        return context

    switch_to = switch

    def current(self) -> Context:
        """Return the active context or fail if none has been selected."""
        if self._active_name is None:
            raise NotFoundError("No context is currently active.")
        return self._lookup(self._active_name)

    get_current = current

    def clear(self) -> None:
        self._active_name = None
