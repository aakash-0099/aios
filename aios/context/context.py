"""
Read-only helpers for an assembled AIOS Context.

``Context`` itself lives in ``aios.core.models`` and is not redefined
here. This module only adds the assembly-facing pieces that sit on
top of that model:

* how a ``Memory`` entry is placed into ``Context.memory``
* a read-only view that does not expose ``Context``'s mutable containers
"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping

from aios.core import Context, Memory, ValidationError


def as_context_memory(entry: Memory) -> Memory:
    """
    Normalize one memory entry for ``Context.memory``.

    ``Context`` accepts a ``Memory`` instance directly, so this step
    does not serialize the entry to a dict. The same instance is
    returned, which makes the builder's round-trip predictable: the
    object passed to ``add_memory_entries`` is the object stored on
    the built context.
    """

    if not isinstance(entry, Memory):
        raise ValidationError(
            "Every memory entry must be a Memory instance."
        )

    return entry


class ContextView:
    """
    Read-only accessors for a built ``Context``.

    ``Context`` is frozen, but its lists and dicts are still mutable.
    These accessors return snapshots — a tuple of mappings, or a
    mapping proxy — so a caller cannot append to or rebind the
    containers the context owns.

    Memory entries are copied. Conversation, working state, and
    metadata mappings are proxies over copies, so item assignment
    fails instead of changing the context.
    """

    def __init__(self, context: Context) -> None:
        if not isinstance(context, Context):
            raise ValidationError(
                "context must be a Context."
            )

        self._context = context

    @property
    def context(self) -> Context:
        """The underlying context. Treat its containers as private."""

        return self._context

    @property
    def system(self) -> str | None:
        """System instructions assembled for the model, if any."""

        return self._context.system

    @property
    def conversation(self) -> tuple[Mapping[str, Any], ...]:
        """Conversation messages as a snapshot of read-only mappings."""

        return tuple(
            MappingProxyType(copy.deepcopy(message))
            for message in self._context.conversation
        )

    @property
    def memory(self) -> tuple[Memory | Mapping[str, Any], ...]:
        """
        Memory entries as a snapshot.

        ``Memory`` items are copied so mutating the snapshot does not
        mutate the entry stored on the context. Dict items are
        returned as read-only mappings.
        """

        snapshots: list[Memory | Mapping[str, Any]] = []

        for item in self._context.memory:
            if isinstance(item, Memory):
                snapshots.append(
                    replace(
                        item,
                        metadata=copy.deepcopy(item.metadata),
                    )
                )
                continue

            if isinstance(item, dict):
                snapshots.append(
                    MappingProxyType(copy.deepcopy(item))
                )
                continue

            raise ValidationError(
                "Every memory item must be a dictionary or Memory instance."
            )

        return tuple(snapshots)

    @property
    def working_state(self) -> Mapping[str, Any]:
        """Working state as a read-only mapping."""

        return MappingProxyType(
            copy.deepcopy(self._context.working_state)
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Context metadata as a read-only mapping."""

        return MappingProxyType(
            copy.deepcopy(self._context.metadata)
        )
