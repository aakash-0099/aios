"""
Assemble an AIOS Context for the LLM layer.

``ContextBuilder`` is the step between "what do we know" and "what
do we send the model." The caller supplies system instructions,
conversation messages, memory entries, and working state. This
module does not fetch memory itself — the kernel (or an
orchestration layer) reads memory from the memory manager and
passes the entries in.

Built contexts are snapshots. Changing the builder after ``build()``
does not change a ``Context`` that was already produced.
"""

from __future__ import annotations

import copy
from typing import Any

from aios.context.context import as_context_memory
from aios.core import Context, Memory, ValidationError
from aios.core.validation import require_non_empty_string


class ContextBuilder:
    """
    Accumulate the pieces of an execution context, then build one.

    ``add_*`` methods append. ``set_*`` methods replace. ``build``
    copies that state into a validated ``Context`` and leaves the
    builder free to keep accumulating.
    """

    def __init__(self) -> None:
        self._instructions: list[str] = []
        self._conversation: list[dict[str, Any]] = []
        self._memory: list[Memory] = []
        self._working_state: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def add_system_instruction(self, text: str) -> None:
        """
        Append one system instruction.

        Instructions are joined, in order, into ``Context.system``
        when the context is built. With none added, ``system`` is
        ``None``.
        """

        require_non_empty_string(text, "system instruction")
        self._instructions.append(text)

    def add_message(self, role: str, content: str) -> None:
        """
        Append one conversation message.

        The message is stored as ``{"role": role, "content": content}``,
        which is the dictionary shape ``Context.conversation`` requires.
        """

        require_non_empty_string(role, "role")
        require_non_empty_string(content, "content")
        self._conversation.append(
            {"role": role, "content": content}
        )

    def add_memory_entries(self, entries: list[Memory]) -> None:
        """
        Append memory entries supplied by the caller.

        Entries must already be ``Memory`` instances. They are stored
        as those instances. ``Context`` accepts ``Memory`` directly,
        so nothing here turns them into dicts, and nothing here asks
        the memory manager for them. The built context therefore
        round-trips the same objects the caller passed in.
        """

        if not isinstance(entries, list):
            raise ValidationError(
                "memory entries must be a list."
            )

        normalized = [as_context_memory(entry) for entry in entries]
        self._memory.extend(normalized)

    def set_working_state(self, state: dict[str, Any]) -> None:
        """
        Replace the working state that will be placed on the context.

        The dict is deep-copied. Later mutation of the caller's dict,
        including nested values, does not change the builder, and
        later mutation of the builder does not change a context
        already built from the previous state.
        """

        self._working_state = _copy_dict(state, "working_state")

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Replace the metadata that will be placed on the context.

        Copied the same way as working state.
        """

        self._metadata = _copy_dict(metadata, "metadata")

    def build(self) -> Context:
        """
        Produce a validated ``Context`` snapshot of the current state.

        The returned object does not share its lists or dicts with
        the builder. Further ``add_*`` or ``set_*`` calls leave it
        unchanged.
        """

        system = (
            "\n".join(self._instructions)
            if self._instructions
            else None
        )

        return Context(
            system=system,
            conversation=copy.deepcopy(self._conversation),
            memory=list(self._memory),
            working_state=copy.deepcopy(self._working_state),
            metadata=copy.deepcopy(self._metadata),
        )


def _copy_dict(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Copy a caller-supplied dict after checking its type."""

    if not isinstance(value, dict):
        raise ValidationError(
            f"{field_name} must be a dictionary."
        )

    return copy.deepcopy(value)
