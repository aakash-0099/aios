"""Deterministic size estimation and compression for :class:`Context`.

This module deliberately has no tokenizer, LLM, or memory-store dependency.
Its estimate is intended for admission/control decisions, not provider billing.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from aios.core import Context, ValidationError

CompressionStrategy = Literal["summarize", "remove"]


class ContextCompressor:
    """Estimate context size and deterministically compact old messages."""

    def __init__(self, chars_per_token: int = 4) -> None:
        if type(chars_per_token) is not int or chars_per_token <= 0:
            raise ValidationError("chars_per_token must be a positive integer.")
        self.chars_per_token = chars_per_token

    def estimate_size(self, context: Context) -> int:
        """Return a stable, tokenizer-free token estimate for ``context``."""
        _require_context(context)
        serialized = _serialize(context)
        return math.ceil(len(serialized) / self.chars_per_token)

    # The longer name reads well at call sites and keeps the API discoverable.
    estimate_context_size = estimate_size

    def compress(
        self,
        context: Context,
        max_size: int,
        strategy: CompressionStrategy = "summarize",
    ) -> Context:
        """Return a compacted snapshot whose estimated size is at most the limit.

        Oldest conversation messages are discarded first. With ``summarize``,
        they are replaced by one deterministic summary message when it fits.
        Newer messages are never removed before older messages.

        A context whose non-conversation fields alone exceed ``max_size`` cannot
        be compressed safely and raises ``ValidationError``.
        """
        _require_context(context)
        if type(max_size) is not int or max_size <= 0:
            raise ValidationError("max_size must be a positive integer.")
        if strategy not in ("summarize", "remove"):
            raise ValidationError("strategy must be 'summarize' or 'remove'.")

        snapshot = _copy_context(context)
        if self.estimate_size(snapshot) <= max_size:
            return snapshot

        messages = copy.deepcopy(snapshot.conversation)
        removed: list[dict[str, Any]] = []

        while messages:
            removed.append(messages.pop(0))
            candidate_messages = list(messages)
            if strategy == "summarize":
                candidate_messages.insert(0, _summary_message(removed))
            candidate = _replace_conversation(snapshot, candidate_messages)
            if self.estimate_size(candidate) <= max_size:
                return candidate

        empty = _replace_conversation(snapshot, [])
        if self.estimate_size(empty) > max_size:
            raise ValidationError(
                "max_size is smaller than the context's non-conversation data."
            )

        if strategy == "summarize" and removed:
            summarized = _replace_conversation(snapshot, [_summary_message(removed)])
            if self.estimate_size(summarized) <= max_size:
                return summarized
        return empty

    compress_context = compress


def _summary_message(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Build a reproducible, bounded summary without semantic inference.

    This records structure rather than pretending to understand message
    meaning. The role sequence is capped so the summary itself cannot grow
    without bound as more history is removed.
    """
    roles = [str(message.get("role", "unknown")) for message in messages]
    shown_roles = roles[:8]
    role_sequence = ", ".join(shown_roles)
    if len(roles) > len(shown_roles):
        role_sequence += f", +{len(roles) - len(shown_roles)} more"
    return {
        "role": "system",
        "content": (
            f"Summary of earlier messages: {len(messages)} message(s) removed; "
            f"roles: {role_sequence}."
        ),
    }


def _replace_conversation(
    context: Context,
    conversation: list[dict[str, Any]],
) -> Context:
    return Context(
        system=context.system,
        conversation=copy.deepcopy(conversation),
        memory=copy.deepcopy(context.memory),
        working_state=copy.deepcopy(context.working_state),
        metadata=copy.deepcopy(context.metadata),
    )


def _copy_context(context: Context) -> Context:
    return _replace_conversation(context, context.conversation)


def _serialize(context: Context) -> str:
    def default(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return str(value)

    payload = {
        "system": context.system,
        "conversation": context.conversation,
        "memory": context.memory,
        "working_state": context.working_state,
        "metadata": context.metadata,
    }
    return json.dumps(
        payload,
        default=default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_context(context: Context) -> None:
    if not isinstance(context, Context):
        raise ValidationError("context must be a Context.")
