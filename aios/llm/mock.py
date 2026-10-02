"""
Deterministic Mock LLM Provider.

Provides a fully local, network-free LLM provider for unit testing and
development. Key properties:

- Deterministic: identical payloads always yield identical results.
- Configurable: success responses and failure scenarios are set at
  construction time and never change after that.
- Observable: every received payload is recorded for later inspection.
- Latency simulation: an optional fixed sleep is injected before each
  response so callers can test timeout/latency handling.

No API keys, external services, or network sockets are used.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .provider import Provider
from .request import LLMPayload
from .response import LLMResult


@dataclass(frozen=True)
class MockScenario:
    """
    A named failure scenario that MockProvider can be configured to use.

    Attributes:
        error_message: Human-readable description of the simulated failure.
            Placed verbatim into LLMResult.error; result text is always
            the empty string for error scenarios.
    """

    error_message: str

    def __post_init__(self) -> None:
        if not isinstance(self.error_message, str):
            raise TypeError("error_message must be a str")
        if not self.error_message.strip():
            raise ValueError("error_message must not be empty")


@dataclass
class MockProvider(Provider):
    """
    Deterministic mock implementation of Provider.

    Args:
        responses: Optional mapping from a response key to the text returned
            when that key exactly matches the last user message in the
            payload.  Falls back to default_response when no key matches.
        default_response: Text returned when no responses entry matches.
            Defaults to 'mock response'.
        scenario: When set, every generate call returns an error result
            whose error field equals scenario.error_message.
        latency_seconds: Non-negative seconds to sleep before returning
            each result.  Defaults to 0.0 (no delay).
    """

    responses: Mapping[str, str] = field(default_factory=dict)
    default_response: str = "mock response"
    scenario: MockScenario | None = None
    latency_seconds: float = 0.0

    _requests: list[LLMPayload] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.responses, Mapping):
            raise TypeError("responses must be a Mapping")
        for key, value in self.responses.items():
            if not isinstance(key, str):
                raise TypeError("responses keys must be str")
            if not isinstance(value, str):
                raise TypeError("responses values must be str")
        if not isinstance(self.default_response, str):
            raise TypeError("default_response must be a str")
        if self.scenario is not None and not isinstance(self.scenario, MockScenario):
            raise TypeError("scenario must be a MockScenario or None")
        if isinstance(self.latency_seconds, bool) or not isinstance(
            self.latency_seconds, (int, float)
        ):
            raise TypeError("latency_seconds must be a numeric value")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")

    @property
    def requests(self) -> list[LLMPayload]:
        """
        Ordered list of every LLMPayload passed to generate since creation
        (or since reset was last called).  Each entry is a deep copy so
        that later mutations by callers do not affect the recorded history.
        """
        return list(self._requests)

    def reset(self) -> None:
        """Clear the recorded request history."""
        self._requests.clear()

    def generate(self, payload: LLMPayload) -> LLMResult:
        """
        Return a deterministic result for payload without mutating it.

        Steps:
        1. Record a deep copy of payload.
        2. Sleep for latency_seconds (if non-zero).
        3. If a scenario is configured, return an error result.
        4. Otherwise look up the last user message in responses;
           fall back to default_response if no key matches.
        5. Compute token counts deterministically from word counts.
        """
        self._requests.append(deepcopy(payload))
        if self.latency_seconds:
            time.sleep(self.latency_seconds)
        if self.scenario is not None:
            return LLMResult(text="", usage={}, error=self.scenario.error_message)
        last_user_content = _last_user_content(payload)
        text = self.responses.get(last_user_content, self.default_response)
        prompt_tokens = sum(len(msg["content"].split()) for msg in payload.messages)
        completion_tokens = len(text.split())
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return LLMResult(text=text, usage=usage)


def _last_user_content(payload: LLMPayload) -> str:
    """Return the content of the last user-role message in payload."""
    for message in reversed(payload.messages):
        if message["role"] == "user":
            return message["content"]
    return ""  # pragma: no cover
