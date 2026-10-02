"""
Deterministic Mock LLM Provider.

Provides a fully local, network-free LLM provider for unit testing and
development. Key properties:

- Deterministic: identical payloads always yield identical results.
- Configurable: success responses and failure scenarios are set at
  construction time and never change after that.
- Observable: every received payload is recorded for later inspection.
- Latency simulation: an optional sleep is injected before each response so
  callers can test timeout/latency handling.

This module serves two call styles that were merged from parallel work, and
both are supported by the single :class:`MockLLM` class:

- ``canned_response`` / ``canned_responses`` / ``canned_error`` / ``latency_ms``
- ``responses`` / ``default_response`` / ``scenario`` / ``latency_seconds``

:class:`MockProvider` is retained as an alias of :class:`MockLLM` so existing
call sites continue to work.

No API keys, external services, or network sockets are used.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .provider import Provider
from .request import LLMPayload
from .response import LLMResult


@dataclass(frozen=True)
class MockScenario:
	"""
	A named failure scenario that the mock provider can be configured to use.

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


class MockLLM(Provider):
	"""
	Deterministic mock implementation of Provider.

	This is the default provider the Kernel's LLM_CALL handler uses until real
	providers are wired in, and it is swappable with the real adapters in
	``openai``, ``groq``, and ``ollama`` without any change at the call site.

	Response text for a successful call is resolved in this order:
	1. ``canned_responses`` when it is a sequence: entries are consumed
		in order, then discarded once exhausted.
	2. ``canned_responses`` when it is a mapping: keyed lookup on the
		last user message.
	3. ``responses``: keyed lookup on the last user message.
	4. ``canned_response``.
	5. ``default_response``.
	6. An echo of the last user message, prefixed with 'Mock response to: '.

	Failure scenarios return an LLMResult whose ``error`` field is set rather
	than raising, as required by the Provider contract. Use ``canned_error``
	(a plain string) or ``scenario`` (a :class:`MockScenario`) to configure one.

	Args:
		canned_response: Fixed text returned for every successful call.
		canned_responses: Either a sequence of texts returned in order, or a
			mapping of last-user-message content to reply text.
		canned_error: When set, every call returns a failed result whose error
			field equals this string.
		responses: Mapping of last-user-message content to reply text.
		default_response: Text returned when no mapping matches.
		scenario: A :class:`MockScenario` whose error_message is used as the
			error field. Equivalent to passing ``canned_error``.
		latency_ms: Simulated latency in milliseconds. Defaults to 0 so tests
			stay fast. Does not affect response content.
		latency_seconds: Simulated latency in seconds. Added to latency_ms.
	"""

	def __init__(
		self,
		canned_response: str | None = None,
		canned_responses: Sequence[str] | Mapping[str, str] | None = None,
		canned_error: str | None = None,
		responses: Mapping[str, str] | None = None,
		default_response: str | None = None,
		scenario: MockScenario | None = None,
		latency_ms: int = 0,
		latency_seconds: float = 0.0,
	) -> None:
		if canned_response is not None and not isinstance(canned_response, str):
			raise TypeError("canned_response must be a str or None")
		if canned_error is not None and not isinstance(canned_error, str):
			raise TypeError("canned_error must be a str or None")
		if scenario is not None and not isinstance(scenario, MockScenario):
			raise TypeError("scenario must be a MockScenario or None")

		if responses is not None:
			if not isinstance(responses, Mapping):
				raise TypeError("responses must be a Mapping")
			for key, value in responses.items():
				if not isinstance(key, str):
					raise TypeError("responses keys must be str")
				if not isinstance(value, str):
					raise TypeError("responses values must be str")

		if canned_responses is not None:
			if isinstance(canned_responses, Mapping):
				for key, value in canned_responses.items():
					if not isinstance(key, str):
						raise TypeError("canned_responses keys must be str")
					if not isinstance(value, str):
						raise TypeError("canned_responses values must be str")
			elif isinstance(canned_responses, Sequence) and not isinstance(
				canned_responses, str
			):
				if any(not isinstance(item, str) for item in canned_responses):
					raise TypeError("canned_responses sequence items must be str")
			else:
				raise TypeError(
					"canned_responses must be a Sequence[str] or Mapping[str, str]"
				)

		if default_response is not None and not isinstance(default_response, str):
			raise TypeError("default_response must be a str or None")

		latencies = (("latency_ms", latency_ms), ("latency_seconds", latency_seconds))
		for name, value in latencies:
			if isinstance(value, bool) or not isinstance(value, (int, float)):
				raise TypeError(f"{name} must be a numeric value")
			if value < 0:
				raise ValueError(f"{name} must be non-negative")

		self._canned_response = canned_response
		self._canned_responses = canned_responses
		self._responses = responses
		self._default_response = default_response
		self._scenario = scenario
		self._canned_error = canned_error
		self._latency = latency_ms / 1000.0 + latency_seconds
		self._calls: list[LLMPayload] = []
		# Consume a private copy so the caller's sequence is left untouched.
		self._remaining: list[str] = (
			list(canned_responses)
			if canned_responses is not None
			and not isinstance(canned_responses, (Mapping, str))
			else []
		)

	def reset(self) -> None:
		"""Clear recorded calls and restore any sequence-based canned responses."""
		self._calls.clear()
		self._remaining = (
			list(self._canned_responses)
			if self._canned_responses is not None
			and not isinstance(self._canned_responses, (Mapping, str))
			else []
		)

	@property
	def calls(self) -> list[LLMPayload]:
		"""Ordered deep copies of every payload passed to generate."""
		return list(self._calls)

	@property
	def requests(self) -> list[LLMPayload]:
		"""Alias of :attr:`calls` for test assertions."""
		return self.calls

	@property
	def received_requests(self) -> list[LLMPayload]:
		"""Alias of :attr:`calls` matching the documented interface contract."""
		return self.calls

	@property
	def call_count(self) -> int:
		"""Number of generate calls recorded since creation or last reset."""
		return len(self._calls)

	@property
	def last_payload(self) -> LLMPayload | None:
		"""Most recently received payload, or None when no call was made."""
		return self._calls[-1] if self._calls else None

	def _resolve_text(self, payload: LLMPayload) -> str:
		if self._remaining:
			return self._remaining.pop(0)
		content = _last_user_content(payload)
		for mapping in (self._canned_responses, self._responses):
			if isinstance(mapping, Mapping) and content in mapping:
				return mapping[content]
		if self._canned_response is not None:
			return self._canned_response
		if self._default_response is not None:
			return self._default_response
		return f"Mock response to: {content}"

	def generate(self, payload: LLMPayload) -> LLMResult:
		"""
		Record the call and return a deterministic result without mutating it.

		Steps:
		1. Record a deep copy of payload.
		2. Sleep for the configured latency (if non-zero).
		3. If a failure scenario is configured, return an error result.
		4. Otherwise resolve the response text and compute token counts.
		"""
		self._calls.append(deepcopy(payload))
		if self._latency:
			time.sleep(self._latency)

		error = self._canned_error
		if error is None and self._scenario is not None:
			error = self._scenario.error_message
		if error is not None:
			return LLMResult(text="", usage={}, error=error)

		text = self._resolve_text(payload)
		prompt_tokens = sum(len(msg["content"].split()) for msg in payload.messages)
		completion_tokens = len(text.split())
		usage: dict[str, Any] = {
			"prompt_tokens": prompt_tokens,
			"completion_tokens": completion_tokens,
			"total_tokens": prompt_tokens + completion_tokens,
		}
		return LLMResult(text=text, usage=usage)


#: Backwards-compatible alias for the merged mock provider.
MockProvider = MockLLM


def _last_user_content(payload: LLMPayload) -> str:
	"""Return the content of the last user-role message in payload."""
	for message in reversed(payload.messages):
		if message["role"] == "user":
			return message["content"]
	return ""  # pragma: no cover