from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from aios.llm import LLMPayload, LLMResult, Provider
from aios.llm.request import PAYLOAD_KEYS
from aios.llm.response import RESULT_KEYS


class FakeProvider(Provider):
	"""Test provider that echoes the last user message and counts words."""

	def generate(self, payload: LLMPayload) -> LLMResult:
		"""Echo the last user message and return simple word-count usage."""
		last_user_message = next(
			message["content"]
			for message in reversed(payload.messages)
			if message["role"] == "user"
		)
		prompt_tokens = sum(
			len(message["content"].split()) for message in payload.messages
		)
		completion_tokens = len(last_user_message.split())
		return LLMResult(
			text=last_user_message,
			usage={
				"prompt_tokens": prompt_tokens,
				"completion_tokens": completion_tokens,
				"total_tokens": prompt_tokens + completion_tokens,
			},
		)


def test_fake_provider_round_trip() -> None:
	"""Round-trip request and result values through their dictionary forms."""
	payload = LLMPayload(
		model="test-model",
		messages=[{"role": "user", "content": "hello from AIOS"}],
		parameters={"temperature": 0.2},
	)
	payload_dict = payload.to_dict()
	restored_payload = LLMPayload.from_dict(payload_dict)
	assert restored_payload == payload

	result = FakeProvider().generate(restored_payload)
	result_dict = result.to_dict()
	restored_result = LLMResult.from_dict(result_dict)
	assert restored_result == result
	assert restored_result.text == "hello from AIOS"


def test_provider_cannot_be_instantiated_directly() -> None:
	"""Require subclasses to implement the abstract generation method."""
	with pytest.raises(TypeError):
		Provider()  # type: ignore[abstract]  # Verify the ABC rejects runtime construction.


def test_provider_subclass_without_generate_fails() -> None:
	"""Reject concrete subclasses that do not implement generate."""
	class IncompleteProvider(Provider):
		"""Provider intentionally missing the abstract method."""

	with pytest.raises(TypeError):
		IncompleteProvider()  # type: ignore[abstract]  # Verify incomplete subclass rejection.


@pytest.mark.parametrize(
	("model", "exception"),
	[("", ValueError), ("   ", ValueError), (None, TypeError)],
)
def test_payload_rejects_invalid_model(
	model: Any, exception: type[Exception]
) -> None:
	"""Reject empty and non-string model names."""
	with pytest.raises(exception):
		LLMPayload(model=model, messages=[{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
	("messages", "exception"),
	[
		([], ValueError),
		([{"content": "hi"}], ValueError),
		([{"role": "tool", "content": "hi"}], ValueError),
		([{"role": "user", "content": 1}], TypeError),
		(["not a message"], TypeError),
	],
)
def test_payload_rejects_invalid_messages(
	messages: Any, exception: type[Exception]
) -> None:
	"""Reject empty or malformed message collections."""
	with pytest.raises(exception):
		LLMPayload(model="test-model", messages=messages)


@pytest.mark.parametrize(
	("parameters", "exception"),
	[
		([], TypeError),
		({"temperature": "warm"}, TypeError),
		({"temperature": -0.1}, ValueError),
		({"temperature": 2.1}, ValueError),
		({"temperature": True}, TypeError),
		({"max_tokens": 1.0}, TypeError),
		({"max_tokens": True}, TypeError),
		({"max_tokens": 0}, ValueError),
		({"max_tokens": -1}, ValueError),
		({1: "invalid key"}, TypeError),
		({"top_p": "high"}, TypeError),
		({"top_p": 1.1}, ValueError),
		({"top_p": False}, TypeError),
	],
)
def test_payload_rejects_malformed_parameters(
	parameters: Any, exception: type[Exception]
) -> None:
	"""Reject invalid types and values for known generation parameters."""
	with pytest.raises(exception):
		LLMPayload(
			model="test-model",
			messages=[{"role": "user", "content": "hi"}],
			parameters=parameters,
		)


def test_payload_allows_unknown_parameter_keys() -> None:
	"""Allow provider-specific parameters that are not validated here."""
	payload = LLMPayload(
		model="test-model",
		messages=[{"role": "user", "content": "hi"}],
		parameters={"provider_option": {"enabled": True}},
	)
	assert payload.parameters["provider_option"] == {"enabled": True}


@pytest.mark.parametrize(
	("text", "usage", "error", "exception"),
	[
		(None, {}, None, TypeError),
		("ok", [], None, TypeError),
		("ok", {"other": 1}, None, ValueError),
		("ok", {"prompt_tokens": -1}, None, ValueError),
		("ok", {"total_tokens": True}, None, TypeError),
		(
			"ok",
			{"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4},
			None,
			ValueError,
		),
		("ok", {}, 1, TypeError),
	],
)
def test_result_rejects_invalid_fields(
	text: Any,
	usage: Any,
	error: Any,
	exception: type[Exception],
) -> None:
	"""Reject malformed text, token usage, and error fields."""
	with pytest.raises(exception):
		LLMResult(text=text, usage=usage, error=error)


@pytest.mark.parametrize(
	("error", "expected"), [(None, True), ("provider failed", False)]
)
def test_result_ok_property(error: str | None, expected: bool) -> None:
	"""Report success only when no error is present."""
	assert LLMResult(text="", usage={}, error=error).ok is expected


@pytest.mark.parametrize(
	("result_type", "data"),
	[
		(LLMPayload, {"messages": []}),
		(LLMResult, {"text": "missing usage"}),
	],
)
def test_from_dict_missing_keys_raises(
	result_type: type[LLMPayload] | type[LLMResult], data: dict[str, Any]
) -> None:
	"""Raise ValueError when required serialized fields are absent."""
	with pytest.raises(ValueError):
		result_type.from_dict(data)


def test_provider_does_not_mutate_payload() -> None:
	"""Ensure generation leaves all payload data unchanged."""
	payload = LLMPayload(
		model="test-model",
		messages=[{"role": "user", "content": "keep this"}],
		parameters={"provider_option": [1, 2]},
	)
	original_payload = deepcopy(payload)
	FakeProvider().generate(payload)
	assert payload == original_payload


def test_payload_keys_contract() -> None:
	"""Keep serializer keys aligned with the protocol payload contracts."""
	payload = LLMPayload(
		model="test-model", messages=[{"role": "user", "content": "hi"}]
	)
	result = LLMResult(text="hi", usage={})
	assert PAYLOAD_KEYS == tuple(payload.to_dict())
	assert RESULT_KEYS == tuple(result.to_dict())
