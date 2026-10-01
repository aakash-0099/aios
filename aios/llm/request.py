from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


PAYLOAD_KEYS = ("model", "messages", "parameters")


@dataclass
class LLMPayload:
	"""Validated request data passed to an LLM provider."""

	model: str
	messages: list[dict[str, str]]
	parameters: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Validate the model, conversation messages, and generation parameters."""
		if not isinstance(self.model, str):
			raise TypeError("model must be a str")
		if not self.model.strip():
			raise ValueError("model must not be empty")

		if not isinstance(self.messages, list):
			raise TypeError("messages must be a list")
		if not self.messages:
			raise ValueError("messages must not be empty")
		for index, message in enumerate(self.messages):
			if not isinstance(message, dict):
				raise TypeError(f"messages[{index}] must be a dict")
			if any(not isinstance(key, str) for key in message):
				raise TypeError(f"messages[{index}] keys must be str")
			if any(not isinstance(value, str) for value in message.values()):
				raise TypeError(f"messages[{index}] values must be str")
			if "role" not in message or "content" not in message:
				raise ValueError(
					f"messages[{index}] must contain role and content keys"
				)
			if message["role"] not in {"system", "user", "assistant"}:
				raise ValueError(f"messages[{index}] has an invalid role")

		if not isinstance(self.parameters, dict):
			raise TypeError("parameters must be a dict")
		if any(not isinstance(key, str) for key in self.parameters):
			raise TypeError("parameter keys must be str")

		if "temperature" in self.parameters:
			temperature = self.parameters["temperature"]
			if isinstance(temperature, bool) or not isinstance(
				temperature, (int, float)
			):
				raise TypeError("temperature must be an int or float")
			if not 0.0 <= temperature <= 2.0:
				raise ValueError("temperature must be between 0.0 and 2.0")

		if "max_tokens" in self.parameters:
			max_tokens = self.parameters["max_tokens"]
			if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
				raise TypeError("max_tokens must be an int")
			if max_tokens <= 0:
				raise ValueError("max_tokens must be greater than zero")

		if "top_p" in self.parameters:
			top_p = self.parameters["top_p"]
			if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
				raise TypeError("top_p must be an int or float")
			if not 0.0 <= top_p <= 1.0:
				raise ValueError("top_p must be between 0.0 and 1.0")

	def to_dict(self) -> dict[str, Any]:
		"""Return a deep-copied, serializable representation of this payload."""
		return deepcopy(
			{
				"model": self.model,
				"messages": self.messages,
				"parameters": self.parameters,
			}
		)

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> LLMPayload:
		"""Build a validated payload from a dictionary."""
		if not isinstance(data, dict):
			raise TypeError("data must be a dict")
		missing_keys = set(PAYLOAD_KEYS) - data.keys()
		if missing_keys:
			raise ValueError(f"missing required keys: {sorted(missing_keys)}")
		return cls(
			model=data["model"],
			messages=data["messages"],
			parameters=data["parameters"],
		)
