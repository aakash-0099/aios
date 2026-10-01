from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


RESULT_KEYS = ("text", "usage", "error")
_USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}


@dataclass
class LLMResult:
	"""Validated text, token usage, and optional provider error."""

	text: str
	usage: dict[str, int]
	error: str | None = None

	def __post_init__(self) -> None:
		"""Validate result text, token usage, and error details."""
		if not isinstance(self.text, str):
			raise TypeError("text must be a str")
		if not isinstance(self.usage, dict):
			raise TypeError("usage must be a dict")
		if any(not isinstance(key, str) for key in self.usage):
			raise TypeError("usage keys must be str")
		if any(key not in _USAGE_KEYS for key in self.usage):
			raise ValueError("usage contains an unsupported key")
		for key, value in self.usage.items():
			if isinstance(value, bool) or not isinstance(value, int):
				raise TypeError(f"usage[{key!r}] must be an int")
			if value < 0:
				raise ValueError(f"usage[{key!r}] must be non-negative")
		if _USAGE_KEYS <= self.usage.keys() and (
			self.usage["total_tokens"]
			!= self.usage["prompt_tokens"] + self.usage["completion_tokens"]
		):
			raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
		if self.error is not None and not isinstance(self.error, str):
			raise TypeError("error must be None or a str")

	@property
	def ok(self) -> bool:
		"""Return whether the result represents a successful provider call."""
		return self.error is None

	def to_dict(self) -> dict[str, Any]:
		"""Return a deep-copied, serializable representation of this result."""
		return deepcopy({"text": self.text, "usage": self.usage, "error": self.error})

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> LLMResult:
		"""Build a validated result from a dictionary."""
		if not isinstance(data, dict):
			raise TypeError("data must be a dict")
		required_keys = {"text", "usage"}
		missing_keys = required_keys - data.keys()
		if missing_keys:
			raise ValueError(f"missing required keys: {sorted(missing_keys)}")
		return cls(text=data["text"], usage=data["usage"], error=data.get("error"))
