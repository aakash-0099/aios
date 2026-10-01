from __future__ import annotations

from abc import ABC, abstractmethod

from .request import LLMPayload
from .response import LLMResult


class Provider(ABC):
	"""Base interface Bhushan and Harsh will implement for LLM providers.

	Implementations must return an LLMResult and must never raise for provider
	failures; they should set the result's error field instead. Implementations
	must not mutate the supplied payload.
	"""

	@abstractmethod
	def generate(self, payload: LLMPayload) -> LLMResult:
		"""Generate a result for a payload without mutating the payload."""
		...
