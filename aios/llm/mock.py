from copy import deepcopy

from .provider import Provider
from .request import LLMPayload
from .response import LLMResult


class MockLLM(Provider):
    """Reference in-memory mock LLM provider for tests and local development.

    Supports canned responses, simulated errors, echo responses, and call tracking
    while adhering strictly to the Provider protocol.
    """

    def __init__(
        self,
        canned_response: str | None = None,
        canned_responses: list[str] | None = None,
        canned_error: str | None = None,
        model: str = "mock-model",
    ) -> None:
        """Initialize MockLLM.

        Args:
            canned_response: Static text to return for every call.
            canned_responses: Sequence of responses to return across sequential calls.
            canned_error: Simulated error string to return in LLMResult.
            model: Default model identifier.
        """
        self.canned_response = canned_response
        self.canned_responses = list(canned_responses) if canned_responses else []
        self.canned_error = canned_error
        self.model = model
        self.calls: list[LLMPayload] = []

    @property
    def call_count(self) -> int:
        """Return the number of generate() calls made to this mock."""
        return len(self.calls)

    @property
    def last_payload(self) -> LLMPayload | None:
        """Return the most recent payload received, or None."""
        return self.calls[-1] if self.calls else None

    def generate(self, payload: LLMPayload) -> LLMResult:
        """Generate a simulated response without mutating the input payload.

        Args:
            payload: Validated LLMPayload request.

        Returns:
            Validated LLMResult.
        """
        # Record deep copy of payload to preserve call history without mutation
        self.calls.append(deepcopy(payload))

        if self.canned_error is not None:
            return LLMResult(text="", usage={}, error=self.canned_error)

        if self.canned_responses:
            text = self.canned_responses.pop(0)
        elif self.canned_response is not None:
            text = self.canned_response
        else:
            # Default behavior: echo the last user message
            user_messages = [
                m["content"] for m in payload.messages if m["role"] == "user"
            ]
            if user_messages:
                text = f"Mock response to: {user_messages[-1]}"
            else:
                text = "Mock response"

        prompt_tokens = sum(
            max(1, len(m.get("content", "").split())) for m in payload.messages
        )
        completion_tokens = max(1, len(text.split()))
        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        return LLMResult(text=text, usage=usage, error=None)

    def reset(self) -> None:
        """Clear call history."""
        self.calls.clear()
