from __future__ import annotations

import logging
import os
from typing import Any

from .errors import ProviderCredentialsError, ProviderError
from .provider import Provider
from .request import LLMPayload
from .response import LLMResult

logger = logging.getLogger(__name__)


class GroqProvider(Provider):
    """Groq Cloud API adapter for AIOS.

    Translates AIOS LLMPayload into Groq chat completions format and converts
    the response into a validated LLMResult.

    Missing credentials do not raise at import or initialization time; an
    actionable ProviderCredentialsError is deferred until generate() is invoked.
    All runtime API, network, and parser errors return an LLMResult with error
    details rather than raising unhandled exceptions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Groq provider.

        Args:
            api_key: Groq API key. If omitted, resolved from GROQ_API_KEY env var.
            model: Default model identifier (e.g. 'llama-3.3-70b-versatile').
            base_url: Optional custom base URL or proxy endpoint.
            timeout: Network timeout in seconds for API calls.
            **kwargs: Extra arguments forwarded to the Groq client.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("GROQ_BASE_URL")
        self.timeout = timeout
        self.extra_kwargs = kwargs
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialized Groq client instance.

        Raises:
            ProviderCredentialsError: If the API key is not configured.
            ProviderError: If the groq library is not installed.
        """
        if self._client is None:
            if not self.api_key:
                raise ProviderCredentialsError(
                    "Groq API key not configured. "
                    "Set api_key in GroqProvider() or "
                    "GROQ_API_KEY environment variable."
                )
            try:
                from groq import Groq
            except ImportError as err:
                raise ProviderError(
                    "The groq package is required for GroqProvider. "
                    "Install it via 'pip install groq'."
                ) from err

            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client_kwargs.update(self.extra_kwargs)

            self._client = Groq(**client_kwargs)

        return self._client

    def _translate_payload(self, payload: LLMPayload) -> dict[str, Any]:
        """Convert LLMPayload into Groq chat.completions.create arguments.

        Does not mutate the input payload.
        """
        model = payload.model if payload.model else self.model
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in payload.messages
        ]

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Extract standard generation parameters if present in payload
        params = payload.parameters
        if "temperature" in params:
            request_kwargs["temperature"] = params["temperature"]
        if "max_tokens" in params:
            request_kwargs["max_tokens"] = params["max_tokens"]
        if "top_p" in params:
            request_kwargs["top_p"] = params["top_p"]

        # Forward any additional Groq-compatible parameters
        for key, value in params.items():
            if key not in {"temperature", "max_tokens", "top_p"}:
                request_kwargs[key] = value

        return request_kwargs

    def _translate_response(self, response: Any) -> LLMResult:
        """Convert Groq ChatCompletion response to a validated LLMResult."""
        try:
            choice = response.choices[0]
            text = choice.message.content or ""

            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return LLMResult(text=text, usage=usage, error=None)
        except Exception as err:
            return LLMResult(
                text="",
                usage={},
                error=f"Failed to parse Groq response: {type(err).__name__}: {err}",
            )

    def generate(self, payload: LLMPayload) -> LLMResult:
        """Generate an LLM response using Groq Cloud chat completions.

        Args:
            payload: Validated LLMPayload request data.

        Returns:
            Validated LLMResult containing completion text, token usage,
            and any error details.

        Raises:
            ProviderCredentialsError: If Groq credentials are not configured.
        """
        # Ensure client is accessible (validates credentials)
        client = self.client

        request_kwargs = self._translate_payload(payload)

        try:
            response = client.chat.completions.create(**request_kwargs)
            return self._translate_response(response)
        except Exception as exc:
            logger.warning("Groq completion request failed: %s", exc)
            return LLMResult(
                text="",
                usage={},
                error=f"Groq error: {type(exc).__name__}: {exc}",
            )
