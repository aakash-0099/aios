from __future__ import annotations

import logging
import os
from typing import Any

from .errors import ProviderCredentialsError, ProviderError
from .provider import Provider
from .request import LLMPayload
from .response import LLMResult

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    """OpenAI API adapter for AIOS.

    Translates AIOS LLMPayload into OpenAI chat completions format and converts
    the response into a validated LLMResult.

    Missing credentials do not raise at import or initialization time; an
    actionable ProviderCredentialsError is deferred until generate() is invoked.
    All runtime API, network, and parser errors return an LLMResult with error
    details rather than raising unhandled exceptions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If omitted, resolved from OPENAI_API_KEY env var.
            model: Default model identifier (e.g. 'gpt-4', 'gpt-4o').
            base_url: Optional custom base URL or proxy endpoint.
            organization: Optional OpenAI organization ID.
            timeout: Network timeout in seconds for API calls.
            **kwargs: Extra arguments forwarded to the OpenAI client.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.timeout = timeout
        self.extra_kwargs = kwargs
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialized OpenAI client instance.

        Raises:
            ProviderCredentialsError: If the API key is not configured.
            ProviderError: If the openai library is not installed.
        """
        if self._client is None:
            if not self.api_key:
                raise ProviderCredentialsError(
                    "OpenAI API key not configured. "
                    "Set api_key in OpenAIProvider() or "
                    "OPENAI_API_KEY environment variable."
                )
            try:
                from openai import OpenAI
            except ImportError as err:
                raise ProviderError(
                    "The openai package is required for OpenAIProvider. "
                    "Install it via 'pip install openai'."
                ) from err

            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.organization:
                client_kwargs["organization"] = self.organization
            client_kwargs.update(self.extra_kwargs)

            self._client = OpenAI(**client_kwargs)

        return self._client

    def _translate_payload(self, payload: LLMPayload) -> dict[str, Any]:
        """Convert LLMPayload into OpenAI chat.completions.create arguments.

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

        # Forward any additional OpenAI-compatible parameters
        for key, value in params.items():
            if key not in {"temperature", "max_tokens", "top_p"}:
                request_kwargs[key] = value

        return request_kwargs

    def _translate_response(self, response: Any) -> LLMResult:
        """Convert OpenAI ChatCompletion response to a validated LLMResult."""
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
                error=f"Failed to parse OpenAI response: {type(err).__name__}: {err}",
            )

    def generate(self, payload: LLMPayload) -> LLMResult:
        """Generate an LLM response using OpenAI chat completions.

        Args:
            payload: Validated LLMPayload request data.

        Returns:
            Validated LLMResult containing completion text, token usage,
            and any error details.

        Raises:
            ProviderCredentialsError: If OpenAI credentials are not configured.
        """
        # Ensure client is accessible (validates credentials)
        client = self.client

        request_kwargs = self._translate_payload(payload)

        try:
            response = client.chat.completions.create(**request_kwargs)
            return self._translate_response(response)
        except Exception as exc:
            logger.warning("OpenAI completion request failed: %s", exc)
            return LLMResult(
                text="",
                usage={},
                error=f"OpenAI error: {type(exc).__name__}: {exc}",
            )
