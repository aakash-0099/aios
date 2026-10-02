from __future__ import annotations

import logging
import os
from typing import Any

from .provider import Provider
from .request import LLMPayload
from .response import LLMResult

logger = logging.getLogger(__name__)


class OllamaProvider(Provider):
    """Ollama API adapter for local or self-hosted LLM endpoints.

    Translates AIOS LLMPayload into Ollama REST API requests (/api/generate or
    /api/chat) and translates the JSON response back into a validated LLMResult.

    Ollama runs locally without authentication by default. Missing or invalid
    service endpoints are caught at generate() time and return a descriptive
    LLMResult error rather than crashing the caller.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "llama3",
        timeout: float = 60.0,
        endpoint: str = "/api/generate",
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL of the Ollama server (e.g. 'http://localhost:11434').
                Defaults to OLLAMA_BASE_URL env var or 'http://localhost:11434'.
            model: Default model identifier (e.g. 'llama3', 'mistral').
            timeout: Network timeout in seconds for API calls.
            endpoint: Ollama endpoint path to use ('/api/generate' or '/api/chat').
        """
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.base_url = raw_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.endpoint = endpoint

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert standard message format into a unified prompt string for Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            Formatted prompt string combining roles and contents.
        """
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

    def _translate_payload(self, payload: LLMPayload) -> dict[str, Any]:
        """Convert LLMPayload into an Ollama REST request body.

        Does not mutate the input payload.
        """
        model = payload.model if payload.model else self.model
        params = payload.parameters

        options: dict[str, Any] = {}
        if "temperature" in params:
            options["temperature"] = params["temperature"]
        if "max_tokens" in params:
            options["num_predict"] = params["max_tokens"]
        if "top_p" in params:
            options["top_p"] = params["top_p"]

        # Forward any custom options in parameters
        for key, value in params.items():
            if key not in {"temperature", "max_tokens", "top_p"}:
                options[key] = value

        if self.endpoint == "/api/chat":
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in payload.messages
                ],
                "stream": False,
            }
        else:
            prompt = self._messages_to_prompt(payload.messages)
            body = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }

        if options:
            body["options"] = options
        if "temperature" in params:
            body["temperature"] = params["temperature"]

        return body

    def _translate_ollama_response(self, response_json: dict[str, Any]) -> LLMResult:
        """Convert Ollama JSON response to a validated LLMResult.

        Handles both /api/generate ('response') and /api/chat ('message.content').
        """
        try:
            if "response" in response_json:
                text = response_json.get("response", "") or ""
            elif "message" in response_json and isinstance(
                response_json["message"], dict
            ):
                text = response_json["message"].get("content", "") or ""
            else:
                text = ""

            prompt_tokens = int(response_json.get("prompt_eval_count", 0) or 0)
            completion_tokens = int(response_json.get("eval_count", 0) or 0)
            total_tokens = prompt_tokens + completion_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            return LLMResult(text=text, usage=usage, error=None)
        except Exception as err:
            return LLMResult(
                text="",
                usage={},
                error=f"Failed to parse Ollama response: {type(err).__name__}: {err}",
            )

    def generate(self, payload: LLMPayload) -> LLMResult:
        """Generate an LLM response using an Ollama REST service.

        Args:
            payload: Validated LLMPayload request data.

        Returns:
            Validated LLMResult containing completion text, token usage,
            and any error details.
        """
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError as err:
            return LLMResult(
                text="",
                usage={},
                error=(
                    "The requests package is required for OllamaProvider. "
                    f"Install it via 'pip install requests': {err}"
                ),
            )

        url = f"{self.base_url}{self.endpoint}"
        body = self._translate_payload(payload)

        try:
            response = requests.post(url, json=body, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return self._translate_ollama_response(data)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Ollama connection error at %s: %s", self.base_url, exc)
            return LLMResult(
                text="",
                usage={},
                error=f"Cannot connect to Ollama at {self.base_url}: {exc}",
            )
        except requests.exceptions.Timeout as exc:
            logger.warning("Ollama request timed out after %ss: %s", self.timeout, exc)
            return LLMResult(
                text="",
                usage={},
                error=f"Ollama request timed out: {exc}",
            )
        except requests.exceptions.HTTPError as exc:
            logger.warning("Ollama HTTP status error: %s", exc)
            return LLMResult(
                text="",
                usage={},
                error=f"Ollama HTTP error: {exc}",
            )
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return LLMResult(
                text="",
                usage={},
                error=f"Ollama error: {type(exc).__name__}: {exc}",
            )
