from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import requests

from aios.llm import (
    GroqProvider,
    LLMPayload,
    LLMResult,
    MockLLM,
    OllamaProvider,
    OpenAIProvider,
    ProviderCredentialsError,
)


class MockChatChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = MagicMock()
        self.message.content = content
        self.message.role = "assistant"
        self.finish_reason = finish_reason


class MockUsage:
    def __init__(self, prompt_tokens: int = 12, completion_tokens: int = 8) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockChatCompletion:
    def __init__(
        self,
        content: str = "Hello from mock provider!",
        prompt_tokens: int = 15,
        completion_tokens: int = 10,
        finish_reason: str = "stop",
    ) -> None:
        self.choices = [MockChatChoice(content, finish_reason)]
        self.usage = MockUsage(prompt_tokens, completion_tokens)


# ==============================================================================
# OpenAIProvider Unit Tests
# ==============================================================================


class TestOpenAIProvider:
    """Unit tests for OpenAIProvider."""

    def test_import_and_init_without_credentials_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure initializing OpenAIProvider without keys does not crash."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(api_key=None)
        assert provider.api_key is None
        assert provider.model == "gpt-4"

    def test_missing_api_key_raises_at_generate_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key raises ProviderCredentialsError when generate() is called."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(api_key=None)
        payload = LLMPayload(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(ProviderCredentialsError) as exc_info:
            provider.generate(payload)

        err_msg = str(exc_info.value)
        assert "OpenAI API key not configured" in err_msg
        assert "OPENAI_API_KEY" in err_msg

    def test_api_key_resolved_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify API key is picked up from OPENAI_API_KEY environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")
        provider = OpenAIProvider()
        assert provider.api_key == "sk-env-test-key"

    def test_payload_translation_to_openai_format(self) -> None:
        """Verify LLMPayload is translated accurately into OpenAI request format."""
        provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4")
        payload = LLMPayload(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AIOS?"},
            ],
            parameters={
                "temperature": 0.5,
                "max_tokens": 256,
                "top_p": 0.9,
                "custom_option": "extra_val",
            },
        )

        translated = provider._translate_payload(payload)

        assert translated["model"] == "gpt-4o-mini"
        assert len(translated["messages"]) == 2
        assert translated["messages"][0]["role"] == "system"
        assert translated["messages"][1]["content"] == "What is AIOS?"
        assert translated["temperature"] == 0.5
        assert translated["max_tokens"] == 256
        assert translated["top_p"] == 0.9
        assert translated["custom_option"] == "extra_val"

    def test_payload_translation_uses_default_model_if_not_specified(self) -> None:
        """Use default model if payload.model is empty/defaulted."""
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4-turbo")
        payload = LLMPayload(
            model="default-placeholder",
            messages=[{"role": "user", "content": "Hi"}],
        )
        translated = provider._translate_payload(payload)
        assert translated["model"] == "default-placeholder"

    def test_openai_response_translation_to_result(self) -> None:
        """Verify OpenAI response object is converted into validated LLMResult."""
        provider = OpenAIProvider(api_key="sk-test")
        mock_resp = MockChatCompletion(
            content="AIOS is an Agent Operating System.",
            prompt_tokens=20,
            completion_tokens=8,
        )

        result = provider._translate_response(mock_resp)

        assert isinstance(result, LLMResult)
        assert result.ok is True
        assert result.error is None
        assert result.text == "AIOS is an Agent Operating System."
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 28

    def test_openai_response_translation_handles_missing_usage(self) -> None:
        """Verify translation succeeds when usage is absent or None."""
        provider = OpenAIProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.choices = [MockChatChoice("Response without usage")]
        mock_resp.usage = None

        result = provider._translate_response(mock_resp)

        assert result.ok is True
        assert result.text == "Response without usage"
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0

    def test_openai_response_translation_handles_malformed_object(self) -> None:
        """Return error result if response object cannot be parsed."""
        provider = OpenAIProvider(api_key="sk-test")
        malformed_resp = "not an object with choices"

        result = provider._translate_response(malformed_resp)

        assert result.ok is False
        assert result.error is not None
        assert "Failed to parse OpenAI response" in result.error
        assert result.text == ""

    def test_generate_with_mocked_openai_api(self) -> None:
        """Verify generate() performs complete successful generation round-trip."""
        provider = OpenAIProvider(api_key="sk-test-key")
        mock_resp = MockChatCompletion("Result from OpenAI", 10, 5)

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            payload = LLMPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Ping"}],
                parameters={"temperature": 0.2},
            )

            result = provider.generate(payload)

            assert result.ok is True
            assert result.text == "Result from OpenAI"
            assert result.usage["total_tokens"] == 15
            mock_client.chat.completions.create.assert_called_once()

    def test_error_handling_on_api_failure(self) -> None:
        """API failures must return LLMResult with error set, not raise."""
        provider = OpenAIProvider(api_key="sk-test-key")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "Rate limit exceeded (503)"
            )
            mock_openai_cls.return_value = mock_client

            payload = LLMPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Ping"}],
            )

            result = provider.generate(payload)

            assert result.ok is False
            assert result.text == ""
            assert result.error is not None
            assert "Rate limit exceeded" in result.error
            assert "OpenAI error: RuntimeError" in result.error

    def test_payload_not_mutated(self) -> None:
        """Ensure generate() leaves the input LLMPayload strictly unchanged."""
        provider = OpenAIProvider(api_key="sk-test-key")
        mock_resp = MockChatCompletion("No mutation")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            payload = LLMPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Immutable"}],
                parameters={"temperature": 0.7, "options": {"deep": [1, 2, 3]}},
            )
            original_payload = deepcopy(payload)

            provider.generate(payload)

            assert payload == original_payload

    def test_openai_custom_base_url_and_organization(self) -> None:
        """Verify custom base_url, organization, and extra kwargs
        forwarded to OpenAI client.
        """
        provider = OpenAIProvider(
            api_key="sk-test-key",
            base_url="https://proxy.example.com/v1",
            organization="org-12345",
            custom_client_opt="enabled",
        )

        with patch("openai.OpenAI") as mock_openai_cls:
            _ = provider.client
            mock_openai_cls.assert_called_once_with(
                api_key="sk-test-key",
                timeout=60.0,
                base_url="https://proxy.example.com/v1",
                organization="org-12345",
                custom_client_opt="enabled",
            )

    def test_openai_missing_library_raises_provider_error(self) -> None:
        """Raise ProviderError if openai library is missing when client is accessed."""
        provider = OpenAIProvider(api_key="sk-test-key")
        from aios.llm.errors import ProviderError

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ProviderError) as exc_info:
                _ = provider.client
            assert "The openai package is required" in str(exc_info.value)


# ==============================================================================
# GroqProvider Unit Tests
# ==============================================================================


class TestGroqProvider:
    """Unit tests for GroqProvider."""

    def test_import_and_init_without_credentials_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure initializing GroqProvider without keys does not crash."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqProvider(api_key=None)
        assert provider.api_key is None
        assert provider.model == "llama-3.3-70b-versatile"

    def test_missing_api_key_raises_at_generate_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key raises ProviderCredentialsError when generate() is called."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqProvider(api_key=None)
        payload = LLMPayload(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(ProviderCredentialsError) as exc_info:
            provider.generate(payload)

        err_msg = str(exc_info.value)
        assert "Groq API key not configured" in err_msg
        assert "GROQ_API_KEY" in err_msg

    def test_api_key_resolved_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify API key is picked up from GROQ_API_KEY environment variable."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-env-test-key")
        provider = GroqProvider()
        assert provider.api_key == "gsk-env-test-key"

    def test_payload_translation_to_groq_format(self) -> None:
        """Verify LLMPayload is translated accurately into Groq request format."""
        provider = GroqProvider(api_key="gsk-test", model="llama-3.3-70b-versatile")
        payload = LLMPayload(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are fast."},
                {"role": "user", "content": "Calculate 42."},
            ],
            parameters={
                "temperature": 0.1,
                "max_tokens": 128,
                "top_p": 0.8,
            },
        )

        translated = provider._translate_payload(payload)

        assert translated["model"] == "mixtral-8x7b-32768"
        assert len(translated["messages"]) == 2
        assert translated["messages"][0]["role"] == "system"
        assert translated["temperature"] == 0.1
        assert translated["max_tokens"] == 128
        assert translated["top_p"] == 0.8

    def test_groq_response_translation_to_result(self) -> None:
        """Verify Groq ChatCompletion response translates into LLMResult."""
        provider = GroqProvider(api_key="gsk-test")
        mock_resp = MockChatCompletion("Fast response from Groq", 14, 6)

        result = provider._translate_response(mock_resp)

        assert isinstance(result, LLMResult)
        assert result.ok is True
        assert result.text == "Fast response from Groq"
        assert result.usage["prompt_tokens"] == 14
        assert result.usage["completion_tokens"] == 6
        assert result.usage["total_tokens"] == 20

    def test_groq_response_translation_handles_malformed_object(self) -> None:
        """Return error result when Groq response structure is unexpected."""
        provider = GroqProvider(api_key="gsk-test")
        result = provider._translate_response({"unexpected": "structure"})

        assert result.ok is False
        assert result.error is not None
        assert "Failed to parse Groq response" in result.error

    def test_generate_with_mocked_groq_api(self) -> None:
        """Verify generate() performs complete successful Groq generation."""
        provider = GroqProvider(api_key="gsk-test-key")
        mock_resp = MockChatCompletion("Groq answer", 8, 4)

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_groq_cls.return_value = mock_client

            payload = LLMPayload(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Compute"}],
            )

            result = provider.generate(payload)

            assert result.ok is True
            assert result.text == "Groq answer"
            assert result.usage["total_tokens"] == 12
            mock_client.chat.completions.create.assert_called_once()

    def test_error_handling_on_api_failure(self) -> None:
        """Groq API failures must return LLMResult with error set."""
        provider = GroqProvider(api_key="gsk-test-key")

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = ConnectionResetError(
                "Connection dropped"
            )
            mock_groq_cls.return_value = mock_client

            payload = LLMPayload(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Ping"}],
            )

            result = provider.generate(payload)

            assert result.ok is False
            assert result.text == ""
            assert result.error is not None
            assert "Connection dropped" in result.error
            assert "Groq error: ConnectionResetError" in result.error

    def test_payload_not_mutated(self) -> None:
        """Ensure GroqProvider leaves payload unchanged."""
        provider = GroqProvider(api_key="gsk-test-key")
        mock_resp = MockChatCompletion("No mutation")

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_groq_cls.return_value = mock_client

            payload = LLMPayload(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Immutable"}],
            )
            original_payload = deepcopy(payload)

            provider.generate(payload)
            assert payload == original_payload

    def test_groq_custom_base_url_and_extra_kwargs(self) -> None:
        """Verify custom base_url and extra kwargs forwarded to Groq client."""
        provider = GroqProvider(
            api_key="gsk-test-key",
            base_url="https://groq-proxy.example.com",
            custom_arg=123,
        )

        with patch("groq.Groq") as mock_groq_cls:
            _ = provider.client
            mock_groq_cls.assert_called_once_with(
                api_key="gsk-test-key",
                timeout=60.0,
                base_url="https://groq-proxy.example.com",
                custom_arg=123,
            )

    def test_groq_missing_library_raises_provider_error(self) -> None:
        """Raise ProviderError if groq library is missing when client is accessed."""
        provider = GroqProvider(api_key="gsk-test-key")
        from aios.llm.errors import ProviderError

        with patch.dict("sys.modules", {"groq": None}):
            with pytest.raises(ProviderError) as exc_info:
                _ = provider.client
            assert "The groq package is required" in str(exc_info.value)

    def test_groq_custom_parameters_forwarded(self) -> None:
        """Verify extra parameters in LLMPayload are forwarded to Groq request."""
        provider = GroqProvider(api_key="gsk-test")
        payload = LLMPayload(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Test"}],
            parameters={"stop": ["\n"], "seed": 42},
        )
        translated = provider._translate_payload(payload)
        assert translated["stop"] == ["\n"]
        assert translated["seed"] == 42


# ==============================================================================
# OllamaProvider Unit Tests
# ==============================================================================


class TestOllamaProvider:
    """Unit tests for OllamaProvider."""

    def test_init_and_base_url_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test base URL defaults and environment overrides."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama3"

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-service:11434/")
        provider_env = OllamaProvider()
        assert provider_env.base_url == "http://ollama-service:11434"

        provider_custom = OllamaProvider(base_url="http://custom-host:8080/")
        assert provider_custom.base_url == "http://custom-host:8080"

    def test_messages_to_prompt_conversion(self) -> None:
        """Verify conversion of multi-turn messages into unified prompt string."""
        provider = OllamaProvider()
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a python function."},
            {"role": "assistant", "content": "def foo(): pass"},
            {"role": "user", "content": "Now add docs."},
        ]

        prompt = provider._messages_to_prompt(messages)

        expected = (
            "System: You are a coding assistant.\n"
            "User: Write a python function.\n"
            "Assistant: def foo(): pass\n"
            "User: Now add docs."
        )
        assert prompt == expected

    def test_payload_translation_generate_endpoint(self) -> None:
        """Verify request body translation for /api/generate endpoint."""
        provider = OllamaProvider(model="llama3", endpoint="/api/generate")
        payload = LLMPayload(
            model="llama3.1",
            messages=[{"role": "user", "content": "Hello Ollama"}],
            parameters={
                "temperature": 0.8,
                "max_tokens": 500,
                "top_p": 0.95,
            },
        )

        body = provider._translate_payload(payload)

        assert body["model"] == "llama3.1"
        assert body["prompt"] == "User: Hello Ollama"
        assert body["stream"] is False
        assert body["temperature"] == 0.8
        assert body["options"]["temperature"] == 0.8
        assert body["options"]["num_predict"] == 500
        assert body["options"]["top_p"] == 0.95

    def test_payload_translation_chat_endpoint(self) -> None:
        """Verify request body translation for /api/chat endpoint."""
        provider = OllamaProvider(model="llama3", endpoint="/api/chat")
        payload = LLMPayload(
            model="llama3",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain gravity."},
            ],
            parameters={"temperature": 0.4},
        )

        body = provider._translate_payload(payload)

        assert body["model"] == "llama3"
        assert body["stream"] is False
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["content"] == "Explain gravity."

    def test_ollama_response_translation_generate_format(self) -> None:
        """Convert Ollama /api/generate JSON response into LLMResult."""
        provider = OllamaProvider()
        response_json = {
            "model": "llama3",
            "response": "Gravity is curvature of spacetime.",
            "done": True,
            "prompt_eval_count": 18,
            "eval_count": 7,
            "total_duration": 4500000000,
        }

        result = provider._translate_ollama_response(response_json)

        assert isinstance(result, LLMResult)
        assert result.ok is True
        assert result.text == "Gravity is curvature of spacetime."
        assert result.usage["prompt_tokens"] == 18
        assert result.usage["completion_tokens"] == 7
        assert result.usage["total_tokens"] == 25

    def test_ollama_response_translation_chat_format(self) -> None:
        """Convert Ollama /api/chat JSON response into LLMResult."""
        provider = OllamaProvider()
        response_json = {
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "Chat formatted completion.",
            },
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = provider._translate_ollama_response(response_json)

        assert result.ok is True
        assert result.text == "Chat formatted completion."
        assert result.usage["total_tokens"] == 15

    def test_ollama_response_translation_handles_missing_fields(self) -> None:
        """Gracefully handle minimal response with zero token metrics."""
        provider = OllamaProvider()
        response_json = {"response": "Short answer"}

        result = provider._translate_ollama_response(response_json)

        assert result.ok is True
        assert result.text == "Short answer"
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0

    def test_generate_with_mocked_ollama_api(self) -> None:
        """Verify successful generate() call with mocked requests.post."""
        provider = OllamaProvider()
        payload = LLMPayload(
            model="llama3",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "4",
            "eval_count": 2,
            "prompt_eval_count": 6,
            "done": True,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = provider.generate(payload)

            assert result.ok is True
            assert result.text == "4"
            assert result.usage["total_tokens"] == 8
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert call_url == "http://localhost:11434/api/generate"

    def test_connection_error_handling(self) -> None:
        """Connection errors return an LLMResult error instead of crashing."""
        provider = OllamaProvider()
        payload = LLMPayload(
            model="llama3",
            messages=[{"role": "user", "content": "Test"}],
        )

        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            result = provider.generate(payload)

            assert result.ok is False
            assert result.text == ""
            assert result.error is not None
            assert "Cannot connect to Ollama at http://localhost:11434" in result.error
            assert "Connection refused" in result.error

    def test_timeout_handling(self) -> None:
        """Timeout errors return an LLMResult error."""
        provider = OllamaProvider(timeout=5.0)
        payload = LLMPayload(
            model="llama3",
            messages=[{"role": "user", "content": "Test"}],
        )

        with patch(
            "requests.post",
            side_effect=requests.exceptions.Timeout("Read timed out"),
        ):
            result = provider.generate(payload)

            assert result.ok is False
            assert result.text == ""
            assert result.error is not None
            assert "Ollama request timed out" in result.error

    def test_http_error_handling(self) -> None:
        """HTTP 4xx/5xx status errors return an LLMResult error."""
        provider = OllamaProvider()
        payload = LLMPayload(
            model="non-existent-model",
            messages=[{"role": "user", "content": "Test"}],
        )

        with patch(
            "requests.post",
            side_effect=requests.exceptions.HTTPError("404 Client Error: Not Found"),
        ):
            result = provider.generate(payload)

            assert result.ok is False
            assert result.text == ""
            assert result.error is not None
            assert "Ollama HTTP error" in result.error

    def test_payload_not_mutated(self) -> None:
        """Ensure OllamaProvider leaves payload unchanged."""
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "done"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            payload = LLMPayload(
                model="llama3",
                messages=[{"role": "user", "content": "Immutable"}],
            )
            original_payload = deepcopy(payload)

            provider.generate(payload)
            assert payload == original_payload

    def test_ollama_missing_requests_package(self) -> None:
        """Return LLMResult error when requests package is not installed."""
        provider = OllamaProvider()
        payload = LLMPayload(model="m", messages=[{"role": "user", "content": "Hi"}])

        with patch.dict("sys.modules", {"requests": None}):
            result = provider.generate(payload)
            assert result.ok is False
            assert "The requests package is required" in (result.error or "")

    def test_ollama_generate_with_chat_endpoint(self) -> None:
        """Verify successful generation using the /api/chat endpoint."""
        provider = OllamaProvider(endpoint="/api/chat")
        payload = LLMPayload(model="m", messages=[{"role": "user", "content": "Hi"}])

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Chat reply"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = provider.generate(payload)
            assert result.ok is True
            assert result.text == "Chat reply"
            assert result.usage["total_tokens"] == 8
            assert mock_post.call_args[0][0] == "http://localhost:11434/api/chat"

    def test_ollama_generic_exception_handling(self) -> None:
        """Catch unexpected generic exceptions and return LLMResult with error."""
        provider = OllamaProvider()
        payload = LLMPayload(model="m", messages=[{"role": "user", "content": "Hi"}])

        with patch("requests.post", side_effect=ValueError("Bad JSON URL")):
            result = provider.generate(payload)
            assert result.ok is False
            assert "Ollama error: ValueError" in (result.error or "")


# ==============================================================================
# MockLLM Unit Tests
# ==============================================================================


class TestMockLLM:
    """Unit tests for MockLLM reference provider."""

    def test_echo_generation_default_behavior(self) -> None:
        """MockLLM echoes the last user message by default."""
        mock = MockLLM()
        payload = LLMPayload(
            model="mock-model",
            messages=[
                {"role": "system", "content": "Sys prompt"},
                {"role": "user", "content": "Echo this text please"},
            ],
        )

        result = mock.generate(payload)

        assert result.ok is True
        assert result.text == "Mock response to: Echo this text please"
        assert result.usage["total_tokens"] > 0
        assert mock.call_count == 1
        assert mock.last_payload == payload

    def test_canned_response(self) -> None:
        """Static canned response returned across calls."""
        mock = MockLLM(canned_response="Fixed answer")
        payload = LLMPayload(
            model="mock-model",
            messages=[{"role": "user", "content": "Question 1"}],
        )

        result1 = mock.generate(payload)
        result2 = mock.generate(payload)

        assert result1.text == "Fixed answer"
        assert result2.text == "Fixed answer"
        assert mock.call_count == 2

    def test_sequential_canned_responses(self) -> None:
        """Sequential canned responses are popped in order."""
        mock = MockLLM(canned_responses=["First", "Second", "Third"])
        payload = LLMPayload(
            model="mock-model",
            messages=[{"role": "user", "content": "Q"}],
        )

        assert mock.generate(payload).text == "First"
        assert mock.generate(payload).text == "Second"
        assert mock.generate(payload).text == "Third"
        # After sequence is exhausted, falls back to echo
        assert "Mock response to: Q" in mock.generate(payload).text

    def test_canned_error(self) -> None:
        """Simulate provider failure."""
        mock = MockLLM(canned_error="Simulated upstream outage")
        payload = LLMPayload(
            model="mock-model",
            messages=[{"role": "user", "content": "Q"}],
        )

        result = mock.generate(payload)

        assert result.ok is False
        assert result.error == "Simulated upstream outage"
        assert result.text == ""

    def test_call_history_and_reset(self) -> None:
        """Verify tracking of payloads and reset() method."""
        mock = MockLLM()
        payload1 = LLMPayload(model="m", messages=[{"role": "user", "content": "1"}])
        payload2 = LLMPayload(model="m", messages=[{"role": "user", "content": "2"}])

        mock.generate(payload1)
        mock.generate(payload2)

        assert mock.call_count == 2
        assert mock.calls[0].messages[0]["content"] == "1"
        assert mock.calls[1].messages[0]["content"] == "2"

        mock.reset()
        assert mock.call_count == 0
        assert mock.last_payload is None

    def test_payload_not_mutated(self) -> None:
        """Ensure MockLLM does not mutate payload."""
        mock = MockLLM()
        payload = LLMPayload(
            model="m",
            messages=[{"role": "user", "content": "Original"}],
            parameters={"temperature": 0.5},
        )
        original_payload = deepcopy(payload)

        mock.generate(payload)
        assert payload == original_payload


# ==============================================================================
# Package Exports and Error Hierarchy Tests
# ==============================================================================


class TestPackageExportsAndCore:
    """Verify package public API and error classes."""

    def test_package_exports(self) -> None:
        """Verify all expected public symbols are exposed in aios.llm."""
        import aios.llm as llm

        expected_symbols = {
            "Provider",
            "LLMPayload",
            "LLMResult",
            "OpenAIProvider",
            "GroqProvider",
            "OllamaProvider",
            "MockLLM",
            "ProviderError",
            "ProviderCredentialsError",
            "ProviderNetworkError",
            "ProviderResponseError",
        }
        for symbol in expected_symbols:
            assert hasattr(llm, symbol), f"Missing public symbol: {symbol}"

    def test_core_module_exports(self) -> None:
        """Verify aios.llm.core module symbols."""
        from aios.llm.core import LLMPayload, LLMResult, MockLLM, Provider

        assert Provider is not None
        assert LLMPayload is not None
        assert LLMResult is not None
        assert MockLLM is not None

    def test_error_hierarchy(self) -> None:
        """Verify custom exception inheritance hierarchy."""
        from aios.llm.errors import (
            ProviderCredentialsError,
            ProviderError,
            ProviderNetworkError,
            ProviderResponseError,
        )

        assert issubclass(ProviderCredentialsError, ProviderError)
        assert issubclass(ProviderNetworkError, ProviderError)
        assert issubclass(ProviderResponseError, ProviderError)
        assert issubclass(ProviderError, Exception)
