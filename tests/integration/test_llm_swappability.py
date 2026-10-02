from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from aios.llm import (
    GroqProvider,
    LLMPayload,
    LLMResult,
    MockLLM,
    OllamaProvider,
    OpenAIProvider,
    Provider,
)
from tests.unit.test_llm_providers import MockChatCompletion

# ==============================================================================
# Swappable Consumer Function: Zero Provider-Specific Code
# ==============================================================================


def ask_agent(provider: Provider, user_query: str) -> LLMResult:
    """Generic agent reasoning step written purely against the Provider protocol.

    This function does NOT know or care whether it is talking to MockLLM,
    OpenAIProvider, GroqProvider, or OllamaProvider.
    """
    payload = LLMPayload(
        model="default-agent-model",
        messages=[
            {"role": "system", "content": "You are a reliable AIOS agent."},
            {"role": "user", "content": user_query},
        ],
        parameters={
            "temperature": 0.3,
            "max_tokens": 150,
        },
    )
    return provider.generate(payload)


def run_multi_turn_dialogue(
    provider: Provider, dialogue: list[dict[str, str]]
) -> LLMResult:
    """Multi-turn interaction executed across any swappable provider."""
    payload = LLMPayload(
        model="default-agent-model",
        messages=dialogue,
        parameters={"temperature": 0.2},
    )
    return provider.generate(payload)


# ==============================================================================
# Swappability and Integration Test Suite
# ==============================================================================


class TestProviderSwappability:
    """Prove that code written against Provider works identically with all backends."""

    def test_code_works_with_mock_llm(self) -> None:
        """Verify generic agent reasoning works with MockLLM."""
        provider = MockLLM(canned_response="4")
        result = ask_agent(provider, "What is 2+2?")

        assert isinstance(result, LLMResult)
        assert result.ok is True
        assert result.text == "4"
        assert result.usage["total_tokens"] > 0
        assert result.error is None

    def test_code_works_with_openai_provider(self) -> None:
        """Verify generic agent reasoning works with OpenAIProvider (mocked API)."""
        mock_resp = MockChatCompletion("4", 12, 3)

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            provider = OpenAIProvider(api_key="test-key")
            result = ask_agent(provider, "What is 2+2?")

            assert isinstance(result, LLMResult)
            assert result.ok is True
            assert result.text == "4"
            assert result.usage["total_tokens"] == 15
            assert result.error is None

    def test_code_works_with_groq_provider(self) -> None:
        """Verify generic agent reasoning works with GroqProvider (mocked API)."""
        mock_resp = MockChatCompletion("4", 12, 3)

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_groq_cls.return_value = mock_client

            provider = GroqProvider(api_key="test-key")
            result = ask_agent(provider, "What is 2+2?")

            assert isinstance(result, LLMResult)
            assert result.ok is True
            assert result.text == "4"
            assert result.usage["total_tokens"] == 15
            assert result.error is None

    def test_code_works_with_ollama_provider(self) -> None:
        """Verify generic agent reasoning works with OllamaProvider (mocked HTTP)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "4",
            "prompt_eval_count": 8,
            "eval_count": 2,
            "done": True,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            provider = OllamaProvider()
            result = ask_agent(provider, "What is 2+2?")

            assert isinstance(result, LLMResult)
            assert result.ok is True
            assert result.text == "4"
            assert result.usage["total_tokens"] == 10
            assert result.error is None

    def test_all_providers_in_identical_consumer_loop(self) -> None:
        """Execute the exact same consumer code in a loop over all 4 providers."""
        providers: list[tuple[str, Provider]] = [
            ("mock", MockLLM(canned_response="Answer 42")),
            ("openai", OpenAIProvider(api_key="sk-test")),
            ("groq", GroqProvider(api_key="gsk-test")),
            ("ollama", OllamaProvider()),
        ]

        mock_openai_resp = MockChatCompletion("Answer 42", 10, 5)
        mock_groq_resp = MockChatCompletion("Answer 42", 10, 5)
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "response": "Answer 42",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        with (
            patch("openai.OpenAI") as mock_openai_cls,
            patch("groq.Groq") as mock_groq_cls,
            patch("requests.post", return_value=mock_ollama_resp),
        ):
            mock_openai_client = MagicMock()
            mock_openai_client.chat.completions.create.return_value = mock_openai_resp
            mock_openai_cls.return_value = mock_openai_client

            mock_groq_client = MagicMock()
            mock_groq_client.chat.completions.create.return_value = mock_groq_resp
            mock_groq_cls.return_value = mock_groq_client

            for name, provider in providers:
                result = ask_agent(provider, "Universal query")
                assert isinstance(result, LLMResult), f"{name} failed instance check"
                assert result.ok is True, f"{name} failed ok check"
                assert result.text == "Answer 42", f"{name} produced incorrect text"
                assert result.usage["total_tokens"] > 0, f"{name} missing usage"
                assert (
                    result.usage["total_tokens"]
                    == result.usage["prompt_tokens"] + result.usage["completion_tokens"]
                ), f"{name} usage sum mismatch"
                assert result.error is None

    def test_multi_turn_dialogue_across_all_providers(self) -> None:
        """Verify multi-turn history translation across all providers."""
        history = [
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": "def add(a, b): return a + b"},
            {"role": "assistant", "content": "Looks good, add type annotations."},
            {"role": "user", "content": "def add(a: int, b: int) -> int: return a + b"},
        ]

        mock_completion = MockChatCompletion("Perfect.", 30, 4)
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "response": "Perfect.",
            "prompt_eval_count": 30,
            "eval_count": 4,
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        with (
            patch("openai.OpenAI") as mock_openai_cls,
            patch("groq.Groq") as mock_groq_cls,
            patch("requests.post", return_value=mock_ollama_resp),
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai_cls.return_value = mock_client
            mock_groq_cls.return_value = mock_client

            providers: list[Provider] = [
                MockLLM(canned_response="Perfect."),
                OpenAIProvider(api_key="test"),
                GroqProvider(api_key="test"),
                OllamaProvider(),
            ]

            for provider in providers:
                result = run_multi_turn_dialogue(provider, history)
                assert result.ok is True
                assert result.text == "Perfect."
                assert result.usage["total_tokens"] > 0
                assert (
                    result.usage["total_tokens"]
                    == result.usage["prompt_tokens"] + result.usage["completion_tokens"]
                )

    def test_error_state_consistency_across_all_providers(self) -> None:
        """All providers must return standard LLMResult on runtime failure."""
        providers: list[tuple[str, Provider]] = [
            ("mock", MockLLM(canned_error="Mock error")),
            ("openai", OpenAIProvider(api_key="test")),
            ("groq", GroqProvider(api_key="test")),
            ("ollama", OllamaProvider()),
        ]

        with (
            patch("openai.OpenAI") as mock_openai_cls,
            patch("groq.Groq") as mock_groq_cls,
            patch(
                "requests.post",
                side_effect=requests.exceptions.ConnectionError("Refused"),
            ),
        ):
            mock_oai = MagicMock()
            mock_oai.chat.completions.create.side_effect = RuntimeError("OpenAI 500")
            mock_openai_cls.return_value = mock_oai

            mock_grq = MagicMock()
            mock_grq.chat.completions.create.side_effect = RuntimeError("Groq 500")
            mock_groq_cls.return_value = mock_grq

            for name, provider in providers:
                result = ask_agent(provider, "Trigger error")
                assert isinstance(result, LLMResult)
                assert result.ok is False, f"{name} should have ok=False"
                assert result.text == "", f"{name} should have empty text"
                assert isinstance(result.error, str), f"{name} error must be str"
                assert len(result.error) > 0

    def test_provider_router_dispatcher_integration(self) -> None:
        """Verify Swarang's LLM Request Layer router pattern.

        A dispatcher holding any combination of providers routes requests
        without knowing the concrete implementations.
        """
        router: dict[str, Provider] = {
            "mock": MockLLM(canned_response="routed-mock"),
            "openai": OpenAIProvider(api_key="test"),
            "groq": GroqProvider(api_key="test"),
            "ollama": OllamaProvider(),
        }

        mock_completion = MockChatCompletion("routed-response", 8, 4)
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.json.return_value = {
            "response": "routed-response",
            "prompt_eval_count": 8,
            "eval_count": 4,
        }
        mock_ollama_resp.raise_for_status = MagicMock()

        with (
            patch("openai.OpenAI") as mock_openai_cls,
            patch("groq.Groq") as mock_groq_cls,
            patch("requests.post", return_value=mock_ollama_resp),
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai_cls.return_value = mock_client
            mock_groq_cls.return_value = mock_client

            payload = LLMPayload(
                model="test",
                messages=[{"role": "user", "content": "Route this"}],
            )

            for route_name, provider in router.items():
                result = provider.generate(payload)
                assert isinstance(result, LLMResult)
                assert result.ok is True
                assert result.error is None
                assert hasattr(result, "text")
                assert hasattr(result, "usage")
