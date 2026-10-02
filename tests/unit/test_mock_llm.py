from __future__ import annotations

import time
from copy import deepcopy

import pytest

from aios.llm import LLMCore, LLMPayload, LLMResult, MockProvider, MockScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload(content: str = "hello", model: str = "test-model") -> LLMPayload:
    return LLMPayload(
        model=model,
        messages=[{"role": "user", "content": content}],
    )


# ---------------------------------------------------------------------------
# MockScenario validation
# ---------------------------------------------------------------------------


def test_mock_scenario_rejects_non_str_error_message() -> None:
    with pytest.raises(TypeError):
        MockScenario(error_message=123)  # type: ignore[arg-type]


def test_mock_scenario_rejects_empty_error_message() -> None:
    with pytest.raises(ValueError):
        MockScenario(error_message="   ")


def test_mock_scenario_stores_message() -> None:
    s = MockScenario(error_message="timeout")
    assert s.error_message == "timeout"


# ---------------------------------------------------------------------------
# MockProvider construction validation
# ---------------------------------------------------------------------------


def test_mock_provider_rejects_non_mapping_responses() -> None:
    with pytest.raises(TypeError):
        MockProvider(responses=["a", "b"])  # type: ignore[arg-type]


def test_mock_provider_rejects_non_str_response_key() -> None:
    with pytest.raises(TypeError):
        MockProvider(responses={1: "answer"})  # type: ignore[arg-type]


def test_mock_provider_rejects_non_str_response_value() -> None:
    with pytest.raises(TypeError):
        MockProvider(responses={"q": 42})  # type: ignore[arg-type]


def test_mock_provider_rejects_non_str_default_response() -> None:
    with pytest.raises(TypeError):
        MockProvider(default_response=99)  # type: ignore[arg-type]


def test_mock_provider_rejects_invalid_scenario_type() -> None:
    with pytest.raises(TypeError):
        MockProvider(scenario="bad")  # type: ignore[arg-type]


def test_mock_provider_rejects_bool_latency() -> None:
    with pytest.raises(TypeError):
        MockProvider(latency_seconds=True)  # type: ignore[arg-type]


def test_mock_provider_rejects_negative_latency() -> None:
    with pytest.raises(ValueError):
        MockProvider(latency_seconds=-0.1)


# ---------------------------------------------------------------------------
# Determinism / repeatability
# ---------------------------------------------------------------------------


def test_default_response_is_returned_when_no_mapping() -> None:
    provider = MockProvider(default_response="static answer")
    result = provider.generate(_payload("anything"))
    assert result.text == "static answer"


def test_response_is_deterministic_for_same_payload() -> None:
    provider = MockProvider(responses={"ping": "pong"})
    p = _payload("ping")
    first = provider.generate(p)
    second = provider.generate(p)
    assert first.text == second.text == "pong"
    assert first.usage == second.usage


def test_mapped_response_returned_for_exact_key_match() -> None:
    provider = MockProvider(responses={"hello": "world", "foo": "bar"})
    assert provider.generate(_payload("hello")).text == "world"
    assert provider.generate(_payload("foo")).text == "bar"


def test_fallback_to_default_for_unrecognised_key() -> None:
    provider = MockProvider(
        responses={"known": "yes"},
        default_response="fallback",
    )
    result = provider.generate(_payload("unknown"))
    assert result.text == "fallback"


def test_response_uses_last_user_message() -> None:
    provider = MockProvider(responses={"second": "found it"})
    payload = LLMPayload(
        model="test-model",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
    )
    assert provider.generate(payload).text == "found it"


def test_token_counts_are_deterministic_and_word_based() -> None:
    provider = MockProvider(responses={"how many words": "three words here"})
    result = provider.generate(_payload("how many words"))
    assert result.usage["prompt_tokens"] == 3        # "how many words"
    assert result.usage["completion_tokens"] == 3    # "three words here"
    assert result.usage["total_tokens"] == 6


def test_result_is_successful_on_happy_path() -> None:
    provider = MockProvider()
    result = provider.generate(_payload())
    assert result.ok
    assert result.error is None


# ---------------------------------------------------------------------------
# Payload immutability
# ---------------------------------------------------------------------------


def test_generate_does_not_mutate_payload() -> None:
    provider = MockProvider()
    payload = _payload("keep me")
    original = deepcopy(payload)
    provider.generate(payload)
    assert payload == original


# ---------------------------------------------------------------------------
# Request recording
# ---------------------------------------------------------------------------


def test_requests_is_empty_before_any_call() -> None:
    provider = MockProvider()
    assert provider.requests == []


def test_each_call_is_recorded_in_order() -> None:
    provider = MockProvider()
    p1 = _payload("first")
    p2 = _payload("second")
    provider.generate(p1)
    provider.generate(p2)
    recorded = provider.requests
    assert len(recorded) == 2
    assert recorded[0].messages[0]["content"] == "first"
    assert recorded[1].messages[0]["content"] == "second"


def test_recorded_requests_are_deep_copies() -> None:
    provider = MockProvider()
    payload = _payload("original")
    provider.generate(payload)
    payload.messages[0]["content"] = "mutated"
    assert provider.requests[0].messages[0]["content"] == "original"


def test_reset_clears_recorded_requests() -> None:
    provider = MockProvider()
    provider.generate(_payload("a"))
    provider.generate(_payload("b"))
    provider.reset()
    assert provider.requests == []


def test_requests_after_reset_does_not_include_pre_reset_calls() -> None:
    provider = MockProvider()
    provider.generate(_payload("before"))
    provider.reset()
    provider.generate(_payload("after"))
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[0]["content"] == "after"


# ---------------------------------------------------------------------------
# Failure scenarios
# ---------------------------------------------------------------------------


def test_error_scenario_returns_error_result() -> None:
    provider = MockProvider(scenario=MockScenario("rate limit"))
    result = provider.generate(_payload())
    assert not result.ok
    assert result.error == "rate limit"
    assert result.text == ""
    assert result.usage == {}


def test_error_scenario_still_records_request() -> None:
    provider = MockProvider(scenario=MockScenario("timeout"))
    provider.generate(_payload("ping"))
    assert len(provider.requests) == 1


def test_error_scenario_overrides_response_mapping() -> None:
    provider = MockProvider(
        responses={"hello": "world"},
        scenario=MockScenario("forced error"),
    )
    result = provider.generate(_payload("hello"))
    assert not result.ok
    assert result.error == "forced error"


# ---------------------------------------------------------------------------
# Latency simulation
# ---------------------------------------------------------------------------


def test_zero_latency_returns_immediately() -> None:
    provider = MockProvider(latency_seconds=0.0)
    start = time.monotonic()
    provider.generate(_payload())
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # generous upper bound


def test_latency_delays_response_by_at_least_configured_amount() -> None:
    delay = 0.1  # 100 ms — comfortably above Windows ~15 ms timer resolution
    provider = MockProvider(latency_seconds=delay)
    start = time.monotonic()
    provider.generate(_payload())
    elapsed = time.monotonic() - start
    # Allow 10 % tolerance for OS timer jitter.
    assert elapsed >= delay * 0.9


def test_latency_applies_to_error_scenarios_too() -> None:
    delay = 0.1  # 100 ms — comfortably above Windows ~15 ms timer resolution
    provider = MockProvider(
        latency_seconds=delay,
        scenario=MockScenario("slow failure"),
    )
    start = time.monotonic()
    result = provider.generate(_payload())
    elapsed = time.monotonic() - start
    assert not result.ok
    # Allow 10 % tolerance for OS timer jitter.
    assert elapsed >= delay * 0.9


# ---------------------------------------------------------------------------
# LLMCore
# ---------------------------------------------------------------------------


def test_llm_core_rejects_non_provider() -> None:
    with pytest.raises(TypeError):
        LLMCore(provider="not a provider")  # type: ignore[arg-type]


def test_llm_core_exposes_provider() -> None:
    mock = MockProvider()
    core = LLMCore(provider=mock)
    assert core.provider is mock


def test_llm_core_generate_delegates_to_provider() -> None:
    mock = MockProvider(default_response="core answer")
    core = LLMCore(provider=mock)
    result = core.generate(_payload())
    assert result.text == "core answer"


def test_llm_core_records_requests_via_provider() -> None:
    mock = MockProvider()
    core = LLMCore(provider=mock)
    core.generate(_payload("recorded"))
    assert len(mock.requests) == 1
    assert mock.requests[0].messages[0]["content"] == "recorded"


def test_llm_core_from_provider_name_mock() -> None:
    core = LLMCore.from_provider_name("mock")
    assert isinstance(core.provider, MockProvider)


def test_llm_core_from_provider_name_mock_case_insensitive() -> None:
    core = LLMCore.from_provider_name("MOCK")
    assert isinstance(core.provider, MockProvider)


def test_llm_core_from_provider_name_mock_with_kwargs() -> None:
    core = LLMCore.from_provider_name("mock", default_response="custom")
    result = core.generate(_payload())
    assert result.text == "custom"


def test_llm_core_from_provider_name_unknown_raises() -> None:
    with pytest.raises(ValueError):
        LLMCore.from_provider_name("unknown_provider")
