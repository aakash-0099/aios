"""
LLM core dispatcher.

Provides LLMCore, a thin orchestration layer that holds a concrete Provider
and forwards generate calls to it.  Higher-level components should depend on
LLMCore rather than importing a concrete provider directly so that the
provider can be swapped without touching call sites.
"""

from __future__ import annotations

from .groq import GroqProvider
from .mock import MockLLM, MockProvider  # noqa: F401  (re-exported)
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .provider import Provider
from .request import LLMPayload
from .response import LLMResult


class LLMCore:
    """
    Thin orchestration layer that owns a concrete LLM provider.

    LLMCore keeps call sites decoupled from the concrete provider
    implementation.  Callers construct an instance via the constructor
    (for full control) or via the factory :meth:`from_provider_name` (for
    string-based configuration, e.g. from AIOSSettings).

    Args:
        provider: Any concrete implementation of Provider.
    """

    def __init__(self, provider: Provider) -> None:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider instance")
        self._provider = provider

    @property
    def provider(self) -> Provider:
        """The underlying concrete provider instance."""
        return self._provider

    def generate(self, payload: LLMPayload) -> LLMResult:
        """Forward a generation request to the underlying provider."""
        return self._provider.generate(payload)

    @classmethod
    def from_provider_name(cls, name: str, **kwargs: object) -> "LLMCore":
        """
        Construct an LLMCore from a provider name string.

        This factory mirrors the string-based configuration in AIOSSettings
        (``llm_provider`` field) so that callers never need to import a
        concrete provider directly.

        The supported names mirror exactly the values accepted by
        ``AIOSSettings.llm_provider`` so that string-based configuration and
        this factory cannot drift apart.

        Args:
            name: Provider name (case-insensitive).  Supported: ``'mock'``,
                ``'openai'``, ``'groq'``, ``'ollama'``.
            **kwargs: Forwarded verbatim to the chosen provider constructor.

        Raises:
            ValueError: When *name* does not match a supported provider.

        Note:
            Constructing a real provider never requires credentials; a missing
            key raises ``ProviderCredentialsError`` only when ``generate()``
            is called.
        """
        name = name.strip().lower()
        providers: dict[str, type[Provider]] = {
            "mock": MockLLM,
            "openai": OpenAIProvider,
            "groq": GroqProvider,
            "ollama": OllamaProvider,
        }
        try:
            provider_cls = providers[name]
        except KeyError:
            supported = ", ".join(repr(key) for key in providers)
            raise ValueError(
                f"unknown provider {name!r}; supported: {supported}"
            ) from None
        return cls(provider_cls(**kwargs))  # type: ignore[arg-type]
