from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all LLM provider failures."""


class ProviderCredentialsError(ProviderError):
    """Raised when a provider's required credentials are missing or invalid.

    This exception is raised at generate() time, never at import time or
    instantiation time, allowing safe initialization across environments.
    """


class ProviderNetworkError(ProviderError):
    """Raised or captured when network operations to a provider endpoint fail."""


class ProviderResponseError(ProviderError):
    """Raised or captured when a provider returns an invalid or unparseable response."""
