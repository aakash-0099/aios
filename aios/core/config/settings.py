"""
AIOS configuration.

The goal at this stage is to establish one central configuration
location rather than having every future component independently
read environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..exceptions import ConfigurationError


@dataclass(frozen=True)
class AIOSSettings:
    """
    Shared application settings.

    These are intentionally minimal for Phase 1.
    """

    environment: str = "development"

    debug: bool = True

    # Root logging level used by the future AIOS runtime.
    log_level: str = "INFO"

    queue_backend: str = "memory"
    llm_provider: str = "openai"
    storage_root: str = "./data/storage"
    request_timeout: int = 30
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ConfigurationError("environment must be a non-empty string")
        if type(self.debug) is not bool:
            raise ConfigurationError("debug must be a boolean")
        if not isinstance(self.log_level, str) or self.log_level not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ConfigurationError("log_level must be a standard logging level")
        if not isinstance(self.queue_backend, str) or self.queue_backend not in {
            "memory",
            "redis",
        }:
            raise ConfigurationError("queue backend must be 'memory' or 'redis'")
        if not isinstance(self.llm_provider, str) or self.llm_provider not in {
            "openai",
            "groq",
            "ollama",
            "mock",
        }:
            raise ConfigurationError(
                "llm provider must be 'openai', 'groq', 'ollama', or 'mock'"
            )
        if not isinstance(self.storage_root, str) or not self.storage_root.strip():
            raise ConfigurationError("storage root must be a non-empty string")
        for name, value in (
            ("request_timeout", self.request_timeout),
            ("connect_timeout", self.connect_timeout),
        ):
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")

    @classmethod
    def from_environment(cls) -> "AIOSSettings":
        """
        Build settings from environment variables.

        We keep environment parsing here so that future modules
        don't each implement their own configuration logic.
        """

        environment = os.getenv(
            "AIOS_ENVIRONMENT",
            "development",
        )

        debug_value = os.getenv(
            "AIOS_DEBUG",
            "true",
        ).lower()

        debug = debug_value in {
            "1",
            "true",
            "yes",
            "on",
        }

        log_level = os.getenv(
            "AIOS_LOG_LEVEL",
            "INFO",
        ).upper()

        return cls(
            environment=environment,
            debug=debug,
            log_level=log_level,
        )