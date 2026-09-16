"""
AIOS configuration.

The goal at this stage is to establish one central configuration
location rather than having every future component independently
read environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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