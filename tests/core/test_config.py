"""
Tests for AIOS configuration.
"""

from aios.core import AIOSSettings


def test_default_settings():
    settings = AIOSSettings()

    assert settings.environment == "development"
    assert settings.debug is True
    assert settings.log_level == "INFO"


def test_settings_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("AIOS_ENVIRONMENT", "test")
    monkeypatch.setenv("AIOS_DEBUG", "false")
    monkeypatch.setenv("AIOS_LOG_LEVEL", "debug")

    settings = AIOSSettings.from_environment()

    assert settings.environment == "test"
    assert settings.debug is False
    assert settings.log_level == "DEBUG"