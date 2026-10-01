"""
Tests for AIOS configuration.
"""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from aios.config import get_settings
from aios.config.settings import load_config
from aios.core import AIOSSettings, ConfigurationError


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


def test_load_config_uses_sensible_defaults():
    settings = load_config()

    assert settings.environment == "development"
    assert settings.debug is True
    assert settings.log_level == "INFO"
    assert settings.queue_backend == "memory"
    assert settings.llm_provider == "openai"
    assert settings.storage_root == "./data/storage"
    assert settings.request_timeout == 30


def test_load_config_invalid_values_raise_clear_error(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("queue:\n  backend: 123\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="queue backend"):
        load_config(config_path=config_path)


def test_env_overrides_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  provider: openai\nstorage:\n  root: ./yaml-storage\n"
        "timeouts:\n  request: 30\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AIOS_LLM_PROVIDER", "groq")
    monkeypatch.setenv("AIOS_STORAGE_ROOT", "/tmp/aios")
    monkeypatch.setenv("AIOS_REQUEST_TIMEOUT", "45")

    settings = load_config(config_path=config_path)

    assert settings.llm_provider == "groq"
    assert settings.storage_root == "/tmp/aios"
    assert settings.request_timeout == 45


def test_get_settings_is_stable_across_calls(monkeypatch):
    monkeypatch.delenv("AIOS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AIOS_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("AIOS_REQUEST_TIMEOUT", raising=False)

    first = get_settings(force_reload=True)
    second = get_settings()

    assert first is not None
    assert second is first
    assert first == second
    assert first.storage_root == second.storage_root
    assert first.llm_provider == second.llm_provider


def test_settings_are_immutable():
    settings = load_config()

    with pytest.raises(FrozenInstanceError):
        settings.storage_root = "/tmp/changed"


def test_importing_config_does_not_import_runtime_packages():
    script = (
        "import sys; import aios.config; "
        "forbidden = ('aios.kernel', 'aios.scheduler', 'aios.llm', 'aios.storage'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in forbidden)"
    )

    subprocess.run([sys.executable, "-c", script], check=True)