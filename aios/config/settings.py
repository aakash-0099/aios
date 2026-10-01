"""Load and cache application settings from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Mapping

import yaml

from aios.core import AIOSSettings, ConfigurationError


_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
_YAML_FIELDS = {
	"environment": "environment",
	"debug": "debug",
	"log_level": "log_level",
	"queue": {"backend": "queue_backend"},
	"llm": {"provider": "llm_provider"},
	"storage": {"root": "storage_root"},
	"timeouts": {
		"request": "request_timeout",
		"connect": "connect_timeout",
	},
}
_ENV_FIELDS = {
	"environment": ("AIOS_ENVIRONMENT",),
	"debug": ("AIOS_DEBUG",),
	"log_level": ("AIOS_LOG_LEVEL",),
	"queue_backend": ("AIOS_QUEUE_BACKEND",),
	"llm_provider": ("AIOS_LLM_PROVIDER",),
	"storage_root": ("AIOS_STORAGE_ROOT",),
	"request_timeout": ("AIOS_TIMEOUTS_REQUEST", "AIOS_REQUEST_TIMEOUT"),
	"connect_timeout": ("AIOS_TIMEOUTS_CONNECT", "AIOS_CONNECT_TIMEOUT"),
}
_BOOLEAN_VALUES = {
	"true": True,
	"1": True,
	"yes": True,
	"on": True,
	"false": False,
	"0": False,
	"no": False,
	"off": False,
}
_settings_cache: AIOSSettings | None = None
_settings_lock = Lock()


def _flatten_config(config: Mapping[object, object]) -> dict[str, object]:
	flattened: dict[str, object] = {}
	for key, value in config.items():
		fields = _YAML_FIELDS.get(key) if isinstance(key, str) else None
		if fields is None:
			raise ConfigurationError(f"unknown configuration section or key: {key}")
		if isinstance(fields, dict):
			if not isinstance(value, Mapping):
				raise ConfigurationError(f"{key} must be a mapping")
			for nested_key, nested_value in value.items():
				attribute = fields.get(nested_key) if isinstance(nested_key, str) else None
				if attribute is None:
					raise ConfigurationError(
						f"unknown configuration key: {key}.{nested_key}"
					)
				flattened[attribute] = nested_value
		else:
			flattened[fields] = value
	return flattened


def _parse_environment_value(name: str, value: str, default: object) -> object:
	if isinstance(default, bool):
		parsed = _BOOLEAN_VALUES.get(value.strip().lower())
		if parsed is None:
			raise ConfigurationError(
				f"{name} must be one of true/false, 1/0, yes/no, or on/off"
			)
		return parsed
	if isinstance(default, int):
		try:
			return int(value)
		except ValueError as error:
			raise ConfigurationError(f"{name} must be an integer") from error
	return value


def _apply_environment(values: dict[str, object]) -> None:
	defaults = AIOSSettings()
	for attribute, names in _ENV_FIELDS.items():
		for name in names:
			value = os.getenv(name)
			if value is not None:
				if attribute == "log_level":
					value = value.upper()
				values[attribute] = _parse_environment_value(
					name, value, getattr(defaults, attribute)
				)
				break


def load_config(config_path: str | Path | None = None) -> AIOSSettings:
	"""Load YAML settings, then apply ``AIOS_*`` environment overrides."""
	path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
	try:
		with path.open(encoding="utf-8") as config_file:
			raw_config = yaml.safe_load(config_file)
	except (OSError, yaml.YAMLError) as error:
		raise ConfigurationError(f"could not load configuration from {path}: {error}") from error

	if raw_config is None:
		raw_config = {}
	if not isinstance(raw_config, Mapping):
		raise ConfigurationError("configuration document must be a mapping")

	values = _flatten_config(raw_config)
	_apply_environment(values)
	try:
		return AIOSSettings(**values)
	except TypeError as error:
		raise ConfigurationError(f"invalid configuration: {error}") from error


def get_settings(*, force_reload: bool = False) -> AIOSSettings:
	"""Return the process-wide immutable settings instance."""
	global _settings_cache
	with _settings_lock:
		if _settings_cache is None or force_reload:
			_settings_cache = load_config()
		return _settings_cache
