"""Validated, authorized execution of registered AIOS tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any

from aios.core.exceptions import (
    AIOSException,
    ExecutionError,
    PermissionError,
    ResourceError,
    ValidationError,
)
from aios.core.ids import ToolID
from aios.tools.registry import ToolRegistry
from aios.tools.tool import BaseTool


Authorizer = Callable[[BaseTool, Mapping[str, Any]], bool]


class ToolExecutor:
    """Resolve, authorize, validate, and execute tools with concurrency limits."""

    def __init__(
        self,
        registry: ToolRegistry,
        authorizer: Authorizer | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise ValidationError("registry must be a ToolRegistry.")
        self.registry = registry
        self.authorizer = authorizer
        self._active: dict[ToolID, int] = {}
        self._conflict_lock = Lock()

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValidationError("arguments must be a mapping.")
        supplied = dict(arguments or {})
        if any(not isinstance(key, str) for key in supplied):
            raise ValidationError("Argument names must be strings.")
        duplicate_keys = supplied.keys() & kwargs.keys()
        if duplicate_keys:
            raise ValidationError(
                f"Arguments supplied more than once: {', '.join(sorted(duplicate_keys))}"
            )
        supplied.update(kwargs)

        tool = self.registry.get_by_name(tool_name)
        if self.authorizer is not None and not self.authorizer(tool, supplied):
            raise PermissionError(f"Execution is not authorized for tool: {tool_name}")

        self._validate_input(tool.input_schema, supplied)
        tool_id = tool.tool.tool_id
        with self._conflict_lock:
            active = self._active.get(tool_id, 0)
            if active >= tool.parallel_limit:
                raise ResourceError(
                    f"Parallel execution limit reached for tool: {tool_name}"
                )
            self._active[tool_id] = active + 1

        try:
            return tool.run(**supplied)
        except AIOSException:
            raise
        except Exception as exc:
            raise ExecutionError(f"Tool execution failed: {tool_name}") from exc
        finally:
            with self._conflict_lock:
                remaining = self._active[tool_id] - 1
                if remaining:
                    self._active[tool_id] = remaining
                else:
                    del self._active[tool_id]

    @classmethod
    def _validate_input(
        cls,
        schema: Mapping[str, Any],
        arguments: Mapping[str, Any],
    ) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ValidationError("Tool input schema is invalid.")
        if any(not isinstance(key, str) for key in properties):
            raise ValidationError("Tool input schema property names must be strings.")
        if any(not isinstance(key, str) for key in required):
            raise ValidationError("Tool input schema required names must be strings.")
        if any(not isinstance(value, Mapping) for value in properties.values()):
            raise ValidationError("Tool input schema property definitions must be mappings.")
        if not isinstance(schema.get("additionalProperties", False), bool):
            raise ValidationError("Tool input schema additionalProperties must be boolean.")

        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValidationError(f"Missing required arguments: {', '.join(missing)}")

        if schema.get("additionalProperties", False) is False:
            unknown = arguments.keys() - properties.keys()
            if unknown:
                raise ValidationError(
                    f"Unexpected arguments: {', '.join(sorted(unknown))}"
                )

        for key, value in arguments.items():
            if key in properties and not cls._matches_type(
                value,
                properties[key].get("type"),
            ):
                expected = properties[key].get("type")
                raise ValidationError(
                    f"Argument '{key}' must have type {expected}."
                )

    @staticmethod
    def _matches_type(value: Any, expected: Any) -> bool:
        type_checks: dict[str, type | tuple[type, ...]] = {
            "array": list,
            "boolean": bool,
            "integer": int,
            "number": (int, float),
            "object": dict,
            "string": str,
        }
        if not isinstance(expected, str):
            raise ValidationError("Tool input schema types must be strings.")
        if expected not in type_checks:
            raise ValidationError(f"Unsupported schema type: {expected}")
        matched = isinstance(value, type_checks[expected])
        if expected in ("integer", "number") and isinstance(value, bool):
            return False
        return matched