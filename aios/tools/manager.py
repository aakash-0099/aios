"""High-level API for registering and executing AIOS tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aios.core.ids import ToolID
from aios.tools.executor import Authorizer, ToolExecutor
from aios.tools.registry import ToolRegistry
from aios.tools.tool import BaseTool


class ToolManager:
    """Manage a tool registry and its execution lifecycle."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        authorizer: Authorizer | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ToolRegistry()
        self.executor = ToolExecutor(self.registry, authorizer)

    def register(self, tool: BaseTool) -> None:
        self.registry.register(tool)

    def unregister(self, name_or_id: str | ToolID) -> BaseTool:
        return self.registry.unregister(name_or_id)

    def get_by_name(self, name: str) -> BaseTool:
        return self.registry.get_by_name(name)

    def get_by_id(self, tool_id: ToolID) -> BaseTool:
        return self.registry.get_by_id(tool_id)

    def list_tools(self) -> list[BaseTool]:
        return self.registry.list_tools()

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.executor.execute(tool_name, arguments, **kwargs)