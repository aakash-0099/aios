"""Thread-safe registry for executable AIOS tools."""

from __future__ import annotations

from threading import RLock

from aios.core.exceptions import NotFoundError, ValidationError
from aios.core.ids import ToolID

from aios.tools.tool import BaseTool


class ToolRegistry:
    """Register tools and resolve them by their unique name or ID."""

    def __init__(self) -> None:
        self._by_id: dict[ToolID, BaseTool] = {}
        self._by_name: dict[str, BaseTool] = {}
        self._lock = RLock()

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise ValidationError("tool must be a BaseTool instance.")

        with self._lock:
            if tool.tool.tool_id in self._by_id:
                raise ValidationError(f"Tool ID is already registered: {tool.tool.tool_id}")
            if tool.tool.name in self._by_name:
                raise ValidationError(f"Tool name is already registered: {tool.tool.name}")
            self._by_id[tool.tool.tool_id] = tool
            self._by_name[tool.tool.name] = tool

    def unregister(self, name_or_id: str | ToolID) -> BaseTool:
        with self._lock:
            if isinstance(name_or_id, ToolID):
                tool = self._by_id.get(name_or_id)
            elif isinstance(name_or_id, str):
                tool = self._by_name.get(name_or_id)
            else:
                raise ValidationError("Tool lookup must be a name or ToolID.")
            if tool is None:
                raise NotFoundError(f"Tool not found: {name_or_id}")
            del self._by_id[tool.tool.tool_id]
            del self._by_name[tool.tool.name]
            return tool

    def get_by_name(self, name: str) -> BaseTool:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Tool name must be a non-empty string.")
        with self._lock:
            tool = self._by_name.get(name)
        if tool is None:
            raise NotFoundError(f"Tool not found: {name}")
        return tool

    def get_by_id(self, tool_id: ToolID) -> BaseTool:
        if not isinstance(tool_id, ToolID):
            raise ValidationError("tool_id must be a ToolID.")
        with self._lock:
            tool = self._by_id.get(tool_id)
        if tool is None:
            raise NotFoundError(f"Tool not found: {tool_id}")
        return tool

    def list_tools(self) -> list[BaseTool]:
        with self._lock:
            return list(self._by_name.values())