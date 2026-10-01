"""Base contract for executable AIOS tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.models import Tool


class BaseTool(ABC):
    """Executable tool backed by the shared AIOS tool metadata model."""

    def __init__(
        self,
        tool: Tool,
        input_schema: dict[str, Any],
    ) -> None:
        if not isinstance(tool, Tool):
            raise ValidationError("tool must be a Tool model.")
        if not isinstance(input_schema, dict):
            raise ValidationError("input_schema must be a dictionary.")

        self.tool = tool
        self.input_schema = input_schema
        parallel_limit = tool.metadata.get("parallel_limit", 1)
        if (
            isinstance(parallel_limit, bool)
            or not isinstance(parallel_limit, int)
            or parallel_limit < 1
        ):
            raise ValidationError("parallel_limit must be a positive integer.")
        self.parallel_limit = parallel_limit

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool using validated keyword arguments."""