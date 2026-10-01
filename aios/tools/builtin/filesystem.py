"""A filesystem tool restricted to a configured root directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import ToolID
from aios.core.models import Tool
from aios.tools.tool import BaseTool


class FilesystemTool(BaseTool):
    """Read, write, and list paths beneath a configured filesystem root."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).expanduser().resolve()
        super().__init__(
            Tool(
                tool_id=ToolID.generate(),
                name="filesystem",
                description="Read, write, or list files under a configured root.",
            ),
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["action", "path"],
                "additionalProperties": False,
            },
        )

    def run(self, action: str, path: str, content: str | None = None) -> Any:
        if action not in {"read", "write", "list"}:
            raise ValidationError("action must be read, write, or list.")
        if action == "write" and content is None:
            raise ValidationError("content is required for write.")
        if action != "write" and content is not None:
            raise ValidationError("content is only accepted for write.")

        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValidationError("Path must remain inside the filesystem root.")

        if action == "read":
            return target.read_text(encoding="utf-8")
        if action == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
            return {
                "path": str(target.relative_to(self.root)),
                "bytes_written": len((content or "").encode("utf-8")),
            }
        if not target.is_dir():
            raise ValidationError("path must refer to a directory for list.")
        return sorted(child.name for child in target.iterdir())