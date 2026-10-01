from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from aios.core.exceptions import (
    ExecutionError,
    NotFoundError,
    PermissionError,
    ResourceError,
    ValidationError,
)
from aios.core.ids import ToolID
from aios.core.models import Tool
from aios.tools import BaseTool, ToolManager
from aios.tools.builtin import CalculatorTool, FilesystemTool


class _FailingTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            Tool(ToolID.generate(), "failure", "Always fails."),
            {"type": "object", "properties": {}, "required": []},
        )

    def run(self, **kwargs: Any) -> None:
        raise RuntimeError("failure")


class _BlockingTool(BaseTool):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__(
            Tool(
                ToolID.generate(),
                "blocking",
                "Waits until released.",
                metadata={"parallel_limit": 1},
            ),
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        self.started = started
        self.release = release

    def run(self, **kwargs: Any) -> str:
        self.started.set()
        self.release.wait(timeout=2)
        return "done"


def test_unknown_tool_fails() -> None:
    with pytest.raises(NotFoundError):
        ToolManager().execute("missing")


def test_invalid_arguments_are_rejected() -> None:
    manager = ToolManager()
    manager.register(CalculatorTool())

    with pytest.raises(ValidationError):
        manager.execute("calculator", expression="1 + 2", typo=True)


def test_execution_failure_is_normalized() -> None:
    manager = ToolManager()
    manager.register(_FailingTool())

    with pytest.raises(ExecutionError) as error:
        manager.execute("failure")
    assert isinstance(error.value.__cause__, RuntimeError)


def test_authorizer_can_reject_execution() -> None:
    manager = ToolManager(authorizer=lambda tool, arguments: False)
    manager.register(CalculatorTool())

    with pytest.raises(PermissionError):
        manager.execute("calculator", expression="1 + 2")


def test_calculator_and_filesystem_execute_end_to_end(tmp_path: Path) -> None:
    manager = ToolManager()
    manager.register(CalculatorTool())
    manager.register(FilesystemTool(tmp_path))

    assert manager.execute("calculator", expression="(2 + 3) * 4") == 20
    result = manager.execute(
        "filesystem",
        action="write",
        path="nested/note.txt",
        content="hello",
    )
    assert result["bytes_written"] == 5
    assert manager.execute(
        "filesystem",
        action="read",
        path="nested/note.txt",
    ) == "hello"
    assert manager.execute("filesystem", action="list", path="nested") == ["note.txt"]


def test_parallel_limit_rejects_a_concurrent_execution() -> None:
    started = Event()
    release = Event()
    manager = ToolManager()
    manager.register(_BlockingTool(started, release))
    results: list[Any] = []

    thread = Thread(target=lambda: results.append(manager.execute("blocking")))
    thread.start()
    assert started.wait(timeout=2)
    try:
        with pytest.raises(ResourceError):
            manager.execute("blocking")
    finally:
        release.set()
        thread.join(timeout=2)

    assert results == ["done"]


def test_duplicate_tool_registration_is_rejected() -> None:
    manager = ToolManager()
    manager.register(CalculatorTool())

    with pytest.raises(ValidationError):
        manager.register(CalculatorTool())