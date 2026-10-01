"""A small calculator that evaluates arithmetic without Python eval."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from aios.core.exceptions import ValidationError
from aios.core.ids import ToolID
from aios.core.models import Tool
from aios.tools.tool import BaseTool


class CalculatorTool(BaseTool):
    """Evaluate arithmetic expressions or basic two-operand operations."""

    def __init__(self) -> None:
        super().__init__(
            Tool(
                tool_id=ToolID.generate(),
                name="calculator",
                description="Evaluate a basic arithmetic expression.",
            ),
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    def run(self, **kwargs: Any) -> int | float:
        if set(kwargs) == {"expression"}:
            expression = kwargs["expression"]
            if not expression.strip() or len(expression) > 256:
                raise ValidationError("expression must contain 1 to 256 characters.")
            try:
                result = self._evaluate(ast.parse(expression, mode="eval").body)
            except (SyntaxError, OverflowError, ZeroDivisionError) as exc:
                raise ValidationError(f"Invalid arithmetic expression: {exc}") from exc
            return self._finite_number(result)

        if set(kwargs) == {"operation", "a", "b"}:
            operations = {
                "add": operator.add,
                "subtract": operator.sub,
                "multiply": operator.mul,
                "divide": operator.truediv,
            }
            operation = kwargs["operation"]
            if operation not in operations:
                raise ValidationError(f"Unsupported operation: {operation}")
            try:
                return self._finite_number(
                    operations[operation](kwargs["a"], kwargs["b"])
                )
            except (OverflowError, ZeroDivisionError) as exc:
                raise ValidationError(f"Calculator operation failed: {exc}") from exc

        raise ValidationError(
            "Provide either expression, or operation together with a and b."
        )

    @classmethod
    def _evaluate(cls, node: ast.expr) -> int | float:
        binary_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        unary_operations = {ast.UAdd: operator.pos, ast.USub: operator.neg}

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operations:
            left = cls._evaluate(node.left)
            right = cls._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValidationError("Exponent magnitude cannot exceed 100.")
            return binary_operations[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operations:
            return unary_operations[type(node.op)](cls._evaluate(node.operand))
        raise ValidationError("Expression contains an unsupported operation.")

    @staticmethod
    def _finite_number(value: Any) -> int | float:
        if type(value) is int:
            return value
        if type(value) is not float or not math.isfinite(value):
            raise ValidationError("Calculator result must be a finite number.")
        return value