"""
tools/calculator.py
-------------------
A safe arithmetic evaluator for the mini AI agent.

Supports: +  -  *  /  **  ()  and floating-point numbers.
Deliberately restricts Python's eval() to a whitelist of
operators and numeric literals so the agent cannot execute
arbitrary code through this tool.
"""

import ast
import operator
from typing import Union

# ── Whitelisted AST node types ──────────────────────────────────────────────
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,      # Python 3.8+ numeric literals
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)

# ── Operator dispatch table ──────────────────────────────────────────────────
_BINARY_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate a whitelisted AST node."""
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants are allowed, got: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    raise ValueError(f"Unhandled node type: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a math expression string and return the result.

    Parameters
    ----------
    expression : str
        A math expression, e.g. ``"(3 + 5) * 2"`` or ``"100 / 4"``.

    Returns
    -------
    str
        A human-readable result string, e.g. ``"Result: 16.0"``
        or an error message if the expression is invalid.

    Examples
    --------
    >>> calculate("2 + 2")
    'Result: 4'
    >>> calculate("10 / 3")
    'Result: 3.3333333333333335'
    >>> calculate("2 ** 8")
    'Result: 256'
    """
    expression = expression.strip()
    if not expression:
        return "Error: Empty expression provided."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        # Return integer-looking floats without the decimal point
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        return f"Result: {result}"
    except ZeroDivisionError as exc:
        return f"Error: {exc}"
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"Error: Could not evaluate expression — {exc}"


# ── Quick smoke-test when run directly ──────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "2 + 2",
        "(3 + 5) * 2",
        "100 / 4",
        "2 ** 10",
        "10 % 3",
        "10 / 0",
        "__import__('os').system('echo hacked')",   # must be rejected
    ]
    for expr in tests:
        print(f"  {expr!r:45s} => {calculate(expr)}")
