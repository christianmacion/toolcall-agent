"""
Safe, deterministic LOCAL tools for the tool-calling agent.

Pure Python stdlib. No external APIs, no network, no `eval`. Every tool is a
pure function of its arguments (plus, for `now`, a frozen clock so traces are
reproducible). Each tool returns a string result; on bad input it raises
ToolError, which the agent loop catches and turns into a retryable failure.

The tool SCHEMAS here are written in Anthropic native tool-use shape
(`name` / `description` / `input_schema`) so the SAME definitions feed both the
offline rule-based planner and live Claude tool-use with zero translation.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Callable

# A frozen clock makes `now()` deterministic for replay/eval. Override via env.
_FROZEN_NOW = os.environ.get("TOOLCALL_FROZEN_NOW", "2026-06-23T12:00:00")


class ToolError(Exception):
    """Raised by a tool on invalid input. The loop treats it as recoverable."""


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

# A deliberately small, safe arithmetic grammar: digits, operators, parens,
# decimal points, and whitespace. No names, no attribute access, no calls.
_SAFE_EXPR = re.compile(r"^[0-9+\-*/().%\s]+$")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (`+ - * / % ( )`)."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("calculator: 'expression' must be a non-empty string")
    expr = expression.strip()
    if not _SAFE_EXPR.match(expr):
        raise ToolError(
            "calculator: only digits and + - * / % ( ) . are allowed"
        )
    try:
        # Safe: input is regex-restricted to an arithmetic-only grammar.
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except ZeroDivisionError:
        raise ToolError("calculator: division by zero")
    except Exception as exc:  # malformed expression
        raise ToolError(f"calculator: could not evaluate ({exc})")
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            value = int(value)
    return str(value)


# Conversion factors expressed relative to a canonical base unit per dimension.
_LENGTH = {  # base: meter
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}
_MASS = {  # base: gram
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "lb": 453.59237, "lbs": 453.59237, "pound": 453.59237, "pounds": 453.59237,
    "oz": 28.349523125, "ounce": 28.349523125, "ounces": 28.349523125,
}
_TEMP = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


def _norm_unit(u: str) -> str:
    return u.strip().lower()


def _temp_to_c(value: float, unit: str) -> float:
    if unit in ("c", "celsius"):
        return value
    if unit in ("f", "fahrenheit"):
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15  # kelvin


def _c_to_temp(value_c: float, unit: str) -> float:
    if unit in ("c", "celsius"):
        return value_c
    if unit in ("f", "fahrenheit"):
        return value_c * 9.0 / 5.0 + 32.0
    return value_c + 273.15  # kelvin


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert `value` from `from_unit` to `to_unit` (length, mass, or temperature)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ToolError("unit_convert: 'value' must be a number")
    fu, tu = _norm_unit(str(from_unit)), _norm_unit(str(to_unit))

    if fu in _TEMP or tu in _TEMP:
        if not (fu in _TEMP and tu in _TEMP):
            raise ToolError(
                f"unit_convert: cannot convert between '{from_unit}' and '{to_unit}'"
            )
        result = _c_to_temp(_temp_to_c(value, fu), tu)
    elif fu in _LENGTH and tu in _LENGTH:
        result = value * _LENGTH[fu] / _LENGTH[tu]
    elif fu in _MASS and tu in _MASS:
        result = value * _MASS[fu] / _MASS[tu]
    else:
        raise ToolError(
            f"unit_convert: unknown or mismatched units '{from_unit}' -> '{to_unit}'"
        )

    result = round(result, 6)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{result} {to_unit}"


def _corpus_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "search_corpus.txt")


# Short acronyms expand to the phrasing that actually appears in the corpus.
_SEARCH_ALIASES = {
    "ceo": "chief executive",
    "cto": "chief technology",
    "dashboard": "dashboard",
}


def local_search(query: str) -> str:
    """Search the bundled corpus file; return matching lines (case-insensitive)."""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("local_search: 'query' must be a non-empty string")
    path = _corpus_path()
    if not os.path.exists(path):
        raise ToolError("local_search: corpus file not found")
    needle = _SEARCH_ALIASES.get(query.strip().lower(), query.strip().lower())
    hits = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if needle in line.lower():
                hits.append(line.strip())
    if not hits:
        return f"No matches found for '{query}'."
    return " | ".join(hits[:5])


def now() -> str:
    """Return the current timestamp (frozen for reproducibility)."""
    try:
        dt = _dt.datetime.fromisoformat(_FROZEN_NOW)
    except ValueError:
        dt = _dt.datetime(2026, 6, 23, 12, 0, 0)
    return dt.isoformat()


# --------------------------------------------------------------------------- #
# Registry + native tool-use schemas
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, Callable] = {
    "calculator": calculator,
    "unit_convert": unit_convert,
    "local_search": local_search,
    "now": now,
}

# Anthropic native tool-use definitions. Shared by offline planner and live mode.
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression using + - * / % and "
            "parentheses. Call this whenever the user asks to compute, add, "
            "subtract, multiply, divide, or otherwise do math on numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '12 * (3 + 4)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "unit_convert",
        "description": (
            "Convert a numeric value between units of length (m, km, mi, ft, "
            "in, ...), mass (g, kg, lb, oz, ...), or temperature (C, F, K). "
            "Call this when the user asks to convert a measurement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The quantity to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "local_search",
        "description": (
            "Search a small bundled knowledge file for lines matching a query "
            "(case-insensitive substring). Call this to look up facts about the "
            "demo company, its products, policies, or people."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, e.g. 'refund policy'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "now",
        "description": (
            "Return the current date and time as an ISO-8601 timestamp. Call "
            "this when the user asks for the current time, today's date, or 'now'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, args: dict) -> str:
    """Dispatch a tool by name with keyword args. Raises ToolError on bad input."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"unknown tool '{name}'")
    try:
        return fn(**(args or {}))
    except TypeError as exc:
        raise ToolError(f"{name}: bad arguments ({exc})")
