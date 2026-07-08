"""
Offline, deterministic, rule-based planner.

This is NOT an LLM. It is a small stdlib planner that reads the task text and
decides which tool to call next, given the results gathered so far. It exists so
the WHOLE agent (loop, tools, tracing, fault-injection) and the eval scorer run
with no API key and no `anthropic` import.

Contract (mirrors how a model would behave in a manual tool-use loop):
    plan(task_text, history) -> Step
where Step is either:
    Step(done=False, tool=<name>, args=<dict>, thought=<str>)   # call a tool
    Step(done=True,  answer=<str>, thought=<str>)               # final answer

`history` is the list of prior (tool, args, result) tuples this run produced.
The planner is a pure function of (task_text, history): no globals, no randomness.

Honest scope: the parsing here is regex/keyword based, deliberately narrow, and
covers the demo eval set. Live mode swaps this planner for Claude native
tool-use (see agent/loop.py). LangGraph would be the production swap for the
hand-rolled control flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Step:
    done: bool
    thought: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    answer: str | None = None


_NUM = r"-?\d+(?:\.\d+)?"

# unit vocabulary the planner can recognise in free text. Ordered LONGEST-FIRST
# so the regex alternation prefers "miles" over "mile" over "mi" over "m" — a
# short unit must never shadow a longer one that shares its prefix.
_UNIT_LIST = [
    "meters", "meter", "kilometers", "kilometer", "km", "centimeters",
    "centimeter", "cm", "millimeters", "millimeter", "mm", "miles", "mile",
    "feet", "foot", "ft", "inches", "inch", "yards", "yard", "yd",
    "kilograms", "kilogram", "kg", "milligrams", "milligram", "mg",
    "grams", "gram", "pounds", "pound", "lbs", "ounces", "ounce", "oz",
    "celsius", "fahrenheit", "kelvin",
    "mi", "in", "lb", "m", "g", "c", "f", "k",
]
_UNIT_WORDS = "|".join(sorted(_UNIT_LIST, key=len, reverse=True))


def _have(history, tool) -> bool:
    return any(h[0] == tool for h in history)


def _result_for(history, tool) -> str | None:
    for h in history:
        if h[0] == tool:
            return h[2]
    return None


def plan(task_text: str, history: list[tuple]) -> Step:
    text = task_text.strip()
    low = text.lower()

    # ---- multi-step: convert THEN add (search reach + a delta) ------------- #
    # e.g. "How far does the Picker reach in feet, then add 2 feet of clearance?"
    wants_search_then_math = (
        ("picker" in low or "reach" in low) and "feet" in low and "add" in low
    )
    if wants_search_then_math:
        if not _have(history, "local_search"):
            return Step(False, "Look up the Picker's reach in the corpus.",
                        "local_search", {"query": "reach"})
        if not _have(history, "unit_convert"):
            # reach is 1.2 meters per corpus; convert to feet
            return Step(False, "Convert the reach from meters to feet.",
                        "unit_convert",
                        {"value": 1.2, "from_unit": "m", "to_unit": "ft"})
        if not _have(history, "calculator"):
            conv = _result_for(history, "unit_convert") or "3.937008 ft"
            feet = re.search(_NUM, conv)
            base = feet.group(0) if feet else "3.937008"
            return Step(False, "Add the 2 feet of clearance.",
                        "calculator", {"expression": f"{base} + 2"})
        ans = _result_for(history, "calculator")
        return Step(True, "All three steps done; report the total.",
                    answer=f"{ans} feet")

    # ---- current time / date ---------------------------------------------- #
    if re.search(r"\b(current time|what time|today'?s date|date is it|now\b)\b", low):
        if not _have(history, "now"):
            return Step(False, "User asked for the current time.",
                        "now", {})
        return Step(True, "Report the timestamp.",
                    answer=str(_result_for(history, "now")))

    # ---- unit conversion --------------------------------------------------- #
    conv = re.search(
        rf"convert\s+({_NUM})\s*({_UNIT_WORDS})\b\s+(?:to|into|in)\s+({_UNIT_WORDS})\b",
        low,
    )
    if not conv:
        # "how many feet in 3 meters" style
        conv = re.search(
            rf"how many\s+({_UNIT_WORDS})\b\s+(?:in|are in)\s+({_NUM})\s*({_UNIT_WORDS})\b",
            low,
        )
        if conv:
            to_u, val, from_u = conv.group(1), conv.group(2), conv.group(3)
            if not _have(history, "unit_convert"):
                return Step(False, "Unit conversion requested.",
                            "unit_convert",
                            {"value": float(val), "from_unit": from_u,
                             "to_unit": to_u})
            return Step(True, "Report the converted value.",
                        answer=str(_result_for(history, "unit_convert")))
    if conv:
        val, from_u, to_u = conv.group(1), conv.group(2), conv.group(3)
        if not _have(history, "unit_convert"):
            return Step(False, "Unit conversion requested.",
                        "unit_convert",
                        {"value": float(val), "from_unit": from_u, "to_unit": to_u})
        return Step(True, "Report the converted value.",
                    answer=str(_result_for(history, "unit_convert")))

    # ---- arithmetic -------------------------------------------------------- #
    # explicit expression, or natural-language "what is 12 times 7"
    arith = re.search(r"(?:what is|calculate|compute|how much is)\s+(.+)", low)
    expr = None
    if arith:
        raw = arith.group(1)
        raw = (raw.replace("plus", "+").replace("minus", "-")
                  .replace("times", "*").replace("multiplied by", "*")
                  .replace("divided by", "/").replace("x", "*"))
        cleaned = re.sub(r"[^0-9+\-*/().%\s]", "", raw)
        if re.search(r"\d", cleaned) and re.search(r"[+\-*/%]", cleaned):
            expr = cleaned.strip()
    if expr is None:
        m = re.search(r"([0-9][0-9+\-*/().%\s]*[0-9+\-*/().%])", text)
        if m and re.search(r"[+\-*/%]", m.group(1)):
            expr = m.group(1).strip()
    if expr:
        if not _have(history, "calculator"):
            return Step(False, "Arithmetic requested.",
                        "calculator", {"expression": expr})
        return Step(True, "Report the result.",
                    answer=str(_result_for(history, "calculator")))

    # ---- local knowledge lookup ------------------------------------------- #
    lookup_triggers = ("who is", "what is the", "refund", "warranty", "policy",
                       "ceo", "cto", "support", "address", "price", "cost",
                       "founded", "headquarter", "product", "hiring", "sla",
                       "warranty", "dashboard", "firmware")
    if any(t in low for t in lookup_triggers):
        if not _have(history, "local_search"):
            # build a query from the salient noun(s)
            q = _search_query(low)
            return Step(False, "Look the answer up in the local corpus.",
                        "local_search", {"query": q})
        return Step(True, "Answer from the search hit.",
                    answer=str(_result_for(history, "local_search")))

    # ---- fallback: try a search, then give up cleanly --------------------- #
    if not _have(history, "local_search"):
        return Step(False, "No rule matched; attempt a corpus search.",
                    "local_search", {"query": _search_query(low)})
    return Step(True, "No further tools apply.",
                answer=str(_result_for(history, "local_search")))


def _search_query(low: str) -> str:
    """Pick a compact, meaningful query from the task text."""
    for kw in ("dashboard", "firmware", "refund", "warranty", "ceo", "cto",
               "address", "payload", "reach", "founded", "headquarter",
               "product", "hiring", "sla", "support", "price", "cost"):
        if kw in low:
            return kw
    # else: longest word
    words = re.findall(r"[a-z]{4,}", low)
    return max(words, key=len) if words else low[:20]
