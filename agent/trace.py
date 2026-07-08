"""
Span-based tracing for the agent loop.

Every step the agent takes is recorded as a span: the thought, the tool it
chose, the arguments, the result, latency, and an estimated cost. Attribute
names follow the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) so the traces line up
with standard observability tooling — e.g. `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.tool.name`, `gen_ai.usage.*`.

Traces are written as JSONL (one span per line) under traces/. Offline runs go
to traces/replay/ so the Streamlit timeline renders with no API key.

Pure stdlib.
"""

from __future__ import annotations

import json
import os
import time
import uuid

# Live-mode pricing for claude-haiku-4-5 (USD per 1M tokens), from the
# claude-api skill model table. Offline mode has no model cost (it's a
# deterministic stdlib planner), so we estimate $0 there but still emit the
# attribute so the schema is uniform.
HAIKU_INPUT_PER_MTOK = 1.00
HAIKU_OUTPUT_PER_MTOK = 5.00


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token usage at claude-haiku-4-5 rates."""
    return round(
        input_tokens / 1_000_000 * HAIKU_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * HAIKU_OUTPUT_PER_MTOK,
        8,
    )


class Span:
    """One step of the agent loop, timed on enter/exit."""

    def __init__(self, trace_id: str, index: int, mode: str, model: str):
        self.attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.system": "anthropic" if mode == "live" else "offline-planner",
            "gen_ai.request.model": model,
            "trace.id": trace_id,
            "span.index": index,
            "agent.mode": mode,
        }
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def record(
        self,
        *,
        thought: str,
        tool: str | None,
        args: dict | None,
        result: str | None,
        is_error: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retry: int = 0,
    ):
        self.attrs.update(
            {
                "agent.thought": thought,
                "gen_ai.tool.name": tool,
                "gen_ai.tool.call.arguments": args or {},
                "gen_ai.tool.call.result": result,
                "error": is_error,
                "agent.retry_count": retry,
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "gen_ai.usage.cost_usd": estimate_cost(input_tokens, output_tokens),
            }
        )
        return self

    def __exit__(self, *exc):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 3)
        self.attrs["latency_ms"] = elapsed_ms
        return False


class Tracer:
    """Collects spans for one task run and persists them as a JSONL file."""

    def __init__(self, mode: str = "offline", model: str = "offline-planner-v1"):
        self.trace_id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.model = model
        self.spans: list[dict] = []
        self._index = 0

    def span(self) -> Span:
        s = Span(self.trace_id, self._index, self.mode, self.model)
        self._index += 1
        return s

    def add(self, span: Span):
        self.spans.append(span.attrs)

    def total_cost(self) -> float:
        return round(sum(s.get("gen_ai.usage.cost_usd", 0.0) for s in self.spans), 8)

    def total_latency_ms(self) -> float:
        return round(sum(s.get("latency_ms", 0.0) for s in self.spans), 3)

    def write(self, task_id: str = "task", replay: bool = True) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        subdir = os.path.join("traces", "replay") if replay else "traces"
        out_dir = os.path.join(root, subdir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{task_id}_{self.trace_id}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for span in self.spans:
                fh.write(json.dumps(span) + "\n")
        return path
