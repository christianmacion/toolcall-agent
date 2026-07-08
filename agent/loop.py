"""
The agent loop: a small ReAct-style controller over the local tools.

Two run modes share ONE loop skeleton (budget, retry-with-backoff, loop guard,
tracing):

  offline (default) -> a deterministic stdlib planner decides each tool call.
                       No key. No `anthropic` import on this path.
  live              -> Claude native tool-use (claude-haiku-4-5) decides each
                       tool call. Lazy-imports `anthropic`, gated on
                       ANTHROPIC_API_KEY.

Resilience features visible in the trace:
  * max-step budget          : the loop never runs unbounded.
  * retry-with-backoff       : tool errors (incl. injected faults) are retried
                               with exponential backoff before giving up.
  * loop-detection guard     : if the planner asks for the exact same
                               (tool, args) twice in a row with no progress, we
                               break instead of spinning.

Pure stdlib on the offline path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import tools as toolmod
from .faults import FaultInjector, InjectedToolError
from .planner_offline import plan as offline_plan
from .trace import Tracer


@dataclass
class RunResult:
    task_id: str
    answer: str | None
    success: bool
    tool_sequence: list[str] = field(default_factory=list)
    arg_sequence: list[dict] = field(default_factory=list)
    steps: int = 0
    recovered_failures: int = 0
    injected_failures: int = 0
    tracer: Tracer | None = None
    trace_path: str | None = None
    note: str = ""


def _execute_with_retry(name, args, injector, span, *, max_retries=3,
                        base_delay=0.0):
    """Run a tool, retrying with exponential backoff on (injected) failure.

    Returns (result_str, is_error, retries_used, injected_seen). `base_delay`
    is 0.0 by default so the eval runs fast; the backoff schedule is still
    computed and recorded so the mechanism is real and visible.
    """
    call_key = f"{name}:{sorted((args or {}).items())}"
    injected_seen = 0
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            injector.maybe_fail(call_key)
            result = toolmod.run_tool(name, args)
            return result, False, attempt, injected_seen
        except InjectedToolError as exc:
            injected_seen += 1
            last_err = str(exc)
        except toolmod.ToolError as exc:
            last_err = str(exc)
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            if delay:
                time.sleep(delay)
    return f"ERROR: {last_err}", True, max_retries, injected_seen


def run_task_offline(task: dict, *, inject: bool = False, fail_rate: float = 1.0,
                     seed: int = 1337, max_steps: int = 8,
                     base_delay: float = 0.0, persist: bool = True) -> RunResult:
    """Run one task with the deterministic offline planner.

    `persist=False` keeps the spans in memory without writing a JSONL file —
    used by the scorer so a scoring pass doesn't litter traces/replay/.
    """
    task_id = task.get("id", "task")
    task_text = task["task"]
    tracer = Tracer(mode="offline", model="offline-planner-v1")
    injector = FaultInjector(enabled=inject, fail_rate=fail_rate, seed=seed)

    history: list[tuple] = []  # (tool, args, result)
    tool_seq, arg_seq = [], []
    recovered = injected_total = 0
    last_call = None
    answer = None

    for _ in range(max_steps):
        step = offline_plan(task_text, history)
        if step.done:
            answer = step.answer
            break

        call_sig = (step.tool, tuple(sorted((step.args or {}).items())))
        if call_sig == last_call:
            # loop-detection guard: planner repeated itself with no progress.
            with tracer.span() as sp:
                sp.record(thought="Loop guard: repeated call with no progress; "
                          "stopping.", tool=step.tool, args=step.args,
                          result=None, is_error=True)
            tracer.add(sp)
            answer = answer or "(stopped: loop detected)"
            break
        last_call = call_sig

        with tracer.span() as sp:
            result, is_error, retries, injected = _execute_with_retry(
                step.tool, step.args, injector, sp, base_delay=base_delay)
            sp.record(thought=step.thought, tool=step.tool, args=step.args,
                      result=result, is_error=is_error, retry=retries)
        tracer.add(sp)

        injected_total += injected
        if injected and not is_error:
            recovered += 1  # a fault was injected but the retry recovered it
        tool_seq.append(step.tool)
        arg_seq.append(step.args)
        history.append((step.tool, step.args, result))

    success = answer is not None and not str(answer).startswith("(stopped")
    trace_path = tracer.write(task_id=task_id, replay=True) if persist else None

    return RunResult(
        task_id=task_id, answer=answer, success=success,
        tool_sequence=tool_seq, arg_sequence=arg_seq, steps=len(tool_seq),
        recovered_failures=recovered, injected_failures=injected_total,
        tracer=tracer, trace_path=trace_path,
    )


def run_task_live(task: dict, *, inject: bool = False, fail_rate: float = 0.6,
                  seed: int = 1337, max_steps: int = 8,
                  base_delay: float = 0.2,
                  model: str = "claude-haiku-4-5-20251001") -> RunResult:
    """Run one task with Claude native tool-use.

    Lazy-imports `anthropic` so the offline path never touches it. Requires
    ANTHROPIC_API_KEY in the environment.
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "live mode requires ANTHROPIC_API_KEY; use offline mode for no-key runs")
    import anthropic  # lazy — only imported in live mode

    client = anthropic.Anthropic()
    task_id = task.get("id", "task")
    tracer = Tracer(mode="live", model=model)
    injector = FaultInjector(enabled=inject, fail_rate=fail_rate, seed=seed)

    messages = [{"role": "user", "content": task["task"]}]
    history_sig: list[tuple] = []
    tool_seq, arg_seq = [], []
    recovered = injected_total = 0
    last_call = None
    answer = None

    for _ in range(max_steps):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=toolmod.TOOL_SCHEMAS,
            messages=messages,
        )
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens

        if resp.stop_reason != "tool_use":
            answer = next((b.text for b in resp.content if b.type == "text"), "")
            with tracer.span() as sp:
                sp.record(thought="Model produced a final answer.", tool=None,
                          args=None, result=answer, input_tokens=in_tok,
                          output_tokens=out_tok)
            tracer.add(sp)
            break

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            name, args = block.name, block.input
            call_sig = (name, tuple(sorted((args or {}).items())))
            if call_sig == last_call:
                with tracer.span() as sp:
                    sp.record(thought="Loop guard: repeated tool call; stopping.",
                              tool=name, args=args, result=None, is_error=True,
                              input_tokens=in_tok, output_tokens=out_tok)
                tracer.add(sp)
                answer = answer or "(stopped: loop detected)"
                tool_results.append({"type": "tool_result",
                                     "tool_use_id": block.id,
                                     "content": "stopped"})
                continue
            last_call = call_sig

            with tracer.span() as sp:
                result, is_error, retries, injected = _execute_with_retry(
                    name, args, injector, sp, base_delay=base_delay)
                sp.record(thought="Model requested a tool call.", tool=name,
                          args=args, result=result, is_error=is_error,
                          retry=retries, input_tokens=in_tok,
                          output_tokens=out_tok)
            tracer.add(sp)

            injected_total += injected
            if injected and not is_error:
                recovered += 1
            tool_seq.append(name)
            arg_seq.append(args)
            history_sig.append(call_sig)
            tool_results.append({"type": "tool_result",
                                 "tool_use_id": block.id,
                                 "content": result, "is_error": is_error})
        messages.append({"role": "user", "content": tool_results})
        if answer:
            break

    success = answer is not None and not str(answer).startswith("(stopped")
    trace_path = tracer.write(task_id=task_id, replay=False)
    return RunResult(
        task_id=task_id, answer=answer, success=success,
        tool_sequence=tool_seq, arg_sequence=arg_seq, steps=len(tool_seq),
        recovered_failures=recovered, injected_failures=injected_total,
        tracer=tracer, trace_path=trace_path,
    )
