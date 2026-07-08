"""
toolcall-agent — Streamlit UI.

A reviewer can: pick a task (or type one), toggle fault-injection, click Run,
and watch the agent's span TIMELINE (thought -> tool -> args -> result ->
latency -> cost), the retry/backoff machinery recover from injected failures,
and the raw OpenTelemetry-GenAI JSON. A second tab shows the honest scorecard
over the 20-task eval set.

Runs fully OFFLINE (deterministic stdlib planner) with no API key. If
ANTHROPIC_API_KEY is set, a "live" toggle swaps in Claude native tool-use
(claude-haiku-4-5). Recorded offline runs persist to traces/replay/*.jsonl.
"""

from __future__ import annotations

import glob
import json
import os

import streamlit as st

from agent.loop import run_task_offline, run_task_live
from agent.score import score as run_score

st.set_page_config(page_title="toolcall-agent", page_icon="🔧", layout="wide")

REPO = os.path.dirname(os.path.abspath(__file__))


def load_tasks() -> list[dict]:
    path = os.path.join(REPO, "eval", "tasks.jsonl")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def render_timeline(spans: list[dict]):
    for sp in spans:
        idx = sp.get("span.index", "?")
        tool = sp.get("gen_ai.tool.name") or "(final answer)"
        err = sp.get("error")
        retry = sp.get("agent.retry_count", 0)
        lat = sp.get("latency_ms", 0.0)
        cost = sp.get("gen_ai.usage.cost_usd", 0.0)
        badge = "🟥 error" if err else ("🟧 recovered" if retry else "🟩 ok")
        header = f"Step {idx} · {tool} · {badge} · {lat} ms · ${cost:.6f}"
        with st.expander(header, expanded=True):
            st.markdown(f"**Thought:** {sp.get('agent.thought','')}")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Arguments**")
                st.json(sp.get("gen_ai.tool.call.arguments", {}))
            with cols[1]:
                st.markdown("**Result**")
                st.code(str(sp.get("gen_ai.tool.call.result", "")), language="text")
            meta = {
                "gen_ai.request.model": sp.get("gen_ai.request.model"),
                "agent.retry_count": retry,
                "gen_ai.usage.input_tokens": sp.get("gen_ai.usage.input_tokens"),
                "gen_ai.usage.output_tokens": sp.get("gen_ai.usage.output_tokens"),
            }
            st.caption(" · ".join(f"{k}={v}" for k, v in meta.items()))


# --------------------------------------------------------------------------- #
st.title("🔧 toolcall-agent")
st.caption("A tool-calling agent that recovers and shows its trace. "
           "ReAct-style loop over 4 safe local tools, with retry/backoff, a "
           "loop guard, and a full span trace.")

has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
tab_run, tab_score, tab_traces = st.tabs(
    ["▶ Run a task", "📊 Scorecard", "🗂 Saved traces"])

with tab_run:
    tasks = load_tasks()
    task_labels = {f"{t['id']} — {t['task'][:60]}": t for t in tasks}

    left, right = st.columns([2, 1])
    with left:
        pick = st.selectbox("Pick an eval task", list(task_labels.keys()))
        custom = st.text_input("…or type your own task (overrides the pick)", "")
    with right:
        inject = st.toggle("Inject tool faults", value=True,
                           help="Randomly fail tool calls so you can watch "
                                "retry + backoff recover.")
        mode = st.radio("Mode", ["offline (no key)", "live (Claude)"],
                        index=0, help="Live mode needs ANTHROPIC_API_KEY.")
        if mode.startswith("live") and not has_key:
            st.warning("ANTHROPIC_API_KEY not set — live mode disabled.")

    if st.button("Run", type="primary"):
        chosen = task_labels[pick]
        task = {"id": chosen["id"], "task": custom.strip() or chosen["task"]}
        try:
            if mode.startswith("live") and has_key:
                res = run_task_live(task, inject=inject)
            else:
                res = run_task_offline(task, inject=inject)
        except Exception as exc:  # surface clean errors in the UI
            st.error(f"Run failed: {exc}")
            res = None

        if res is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Answer", str(res.answer)[:18])
            c2.metric("Steps", res.steps)
            c3.metric("Injected faults", res.injected_failures)
            c4.metric("Recovered", res.recovered_failures)
            st.success(f"Final answer: **{res.answer}**"
                       if res.success else f"Stopped: {res.answer}")

            st.subheader("Span timeline")
            render_timeline(res.tracer.spans)

            st.subheader("Raw trace JSON (OpenTelemetry GenAI attributes)")
            st.code("\n".join(json.dumps(s) for s in res.tracer.spans),
                    language="json")
            st.caption(f"Trace written to `{res.trace_path}`")

with tab_score:
    st.markdown("Honest metric, computed by the **offline planner** over the "
                "20-task eval set in `eval/tasks.jsonl`.")
    if st.button("Compute scorecard"):
        sc = run_score(verbose=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tool correctness", f"{sc['tool_correctness']:.0%}")
        c2.metric("Argument correctness", f"{sc['argument_correctness']:.0%}")
        c3.metric("Task completion", f"{sc['task_completion_rate']:.0%}")
        c4, c5 = st.columns(2)
        c4.metric("Error-recovery rate",
                  f"{sc['error_recovery_rate']:.0%}",
                  f"{sc['recovered_failures']}/{sc['injected_failures']} faults")
        c5.metric("Avg steps to completion", sc["avg_steps_to_completion"])
        st.dataframe(sc["rows"], use_container_width=True)
    else:
        path = os.path.join(REPO, "scorecard.json")
        if os.path.exists(path):
            st.info("Showing the last saved scorecard. Click to recompute.")
            st.json(json.load(open(path)))

with tab_traces:
    st.markdown("Recorded runs under `traces/` (offline runs land in "
                "`traces/replay/`).")
    files = sorted(glob.glob(os.path.join(REPO, "traces", "**", "*.jsonl"),
                             recursive=True), reverse=True)
    if not files:
        st.info("No traces yet — run a task first.")
    for f in files[:25]:
        with st.expander(os.path.relpath(f, REPO)):
            spans = [json.loads(line) for line in open(f) if line.strip()]
            render_timeline(spans)

st.divider()
st.caption("Christian Macion — AI / Agent Engineer")
