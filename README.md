# 🔧 toolcall-agent

**A tool-calling agent that recovers and shows its trace.**

A small ReAct-style agent over 4 safe local tools. It completes multi-step
tasks, recovers from tool failures with retry + backoff, guards against loops,
and renders a full span trace plus an honest tool-call scorecard.

---

## TL;DR + headline metric

On the in-repo 20-task eval set, computed by the offline planner:

> **tool-correctness 1.00 · argument-correctness 1.00 · task-completion 1.00 · recovered from 6/6 injected tool failures · avg 1.2 steps/task**

Every step is a span (`thought → tool → args → result → latency_ms → est_cost`)
emitted with **OpenTelemetry GenAI** attribute names (`gen_ai.tool.name`,
`gen_ai.usage.*`, …) and logged to `traces/*.jsonl`.

Reproduce the metric with one command:

```bash
python3 -m agent.score        # or: python3 eval/score.py
```

It prints the scorecard and writes `scorecard.json`. **No API key needed.**

---

## How a reviewer clicks it

```bash
pip install -r requirements.txt
streamlit run app.py
```

In the UI: pick a task (or type your own) → leave **Inject tool faults** on →
click **Run**. Watch the span timeline render, see the 🟧 *recovered* badges
where the injected failure was retried away, and expand the raw GenAI JSON.
The **Scorecard** tab reproduces the headline metric; the **Saved traces** tab
replays the bundled `traces/replay/*.jsonl` runs (which is why the timeline
renders with no key).

---

## The tools (4 deterministic, local, no network)

| Tool | What it does |
|------|--------------|
| `calculator`   | Evaluates a safe arithmetic expression (`+ - * / % ( )`). |
| `unit_convert` | Length / mass / temperature conversions. |
| `local_search` | Substring search over the bundled `data/search_corpus.txt`. |
| `now`          | Current timestamp (frozen for reproducible traces). |

The agent loop enforces a **max-step budget**, does **retry-with-backoff** on
any tool error (including injected faults), and has a **loop-detection guard**
that stops if the planner repeats a call with no progress.

---

## Offline vs live mode

The project runs two ways behind one loop skeleton:

- **Offline (default, no key).** A deterministic, rule-based **stdlib planner**
  (`agent/planner_offline.py`) reads the task text and decides each tool call.
  The whole agent — loop, tools, tracing, fault-injection, **and the eval
  scorer** — runs with **no API key**, and the offline path never imports
  `anthropic`. Offline runs persist to `traces/replay/*.jsonl`.
- **Live (gated on `ANTHROPIC_API_KEY`).** Swaps in **Claude native tool-use**
  (`claude-haiku-4-5`) to decide tool calls in a manual agentic loop. The same
  tool schemas, tracing, retry/backoff, and loop guard apply. `anthropic` is
  **lazy-imported** only on this path.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # only needed for live mode
```

Live-mode spans carry real `gen_ai.usage.input_tokens` / `output_tokens` and a
cost estimate at claude-haiku-4-5 rates ($1.00 / $5.00 per 1M tok).

---

## The honest metric

`python3 -m agent.score` runs the **offline planner** over `eval/tasks.jsonl`
(20 tasks, each with a gold tool sequence + expected answer) and reports:

- **Tool Correctness** — % of tasks whose tool *sequence* exactly matches gold.
- **Argument Correctness** — % whose arguments match gold on every step.
- **Task Completion rate** — % finished with the expected answer text.
- **Error-recovery rate** — of injected faults, the fraction retry/backoff
  recovered.
- **Avg steps-to-completion** — over completed tasks.

At least one task (`t19`/`t20`) exercises a **3-tool multi-step sequence**
(`local_search → unit_convert → calculator`), and `t20` runs it with faults
injected on every step so the recovery path is measured, not just claimed.

---

## Honest scope

- The **offline planner is deterministic, not an LLM** — narrow regex/keyword
  rules that cover the demo eval set. It exists so the agent + metric run with
  no key and reproduce exactly. It is not a general reasoner.
- **Live mode swaps in Claude tool-use** (`claude-haiku-4-5`) for genuine
  model-driven planning over the identical tool surface.
- The hand-rolled control loop (budget, retry, loop guard) is intentionally
  small and transparent. The production swap for it is **LangGraph** — a
  graph-structured agent runtime with built-in retries, checkpointing, and
  human-in-the-loop, where these same tools and trace attributes would plug in.

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `app.py`.
3. (Optional) add `ANTHROPIC_API_KEY` in the app's **Secrets** to enable live
   mode. Without it the app runs fully offline on the bundled replay traces.

Theme is set in `.streamlit/config.toml` (primary `#0b3d5c`).

---

## Layout

```
app.py                      Streamlit UI (timeline + scorecard + saved traces)
agent/
  loop.py                   ReAct loop: budget, retry/backoff, loop guard
  tools.py                  4 local tools + Anthropic native tool schemas
  planner_offline.py        deterministic stdlib planner (no LLM, no key)
  faults.py                 seeded fault injector (transient, recoverable)
  trace.py                  span tracing w/ OpenTelemetry GenAI attributes
  score.py                  CLI scorer  (python -m agent.score)
eval/
  tasks.jsonl               20 tasks w/ gold tool seq + expected answer
  score.py                  alt CLI entry (python eval/score.py)
traces/replay/*.jsonl       recorded offline runs (UI renders these, no key)
data/search_corpus.txt      bundled corpus for local_search
requirements.txt            streamlit, anthropic (live mode only)
.streamlit/config.toml      theme
```

---

*Christian Macion — AI / Agent Engineer*
