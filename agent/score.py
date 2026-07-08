"""
Honest scorecard for the tool-calling agent.

Runs the OFFLINE deterministic planner over eval/tasks.jsonl and computes:

  * Tool Correctness      : % of tasks whose tool SEQUENCE matches the gold
                            sequence exactly.
  * Argument Correctness  : % of tasks whose arguments match gold on every step.
  * Task Completion rate  : % of tasks the agent finished with a non-empty,
                            non-aborted answer that contains the expected text.
  * Error-recovery rate   : of the injected-fault calls, the fraction the
                            retry/backoff machinery recovered.
  * Avg steps-to-completion (over completed tasks).

Prints a scorecard and writes scorecard.json next to the repo root.

Run:  python -m agent.score        (from the repo root)
  or  python eval/score.py
Everything here is pure stdlib — no key, no `anthropic` import.
"""

from __future__ import annotations

import json
import os

from .loop import run_task_offline


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def load_tasks() -> list[dict]:
    path = os.path.join(_repo_root(), "eval", "tasks.jsonl")
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _args_match(got: list[dict], gold: list[dict]) -> bool:
    if len(got) != len(gold):
        return False
    for g, gd in zip(got, gold):
        # compare on the keys the gold specifies; numeric-aware
        for k, v in gd.items():
            if k not in g:
                return False
            a, b = g[k], v
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                if abs(float(a) - float(b)) > 1e-6:
                    return False
            elif str(a).strip().lower() != str(b).strip().lower():
                return False
    return True


def score(verbose: bool = True) -> dict:
    tasks = load_tasks()
    n = len(tasks)

    tool_correct = arg_correct = completed = answer_correct = 0
    injected_total = recovered_total = 0
    steps_completed = []
    rows = []

    for task in tasks:
        res = run_task_offline(task, inject=bool(task.get("inject", False)),
                               persist=False)

        gold_tools = task.get("gold_tools", [])
        gold_args = task.get("gold_args", [])
        expected = str(task.get("expected_answer", "")).strip().lower()

        tools_ok = res.tool_sequence == gold_tools
        args_ok = tools_ok and _args_match(res.arg_sequence, gold_args)
        ans = (res.answer or "")
        ans_ok = res.success and expected in ans.lower()

        if tools_ok:
            tool_correct += 1
        if args_ok:
            arg_correct += 1
        if res.success:
            completed += 1
            steps_completed.append(res.steps)
        if ans_ok:
            answer_correct += 1

        injected_total += res.injected_failures
        recovered_total += res.recovered_failures

        rows.append({
            "id": res.task_id,
            "tools_ok": tools_ok,
            "args_ok": args_ok,
            "answer_ok": ans_ok,
            "steps": res.steps,
            "injected": res.injected_failures,
            "recovered": res.recovered_failures,
            "answer": ans,
        })

    recovery_rate = (recovered_total / injected_total) if injected_total else 1.0
    avg_steps = (sum(steps_completed) / len(steps_completed)
                 if steps_completed else 0.0)

    scorecard = {
        "n_tasks": n,
        "tool_correctness": round(tool_correct / n, 4),
        "argument_correctness": round(arg_correct / n, 4),
        "task_completion_rate": round(answer_correct / n, 4),
        "error_recovery_rate": round(recovery_rate, 4),
        "injected_failures": injected_total,
        "recovered_failures": recovered_total,
        "avg_steps_to_completion": round(avg_steps, 3),
        "rows": rows,
    }

    out_path = os.path.join(_repo_root(), "scorecard.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)

    if verbose:
        _print_scorecard(scorecard, out_path)
    return scorecard


def _print_scorecard(sc: dict, out_path: str) -> None:
    print("=" * 60)
    print("  TOOLCALL-AGENT  —  OFFLINE SCORECARD")
    print("=" * 60)
    print(f"  Tasks evaluated         : {sc['n_tasks']}")
    print(f"  Tool correctness        : {sc['tool_correctness']:.2%}")
    print(f"  Argument correctness    : {sc['argument_correctness']:.2%}")
    print(f"  Task completion rate    : {sc['task_completion_rate']:.2%}")
    print(f"  Error-recovery rate     : {sc['error_recovery_rate']:.2%} "
          f"({sc['recovered_failures']}/{sc['injected_failures']} injected faults)")
    print(f"  Avg steps to completion : {sc['avg_steps_to_completion']}")
    print("-" * 60)
    print("  per-task:")
    for r in sc["rows"]:
        flag = "ok " if r["answer_ok"] else "FAIL"
        inj = f" inj={r['injected']}/rec={r['recovered']}" if r["injected"] else ""
        print(f"   [{flag}] {r['id']:<22} tools={r['tools_ok']!s:<5} "
              f"args={r['args_ok']!s:<5} steps={r['steps']}{inj}")
    print("-" * 60)
    print(f"  scorecard written to    : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    score()
