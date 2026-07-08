"""
Thin CLI wrapper so `python eval/score.py` works from the repo root, in
addition to `python -m agent.score`. Adds the repo root to sys.path so the
`agent` package imports cleanly, then delegates to agent.score.score().
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.score import score  # noqa: E402

if __name__ == "__main__":
    score()
