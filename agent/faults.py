"""
Fault injection for tool calls.

The reviewer should SEE the retry/backoff machinery actually fire, so we expose
a toggle that randomly fails a tool call before it runs. The agent loop catches
the injected failure, backs off, and retries — and every attempt is traced.

Determinism: the injector is seeded, so an injected run is reproducible. The
default seed is fixed; the eval scorer drives a separate, seeded injector so
the recovery-rate metric is stable across runs.

Pure stdlib.
"""

from __future__ import annotations

import random


class InjectedToolError(Exception):
    """A synthetic, recoverable tool failure (network blip simulation)."""


class FaultInjector:
    """Fails a fraction of tool calls, transiently, in a seeded/reproducible way.

    `fail_rate` is the per-call probability of a transient failure. To keep a
    failure *recoverable* rather than fatal, the injector only fails a given
    (call_key) the first `max_consecutive` times it is seen, then lets it
    through — modelling a blip that clears on retry.
    """

    def __init__(self, enabled: bool = False, fail_rate: float = 0.5,
                 seed: int = 1337, max_consecutive: int = 1):
        self.enabled = enabled
        self.fail_rate = fail_rate
        self.max_consecutive = max_consecutive
        self._rng = random.Random(seed)
        self._fail_counts: dict[str, int] = {}

    def maybe_fail(self, call_key: str) -> None:
        """Raise InjectedToolError if this call is selected to fail transiently."""
        if not self.enabled:
            return
        already = self._fail_counts.get(call_key, 0)
        if already >= self.max_consecutive:
            return  # blip has cleared; let the retry succeed
        if self._rng.random() < self.fail_rate:
            self._fail_counts[call_key] = already + 1
            raise InjectedToolError(
                f"injected transient failure on '{call_key}' (attempt {already + 1})"
            )
