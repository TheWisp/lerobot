# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The wait the browser suites are built on, tested without a browser.

Every flaky failure in those suites has been one shape: a wait expressed as a
number of seconds, which on a loaded four-vCPU runner decoding video in
software measures the runner rather than the product. `wait_while_decoding`
waits on the player's own progress instead, so the cases that matter are that
a slow run waits longer instead of failing, and that a stopped one is still
reported promptly. Both are driven here against a scripted page, so they are
deterministic and cost no decoding.
"""

from __future__ import annotations

import pytest

from tests.gui import chunk_fixtures


class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


class _Page:
    """A page that answers the condition and the progress from a script.

    Each step is `(condition, progress)`, consumed one per poll; the last step
    repeats forever, which is how a stalled player behaves.
    """

    def __init__(self, steps, clock):
        self.steps = list(steps)
        self.clock = clock
        self.polls = 0

    def _step(self):
        return self.steps[min(self.polls, len(self.steps) - 1)]

    def evaluate(self, expression, arg=None):
        if expression == chunk_fixtures._PROGRESS:
            return self._step()[1]
        if expression.startswith("() => (window.__chunkPlayer"):
            return None  # the evidence dump
        return self._step()[0]

    def wait_for_timeout(self, ms):
        self.polls += 1
        self.clock.t += ms / 1000.0


class _Media:
    def dump(self):
        return "(no media)"


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(chunk_fixtures, "time", c)
    return c


def test_a_slow_decode_waits_longer_rather_than_failing(clock):
    """The property the whole change exists for.

    Progress creeps for far longer than any deadline these suites used to
    carry, and the wait still returns when the condition comes true.
    """
    steps = [(False, [n, 0]) for n in range(400)] + [(True, [400, 1])]
    page = _Page(steps, clock)

    chunk_fixtures.wait_while_decoding(page, _Media(), "() => cond", "never seen", quiet_s=25.0)

    assert clock.t > 90, f"the run has to outlast the old budgets to prove anything, got {clock.t}s"


def test_a_player_that_stops_getting_anywhere_is_reported_within_the_quiet_window(clock):
    page = _Page([(False, [7, 1])], clock)  # frozen from the first poll

    with pytest.raises(AssertionError, match="stopped getting anywhere"):
        chunk_fixtures.wait_while_decoding(page, _Media(), "() => cond", "the tab never painted", quiet_s=5.0)

    assert clock.t < 30, f"a stalled player must be reported promptly, took {clock.t}s"


def test_progress_that_resumes_clears_the_quiet_window(clock):
    """A pause inside the window is not a stall; only silence to the end is."""
    steps = [(False, [1, 0])] * 12 + [(False, [2, 0])] * 12 + [(True, [3, 1])]
    page = _Page(steps, clock)

    chunk_fixtures.wait_while_decoding(page, _Media(), "() => cond", "never seen", quiet_s=4.0)


def test_the_cap_ends_a_wait_that_progresses_forever_without_arriving(clock):
    """Progress alone is not success: a player can decode and never satisfy the
    condition, and the run must still end."""
    page = _Page([(False, [n, 0]) for n in range(100_000)], clock)

    with pytest.raises(AssertionError, match="going nowhere at the cap"):
        chunk_fixtures.wait_while_decoding(page, _Media(), "() => cond", "never true", cap_s=60.0)

    assert clock.t >= 60
