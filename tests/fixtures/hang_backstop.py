#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hand a test the signal timeout could not end to pytest-timeout's thread method.

pyproject.toml asks for `timeout_method = "signal"`: the stuck test fails at the
line it hung on, every thread's stack is printed, and the run carries on. Two
kinds of hang never let that happen. A wait inside an asyncio event loop
swallows the failure -- pytest's Failed is a BaseException, and asyncio's
callback runner catches every BaseException but SystemExit and
KeyboardInterrupt, logs it and keeps going -- and every Playwright call waits in
one. A thread blocked in native code never runs the handler at all. Either way
the test held its worker until the CI job's own limit, and nothing said which
test it was.

A test still running a grace period after its timeout is taken to be one of
those, and ends the way the thread method ends one: capture suspended, every
thread's stack dumped, the process exited. Under xdist that reads as "worker
crashed while running <test>". It stands down wherever pytest-timeout does --
no timeout, or a debugger attached.
"""

import threading

import pytest

try:
    from pytest_timeout import Settings, is_debugging, timeout_timer
except ImportError:  # the backstop extends pytest-timeout; without it there is nothing to extend
    timeout_timer = None

# Long enough for a test the signal did fail to unwind through its teardown,
# which for a GUI test includes shutting its server down.
GRACE_S = 60.0


def _timeout_for(item: pytest.Item) -> float:
    """The timeout pytest-timeout applies to this item: marker, then option, then ini."""
    marker = item.get_closest_marker("timeout")
    if marker is not None:
        value = marker.args[0] if marker.args else marker.kwargs.get("timeout")
    else:
        value = item.config.getoption("timeout", None)
        if value is None:
            value = item.config.getini("timeout")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    timeout = _timeout_for(item) if timeout_timer is not None else 0.0
    backstop = None
    if timeout > 0 and not is_debugging():
        settings = Settings(
            timeout=timeout, method="thread", func_only=False, disable_debugger_detection=False
        )
        backstop = threading.Timer(timeout + GRACE_S, timeout_timer, args=(item, settings))
        backstop.name = "timeout-backstop"
        backstop.daemon = True
        backstop.start()
    try:
        yield
    finally:
        if backstop is not None:
            # cancel() only asks the timer to stop. Waiting for it keeps the
            # backstop invisible to the next test: one still exiting there shows
            # up in any thread count taken as that test starts.
            backstop.cancel()
            backstop.join()
