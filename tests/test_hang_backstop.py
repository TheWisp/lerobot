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
"""The timeout backstop must be invisible to the test it guards."""

import contextlib
import threading

from tests.fixtures import hang_backstop


def test_the_backstop_thread_is_gone_when_its_test_ends(monkeypatch):
    """A test that counts threads saw one more at its start than at its end on
    CI: the previous test's backstop, cancelled but still exiting. cancel()
    only asks a timer to stop, so the protocol must not finish until it has --
    otherwise every thread count taken early in the next test is off by one,
    and only on a machine slow enough for the exit to lag.

    Checked at the instant the protocol ends rather than by counting in a
    later test, because on a fast machine the window closes in microseconds.
    """
    monkeypatch.setattr(hang_backstop, "_timeout_for", lambda item: 300.0)
    already = {t for t in threading.enumerate() if t.name == "timeout-backstop"}

    protocol = hang_backstop.pytest_runtest_protocol(item=None, nextitem=None)
    next(protocol)
    armed = [t for t in threading.enumerate() if t.name == "timeout-backstop" and t not in already]
    assert len(armed) == 1, armed
    with contextlib.suppress(StopIteration):
        protocol.send(None)

    assert not armed[0].is_alive(), "the backstop outlived the test it guarded"
