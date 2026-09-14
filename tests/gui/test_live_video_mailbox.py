"""The mailbox between the tap and each encoder: one value, newest wins.

It is what makes "nothing holds more than one frame" a property of the
pipeline rather than a hope, so the tests are about the handoff's rules, not
about frames.
"""

import threading
import time

from lerobot.gui.live_video.mailbox import Mailbox


def test_take_returns_the_newest_of_what_was_put():
    m = Mailbox()
    m.put("a")
    m.put("b")
    m.put("c")
    assert m.take(timeout=0) == "c"


def test_a_taken_value_is_gone():
    m = Mailbox()
    m.put(1)
    assert m.take(timeout=0) == 1
    assert m.take(timeout=0.01) is None


def test_replaced_values_are_counted_as_dropped():
    m = Mailbox()
    m.put(1)
    m.put(2)
    m.put(3)
    assert m.take(timeout=0) == 3
    assert m.dropped == 2


def test_take_waits_for_a_put_from_another_thread():
    m = Mailbox()
    threading.Timer(0.05, m.put, args=("late",)).start()
    t0 = time.perf_counter()
    assert m.take(timeout=2.0) == "late"
    assert time.perf_counter() - t0 < 1.0


def test_close_wakes_a_waiting_taker_with_nothing():
    m = Mailbox()
    threading.Timer(0.05, m.close).start()
    assert m.take(timeout=2.0) is None
    assert m.closed


def test_an_unread_value_survives_close_but_a_later_put_does_not():
    m = Mailbox()
    m.put("before")
    m.close()
    m.put("after")
    assert m.take(timeout=0) == "before"
    assert m.take(timeout=0) is None


def test_a_slow_taker_sees_only_newer_values_and_ends_on_the_last():
    """Every put is either taken or dropped, taken values only move forward,
    and the last one put is the last one taken."""
    m = Mailbox()
    n = 20000

    def produce():
        for i in range(n):
            m.put(i)
        m.close()

    taken: list[int] = []
    t = threading.Thread(target=produce)
    t.start()
    while True:
        v = m.take(timeout=5.0)
        if v is None:
            break
        taken.append(v)
        time.sleep(0)
    t.join()
    assert taken, "nothing was taken"
    assert taken == sorted(taken)
    assert len(set(taken)) == len(taken)
    assert taken[-1] == n - 1
    assert len(taken) + m.dropped == n
