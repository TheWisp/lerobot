"""TEMPORARY -- DO NOT MERGE. Proves the timeout backstop names an asyncio hang on CI.

Its loop is almost always inside a callback, as Playwright's is while it waits,
so the signal timeout's Failed lands in the callback and asyncio swallows it.
Only the backstop can end it."""

import asyncio
import time


def test_probe_before():
    pass


def test_probe_asyncio_swallows_the_signal():
    loop = asyncio.new_event_loop()

    def tick():
        time.sleep(0.05)
        loop.call_soon(tick)

    loop.call_soon(tick)
    loop.run_forever()
