"""TEMPORARY -- DO NOT MERGE. Proves the pyproject timeout names a hang on CI.

Hangs the way test_concurrent_launches_serialize does: an asyncio wait that
nothing will ever satisfy, with a background thread alive beside it. No
timeout marker, so only [tool.pytest.ini_options] can end it.
"""

import asyncio
import threading


def test_probe_before():
    pass


def test_probe_hangs_on_an_event_nobody_sets():
    threading.Thread(target=threading.Event().wait, name="probe-background-thread", daemon=True).start()

    async def run():
        await asyncio.Event().wait()

    asyncio.run(run())


def test_probe_after():
    pass
