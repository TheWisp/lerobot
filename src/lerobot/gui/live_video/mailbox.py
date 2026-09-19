"""A one-value handoff in which a new value replaces an unread one.

The stage after it always takes the newest, and nothing waits behind a slow
value. It can never hold two, which is what makes the pipeline's age a
property of its structure rather than of its timing.
"""

from __future__ import annotations

import threading

_EMPTY = object()


class Mailbox[T]:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: object = _EMPTY
        self._closed = False
        self._dropped = 0

    def put(self, value: T) -> None:
        """Replace whatever is unread. Ignored once closed."""
        with self._cond:
            if self._closed:
                return
            if self._value is not _EMPTY:
                self._dropped += 1
            self._value = value
            self._cond.notify()

    def take(self, timeout: float | None = None) -> T | None:
        """The newest value, waiting up to ``timeout`` seconds for one.

        Postcondition: the mailbox is empty. None means nothing arrived in
        time, or the mailbox is closed and empty.
        """
        with self._cond:
            if self._value is _EMPTY and not self._closed:
                self._cond.wait_for(lambda: self._value is not _EMPTY or self._closed, timeout)
            if self._value is _EMPTY:
                return None
            value, self._value = self._value, _EMPTY
            return value  # type: ignore[return-value]

    def close(self) -> None:
        """Wake any waiting taker; an unread value is still theirs to take."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def dropped(self) -> int:
        """Values replaced before anyone took them."""
        return self._dropped
