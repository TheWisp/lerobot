# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Driving the app's own confirm / alert / prompt from a Playwright test.

The GUI stopped using ``window.confirm`` (see ``static/dialogs.js``), so
``page.on("dialog", ...)`` no longer sees anything -- the question is DOM inside
the page now. Clicking the real button is also a stronger test than stubbing the
call away: it proves the dialog opened, that its affirmative button is reachable,
and that the handler resumed after the answer.
"""

from __future__ import annotations

OPEN_DIALOG = "dialog.app-dialog[open]"


def answer_dialog(page, *, accept: bool = True, text: str | None = None, timeout: int = 10_000) -> None:
    """Answer the app dialog that is open, and wait for that one to go away.

    Pre: a ``dialog.app-dialog`` is open, or becomes open within ``timeout`` ms.
    ``text`` is typed into the field first and is meaningful only for a prompt;
    ``accept=False`` needs a dialog that has a cancel button, which an alert
    does not.

    Post: the dialog answered here is detached. A handler that raises a second
    question -- ``launchRun`` asks about FPS and about the robot back to back --
    has already put it up by then, so the wait is scoped to the element that was
    answered rather than to "no app dialog anywhere", which that case never
    satisfies.
    """
    page.wait_for_selector(OPEN_DIALOG, timeout=timeout)
    # The LAST open dialog, not the first: they stack in the top layer in the
    # order they opened, so the newest is the one on top and the only one a user
    # could reach. Answering `.first` clicks at a button the one above it covers.
    dialog = page.locator(OPEN_DIALOG).last
    # A handle to this exact element, held across the answer. Waiting on the
    # locator instead would prove nothing: its selector carries `[open]`, which
    # stops matching the instant the dialog closes -- one turn before the module
    # takes the element out of the document.
    element = dialog.element_handle()
    if text is not None:
        dialog.locator(".app-dialog-input").fill(text)
    if accept:
        dialog.locator(".app-dialog-btn.primary").click()
    else:
        cancel = dialog.locator(".app-dialog-btn:not(.primary)")
        assert cancel.count() == 1, (
            "accept=False needs a cancel button; an alert is built with only its "
            "primary button, so this call cannot do what it is asking for"
        )
        cancel.click()
    page.wait_for_function("el => !el.isConnected", arg=element, timeout=timeout)


def dialog_is_open(page) -> bool:
    """Whether the app is currently asking something.

    The counterpart to :func:`answer_dialog`, for asserting that a question was
    *not* raised.
    """
    return page.query_selector(OPEN_DIALOG) is not None
