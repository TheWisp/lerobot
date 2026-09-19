# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The app's confirm / alert / prompt, driven the way a user drives them.

These replaced ``window.confirm`` and friends. The interesting properties are
not that a box appears -- it is that the promise settles the way the blocking
call it replaced returned, that Escape and the backdrop mean *cancel* rather
than "accept by accident", and that a destructive question does not open with
the destructive button under the Enter key.

``page.on("dialog", ...)`` is registered throughout as a tripwire: any test here
that provoked a real browser dialog would be testing nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

from tests.gui.app_dialogs import OPEN_DIALOG, answer_dialog, dialog_is_open  # noqa: E402

pytestmark = pytest.mark.requires_playwright


@pytest.fixture
def page(gui_page):
    """gui_page, with a tripwire on any browser-native dialog."""
    seen = []
    gui_page.on("dialog", lambda d: (seen.append(d.message), d.dismiss()))
    gui_page.native_dialogs = seen
    yield gui_page
    assert seen == [], f"a browser-native dialog was raised: {seen}"


def _ask(page, expr: str):
    """Start a Dialogs call and hand back a handle to its promise.

    ``void`` on the call itself: evaluate() awaits a returned promise, and this
    one cannot settle until the dialog is answered.
    """
    page.evaluate(f"() => {{ window._answer = {expr}; }}")
    page.wait_for_selector(OPEN_DIALOG, timeout=10_000)


def _result(page):
    return page.evaluate("() => window._answer")


def test_confirm_resolves_true_when_accepted(page):
    _ask(page, "Dialogs.confirm('Proceed?')")
    answer_dialog(page)
    assert _result(page) is True


def test_confirm_resolves_false_when_cancelled(page):
    _ask(page, "Dialogs.confirm('Proceed?')")
    answer_dialog(page, accept=False)
    assert _result(page) is False


def test_escape_cancels(page):
    """`window.confirm` could not be dismissed by accident into a yes, and
    neither may this: Escape is the same answer as the Cancel button."""
    _ask(page, "Dialogs.confirm('Proceed?')")
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is False


def test_clicking_the_backdrop_cancels(page):
    """The dialog carries no padding of its own precisely so that a click
    landing on the <dialog> element is a click on the backdrop and nothing
    else -- padding there would make the frame around the content dismiss it."""
    _ask(page, "Dialogs.confirm('Proceed?')")
    page.mouse.click(5, 5)
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is False


def test_a_destructive_confirm_opens_with_the_focus_on_cancel(page):
    """Enter is muscle memory. On a question that deletes something, the key
    that fires by reflex must not be the one that deletes."""
    _ask(page, "Dialogs.confirm('Gone forever', {title: 'Delete?', danger: true})")
    assert page.evaluate("() => document.activeElement.textContent") == "Cancel"
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is False


def test_an_ordinary_confirm_opens_with_the_focus_on_the_affirmative(page):
    _ask(page, "Dialogs.confirm('Carry on?', {confirmLabel: 'Continue'})")
    assert page.evaluate("() => document.activeElement.textContent") == "Continue"
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is True


def test_prompt_returns_the_typed_text(page):
    _ask(page, "Dialogs.prompt('Name:', 'seed')")
    assert page.input_value(f"{OPEN_DIALOG} .app-dialog-input") == "seed", (
        "the default value must be in the field, and selected, the way window.prompt did it"
    )
    answer_dialog(page, text="typed")
    assert _result(page) == "typed"


def test_prompt_returns_null_when_cancelled(page):
    """Distinguishable from an empty answer: `window.prompt` returned "" for
    an emptied field and null for Cancel, and callers branch on that."""
    _ask(page, "Dialogs.prompt('Name:', 'seed')")
    answer_dialog(page, accept=False)
    assert _result(page) is None


def test_prompt_treats_an_emptied_field_as_an_answer(page):
    _ask(page, "Dialogs.prompt('Name:', 'seed')")
    answer_dialog(page, text="")
    assert _result(page) == ""


def test_enter_in_the_prompt_field_accepts(page):
    """The field is inside a form whose first submit button is Cancel, so
    implicit submission would answer the opposite of what was typed."""
    _ask(page, "Dialogs.prompt('Name:', '')")
    page.fill(f"{OPEN_DIALOG} .app-dialog-input", "typed")
    page.keyboard.press("Enter")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) == "typed"


def test_alert_has_one_button_and_settles_when_dismissed(page):
    page.evaluate(
        "() => { window._answer = 'pending'; Dialogs.alert('Told you').then(() => { window._answer = 'done'; }); }"
    )
    page.wait_for_selector(OPEN_DIALOG, timeout=10_000)
    assert page.locator(f"{OPEN_DIALOG} .app-dialog-btn").count() == 1
    answer_dialog(page)
    page.wait_for_function("() => window._answer === 'done'", timeout=5_000)


def test_the_message_is_inserted_as_text_not_markup(page):
    """Nearly every message interpolates a path, a repo id, or a server error
    string. Rendering those as HTML would be a scripting hole on content the
    GUI does not control."""
    _ask(page, "Dialogs.confirm('<img src=x onerror=\"window.__xss=1\">')")
    assert page.locator(f"{OPEN_DIALOG} .app-dialog-message img").count() == 0
    assert page.evaluate("() => window.__xss") is None
    assert "<img" in page.inner_text(f"{OPEN_DIALOG} .app-dialog-message")
    answer_dialog(page)


def test_the_dialog_is_removed_from_the_document_after_answering(page):
    """One left attached per question would accumulate over a session, and
    every `dialog.app-dialog` selector after the first would be ambiguous."""
    _ask(page, "Dialogs.confirm('one')")
    answer_dialog(page)
    _ask(page, "Dialogs.confirm('two')")
    assert page.locator("dialog.app-dialog").count() == 1
    answer_dialog(page)
    assert not dialog_is_open(page)


def test_the_page_keyboard_shortcuts_stand_down_while_a_dialog_is_up(page):
    """`window.confirm` froze the event loop, so the page saw nothing while it
    was up. A modal <dialog> is inert to focus and hit-testing but its events
    still bubble to `document`, where this app keeps its shortcuts -- so Space
    played the episode behind a delete confirm, Delete staged an episode
    deletion, and the arrows moved to a different one.
    """
    # Measured at `document`, which is where every one of the app's shortcut
    # handlers is bound. Stubbing the individual shortcuts instead would pass
    # for the wrong reason here: this fixture has no episode open, so most of
    # them return early and would record nothing either way.
    page.evaluate(
        "() => { window._reached = [];"
        "  document.addEventListener('keydown', (e) => window._reached.push(e.key)); }"
    )
    _ask(page, "Dialogs.confirm('Gone forever', {title: 'Delete?', danger: true})")
    # Space is left to the test below: it legitimately activates the focused
    # button, so it is the one key that SHOULD change what is on screen.
    for key in ("Delete", "ArrowRight", "ArrowLeft", "r"):
        page.keyboard.press(key)
    assert page.evaluate("() => window._reached") == [], (
        "keystrokes reached the page's document-level handlers behind the dialog"
    )
    assert page.locator(OPEN_DIALOG).count() == 1, "a page handler dismissed the dialog"
    answer_dialog(page, accept=False)


def test_space_answers_the_dialog_rather_than_the_page(page):
    """The page's Space handler called preventDefault, which cancelled the
    focused button's own activation -- so the dialog could not be answered from
    the keyboard at all while the episode played behind it."""
    _ask(page, "Dialogs.confirm('Carry on?', {confirmLabel: 'Continue'})")
    page.keyboard.press("Space")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is True


def test_escape_does_not_reach_the_page_behind_the_dialog(page):
    """Declining a question about a frame range must not also clear the range."""
    page.evaluate(
        "() => { window._escapes = 0;"
        "  document.addEventListener('keydown',"
        "    (e) => { if (e.key === 'Escape') window._escapes++; }); }"
    )
    _ask(page, "Dialogs.confirm('Large edit')")
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.app-dialog", state="detached", timeout=5_000)
    assert _result(page) is False
    assert page.evaluate("() => window._escapes") == 0


def test_a_selection_drag_out_of_the_field_does_not_cancel(page):
    """A `click` is delivered to the nearest common ancestor of its mousedown and
    mouseup, so dragging a selection out of the field and releasing on the
    backdrop reports the <dialog> itself. Read as a backdrop click, that threw
    away what had just been typed."""
    _ask(page, "Dialogs.prompt('New folder name:', 'my_dataset_copy')")
    field = page.locator(f"{OPEN_DIALOG} .app-dialog-input").bounding_box()
    dlg = page.locator(OPEN_DIALOG).bounding_box()
    page.mouse.move(field["x"] + field["width"] * 0.6, field["y"] + field["height"] / 2)
    page.mouse.down()
    page.mouse.move(dlg["x"] - 60, field["y"] + field["height"] / 2, steps=10)
    page.mouse.up()
    assert page.locator(OPEN_DIALOG).count() == 1, "the selection drag dismissed the dialog"
    answer_dialog(page, text="kept")
    assert _result(page) == "kept"


def test_a_capture_phase_page_listener_also_stands_down(page):
    """The reason the guard wraps listeners instead of stopping events at the
    dialog's edge. A `document` listener in the capture phase runs *before* the
    event reaches the dialog, so nothing the dialog does can contain it -- and
    this app registers five of them."""
    page.evaluate(
        "() => { window._captured = [];"
        "  document.addEventListener('click',"
        "    (e) => window._captured.push(e.type), true);"
        "  document.addEventListener('mousedown',"
        "    (e) => window._captured.push(e.type), true); }"
    )
    _ask(page, "Dialogs.confirm('Proceed?')")
    page.locator(f"{OPEN_DIALOG} .app-dialog-message").click()
    assert page.evaluate("() => window._captured") == [], "a capture-phase page listener saw the interaction"
    answer_dialog(page)


def test_the_guard_is_not_limited_to_a_list_of_event_types(page):
    """`wheel` and `dblclick` were on no hand-written list, which is the point:
    the guard is around the listener, so what it covers does not depend on
    anyone having thought of the event type in advance."""
    page.evaluate(
        "() => { window._seen = [];"
        "  for (const t of ['wheel', 'dblclick', 'auxclick', 'pointerdown'])"
        "    document.addEventListener(t, (e) => window._seen.push(e.type)); }"
    )
    _ask(page, "Dialogs.confirm('Proceed?')")
    body = page.locator(f"{OPEN_DIALOG} .app-dialog-message")
    body.dblclick()
    page.mouse.wheel(0, 100)
    assert page.evaluate("() => window._seen") == [], "a page listener saw a gated event"
    answer_dialog(page)


def test_the_page_wakes_up_again_once_the_dialog_is_answered(page):
    """Standing down has to be temporary, and it has to survive nesting: a
    question raised while another is up must not wake the page when only the
    inner one is answered."""
    page.evaluate(
        "() => { window._clicks = 0;  document.addEventListener('click', () => { window._clicks += 1; }); }"
    )
    page.evaluate("() => { window._outer = Dialogs.confirm('outer'); }")
    page.wait_for_selector(OPEN_DIALOG, timeout=5_000)
    page.evaluate("() => { window._inner = Dialogs.confirm('inner'); }")
    page.wait_for_function("() => document.querySelectorAll('dialog.app-dialog').length === 2")
    assert page.evaluate("() => Dialogs.isOpen()") is True

    answer_dialog(page)  # the inner one, on top
    assert page.evaluate("() => Dialogs.isOpen()") is True, "one dialog left, page still down"
    answer_dialog(page)
    assert page.evaluate("() => Dialogs.isOpen()") is False

    page.locator("#play-btn").click()
    assert page.evaluate("() => window._clicks") >= 1, "the page never woke up"


def test_removing_a_guarded_listener_still_works(page):
    """The browser holds a wrapper, not the function the caller passed, so a
    removal that is not mapped back silently does nothing -- and the splitter
    drags rely on removing their own document handlers."""
    page.evaluate(
        "() => { window._n = 0;"
        "  window._h = () => { window._n += 1; };"
        "  document.addEventListener('click', window._h); }"
    )
    page.locator("#play-btn").click()
    assert page.evaluate("() => window._n") == 1
    page.evaluate("() => document.removeEventListener('click', window._h)")
    page.locator("#play-btn").click()
    assert page.evaluate("() => window._n") == 1, "the listener was not removed"


def test_the_rule_classifies_event_types_nobody_wrote_down(page):
    """The guard asks the DOM what kind of event it is rather than matching a
    list of type names, so a type this repository has never mentioned is still
    classified. A list would answer "not on it" and let the event through.
    """
    verdicts = page.evaluate(
        """() => {
          const cases = {
            // Starts something -> must stand down.
            keydown: new KeyboardEvent('keydown'),
            auxclick: new MouseEvent('auxclick'),
            dblclick: new MouseEvent('dblclick'),
            wheel: new WheelEvent('wheel'),
            pointerdown: new PointerEvent('pointerdown'),
            // Finishes a gesture already in flight -> must be let through, or a
            // drag interrupted by a dialog never ends.
            mousemove: new MouseEvent('mousemove'),
            pointerup: new PointerEvent('pointerup'),
            mouseleave: new MouseEvent('mouseleave'),
            // Not user input at all -> never gated; a websocket message has to
            // keep arriving while a question is on screen.
            message: new MessageEvent('message'),
            visibilitychange: new Event('visibilitychange'),
          };
          const seen = {};
          const names = Object.keys(cases);
          for (const n of names) document.addEventListener(n, () => { seen[n] = true; });
          return {names, cases: Object.fromEntries(names.map((n) => [n, cases[n]]))};
        }"""
    )
    names = verdicts["names"]
    _ask(page, "Dialogs.confirm('Proceed?')")
    reached = page.evaluate(
        """(names) => {
          const seen = [];
          for (const n of names) {
            const e = n === 'wheel' ? new WheelEvent(n)
                    : n.startsWith('pointer') ? new PointerEvent(n)
                    : n === 'message' ? new MessageEvent(n)
                    : n === 'visibilitychange' ? new Event(n)
                    : n === 'keydown' ? new KeyboardEvent(n)
                    : new MouseEvent(n);
            window.__hit = false;
            const h = () => { window.__hit = true; };
            document.addEventListener(n, h);
            document.dispatchEvent(e);
            document.removeEventListener(n, h);
            if (window.__hit) seen.push(n);
          }
          return seen;
        }""",
        names,
    )
    assert sorted(reached) == ["message", "mouseleave", "mousemove", "pointerup", "visibilitychange"], (
        f"wrong events reached the page while a dialog was up: {sorted(reached)}"
    )
    answer_dialog(page)
