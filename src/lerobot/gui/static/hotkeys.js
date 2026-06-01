// Browser-side hotkey handler — replaces the global pynput keyboard
// listener for GUI sessions.
//
// Captures `keydown` events with proper focus-awareness (events only
// fire when the GUI tab is focused; minimised browser → no events;
// other app focused → no events) and POSTs the bound action(s) to
// `/api/run/control`, the same endpoint the GUI's Next Episode and
// Rerecord buttons already use. The subprocess sees them on stdin via
// its control channel.
//
// First-cut behaviour:
//   - Hotkeys are active ONLY when the Run tab is the current tab AND
//     a subprocess is running. Other tabs (Data, Model, Robot) are
//     unaffected so the user can type into search boxes / forms
//     without triggering rollout actions.
//   - Text inputs are skipped (input / textarea / select / contenteditable)
//     so typing into form fields never fires a hotkey, even when the
//     Run tab is current.
//   - Keyboard bindings are hardcoded here — match the
//     ``init_keyboard_listener`` defaults so the user sees the same
//     behaviour they'd get in the CLI. The hotkeys.json layer
//     (CONTROL_CHANNEL.md P2) will eventually own this map.
//
// See ``src/lerobot/common/CONTROL_CHANNEL.md`` ("Source location:
// browser vs Python") for the full design.

(function () {
    'use strict';

    // KeyboardEvent.key → channel action name(s).
    // Multi-action entries reproduce the legacy compound:
    //   - left arrow fires both rerecord_episode AND exit_early
    //   - esc fires both stop_recording AND exit_early
    // The browser POSTs each verb back-to-back; the subprocess's
    // channel sees them in order.
    const _KEYBOARD_BINDINGS = {
        'ArrowRight': ['exit_early'],
        'ArrowLeft':  ['rerecord_episode', 'exit_early'],
        'Escape':     ['stop_recording', 'exit_early'],
    };

    function _isTextInput(target) {
        if (!target) return false;
        const tag = (target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
        if (target.isContentEditable) return true;
        return false;
    }

    function _isHotkeyActive() {
        // Hotkeys only fire on the Run tab while a subprocess is
        // running. `_isRunning` is the module-level mirror set by
        // ``updateRunUI`` in run.js — covers both launch-side and
        // poll-side state updates.
        const tabRun = document.getElementById('tab-run');
        if (!tabRun || !tabRun.classList.contains('active')) return false;
        return typeof _isRunning !== 'undefined' && _isRunning;
    }

    async function _send(cmd) {
        try {
            const res = await fetch('/api/run/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cmd}),
            });
            if (!res.ok) {
                // Don't toast on hotkey errors — they're high-frequency and
                // a 409 ("no active process") is a normal race during stop.
                const err = await res.json().catch(() => ({}));
                console.warn('hotkey send failed:', cmd, err.detail || res.statusText);
            }
        } catch (e) {
            console.warn('hotkey send threw:', cmd, e);
        }
    }

    document.addEventListener('keydown', (event) => {
        // Don't intercept when the user is typing into a form field.
        if (_isTextInput(event.target)) return;

        // Don't intercept on tabs that aren't the running orchestrator.
        if (!_isHotkeyActive()) return;

        const verbs = _KEYBOARD_BINDINGS[event.key];
        if (!verbs) return;

        event.preventDefault();
        for (const cmd of verbs) {
            _send(cmd);
        }
    });
})();
