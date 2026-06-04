// Browser-side hotkey handler — replaces the global pynput keyboard
// listener for GUI sessions.
//
// Loads bindings from /api/hotkeys/bindings on startup (and again
// after the Settings page saves), captures `keydown` events with
// proper focus-awareness (events only fire when the GUI tab is
// focused; minimised browser → no events; other app focused → no
// events), and POSTs the bound action(s) to `/api/run/control` —
// the same endpoint the GUI's Next Episode and Rerecord buttons
// already use.
//
// Active only when:
//   - The Run tab is the current tab (other tabs let the user type
//     into search boxes without firing record-loop actions).
//   - A subprocess is running (no point firing hotkeys when there's
//     no orchestrator listening).
//   - The keydown target isn't a form field (input / textarea /
//     contenteditable) — typing in a focused field always wins.
//
// See ``src/lerobot/common/CONTROL_CHANNEL.md`` ("Source location:
// browser vs Python") for the full design.

(function () {
    'use strict';

    // {KeyboardEvent.key: [action_name, ...]}. Populated from
    // /api/hotkeys/bindings on init and on every Save from the
    // Settings page. Empty until the first fetch resolves — until
    // then the handler is a no-op (matches the legacy behaviour
    // where keys did nothing until pynput attached).
    let _keyboardBindings = {};

    async function _loadBindings() {
        try {
            const res = await fetch('/api/hotkeys/bindings');
            if (!res.ok) {
                console.warn('hotkey bindings fetch failed:', res.status);
                return;
            }
            const body = await res.json();
            const map = {};
            for (const row of (body.by_source && body.by_source.keyboard) || []) {
                (map[row.binding] = map[row.binding] || []).push(row.action);
            }
            _keyboardBindings = map;
        } catch (e) {
            console.warn('hotkey bindings fetch threw:', e);
        }
    }

    // Settings page calls this after saving so the new bindings take
    // effect immediately without a page reload.
    window.reloadHotkeyBindings = _loadBindings;

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
                // Don't toast on hotkey errors — 409 ("no active process")
                // is a normal race during stop.
                const err = await res.json().catch(() => ({}));
                console.warn('hotkey send failed:', cmd, err.detail || res.statusText);
            }
        } catch (e) {
            console.warn('hotkey send threw:', cmd, e);
        }
    }

    document.addEventListener('keydown', (event) => {
        if (_isTextInput(event.target)) return;
        if (!_isHotkeyActive()) return;

        const verbs = _keyboardBindings[event.key];
        if (!verbs || !verbs.length) return;

        event.preventDefault();
        for (const cmd of verbs) {
            _send(cmd);
        }
    });

    // Kick off the initial fetch as soon as the page is ready.
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _loadBindings);
    } else {
        _loadBindings();
    }
})();
