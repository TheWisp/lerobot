// Hotkeys settings page — renders the keyboard list, Quest controllers,
// and Gamepad SVGs against /api/hotkeys/{actions,bindings,status}.
//
// The page is a *view* of three things: the action manifest from the
// backend, the current bindings file, and live connection state per
// source. Edits stage locally, then "Save" POSTs the full new set.

(function () {
    'use strict';

    let _actions = [];          // [{name, description, default_keyboard_keys}, ...]
    let _bindings = [];         // [{action, source, binding}, ...] — staged set
    let _serverBindings = [];   // last loaded set (for "Reset to saved")
    let _initialised = false;

    // Buttons that are owned by the Quest VR teleop for continuous-state
    // semantics (clutch / gripper / reset). The UI shows them as
    // "RESERVED" so the user knows they're not configurable here.
    const QUEST_RESERVED = new Set([
        '1',  // grip — clutch
        '0',  // trigger — gripper
        '4',  // A (right) / X (left) — reset ramp
    ]);

    // Configurable buttons per Quest controller. Index → label.
    const QUEST_BUTTONS_LEFT = {
        5: 'Y',
        3: 'joy click',
    };
    const QUEST_BUTTONS_RIGHT = {
        5: 'B',
        3: 'joy click',
    };

    // Generic Xbox-style gamepad button layout (Web Gamepad API indices).
    const GAMEPAD_BUTTONS = {
        0: { label: 'A', cx: 220, cy: 130 },
        1: { label: 'B', cx: 250, cy: 100 },
        2: { label: 'X', cx: 190, cy: 100 },
        3: { label: 'Y', cx: 220, cy: 70 },
        4: { label: 'LB', cx: 70, cy: 25 },
        5: { label: 'RB', cx: 220, cy: 25 },
        12: { label: '↑', cx: 70, cy: 70 },
        13: { label: '↓', cx: 70, cy: 130 },
        14: { label: '←', cx: 40, cy: 100 },
        15: { label: '→', cx: 100, cy: 100 },
    };

    async function _fetchActions() {
        const res = await fetch('/api/hotkeys/actions');
        if (!res.ok) throw new Error('GET /api/hotkeys/actions failed');
        const body = await res.json();
        _actions = body.actions || [];
    }

    async function _fetchBindings() {
        const res = await fetch('/api/hotkeys/bindings');
        if (!res.ok) throw new Error('GET /api/hotkeys/bindings failed');
        const body = await res.json();
        _bindings = (body.flat || []).map(b => ({...b}));
        _serverBindings = JSON.parse(JSON.stringify(_bindings));
    }

    async function _fetchStatus() {
        try {
            const res = await fetch('/api/hotkeys/status');
            const body = await res.json();
            for (const [source, info] of Object.entries(body.sources)) {
                const badge = document.querySelector(`.source-badge[data-source="${source}"]`);
                if (!badge) continue;
                if (info.connected === true) {
                    badge.classList.add('connected');
                    badge.classList.remove('disconnected');
                } else if (info.connected === false) {
                    badge.classList.add('disconnected');
                    badge.classList.remove('connected');
                } else {
                    // Detected-by-client: probe locally.
                    if (source === 'gamepad') {
                        const has = (navigator.getGamepads ? Array.from(navigator.getGamepads()) : []).some(g => g);
                        badge.classList.toggle('connected', has);
                        badge.classList.toggle('disconnected', !has);
                    } else if (source === 'quest') {
                        // Best-effort: WebXR session existence isn't visible from this page
                        // (Quest connects to the teleop's own server, not the GUI). Show
                        // grey until we have a proper signal.
                        badge.classList.add('disconnected');
                    }
                }
            }
        } catch (e) {
            console.warn('hotkey status fetch failed:', e);
        }
    }

    function _bindingsFor(source, binding) {
        return _bindings.filter(b => b.source === source && b.binding === binding).map(b => b.action);
    }

    function _actionsForSource(source) {
        // Group bindings by action for the keyboard list.
        const byAction = {};
        for (const a of _actions) byAction[a.name] = [];
        for (const b of _bindings) {
            if (b.source === source && byAction[b.action] !== undefined) {
                byAction[b.action].push(b.binding);
            }
        }
        return byAction;
    }

    function _renderKeyboard() {
        const container = document.getElementById('hotkey-keyboard-list');
        if (!container) return;
        container.innerHTML = '';
        const grouped = _actionsForSource('keyboard');
        for (const action of _actions) {
            const row = document.createElement('div');
            row.className = 'hotkey-keyboard-row';
            row.dataset.action = action.name;

            const nameEl = document.createElement('div');
            nameEl.className = 'action-name';
            nameEl.textContent = action.name;
            row.appendChild(nameEl);

            const descEl = document.createElement('div');
            descEl.className = 'action-desc';
            descEl.textContent = action.description;
            row.appendChild(descEl);

            const chipsEl = document.createElement('div');
            chipsEl.className = 'key-chips';
            const keys = grouped[action.name] || [];
            for (const key of keys) {
                const chip = document.createElement('span');
                chip.className = 'key-chip';
                chip.innerHTML = `<span>${_keyLabel(key)}</span><span class="remove" title="Remove">×</span>`;
                chip.querySelector('.remove').addEventListener('click', () => {
                    _removeBinding(action.name, 'keyboard', key);
                });
                chipsEl.appendChild(chip);
            }
            const addBtn = document.createElement('button');
            addBtn.className = 'key-add';
            addBtn.textContent = '+ Add key';
            addBtn.addEventListener('click', () => _captureKey(action.name, addBtn));
            chipsEl.appendChild(addBtn);

            row.appendChild(chipsEl);
            container.appendChild(row);
        }
    }

    function _keyLabel(key) {
        const map = { 'ArrowRight': '→', 'ArrowLeft': '←', 'ArrowUp': '↑', 'ArrowDown': '↓', 'Escape': 'esc', ' ': 'space' };
        return map[key] || key;
    }

    function _captureKey(actionName, btn) {
        btn.classList.add('capturing');
        btn.textContent = 'Press a key…';
        const onKey = (event) => {
            event.preventDefault();
            event.stopPropagation();
            document.removeEventListener('keydown', onKey, true);
            btn.classList.remove('capturing');
            btn.textContent = '+ Add key';
            if (event.key === 'Escape' && event.target === document.body) {
                // Allow Escape to cancel capture if it wasn't the intended binding.
                if (!confirm(`Bind Escape to "${actionName}"?`)) return;
            }
            _addBinding(actionName, 'keyboard', event.key);
        };
        document.addEventListener('keydown', onKey, true);
    }

    function _addBinding(action, source, binding) {
        // Don't duplicate.
        if (_bindings.some(b => b.action === action && b.source === source && b.binding === binding)) return;
        _bindings.push({action, source, binding});
        _markDirty();
        _render();
    }

    function _removeBinding(action, source, binding) {
        _bindings = _bindings.filter(b => !(b.action === action && b.source === source && b.binding === binding));
        _markDirty();
        _render();
    }

    // ── Quest controller SVGs ──────────────────────────────────────────

    function _renderQuestSide(hostId, hand, buttonMap) {
        const host = document.getElementById(hostId);
        if (!host) return;
        host.innerHTML = '';
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 180 240');
        svg.setAttribute('width', '180');
        svg.setAttribute('height', '240');

        // Controller body — simplified Touch Plus shape.
        const body = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        body.setAttribute('cx', '90'); body.setAttribute('cy', '130');
        body.setAttribute('rx', '60'); body.setAttribute('ry', '90');
        body.setAttribute('fill', '#10182a'); body.setAttribute('stroke', '#2a3a66'); body.setAttribute('stroke-width', '1.5');
        svg.appendChild(body);

        // Ring on top (tracking ring).
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        ring.setAttribute('cx', '90'); ring.setAttribute('cy', '60');
        ring.setAttribute('rx', '50'); ring.setAttribute('ry', '18');
        ring.setAttribute('fill', 'none'); ring.setAttribute('stroke', '#2a3a66'); ring.setAttribute('stroke-width', '2.5');
        svg.appendChild(ring);

        // Trigger (reserved): drawn as a label on the upper-front of the body.
        _drawQuestButton(svg, hand, '0', 90, 165, 24, 12, 'trigger', true);
        // Grip (reserved): side button.
        _drawQuestButton(svg, hand, '1', hand === 'right' ? 40 : 140, 140, 16, 26, 'grip', true);
        // Reset (A on right, X on left) — reserved.
        _drawQuestButton(svg, hand, '4', hand === 'right' ? 105 : 75, 105, 12, 12, hand === 'right' ? 'A' : 'X', true);
        // Configurable face button (B/Y).
        _drawQuestButton(svg, hand, '5', hand === 'right' ? 75 : 105, 105, 12, 12, buttonMap[5], false);
        // Joystick (configurable for clicks).
        _drawQuestButton(svg, hand, '3', 90, 80, 14, 14, 'joy', false);

        host.appendChild(svg);
    }

    function _drawQuestButton(svg, hand, idx, x, y, w, h, label, reserved) {
        const NS = 'http://www.w3.org/2000/svg';
        const binding = `${hand}:${idx}`;
        const boundActions = _bindingsFor('quest', binding);
        const rect = document.createElementNS(NS, 'rect');
        rect.setAttribute('x', x - w / 2); rect.setAttribute('y', y - h / 2);
        rect.setAttribute('width', w); rect.setAttribute('height', h);
        rect.setAttribute('rx', '3');
        rect.setAttribute('class', 'btn-overlay' + (reserved ? ' reserved' : (boundActions.length ? ' bound' : '')));
        rect.dataset.source = 'quest';
        rect.dataset.binding = binding;
        if (!reserved) {
            rect.addEventListener('click', () => _onSourceButtonClick('quest', binding, label));
        }
        svg.appendChild(rect);

        const lab = document.createElementNS(NS, 'text');
        lab.setAttribute('x', x); lab.setAttribute('y', y - 2);
        lab.setAttribute('class', 'btn-label' + (reserved ? ' reserved' : ''));
        lab.textContent = label;
        svg.appendChild(lab);

        if (!reserved && boundActions.length) {
            const action = document.createElementNS(NS, 'text');
            action.setAttribute('x', x); action.setAttribute('y', y + 8);
            action.setAttribute('class', 'btn-label action');
            action.textContent = boundActions[0];
            svg.appendChild(action);
        }
    }

    // ── Gamepad SVG ────────────────────────────────────────────────────

    function _renderGamepad() {
        const host = document.getElementById('hotkey-gamepad-svg');
        if (!host) return;
        host.innerHTML = '';
        const NS = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(NS, 'svg');
        svg.setAttribute('viewBox', '0 0 320 180');
        svg.setAttribute('width', '480');
        svg.setAttribute('height', '270');

        // Body — Xbox-ish dual grip silhouette.
        const path = document.createElementNS(NS, 'path');
        path.setAttribute('d', 'M40 40 Q20 60 30 110 Q40 165 90 160 Q160 158 230 160 Q280 165 290 110 Q300 60 280 40 Q260 30 230 35 Q160 40 90 35 Q60 30 40 40 Z');
        path.setAttribute('fill', '#10182a'); path.setAttribute('stroke', '#2a3a66'); path.setAttribute('stroke-width', '1.5');
        svg.appendChild(path);

        // LB / RB hint bars on the shoulders.
        for (const [x, label] of [[70, 'LB'], [220, 'RB']]) {
            const sh = document.createElementNS(NS, 'rect');
            sh.setAttribute('x', x - 22); sh.setAttribute('y', 18);
            sh.setAttribute('width', '44'); sh.setAttribute('height', '14');
            sh.setAttribute('rx', '6');
            sh.setAttribute('class', 'btn-overlay');
            sh.dataset.source = 'gamepad'; sh.dataset.binding = (label === 'LB' ? '4' : '5');
            const idx = (label === 'LB' ? '4' : '5');
            const bound = _bindingsFor('gamepad', idx);
            if (bound.length) sh.classList.add('bound');
            sh.addEventListener('click', () => _onSourceButtonClick('gamepad', idx, label));
            svg.appendChild(sh);
            const lab = document.createElementNS(NS, 'text');
            lab.setAttribute('x', x); lab.setAttribute('y', 26);
            lab.setAttribute('class', 'btn-label');
            lab.textContent = label;
            svg.appendChild(lab);
        }

        for (const [idx, meta] of Object.entries(GAMEPAD_BUTTONS)) {
            if (idx === '4' || idx === '5') continue; // already drawn as shoulder bars
            const bound = _bindingsFor('gamepad', idx);
            const btn = document.createElementNS(NS, 'circle');
            btn.setAttribute('cx', meta.cx); btn.setAttribute('cy', meta.cy);
            btn.setAttribute('r', '12');
            btn.setAttribute('class', 'btn-overlay' + (bound.length ? ' bound' : ''));
            btn.dataset.source = 'gamepad'; btn.dataset.binding = idx;
            btn.addEventListener('click', () => _onSourceButtonClick('gamepad', idx, meta.label));
            svg.appendChild(btn);
            const lab = document.createElementNS(NS, 'text');
            lab.setAttribute('x', meta.cx); lab.setAttribute('y', meta.cy - 2);
            lab.setAttribute('class', 'btn-label');
            lab.textContent = meta.label;
            svg.appendChild(lab);
            if (bound.length) {
                const al = document.createElementNS(NS, 'text');
                al.setAttribute('x', meta.cx); al.setAttribute('y', meta.cy + 8);
                al.setAttribute('class', 'btn-label action');
                al.textContent = bound[0];
                svg.appendChild(al);
            }
        }

        host.appendChild(svg);
    }

    function _onSourceButtonClick(source, binding, label) {
        // Pop a tiny picker: pick an action to bind to this button.
        const current = _bindingsFor(source, binding);
        const choices = _actions.map(a => a.name);
        const msg = `Bind ${source} button "${label}" (${binding}) to which action?\n\n` +
            `Currently bound to: ${current.length ? current.join(', ') : '(none)'}\n\n` +
            `Actions:\n${choices.map((a, i) => `  ${i + 1}. ${a}`).join('\n')}\n\n` +
            `Enter the action name, or blank to unbind:`;
        const input = prompt(msg, current[0] || '');
        if (input === null) return; // cancelled
        const trimmed = input.trim();
        // Clear existing bindings on this button.
        _bindings = _bindings.filter(b => !(b.source === source && b.binding === binding));
        if (trimmed) {
            if (!choices.includes(trimmed)) {
                alert(`Unknown action: ${trimmed}\n\nKnown: ${choices.join(', ')}`);
                return;
            }
            _bindings.push({action: trimmed, source, binding});
        }
        _markDirty();
        _render();
    }

    // ── Save / load ────────────────────────────────────────────────────

    function _markDirty() {
        const status = document.getElementById('hotkey-save-status');
        if (status) {
            status.textContent = 'Unsaved changes';
            status.className = 'settings-status err';
        }
    }

    async function saveHotkeys() {
        try {
            const res = await fetch('/api/hotkeys/bindings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({bindings: _bindings}),
            });
            const status = document.getElementById('hotkey-save-status');
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                status.textContent = 'Save failed: ' + (err.detail || res.statusText);
                status.className = 'settings-status err';
                return;
            }
            _serverBindings = JSON.parse(JSON.stringify(_bindings));
            status.textContent = 'Saved.';
            status.className = 'settings-status ok';
            // Tell the browser keyboard handler to reload its binding cache.
            if (typeof window.reloadHotkeyBindings === 'function') {
                window.reloadHotkeyBindings();
            }
        } catch (e) {
            const status = document.getElementById('hotkey-save-status');
            status.textContent = 'Save threw: ' + e.message;
            status.className = 'settings-status err';
        }
    }

    async function resetHotkeysToDefaults() {
        if (!confirm('Reset all hotkey bindings to defaults? Unsaved changes will be lost.')) return;
        try {
            // Empty list — server falls back to defaults next load.
            await fetch('/api/hotkeys/bindings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({bindings: []}),
            });
            await _fetchBindings();
            _render();
            const status = document.getElementById('hotkey-save-status');
            status.textContent = 'Reset to defaults.';
            status.className = 'settings-status ok';
            if (typeof window.reloadHotkeyBindings === 'function') {
                window.reloadHotkeyBindings();
            }
        } catch (e) {
            console.error('reset threw:', e);
        }
    }

    function _render() {
        _renderKeyboard();
        _renderQuestSide('quest-left-svg', 'left', QUEST_BUTTONS_LEFT);
        _renderQuestSide('quest-right-svg', 'right', QUEST_BUTTONS_RIGHT);
        _renderGamepad();
    }

    async function _initIfNeeded() {
        if (_initialised) return;
        try {
            await _fetchActions();
            await _fetchBindings();
            await _fetchStatus();
            _render();
            _initialised = true;
        } catch (e) {
            console.error('hotkeys settings init failed:', e);
        }
    }

    // Mount when the Settings tab is shown. Hooks into the existing
    // switchTab function — when the user navigates to Settings, run
    // initial fetches (idempotent).
    const _origSwitchTab = window.switchTab;
    window.switchTab = function (tabName) {
        if (typeof _origSwitchTab === 'function') _origSwitchTab(tabName);
        if (tabName === 'settings') _initIfNeeded();
    };

    // Expose handlers for inline onclick=.
    window.saveHotkeys = saveHotkeys;
    window.resetHotkeysToDefaults = resetHotkeysToDefaults;
})();
