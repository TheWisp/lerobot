// Overlays panel — run a processing step on the current observation and draw the
// result on the camera tiles. One shell, two drivers (data run-on-episode + live
// standalone). Ports the prototype's control UI: open-vocab object rows with a
// +/- sign, a colour palette per object, a Background colour, and camera toggle
// buttons. See gui/docs/overlays.md.

(function () {
    let MODELS = [];
    let SEGMENTERS = [];    // model keys valid for data editing (text-prompted segmenters)
    let RESOLUTIONS = [];   // SAM resolution presets [{value,label}] — a load-time knob (change = respawn)
    let TREATMENTS = [];  // per-region treatments (from /api/process/treatments); Tint/Random/Blur/None
    const panels = [];
    let livePanel = null;

    const PALETTE = [[239, 68, 68], [34, 197, 94], [59, 130, 246], [234, 179, 8], [168, 85, 247], [20, 184, 166]];
    const MAX_OBJECTS = 6;

    // Per-tab identity for the data overlay's single-owner lease (the model + obs
    // stream are shared, so one tab drives at a time). sessionStorage keeps ownership
    // across a reload; a new tab gets a new token. Sent as X-Overlay-Session.
    const OVL_SESSION = (() => {
        try {
            let s = sessionStorage.getItem('ovlSession');
            if (!s) { s = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : 'ovl-' + Math.random().toString(36).slice(2); sessionStorage.setItem('ovlSession', s); }
            return s;
        } catch (e) { return 'ovl-' + Math.random().toString(36).slice(2); }
    })();
    const ovlHeaders = (extra) => Object.assign({ 'X-Overlay-Session': OVL_SESSION }, extra || {});
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const safeCam = (cam) => cam.replace(/\./g, '-');

    // Shared "open log" viewer: a roomy, dismissible modal (created once) with Copy +
    // Refresh, showing the model's persistent log — the live standalone's log file, or
    // the data state. A loader fn produces the text, so Refresh can re-run it.
    let logLoader = null;
    let logTimer = null;  // poll handle while the log modal is open (live tail)
    function runLogLoader(silent) {
        const p = document.querySelector('#overlays-log-modal .overlays-log-pre');
        if (!p || !logLoader) return;
        const pinned = p.scrollTop + p.clientHeight >= p.scrollHeight - 4;  // following the tail?
        if (!silent) p.textContent = 'Loading…';
        Promise.resolve(logLoader())
            .then((t) => {
                p.textContent = (t && String(t).trim()) ? t : '(empty)';
                if (pinned) p.scrollTop = p.scrollHeight;  // keep the newest line in view unless scrolled up
            })
            .catch(() => { if (!silent) p.textContent = '(failed to read log)'; });
    }
    // Fallback for NON-secure contexts (GUI over a LAN IP on plain HTTP), where
    // navigator.clipboard is undefined: copy via a temporary textarea + execCommand
    // (which is not secure-context-gated). Flashes only if the copy actually succeeded.
    function selectPre(pre, after) {
        const ta = document.createElement('textarea');
        ta.value = pre.textContent;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '0';
        ta.style.left = '0';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        if (ok && after) after();
    }
    function logModalEl() {
        let m = document.getElementById('overlays-log-modal');
        if (!m) {
            m = document.createElement('div');
            m.id = 'overlays-log-modal';
            m.className = 'overlays-log-modal';
            m.innerHTML = '<div class="overlays-log-box"><div class="overlays-log-head">'
                + '<span class="overlays-log-title"></span>'
                + '<span class="overlays-log-actions">'
                + '<button class="overlays-log-btn overlays-log-copy">Copy</button>'
                + '<button class="overlays-log-btn overlays-log-refresh">Refresh</button>'
                + '<button class="overlays-log-close" title="close (Esc)">&times;</button>'
                + '</span></div><pre class="overlays-log-pre"></pre></div>';
            document.body.appendChild(m);
            const close = () => { m.style.display = 'none'; clearInterval(logTimer); logTimer = null; };
            m.addEventListener('click', (e) => { if (e.target === m) close(); });
            m.querySelector('.overlays-log-close').addEventListener('click', close);
            m.querySelector('.overlays-log-refresh').addEventListener('click', runLogLoader);
            m.querySelector('.overlays-log-copy').addEventListener('click', () => {
                const pre = m.querySelector('.overlays-log-pre');
                const btn = m.querySelector('.overlays-log-copy');
                const flash = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(pre.textContent).then(flash).catch(() => selectPre(pre, flash));
                } else { selectPre(pre, flash); }
            });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
        }
        return m;
    }
    function showLog(title, loader) {
        const m = logModalEl();
        m.querySelector('.overlays-log-title').textContent = title;
        logLoader = loader;
        runLogLoader();
        m.style.display = 'flex';
        clearInterval(logTimer);  // follow the tail live while open; the × / Esc / click-out close stops it
        logTimer = setInterval(() => runLogLoader(true), 1500);
    }

    function init() {
        const roots = [
            { el: document.getElementById('overlays-panel'), mode: 'data' },
            { el: document.getElementById('overlays-panel-run'), mode: 'live' },
        ].filter((r) => r.el);
        if (!roots.length) return;
        // The data-editing effects live in the overlay panel now (they drive the live
        // preview); fetch them once so the picker is ready when a model is chosen.
        fetch('/api/process/treatments').then((r) => r.json()).then((d) => { TREATMENTS = d.treatments || []; }).catch(() => {});
        fetch('/api/overlays/models').then((r) => r.json())
            .then((d) => { MODELS = d.models || []; SEGMENTERS = d.segmenters || []; RESOLUTIONS = d.resolutions || []; build(roots); })
            .catch(() => build(roots));
    }

    function build(roots) {
        for (const r of roots) {
            const p = new Panel(r.el, r.mode);
            panels.push(p);
            if (r.mode === 'live') livePanel = p;
        }
        // The data-editing menu lives alongside the data overlay (shares its
        // objects). Init it once; reflect the active-job count on the button.
        if (window.ProcessData) window.ProcessData.init({ onCountChange: updateProcessButtons });
    }

    // Show the running-job count on every "Process dataset…" button so a user
    // who closed the menu still sees work is in flight.
    function updateProcessButtons(count) {
        document.querySelectorAll('.overlays-process').forEach((b) => {
            b.textContent = count > 0 ? `⚙ Process dataset… (${count} running)` : '⚙ Process dataset…';
            b.classList.toggle('busy', count > 0);
        });
    }

    function Panel(root, mode) {
        const q = (cls) => root.querySelector('.' + cls);
        const els = {
            picker: q('overlays-picker'), modelBody: q('overlays-model-body'),
            action: q('overlays-action'), badge: q('overlays-badge'),
            caret: q('overlays-caret'), logLink: q('overlays-log-link'), header: q('overlays-header'),
        };
        let current = '';
        // Monitored objects: open-vocab name + colour + sign (+ include / − exclude).
        // Data mode: each object is a region with its own treatment; the background is a
        // region too (see backgroundTreatment). Objects default to None (kept as-is);
        // background defaults to Random → the GreenAug recipe is zero-click.
        let objects = [{ name: '', sign: '+', treatment: { key: 'none', params: {} } }];
        let backgroundTreatment = { key: 'random', params: {} };
        let multiInstance = true;               // data mode: segment ALL instances of each object (both arms) vs largest
        // SAM inference resolution (a LOAD-TIME knob: changing it respawns the worker; the
        // batch job inherits it so preview == commit). Default to the backend's default preset.
        let resolution = (RESOLUTIONS.find((r) => /default/i.test(r.label || '')) || RESOLUTIONS[0] || { value: null }).value;
        let background = null;            // run mode only: contour-view fill; null = transparent, else [r,g,b]
        let nameTimer = null;
        let status = { state: 'idle' };
        let pollTimer = null;
        let overlayES = null;            // data: SSE stream that pushes overlay-ready → instant re-pull (vs the 500ms poll)
        const pullGate = window.OverlayPullGate ? window.OverlayPullGate.create() : null;  // seq-gated overlay pulls (see overlay_pull_gate.js)
        let wasBusy = false;             // data: another tab owns the overlay (lease); auto-resume when freed
        let started = false;             // live: standalone launched
        let dataVersion = 0;             // data: cache-buster, bumped on config change so scrubbing re-pulls
        let frameTick = 0;               // data: increments each overlay re-pull so the lagging worker result refreshes
        let selectedCameras = null;      // Set<camera key>; null until first loadCameras
        // Per-MODEL control values (the model body's select/slider state — e.g. policy_saliency's
        // style/method/smoothing), grouped in one object so they don't mix with the panel-generic
        // state above. A control maps to its slot by shape, not by model name, so new steps reuse it.
        const ctl = { style: null, smooth: null, method: null };
        const ctlSlot = (c) => (c.type === 'slider' ? 'smooth' : (c.key === 'method' ? 'method' : 'style'));
        let availCameras = [];
        let camRetry;                    // retry timer while the live obs stream's cameras aren't up yet
        let lastDiag = '';               // last frontend-state signature reported to the server log (dedup)

        for (const m of MODELS) {
            // Data mode edits pixels via a segmenter; overlay-only steps (policy saliency
            // reads the RUNNING policy) can't produce anything there — don't offer them.
            if (mode === 'data' && SEGMENTERS.length && !SEGMENTERS.includes(m.key)) continue;
            const o = document.createElement('option');
            o.value = m.key; o.textContent = m.label;
            els.picker.appendChild(o);
        }
        els.picker.addEventListener('change', (e) => onPick(e.target.value));
        // Collapse only via the caret. The status badge + log link live in the header; they must
        // NOT turn a stray click into a panel toggle (the 'off' badge read as a button and ate
        // clicks meant for 'log').
        els.caret.addEventListener('click', (e) => {
            e.stopPropagation();
            root.classList.toggle('collapsed');
            els.caret.textContent = root.classList.contains('collapsed') ? '▸' : '▾';
        });
        els.logLink.addEventListener('click', (e) => { e.stopPropagation(); openLog(); });
        renderBody();

        const modelSpec = (k) => MODELS.find((m) => m.key === k);
        const namedObjects = () => objects.filter((o) => (o.name || '').trim());
        // A step needs a named object only if it declares an "objects" control (SAM3). A no-objects
        // step like policy_attention is "ready" without one — don't gate its launch on the object field.
        const requiresObjects = () => (modelSpec(current)?.controls || []).some((c) => c.type === 'objects');
        const objectsReady = () => !requiresObjects() || namedObjects().length > 0;
        // What goes to the backend: named objects with their per-region treatment + sign.
        function payloadObjects() {
            const tr = (o) => o.treatment || { key: 'none', params: {} };
            const named = objects.filter((o) => (o.name || '').trim())
                .map((o) => ({ name: o.name.trim(), sign: o.sign || '+', treatment: tr(o) }));
            if (named.length) return named;
            const o = objects[0] || {};
            return [{ name: 'object', sign: o.sign || '+', treatment: tr(o) }];
        }
        const bgPayload = () => ({ color: background });  // run mode only: contour-view fill
        const camsArg = () => (selectedCameras && selectedCameras.size ? [...selectedCameras] : null);

        function onPick(key) {
            if (mode === 'live' && started) { fetch('/api/overlays/live/stop', { method: 'POST' }).catch(() => {}); started = false; stopPoll(); }
            current = key;
            // Model-specific control values must not leak across models (a saliency style/smooth
            // would otherwise ride along in the next model's /live/start body).
            ctl.style = ctl.smooth = ctl.method = null;
            if (!key) {
                stopPoll();
                if (mode === 'data') fetch('/api/overlays/data/cancel', { method: 'POST', headers: ovlHeaders() }).catch(() => {});
                clearOverlays();
                setBadge('off', 'off');
            }
            renderBody();
            refreshStatus();
            sync();
        }

        // ---- body (per-model config) ----
        function renderBody() {
            if (!current) {
                els.modelBody.innerHTML = '<div class="overlays-hint">Pick a processing step.</div>';
                els.action.innerHTML = '';
                return;
            }
            const controls = modelSpec(current)?.controls || [];
            const ctrl = controls[0] || {};
            if (ctrl.type === 'objects' || ctrl.type === 'text') {
                const hint = mode === 'data'
                    ? 'Each object is a region with a treatment; the <b>Background</b> row is a region too. Tile shows the live WYSIWYG result — the glow + label is a detection aid, not part of the output.'
                    : 'Open-vocab names, each in its own colour. <b>+</b> include / <b>−</b> exclude. Name edits apply ~1s after you stop typing; colour/sign are instant.';
                els.modelBody.innerHTML = `
                    <label class="overlays-label">${esc(ctrl.label || 'Objects')}</label>
                    <div class="overlays-hint">${hint}</div>
                    <div class="overlays-objrows"></div>
                    <button class="overlays-add-obj">+ Add object</button>
                    ${mode === 'data' ? `<label class="overlays-check" title="On: keep every instance of each object (e.g. both robot arms). Off: keep only the single largest.">
                        <input type="checkbox" class="overlays-multi"${multiInstance ? ' checked' : ''}> Segment all instances (e.g. both arms)</label>` : ''}
                    ${RESOLUTIONS.length ? `<label class="overlays-label" title="SAM inference resolution — lower is faster; Balanced measured equal-or-better masks than Full at ~1.8× the speed. Changing it reloads the model.">Quality</label>
                    <select class="overlays-select overlays-res">${RESOLUTIONS.map((r) => `<option value="${r.value}"${r.value === resolution ? ' selected' : ''}>${esc(r.label)}</option>`).join('')}</select>` : ''}
                    <label class="overlays-label">cameras</label>
                    <div class="overlays-cameras"></div>
                    ${mode === 'data' ? '<button class="overlays-process" title="Apply these per-region treatments to every episode as a new dataset">⚙ Process dataset…</button>' : ''}`;
                els.modelBody.querySelector('.overlays-add-obj').addEventListener('click', addObject);
                const procBtn = els.modelBody.querySelector('.overlays-process');
                if (procBtn) procBtn.addEventListener('click', openProcess);
                const multiCb = els.modelBody.querySelector('.overlays-multi');
                if (multiCb) multiCb.addEventListener('change', () => { multiInstance = multiCb.checked; applyInstant(); });
                const resSel = els.modelBody.querySelector('.overlays-res');
                if (resSel) resSel.addEventListener('change', () => {
                    resolution = Number(resSel.value) || null;
                    // Resolution is baked into the model at load — a running live worker must
                    // RESTART (a control push can't apply it); the data path's re-configure
                    // respawns server-side when the resolution differs.
                    if (mode === 'live' && started) { fetch('/api/overlays/live/stop', { method: 'POST' }).catch(() => {}); started = false; }
                    applyInstant();
                });
                renderObjects();
            } else {
                // simple controls (select, slider, ...) rendered in order, then the camera picker
                els.modelBody.innerHTML = controls.map(controlHTML).filter(Boolean).join('')
                    + '<label class="overlays-label">cameras</label><div class="overlays-cameras"></div>';
                controls.forEach(attachControl);
            }
            loadCameras();
            renderAction();
        }

        // ---- simple controls (select / slider) for non-objects steps ----
        function controlHTML(c) {
            if (c.type === 'select') {
                const opts = c.options || [];
                let cur = ctl[ctlSlot(c)];
                if (!opts.some((o) => o.value === cur)) { cur = c.default ?? (opts[0] && opts[0].value) ?? null; ctl[ctlSlot(c)] = cur; }
                return `<label class="overlays-label">${esc(c.label || c.key)}</label>
                    <select class="overlays-select" data-key="${esc(c.key)}">${opts.map((o) => `<option value="${esc(o.value)}"${o.value === cur ? ' selected' : ''}>${esc(o.label)}</option>`).join('')}</select>`;
            }
            if (c.type === 'slider') {
                if (ctl.smooth === null) ctl.smooth = c.default ?? 0;
                return `<label class="overlays-label">${esc(c.label || 'Smoothing')} <span class="overlays-sliderval">${(+ctl.smooth).toFixed(1)}</span></label>
                    <input type="range" class="overlays-slider" min="${c.min ?? 0}" max="${c.max ?? 3}" step="${c.step ?? 0.1}" value="${ctl.smooth}">`;
            }
            return '';
        }

        function attachControl(c) {
            if (c.type === 'select') {
                const el = els.modelBody.querySelector(`.overlays-select[data-key="${c.key}"]`);
                if (el) el.addEventListener('change', (e) => { ctl[ctlSlot(c)] = e.target.value; applyInstant(); });
            } else if (c.type === 'slider') {
                const el = els.modelBody.querySelector('.overlays-slider');
                if (!el) return;
                el.addEventListener('input', (e) => { ctl.smooth = parseFloat(e.target.value); const v = els.modelBody.querySelector('.overlays-sliderval'); if (v) v.textContent = ctl.smooth.toFixed(1); });
                el.addEventListener('change', () => applyInstant());  // send on release (latest-wins), not every drag tick
            }
        }

        // ---- object rows: [sign][name][palette][trash] + a Background row ----
        const swatch = (rgb, sel) => `<span class="overlays-swatch${sel ? ' sel' : ''}" data-rgb="${rgb.join(',')}" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>`;
        const paletteHTML = (s) => PALETTE.map((c) => swatch(c, s && c[0] === s[0] && c[1] === s[1] && c[2] === s[2])).join('');

        // ---- per-region treatment widget (data mode): [ Tint | Random | Blur | None ] ----
        const TINT_PRESETS = [[239, 68, 68], [34, 197, 94], [59, 130, 246], [234, 179, 8], [168, 85, 247], [20, 184, 166], [255, 255, 255], [15, 23, 42]];
        const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
        const toHex = (c) => '#' + c.map((x) => Math.max(0, Math.min(255, x | 0)).toString(16).padStart(2, '0')).join('');
        const fromHex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
        const regionTreatment = (r) => (r === 'bg' ? backgroundTreatment : (objects[r] || {}).treatment) || { key: 'none', params: {} };

        // Compact ICON set (best-practice glyphs): ∅ none · colour square = tint (click to
        // pick) · dice = random · fading circle = blur. The SELECTED button gets a filled
        // accent so the active treatment is unambiguous regardless of the icons.
        const TREAT_SVG = {
            none: '<svg viewBox="0 0 16 16" class="ti"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.4"/><line x1="4.2" y1="11.8" x2="11.8" y2="4.2" stroke="currentColor" stroke-width="1.4"/></svg>',
            random: '<svg viewBox="0 0 16 16" class="ti"><rect x="2.3" y="2.3" width="11.4" height="11.4" rx="2.6" fill="none" stroke="currentColor" stroke-width="1.3"/><g fill="currentColor"><circle cx="5.6" cy="5.6" r="1.05"/><circle cx="10.4" cy="5.6" r="1.05"/><circle cx="8" cy="8" r="1.05"/><circle cx="5.6" cy="10.4" r="1.05"/><circle cx="10.4" cy="10.4" r="1.05"/></g></svg>',
            blur: '<svg viewBox="0 0 16 16" class="ti"><path d="M8 1.6 C 8 1.6 3.4 7.6 3.4 10 a 4.6 4.6 0 1 0 9.2 0 C 12.6 7.6 8 1.6 8 1.6 Z" fill="currentColor"/></svg>',
        };
        const treatIcon = (key, tr) => key === 'tint'
            ? `<span class="overlays-tint-chip" style="background:${rgbCss((tr && tr.params && tr.params.color) || TINT_PRESETS[2])}"></span>`
            : (TREAT_SVG[key] || esc((key || '?')[0]));
        // `selAttr` identifies the region for delegated handlers (data-obj="i" / data-bg="1").
        function treatWidget(tr, selAttr) {
            const cur = (tr && tr.key) || 'none';
            const btns = TREATMENTS.map((t) => `<button class="overlays-treat-btn${t.key === cur ? ' sel' : ''}" data-key="${t.key}" title="${esc(t.label)}" aria-label="${esc(t.label)}">${treatIcon(t.key, tr)}</button>`).join('');
            return `<span class="overlays-treat" ${selAttr}>${btns}</span>`;
        }
        function setTreatment(region, key) {
            const t = regionTreatment(region);
            const params = Object.assign({}, t.params);
            if (key === 'tint' && !params.color) params.color = TINT_PRESETS[2];
            const nt = { key, params: (key === 'tint' || key === 'blur') ? params : {} };
            if (region === 'bg') backgroundTreatment = nt; else objects[region].treatment = nt;
            renderObjects(); applyInstant();
        }
        // Update a tint region's colour IN PLACE — no re-render (re-rendering would destroy
        // the open native colour picker mid-interaction, which dropped custom colours). Just
        // repaint the bar + push to the worker.
        // Update a tint region's colour IN PLACE (no re-render — that would destroy the open
        // native picker). Repaint the icon's colour chip immediately; DEBOUNCE the worker push,
        // because the native picker fires 'input' continuously and each push re-segments (laggy).
        let tintPushTimer = null;
        function setTintColor(region, rgb, immediate) {
            const t = regionTreatment(region);
            t.params = Object.assign({}, t.params, { color: rgb });
            if (region === 'bg') backgroundTreatment = t; else objects[region].treatment = t;
            const sel = region === 'bg' ? '[data-bg="1"]' : `[data-obj="${region}"]`;
            const chip = els.modelBody.querySelector(`.overlays-treat${sel} .overlays-treat-btn[data-key="tint"] .overlays-tint-chip`);
            if (chip) chip.style.background = rgbCss(rgb);
            clearTimeout(tintPushTimer);
            if (immediate) applyInstant();
            else tintPushTimer = setTimeout(() => applyInstant(), 250);
        }

        // Shared Tint colour popover — created once at body level so panel re-renders don't
        // kill it. Presets + a custom picker; click-outside closes.
        let tintPop = null;
        function closeTintPop() { if (tintPop) tintPop.style.display = 'none'; }
        function tintPopEl() {
            if (tintPop) return tintPop;
            tintPop = document.createElement('div');
            tintPop.className = 'overlays-tint-pop';
            tintPop.style.display = 'none';
            document.body.appendChild(tintPop);
            document.addEventListener('click', (e) => {
                if (tintPop.style.display === 'none') return;
                const onTint = e.target.closest && e.target.closest('.overlays-treat-btn[data-key="tint"]');
                if (!tintPop.contains(e.target) && !onTint) closeTintPop();
            });
            return tintPop;
        }
        const paintPopSel = (el, rgb) => el.querySelectorAll('.overlays-tint-sw').forEach((sw) => sw.classList.toggle('sel', sw.dataset.rgb === rgb.join(',')));
        function openTintPop(region, anchor) {
            const el = tintPopEl();
            const cur = (regionTreatment(region).params || {}).color || TINT_PRESETS[2];
            el.innerHTML = `<div class="overlays-tint-sws">${TINT_PRESETS.map((c) => `<span class="overlays-tint-sw${c.join(',') === cur.join(',') ? ' sel' : ''}" data-rgb="${c.join(',')}" style="background:${rgbCss(c)}"></span>`).join('')}</div>`
                + `<label class="overlays-tint-custom-row">Custom <input type="color" class="overlays-tint-custom" value="${toHex(cur)}"></label>`;
            el.querySelectorAll('.overlays-tint-sw').forEach((sw) => sw.addEventListener('click', () => { const rgb = sw.dataset.rgb.split(',').map(Number); setTintColor(region, rgb, true); paintPopSel(el, rgb); el.querySelector('.overlays-tint-custom').value = toHex(rgb); }));
            const ci = el.querySelector('.overlays-tint-custom');
            ci.addEventListener('input', (e) => { const rgb = fromHex(e.target.value); setTintColor(region, rgb); paintPopSel(el, rgb); });  // debounced push while dragging
            ci.addEventListener('change', (e) => setTintColor(region, fromHex(e.target.value), true));  // final push on close
            el.style.display = 'block';
            const r = anchor.getBoundingClientRect();
            el.style.left = Math.max(6, Math.min(r.left, window.innerWidth - el.offsetWidth - 8)) + 'px';
            el.style.top = (r.bottom + 4) + 'px';
        }
        function wireTreatments(box) {
            const regionOf = (el) => { const s = el.closest('.overlays-treat'); return s.dataset.bg ? 'bg' : +s.dataset.obj; };
            box.querySelectorAll('.overlays-treat-btn').forEach((b) => b.addEventListener('click', (e) => {
                e.stopPropagation();
                const region = regionOf(b);
                if (b.dataset.key === 'tint') {
                    setTreatment(region, 'tint');  // re-renders; anchor the popover to the fresh button
                    const sel = region === 'bg' ? '[data-bg="1"]' : `[data-obj="${region}"]`;
                    const fresh = els.modelBody.querySelector(`.overlays-treat${sel} .overlays-treat-btn[data-key="tint"]`);
                    if (fresh) openTintPop(region, fresh);
                } else { closeTintPop(); setTreatment(region, b.dataset.key); }
            }));
        }

        function renderObjects() {
            const box = els.modelBody.querySelector('.overlays-objrows');
            if (!box) return;
            const anyNamed = objects.some((o) => (o.name || '').trim());
            const signBtn = (o, i) => `<button class="overlays-obj-btn sign${o.sign === '-' ? ' neg' : ''}" data-i="${i}" title="${o.sign === '-' ? 'excluded — click to include' : 'included — click to exclude'}">${o.sign === '-' ? '−' : '+'}</button>`;
            const nameInput = (o, i) => `<input class="overlays-obj-name" type="text" data-i="${i}" placeholder="${(i === 0 && !anyNamed) ? 'object' : 'object name (e.g. robot arm)'}" value="${esc(o.name)}">`;
            const trail = (i) => objects.length > 1 ? `<button class="overlays-obj-btn rm" data-i="${i}" title="remove">✕</button>` : '<span class="overlays-obj-slot"></span>';

            if (mode === 'data') {
                // One line per region: [+/− polarity] [name] [treatment icons] [× remove].
                // The polarity pill is a first-class per-concept filter: green + = add to the
                // foreground, red − = SUPPRESS (subtract from it — e.g. arm − gripper). The
                // Background row uses slot placeholders so its columns line up.
                const pol = (o, i) => `<button class="overlays-pol ${o.sign === '-' ? 'neg' : 'pos'}" data-i="${i}" title="${o.sign === '-' ? '− suppress: subtracted from the foreground — click to add' : '+ foreground: added — click to suppress'}">${o.sign === '-' ? '−' : '+'}</button>`;
                const rmBtn = (i) => objects.length > 1 ? `<span class="overlays-obj-rm" data-i="${i}" title="remove">&times;</span>` : '<span class="overlays-obj-slot"></span>';
                const rows = objects.map((o, i) => {
                    const excl = o.sign === '-';
                    const mid = excl
                        ? '<span class="overlays-treat-na" title="a − concept is subtracted from the foreground, not treated">subtracted</span>'
                        : treatWidget(o.treatment, `data-obj="${i}"`);
                    return `<div class="overlays-objrow data${excl ? ' excl' : ''}">${pol(o, i)}${nameInput(o, i)}${mid}${rmBtn(i)}</div>`;
                }).join('');
                const bgrow = `<div class="overlays-objrow data bg"><span class="overlays-obj-slot"></span><span class="overlays-bg-label">Background</span>${treatWidget(backgroundTreatment, 'data-bg="1"')}<span class="overlays-obj-slot"></span></div>`;
                box.innerHTML = rows + bgrow;
                wireTreatments(box);
                box.querySelectorAll('.overlays-pol').forEach((b) => b.addEventListener('click', () => { objects[+b.dataset.i].sign = objects[+b.dataset.i].sign === '-' ? '+' : '-'; renderObjects(); applyInstant(); }));
                box.querySelectorAll('.overlays-obj-rm').forEach((b) => b.addEventListener('click', () => { if (objects.length > 1) { objects.splice(+b.dataset.i, 1); renderObjects(); applyInstant(); } }));
            } else {
                // Run tab: the debug-contour view keeps per-object colours + a Background fill.
                const rows = objects.map((o, i) => {
                    const neg = o.sign === '-';
                    return `<div class="overlays-objrow">${signBtn(o, i)}${nameInput(o, i)}<span class="overlays-palette${neg ? ' disabled' : ''}" data-i="${i}" title="${neg ? 'a − concept is subtracted, not drawn — colour unused' : ''}">${paletteHTML(o.color)}</span>${trail(i)}</div>`;
                }).join('');
                const bgrow = `<div class="overlays-objrow"><span class="overlays-obj-slot"></span><span class="overlays-bg-label">Background</span><span class="overlays-palette" data-bg="1">${paletteHTML(background)}</span><button class="overlays-obj-btn bg-clear${!background ? ' on' : ''}" title="transparent (don't paint)">∅</button></div>`;
                box.innerHTML = rows + bgrow;
                box.querySelectorAll('.overlays-palette[data-i] .overlays-swatch').forEach((sw) => sw.addEventListener('click', () => { objects[+sw.parentElement.dataset.i].color = sw.dataset.rgb.split(',').map(Number); renderObjects(); applyInstant(); }));
                box.querySelectorAll('.overlays-palette[data-bg] .overlays-swatch').forEach((sw) => sw.addEventListener('click', () => { background = sw.dataset.rgb.split(',').map(Number); renderObjects(); applyInstant(); }));
                const bgClear = box.querySelector('.overlays-obj-btn.bg-clear');
                if (bgClear) bgClear.addEventListener('click', () => { background = null; renderObjects(); applyInstant(); });
            }

            box.querySelectorAll('.overlays-obj-btn.sign').forEach((b) => b.addEventListener('click', () => { objects[+b.dataset.i].sign = objects[+b.dataset.i].sign === '-' ? '+' : '-'; renderObjects(); applyInstant(); }));
            box.querySelectorAll('.overlays-obj-btn.rm').forEach((b) => b.addEventListener('click', () => { if (objects.length > 1) { objects.splice(+b.dataset.i, 1); renderObjects(); applyInstant(); } }));
            box.querySelectorAll('.overlays-obj-name').forEach((inp) => inp.addEventListener('input', () => { objects[+inp.dataset.i].name = inp.value; renderAction(); scheduleApply(); }));

            const add = els.modelBody.querySelector('.overlays-add-obj');
            if (add) { add.disabled = objects.length >= MAX_OBJECTS; add.textContent = `+ Add object (${objects.length}/${MAX_OBJECTS})`; }

            renderProcessGate();
        }

        // Gate "Process dataset…": needs a named object AND at least one real treatment.
        // Called from renderObjects() (rows/treatments changed) AND renderAction()
        // (name edits, which do NOT re-render the rows). Living only in renderObjects
        // was a bug: typing a valid object name left the button greyed out — with a
        // tooltip telling you to do the thing you had just done — until you happened to
        // click a treatment button and trigger a row re-render.
        function renderProcessGate() {
            const procBtn = els.modelBody && els.modelBody.querySelector('.overlays-process');
            if (!procBtn) return;
            const ok = namedObjects().length > 0 && hasTreatment();
            procBtn.disabled = !ok;
            procBtn.title = ok ? 'Apply these per-region treatments to every episode as a new dataset'
                : 'Name an object and set at least one treatment (an object or the Background) first';
        }

        function addObject() {
            if (objects.length >= MAX_OBJECTS) return;
            objects.push({ name: '', sign: '+', treatment: { key: 'none', params: {} } });
            renderObjects();  // no apply — the new row has no name yet
        }

        // Whether any region carries a real (non-None) treatment — the commit needs one.
        const hasTreatment = () => (backgroundTreatment.key && backgroundTreatment.key !== 'none')
            || objects.some((o) => (o.name || '').trim() && o.sign !== '-' && o.treatment && o.treatment.key && o.treatment.key !== 'none');

        // Open the data-editing menu with the panel's current per-region treatments +
        // selected cameras. Segmentation is the same SAM3 the tile previews, so what you
        // see (minus the detection chrome) is exactly what gets committed.
        function openProcess() {
            if (!window.ProcessData || !window.currentDataset || !hasTreatment()) return;
            window.ProcessData.open({
                datasetId: window.currentDataset,
                objects: payloadObjects(),
                backgroundTreatment: backgroundTreatment,
                cameras: camsArg(),
                multiInstance: multiInstance,
                model: current,          // the batch job runs the SAME segmenter + resolution
                resolution,              // as this live preview (preview == commit)
                computeMs: (status && status.compute_ms) || null,  // measured ms/frame/cam from THIS preview (null = unmeasured)
            });
        }

        // Name edits restart tracking, so debounce; colour/sign/remove are display-only → instant.
        function scheduleApply() { clearTimeout(nameTimer); nameTimer = setTimeout(() => { sync(); renderAction(); }, 1000); }
        function applyInstant() { clearTimeout(nameTimer); sync(); renderAction(); }

        // ---- camera selection: toggle buttons ----
        function loadCameras() {
            const container = els.modelBody.querySelector('.overlays-cameras');
            if (!container) return;
            const apply = (cams) => {
                availCameras = cams || [];
                // Live: the obs stream may not be up yet — teleop's cameras can take ~10s to
                // initialise (the RealSense especially). Latching an empty selection here was the
                // bug that made the overlay silently never draw: retry until cameras appear, THEN
                // select. (Data fixtures are always available, so they apply immediately.)
                if (mode === 'live' && availCameras.length === 0) {
                    console.log('[overlays] live: obs stream has no cameras yet — retrying in 1.5s');
                    renderCameras(container);
                    clearTimeout(camRetry);
                    if (current) camRetry = setTimeout(loadCameras, 1500);
                    return;
                }
                // Live defaults to ONE camera for an expensive per-camera model (SAM3's VLM = 4x cost
                // per tile); a 'fast' model like policy_saliency has no model of its own (it just
                // colorizes the running policy's per-camera saliency), so it shows ALL cameras like
                // data mode — otherwise only the first tile ever drew.
                const allCams = mode === 'data' || modelSpec(current)?.load_cost === 'fast';
                if (selectedCameras === null) {
                    selectedCameras = new Set(allCams ? availCameras : [availCameras[0]]);
                } else {
                    // A dataset switch may have changed the camera set — drop selections that no
                    // longer exist, else the panel offers a ghost camera the new dataset lacks.
                    selectedCameras = new Set([...selectedCameras].filter((c) => availCameras.includes(c)));
                    if (!selectedCameras.size) selectedCameras = new Set(allCams ? availCameras : [availCameras[0]]);
                }
                console.log('[overlays] cameras=', availCameras, 'selected=', [...selectedCameras]);
                renderCameras(container);
                sync();  // cameras known — (re)drive the active mode
            };
            if (mode === 'data') {
                const ds = window.datasets && window.datasets[window.currentDataset];
                apply(ds ? ds.camera_keys : []);
            } else {
                fetch('/api/run/obs-stream/meta').then((r) => r.json())
                    .then((m) => apply(m && m.available ? Object.keys(m.image_keys) : []))
                    .catch(() => apply([]));
            }
        }

        function renderCameras(container) {
            if (!availCameras.length) { container.innerHTML = '<span class="overlays-hint">no cameras available yet</span>'; return; }
            container.innerHTML = availCameras.map((c) => `<button class="overlays-cam-btn${selectedCameras.has(c) ? ' on' : ''}" data-cam="${esc(c)}" title="${esc(c)}">${esc(c.split(/[./]/).filter(Boolean).pop() || c)}</button>`).join('');
            container.querySelectorAll('.overlays-cam-btn').forEach((b) => b.addEventListener('click', () => {
                const c = b.dataset.cam;
                if (selectedCameras.has(c)) selectedCameras.delete(c); else selectedCameras.add(c);
                b.classList.toggle('on');
                if (mode === 'live' && started) fetch('/api/overlays/live/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cameras: camsArg() }) }).catch(() => {});
                else if (mode === 'data') sync();  // re-sync the worker's active cameras + show/hide
            }));
        }

        // ---- action + status (mode-driven) ----
        function renderAction() {
            renderProcessGate();  // name edits land here, not in renderObjects()
            if (!current) { els.action.innerHTML = ''; return; }
            const hasObj = namedObjects().length > 0;
            if (mode === 'data') {
                let txt = '';
                if (!hasObj) txt = 'name an object';
                else if (status.state === 'loading') txt = 'loading…';
                else if (status.state === 'error') txt = 'error — see log';
                // else: the tile IS the feedback — no redundant status line.
                els.action.innerHTML = txt ? `<div class="overlays-status">${esc(txt)}</div>` : '';
            } else {
                let txt;
                if (!hasObj) txt = 'name an object to start';
                else if (status.state === 'starting') txt = 'starting… (waiting for cameras)';
                else if (status.state === 'active') txt = 'live';  // fps/util/VRAM live in the badge
                else if (status.state === 'error') txt = 'error — see log';
                else txt = 'live';
                els.action.innerHTML = `<div class="overlays-status">${esc(txt)}</div>`;
            }
        }

        // ---- data: pull-based, on-demand (no button) ----
        function sync() { if (mode === 'live') syncLive(); else syncData(); }

        function syncData() {
            if (mode !== 'data') return;
            if (!current || !objectsReady() || !window.currentDataset) {
                fetch('/api/overlays/data/cancel', { method: 'POST', headers: ovlHeaders() }).catch(() => {});
                dataVersion++;
                stopPoll();
                clearOverlays();
                setBadge('off', 'off');
                return;
            }
            dataVersion++;  // bust the per-frame img cache so changed objects/colours re-pull
            if (pullGate) pullGate.reset();
            if (pullLoader) pullLoader.reset();
            fetch('/api/overlays/data/configure', {
                method: 'POST', headers: ovlHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({ dataset_id: window.currentDataset, model: current, objects: payloadObjects(), background_treatment: backgroundTreatment, multi_instance: multiInstance, resolution, cameras: selectedCameras ? [...selectedCameras] : [] }),
            }).then(async (r) => {
                if (r.status === 409) {
                    // The overlay mutex is held by another client (another data tab/machine,
                    // or the run overlay). Show it and keep polling so we auto-resume when freed.
                    const d = await r.json().catch(() => ({}));
                    const holder = d && d.detail && d.detail.holder;
                    wasBusy = true;
                    setBadge('busy: ' + (holder || 'another client'), 'idle');
                    clearOverlays();
                    startPoll();
                    return;
                }
                startPoll(); onFrame();
            }).catch(() => {});
        }

        // ---- live: start/stop/control the standalone ----
        function syncLive() {
            if (mode !== 'live') return;
            reportLiveDiag(status);  // log the launch decision (incl. *why* it isn't starting) on every change
            if (!current || !objectsReady()) {
                if (started) { fetch('/api/overlays/live/stop', { method: 'POST' }).catch(() => {}); started = false; stopPoll(); setBadge('off', 'off'); }
                return;
            }
            if (selectedCameras === null) return;  // wait for loadCameras
            if (!started) {
                started = true;
                fetch('/api/overlays/live/start', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: current, objects: payloadObjects(), background: bgPayload(), cameras: camsArg(), style: ctl.style, smooth: ctl.smooth, method: ctl.method, resolution }),
                }).then(async (r) => {
                    if (r.status === 409) {
                        // The aux-GPU slot is held by another activity (a data client, or a
                        // batch job) — can't start the run overlay. Show it and keep polling.
                        const d = await r.json().catch(() => ({}));
                        started = false; wasBusy = true;
                        setBadge('busy: ' + ((d.detail && d.detail.holder) || 'another client'), 'idle');
                        startPoll();
                        return;
                    }
                    startPoll();
                }).catch(() => {});
                setBadge('starting…', 'loading');
            } else {
                fetch('/api/overlays/live/control', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ objects: payloadObjects(), background: bgPayload(), style: ctl.style, smooth: ctl.smooth, method: ctl.method }),
                }).catch(() => {});
            }
        }

        // Draw iff the backend machine says ACTIVE — single source of truth. `started` (below) is
        // only the frontend's launch *request*, not a second copy of "is it running". The decision
        // lives in OverlayGate.shouldDraw (overlay_gate.js) so it is unit-tested in isolation.
        function isLiveOn() { return !status.busy && OverlayGate.shouldDraw(mode, current, objectsReady(), status.state); }

        // ---- status polling + badge ----
        function startPoll() { stopPoll(); pollTimer = setInterval(refreshStatus, 500); refreshStatus(); if (mode === 'data') startOverlayStream(); }
        function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } stopOverlayStream(); lastDiag = ''; }

        // Event-driven overlay delivery: the server SSE-pushes {cam, seq} the instant the
        // worker writes a new overlay, so we re-pull that camera immediately instead of on
        // the ~2 Hz status poll (the dominant felt lag). The 500 ms poll stays as a fallback.
        function startOverlayStream() {
            if (overlayES || mode !== 'data') return;
            try {
                overlayES = new EventSource('/api/overlays/data/events');
                overlayES.onmessage = (e) => {
                    let d; try { d = JSON.parse(e.data); } catch (_) { return; }
                    diag.lastSseAt = performance.now();
                    // Pull only when this camera's overlay SEQ actually advanced — the
                    // worker produces ~8 overlays/s; anything more is wasted PNG decodes.
                    if (d && d.cam && (!pullGate || pullGate.onSse(d.cam, d.seq))) pullOverlayCam(d.cam, d.seq);
                };
                // onerror: the browser auto-reconnects; nothing to do.
            } catch (_) { overlayES = null; }
        }
        function stopOverlayStream() { if (overlayES) { try { overlayES.close(); } catch (_) { /* */ } overlayES = null; } }
        // At most one in-flight overlay load per camera, latest-wins (createLoader in
        // overlay_pull_gate.js): reassigning src mid-download ABORTS the fetch, so
        // unthrottled reassignment under remote-browser bandwidth means no load ever
        // completes and the tile freezes while the worker badge stays healthy.
        const pullLoader = window.OverlayPullGate && window.OverlayPullGate.createLoader
            ? window.OverlayPullGate.createLoader() : null;
        function assignOverlaySrc(cam, img, url) {
            img.onload = () => {
                img.style.display = 'block';
                diag.lastLoadDoneAt = performance.now();
                const next = pullLoader && pullLoader.done(cam);
                if (next) assignOverlaySrc(cam, img, next);
            };
            img.onerror = () => {
                const next = pullLoader && pullLoader.done(cam);
                if (next) assignOverlaySrc(cam, img, next);
            };
            img.src = url;
        }
        function pullOverlayCam(cam, seq) {
            if (!current || !objectsReady() || !window.currentDataset || window.currentEpisode === null || status.busy) return;
            if (!(selectedCameras && selectedCameras.has(cam))) return;
            const img = document.getElementById(`overlay-${safeCam(cam)}`);
            if (!img) return;
            // Cache-key by overlay seq when known (one fetch per produced overlay);
            // frameTick only paces the SSE-down fallback pulls.
            diag.lastPullAt = performance.now();
            const tick = (seq === undefined || seq === null) ? `t${frameTick++}` : `s${seq}`;
            const url = `/api/overlays/data/${encodeURIComponent(window.currentDataset)}/frame/${window.currentEpisode}/${window.currentFrame}?camera=${encodeURIComponent(cam)}&v=${dataVersion}-${tick}`;
            if (!pullLoader) {
                img.onload = () => { img.style.display = 'block'; };
                img.src = url;
                return;
            }
            const now = pullLoader.request(cam, url);
            if (now) assignOverlaySrc(cam, img, now);
        }

        function refreshStatus() {
            const url = mode === 'live'
                ? '/api/overlays/live/status?model=' + encodeURIComponent(current || '')  // per-model state
                : '/api/overlays/data/status';
            fetch(url, { headers: ovlHeaders() }).then((r) => r.json()).then((s) => {
                status = s;
                // Overlay mutex: if another client holds the shared worker (another data
                // tab/machine, or the run overlay), don't draw/publish — keep polling so
                // we auto-resume the instant it frees. Same for both panels.
                if (s.busy) {
                    wasBusy = true;
                    setBadge('busy: ' + (s.holder || 'another client'), 'idle');
                    renderAction();
                    clearOverlays();
                    if (!pollTimer) startPoll();
                    return;
                }
                if (wasBusy && !s.busy) {
                    wasBusy = false;  // freed — retry to take the mutex
                    sync();
                    return;
                }
                // (The backend re-pushes the data config on every poll while the worker
                // is up, so an effect chosen during its load window is delivered reliably
                // once the shm buffer exists — no frontend reconcile needed here.)
                // `started` is ONLY the launch request — it picks /live/start vs /live/control, nothing
                // more. The draw gate (isLiveOn) reads the backend machine's ACTIVE state directly, so the
                // worker's own INACTIVE→LOADING→ACTIVE warm-up needs no syncing here (the old reconcile
                // mirrored `started` to the backend and a transient spawn 'inactive' latched it false
                // forever). Clear the request only on a real crash, so a later change re-fires /live/start.
                if (mode === 'live' && started && s.state === 'error') started = false;
                applyBadge(s);
                if (mode === 'live' && current) reportLiveDiag(s);
                renderAction();
                // Keep a status poll alive while the overlay is working so the badge
                // (fps / util / VRAM / cached) stays live; stop once it's off.
                const busy = mode === 'live'
                    ? !!current  // poll while a model is picked so the launch decision keeps logging, even off
                    : (current && objectsReady());
                if (busy) { if (!pollTimer) startPoll(); } else { stopPoll(); }
                if (mode === 'data') onFrame();
            }).catch(() => {});
        }

        // Report the live overlay's *frontend* state to the server log (only on change), so a
        // failure is visible server-side, not just in the browser console: selected=[] means no
        // cameras chosen → nothing draws; 'blank' lists selected cameras whose overlay <img>
        // hasn't rendered. Closes the consumer-side logging gap behind 'no overlay shows up'.
        function reportLiveDiag(s) {
            const layers = [...document.querySelectorAll('.overlay-layer')]
                .map((o) => ({ cam: (o.getAttribute('src') || '').split('/frame/').pop().split('?')[0], on: o.naturalWidth > 0 }))
                .filter((l) => l.cam);
            const objs = namedObjects().length;
            const reason = !current ? 'no model selected'
                : !objectsReady() ? 'no object named'
                : selectedCameras === null ? 'waiting for obs-stream cameras'
                : !started ? 'ready (about to start)'
                : (s && s.state === 'active') ? 'running'
                : `warming up (${s ? s.state : '?'})`;
            const payload = {
                model: current, fps: s ? s.fps : null, objects: objs, started: !!started, reason,
                available: availCameras, selected: selectedCameras === null ? null : [...selectedCameras],
                drawn: layers.filter((l) => l.on).map((l) => l.cam),
                blank: layers.filter((l) => !l.on).map((l) => l.cam),
            };
            const sig = JSON.stringify([reason, payload.selected, payload.drawn, payload.blank, objs, payload.started]);
            if (sig === lastDiag) return;
            lastDiag = sig;
            fetch('/api/overlays/live/diag', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
            }).catch(() => {});
        }

        // Each badge number has a hover tooltip clarifying scope — fps + VRAM are the
        // model's own; util is its share of the card (nvidia-smi pmon SM%).
        function applyBadge(s) {
            if (!current) { setBadge(mode === 'live' ? 'inactive' : 'off', 'off'); return; }
            const fpsPart = () => ({ t: `${s.fps || 0} fps`, title: 'Overlay inference rate — frames/sec the model actually processes (0 when the stream is idle).' });
            const utilPart = () => ({ t: `${s.util || 0}% gpu`, title: "This model's own GPU utilization (SM %, nvidia-smi pmon) — its share of the card, not the whole device. 0% = loaded but not computing." });
            const vramPart = () => ({ t: `${s.vram} GB`, title: "The model's live tensor allocations (torch memory_allocated) — a leak shows as steady growth. Excludes the CUDA context + allocator cache, so nvidia-smi reads higher for the process." });
            if (mode === 'live') {
                // The badge renders the backend lifecycle state machine — the single source of truth
                // (inactive / loading / active / stopping / error). We never assemble a string state here.
                if (s.state === 'loading') { setBadge('loading…', 'loading'); return; }
                if (s.state === 'stopping') { setBadge('stopping…', 'loading'); return; }
                if (s.state === 'error') { setBadge('error', 'error'); return; }
                if (s.state === 'active') {
                    // active = model loaded; fps/util read 0 when idle (no input frames) — still active.
                    const parts = [fpsPart(), utilPart()];
                    if (s.vram) parts.push(vramPart());
                    if (typeof s.sal_ms === 'number') parts.push({ t: `sal ${s.sal_ms} ms`, title: 'Saliency pass wall time in the POLICY process — the net cost added to its inference thread on publishing inferences (demand-gated, every Nth). The fps/gpu numbers are the worker\'s and do not include this.' });
                    setBadgeParts(parts, s.fps ? 'ok' : 'idle');
                    return;
                }
                setBadge('inactive', 'off');  // not loaded (a concept-required model shows the gap via the red field)
                return;
            }
            // Data renders the SAME worker lifecycle state machine as live (the worker is identical).
            if (s.state === 'loading') { setBadge('loading…', 'loading'); return; }
            if (s.state === 'stopping') { setBadge('stopping…', 'loading'); return; }
            if (s.state === 'error') { setBadge('error', 'error'); return; }
            if (s.state === 'active') {
                const parts = [fpsPart(), utilPart()];
                if (s.vram) parts.push(vramPart());
                setBadgeParts(parts, s.fps ? 'ok' : 'idle');
                return;
            }
            setBadge('off', 'off');
        }

        function setBadge(text, cls) { els.badge.className = 'overlays-badge ' + cls; els.badge.removeAttribute('title'); els.badge.textContent = text; }
        function setBadgeParts(parts, cls) {
            els.badge.className = 'overlays-badge ' + cls;
            els.badge.removeAttribute('title');
            els.badge.innerHTML = parts.map((p) => `<span title="${esc(p.title)}">${esc(p.t)}</span>`).join(' · ');
        }

        // ---- data: playback-stall diagnostics ----
        // The worker badge can read a healthy fps while the page itself is stuck (the
        // worker re-sweeps whatever was last published), so a frozen view needs
        // CLIENT-side facts. Track frame advance here; on a stall during playback,
        // console.warn a snapshot (throttled) and keep the latest one readable at
        // window.__ovlDiag for remote inspection.
        const diag = { frame: null, changedAt: 0, sse: null, lastSseAt: 0, lastPullAt: 0, lastLoadDoneAt: 0, warnedAt: 0 };
        function diagSnapshot() {
            return {
                frame: window.currentFrame, episode: window.currentEpisode, playing: !!window.isPlaying,
                msSinceFrameChange: Math.round(performance.now() - diag.changedAt),
                sseState: overlayES ? overlayES.readyState : null,
                msSinceSse: diag.lastSseAt ? Math.round(performance.now() - diag.lastSseAt) : null,
                msSincePull: diag.lastPullAt ? Math.round(performance.now() - diag.lastPullAt) : null,
                msSinceLoadDone: diag.lastLoadDoneAt ? Math.round(performance.now() - diag.lastLoadDoneAt) : null,
                workerState: status.state, workerFps: status.fps || null, busy: !!status.busy,
            };
        }
        function diagTick(showable) {
            const now = performance.now();
            if (window.currentFrame !== diag.frame) { diag.frame = window.currentFrame; diag.changedAt = now; return; }
            const stalled = showable && window.isPlaying && diag.changedAt && now - diag.changedAt > 2000;
            if (stalled && now - diag.warnedAt > 5000) {
                diag.warnedAt = now;
                window.__ovlDiag = diagSnapshot();
                console.warn('[overlays] playback stalled while playing:', JSON.stringify(window.__ovlDiag));
            }
        }

        // ---- data: per-frame overlay renderer (hooked from app.js loadAllFrames) ----
        function onFrame() {
            if (mode !== 'data') return;
            const ds = window.datasets && window.datasets[window.currentDataset];
            if (!ds) return;
            const showable = current && objectsReady() && window.currentDataset && window.currentEpisode !== null;
            diagTick(showable);
            if (showable) {
                // Feed the worker the current frame: the backend decodes it + publishes it to the obs
                // stream (no-op if the frame is unchanged). Called on frame change AND the status poll,
                // so a re-visited frame re-publishes and the overlay is never stale.
                fetch('/api/overlays/data/publish', {
                    method: 'POST', headers: ovlHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({ dataset_id: window.currentDataset, episode: window.currentEpisode, frame: window.currentFrame }),
                }).catch(() => {});
            }
            for (const cam of ds.camera_keys) {
                const img = document.getElementById(`overlay-${safeCam(cam)}`);
                if (!img) continue;
                if (!showable || !(selectedCameras && selectedCameras.has(cam))) { img.style.display = 'none'; img.src = ''; continue; }
                // onload/onerror belong to assignOverlaySrc (the completion-gated loader);
                // overriding them here would orphan its in-flight bookkeeping.
                // Freshness is SSE-driven (pull per NEW overlay seq — see overlay_pull_gate.js);
                // this per-tick path only pulls as a rate-limited fallback while SSE is down.
                // Unconditional per-tick re-pulls cost ~4 fps of worker throughput (measured).
                if (!pullGate || pullGate.onTick(cam, !!(overlayES && overlayES.readyState === 1))) pullOverlayCam(cam);
            }
        }

        function clearOverlays() {
            if (mode === 'data') document.querySelectorAll('#camera-grid .overlay-layer').forEach((i) => { i.style.display = 'none'; i.src = ''; });
            if (pullLoader) pullLoader.reset();
        }

        function openLog() {
            const title = `Overlays · ${mode} · ${current || 'none'} · ${status.state || 'idle'}`;
            if (mode === 'live' && started) {
                // The standalone writes a real log file; tail it (Refresh re-fetches).
                showLog(title, () => fetch('/api/overlays/live/log').then((r) => r.json())
                    .then((d) => (d.log && d.log.trim()) ? d.log : '(live log is empty — the standalone has not written anything yet)'));
            } else {
                // Data runs in-process; tail its saved adapter log (detections, seeds, errors).
                showLog(title, () => fetch('/api/overlays/data/log').then((r) => r.json()).then((d) => {
                    const head = status.message ? `state: ${status.state} — ${status.message}\n\n` : '';
                    return head + ((d.log && d.log.trim()) ? d.log : '(no data-overlay log yet — pick a model and scrub)');
                }));
            }
        }

        // A dataset switch re-reads the new dataset's cameras (loadCameras drops stale selections
        // and re-syncs the worker). Only the data panel tracks a dataset; the live panel ignores it.
        this.refreshCameras = () => { if (mode === 'data' && current) loadCameras(); };
        this.onFrame = onFrame;
        this.isLiveOn = isLiveOn;
        this.isCameraOn = (cam) => !!(selectedCameras && selectedCameras.has(cam));
    }

    function liveFrameUrl(camKey, seq) {
        if (!livePanel || !livePanel.isLiveOn() || !livePanel.isCameraOn(camKey)) return null;
        return `/api/overlays/live/frame/${encodeURIComponent(camKey)}?_=${seq}`;
    }

    window.Overlays = {
        init,
        onFrame: () => panels.forEach((p) => p.onFrame && p.onFrame()),
        refreshCameras: () => panels.forEach((p) => p.refreshCameras && p.refreshCameras()),
        liveFrameUrl,
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
