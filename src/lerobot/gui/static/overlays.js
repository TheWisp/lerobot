// Overlays panel — run a processing step on the current observation and draw the
// result on the camera tiles. One shell, two drivers (data run-on-episode + live
// standalone). Ports the prototype's control UI: open-vocab object rows with a
// +/- sign, a colour palette per object, a Background colour, and camera toggle
// buttons. See gui/docs/overlays.md.

(function () {
    let MODELS = [];
    const panels = [];
    let livePanel = null;

    const PALETTE = [[239, 68, 68], [34, 197, 94], [59, 130, 246], [234, 179, 8], [168, 85, 247], [20, 184, 166]];
    const MAX_OBJECTS = 6;
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
        fetch('/api/overlays/models').then((r) => r.json())
            .then((d) => { MODELS = d.models || []; build(roots); })
            .catch(() => build(roots));
    }

    function build(roots) {
        for (const r of roots) {
            const p = new Panel(r.el, r.mode);
            panels.push(p);
            if (r.mode === 'live') livePanel = p;
        }
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
        let objects = [{ name: '', color: PALETTE[0], sign: '+' }];
        let background = null;            // null = transparent; else [r,g,b]
        let nameTimer = null;
        let status = { state: 'idle' };
        let pollTimer = null;
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
        // What goes to the backend: named objects (with colour/sign), or the implicit
        // "object" default coloured from row 0 so the palette still drives it.
        function payloadObjects() {
            const named = objects.filter((o) => (o.name || '').trim())
                .map((o) => ({ name: o.name.trim(), color: o.color, sign: o.sign || '+' }));
            if (named.length) return named;
            const o = objects[0] || {};
            return [{ name: 'object', color: o.color || PALETTE[0], sign: o.sign || '+' }];
        }
        const bgPayload = () => ({ color: background });
        const camsArg = () => (selectedCameras && selectedCameras.size ? [...selectedCameras] : null);

        function onPick(key) {
            if (mode === 'live' && started) { fetch('/api/overlays/live/stop', { method: 'POST' }).catch(() => {}); started = false; stopPoll(); }
            current = key;
            // Model-specific control values must not leak across models (a saliency style/smooth
            // would otherwise ride along in the next model's /live/start body).
            ctl.style = ctl.smooth = ctl.method = null;
            if (!key) {
                stopPoll();
                if (mode === 'data') fetch('/api/overlays/data/cancel', { method: 'POST' }).catch(() => {});
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
                els.modelBody.innerHTML = `
                    <label class="overlays-label">${esc(ctrl.label || 'Objects')}</label>
                    <div class="overlays-hint">Open-vocab names, each in its own colour. <b>+</b> include / <b>−</b> exclude. Name edits apply ~1s after you stop typing; colour/sign are instant.</div>
                    <div class="overlays-objrows"></div>
                    <button class="overlays-add-obj">+ Add object</button>
                    <label class="overlays-label">cameras</label>
                    <div class="overlays-cameras"></div>`;
                els.modelBody.querySelector('.overlays-add-obj').addEventListener('click', addObject);
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

        function renderObjects() {
            const box = els.modelBody.querySelector('.overlays-objrows');
            if (!box) return;
            const anyNamed = objects.some((o) => (o.name || '').trim());
            const rows = objects.map((o, i) => {
                const neg = o.sign === '-';
                const ph = (i === 0 && !anyNamed) ? 'object' : 'object name (e.g. robot arm)';
                const trail = objects.length > 1
                    ? `<button class="overlays-obj-btn rm" data-i="${i}" title="remove">✕</button>`
                    : '<span class="overlays-obj-slot"></span>';
                // A − concept is subtracted, never drawn, so its colour is unused — grey the
                // palette out (kept in place so flipping +/− doesn't shift the row).
                return `<div class="overlays-objrow">
                    <button class="overlays-obj-btn sign${neg ? ' neg' : ''}" data-i="${i}" title="${neg ? 'excluded — click to include' : 'included — click to exclude'}">${neg ? '−' : '+'}</button>
                    <input class="overlays-obj-name" type="text" data-i="${i}" placeholder="${ph}" value="${esc(o.name)}">
                    <span class="overlays-palette${neg ? ' disabled' : ''}" data-i="${i}" title="${neg ? 'a − concept is subtracted, not drawn — colour unused' : ''}">${paletteHTML(o.color)}</span>${trail}</div>`;
            }).join('');
            const bgrow = `<div class="overlays-objrow">
                <span class="overlays-obj-slot"></span>
                <span class="overlays-bg-label">Background</span>
                <span class="overlays-palette" data-bg="1">${paletteHTML(background)}</span>
                <button class="overlays-obj-btn bg-clear${!background ? ' on' : ''}" title="transparent (don't paint)">∅</button></div>`;
            box.innerHTML = rows + bgrow;

            box.querySelectorAll('.overlays-obj-btn.sign').forEach((b) => b.addEventListener('click', () => { objects[+b.dataset.i].sign = objects[+b.dataset.i].sign === '-' ? '+' : '-'; renderObjects(); applyInstant(); }));
            box.querySelectorAll('.overlays-obj-btn.rm').forEach((b) => b.addEventListener('click', () => { if (objects.length > 1) { objects.splice(+b.dataset.i, 1); renderObjects(); applyInstant(); } }));
            box.querySelectorAll('.overlays-obj-name').forEach((inp) => inp.addEventListener('input', () => { objects[+inp.dataset.i].name = inp.value; renderAction(); scheduleApply(); }));
            box.querySelectorAll('.overlays-palette[data-i] .overlays-swatch').forEach((sw) => sw.addEventListener('click', () => { objects[+sw.parentElement.dataset.i].color = sw.dataset.rgb.split(',').map(Number); renderObjects(); applyInstant(); }));
            box.querySelectorAll('.overlays-palette[data-bg] .overlays-swatch').forEach((sw) => sw.addEventListener('click', () => { background = sw.dataset.rgb.split(',').map(Number); renderObjects(); applyInstant(); }));
            box.querySelector('.overlays-obj-btn.bg-clear').addEventListener('click', () => { background = null; renderObjects(); applyInstant(); });

            const add = els.modelBody.querySelector('.overlays-add-obj');
            if (add) { add.disabled = objects.length >= MAX_OBJECTS; add.textContent = `+ Add object (${objects.length}/${MAX_OBJECTS})`; }
        }

        function addObject() {
            if (objects.length >= MAX_OBJECTS) return;
            objects.push({ name: '', color: PALETTE[objects.length % PALETTE.length], sign: '+' });
            renderObjects();  // no apply — the new row has no name yet
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
            if (!current) { els.action.innerHTML = ''; return; }
            const hasObj = namedObjects().length > 0;
            if (mode === 'data') {
                let txt;
                if (!hasObj) txt = 'name an object';
                else if (status.state === 'loading') txt = 'loading…';
                else if (status.state === 'error') txt = 'error — see log';
                else txt = 'following scrub';
                els.action.innerHTML = `<div class="overlays-status">${esc(txt)}</div>`;
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
                fetch('/api/overlays/data/cancel', { method: 'POST' }).catch(() => {});
                dataVersion++;
                stopPoll();
                clearOverlays();
                setBadge('off', 'off');
                return;
            }
            dataVersion++;  // bust the per-frame img cache so changed objects/colours re-pull
            fetch('/api/overlays/data/configure', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: window.currentDataset, model: current, objects: payloadObjects(), background: bgPayload(), cameras: selectedCameras ? [...selectedCameras] : [] }),
            }).then((r) => {
                // A run owns the obs stream (one writer at a time) — surface it, don't silently fail.
                if (r.status === 409) { setBadge('run active', 'error'); stopPoll(); clearOverlays(); return; }
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
                    body: JSON.stringify({ model: current, objects: payloadObjects(), background: bgPayload(), cameras: camsArg(), style: ctl.style, smooth: ctl.smooth, method: ctl.method }),
                }).then(() => startPoll()).catch(() => {});
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
        function isLiveOn() { return OverlayGate.shouldDraw(mode, current, objectsReady(), status.state); }

        // ---- status polling + badge ----
        function startPoll() { stopPoll(); pollTimer = setInterval(refreshStatus, 500); refreshStatus(); }
        function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } lastDiag = ''; }

        function refreshStatus() {
            const url = mode === 'live'
                ? '/api/overlays/live/status?model=' + encodeURIComponent(current || '')  // per-model state
                : '/api/overlays/data/status';
            fetch(url).then((r) => r.json()).then((s) => {
                status = s;
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

        // ---- data: per-frame overlay renderer (hooked from app.js loadAllFrames) ----
        function onFrame() {
            if (mode !== 'data') return;
            const ds = window.datasets && window.datasets[window.currentDataset];
            if (!ds) return;
            const showable = current && objectsReady() && window.currentDataset && window.currentEpisode !== null;
            if (showable) {
                // Feed the worker the current frame: the backend decodes it + publishes it to the obs
                // stream (no-op if the frame is unchanged). Called on frame change AND the status poll,
                // so a re-visited frame re-publishes and the overlay is never stale.
                fetch('/api/overlays/data/publish', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_id: window.currentDataset, episode: window.currentEpisode, frame: window.currentFrame }),
                }).catch(() => {});
            }
            for (const cam of ds.camera_keys) {
                const img = document.getElementById(`overlay-${safeCam(cam)}`);
                if (!img) continue;
                if (!showable || !(selectedCameras && selectedCameras.has(cam))) { img.style.display = 'none'; img.src = ''; continue; }
                img.onload = () => { img.style.display = 'block'; };
                img.onerror = () => { img.style.display = 'none'; };
                // Re-pull each tick so the worker's result (which lags playback, like the live feed)
                // refreshes; the backend PNG-caches by overlay seq so an unchanged result is cheap.
                img.src = `/api/overlays/data/${encodeURIComponent(window.currentDataset)}/frame/${window.currentEpisode}/${window.currentFrame}?camera=${encodeURIComponent(cam)}&v=${dataVersion}-${frameTick++}`;
            }
        }

        function clearOverlays() {
            if (mode === 'data') document.querySelectorAll('#camera-grid .overlay-layer').forEach((i) => { i.style.display = 'none'; i.src = ''; });
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
