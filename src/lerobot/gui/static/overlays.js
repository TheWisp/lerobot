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
        // Data tab: background defaults Random -> the GreenAug recipe is zero-click.
        // Run tab: everything defaults None -> pure observability (chrome only).
        // Both tabs start inert: every region, background included, treated as None. The data
        // tab used to default the background to Random, so picking a model immediately buried
        // the scene in static and the two tabs behaved differently for no reason a user could see.
        let backgroundTreatment = { key: 'none', params: {} };
        // Segment ALL instances of each object (both arms) vs the single largest. Same control
        // on both tabs; the DEFAULT differs on purpose. Data edits pixels, and a treatment that
        // protects only one of two arms silently corrupts the written dataset, so it starts All.
        // The run tab is a live debug view where every extra masklet costs frame rate, so it
        // starts Largest — visible and one click away, not hidden.
        let multiInstance = mode === 'data';
        let boxMethod = 'tracker';  // which SAM3 box API a drag uses; see the button tooltips
        // SAM inference resolution (a LOAD-TIME knob: changing it respawns the worker; the
        // batch job inherits it so preview == commit). Default to the backend's default preset.
        let resolution = (RESOLUTIONS.find((r) => /default/i.test(r.label || '')) || RESOLUTIONS[0] || { value: null }).value;
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
        // Per-dataset scope (data mode): episodes within a dataset share a scene and
        // task, datasets do not — so objects/treatments/signs/instances/cameras are
        // remembered PER DATASET and swapped on switch. Model + resolution stay
        // global: they are worker identity, and per-dataset values would respawn the
        // worker on every switch. Unbounded map by decision (LRU later if it matters);
        // in-memory, so a reload starts clean.
        const datasetConfigs = new Map();  // dataset id -> snapshot
        let scopedDatasetId = (mode === 'data' && window.currentDataset) || null;
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
        // A segmenter is ready with nothing named — the tiles are its input. Requiring a
        // word would mean no worker, and with no worker there is nothing to click.
        // Run tab only. A teleop stream never wraps and never changes episode, so a clicked
        // object's tracker state is never invalidated. The data tab loops episodes constantly,
        // and a click cannot be re-derived on a frame it was not made on — measured, it comes
        // back at IoU ~0.93 across a wrap and has no mechanism at all across episodes. Rather
        // than ship a designation that quietly stops being true, the data tab keeps text
        // prompts only until the gesture is stored outside the worker.
        const clickCapable = () => mode === 'live' && SEGMENTERS.includes(current);
        const objectsReady = () => !requiresObjects() || namedObjects().length > 0
            || clickCapable() || objects.some((o) => o.clicked);
        // With nothing designated the background is the whole frame, so picking a model
        // would bury the scene in static — and "Process dataset" would write it.
        const bgPayload = () => (objects.some((o) => (o.name || '').trim() || o.clicked)
            ? backgroundTreatment : { key: 'none', params: {} });
        // Only run the text detector when there is a word to look for. A clicked object's
        // name is a label, never a query.
        const textDetectionOn = () => objects.some((o) => (o.name || '').trim() && !o.clicked);
        // What goes to the backend: named objects with their per-region treatment + sign.
        function payloadObjects() {
            const tr = (o) => o.treatment || { key: 'none', params: {} };
            // Clicked objects are included so their treatment applies (keyed by name
            // worker-side); the `clicked` flag keeps them out of the text prompt.
            // Nothing named means no text concepts — send an empty list. A fallback here
            // once sent a concept literally named "object", which matched a 47k px blob,
            // lost it, and rebuilt the tracker on every loss. That was the reported
            // "detection is not sticky".
            return objects.filter((o) => (o.name || '').trim())
                .map((o) => ({ name: o.name.trim(), sign: o.sign || '+', treatment: tr(o),
                               ...(o.clicked ? { clicked: true } : {}) }));
        }
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
                // One line: the panel is narrow and this sits above the rows. Detail is
                // on hover.
                const hint = clickCapable()
                    ? 'Describe an object, or <b>click / box it on a camera tile</b>.'
                    : 'Describe an object to detect.';
                const hintFull = mode === 'data'
                    ? 'Each object is a region with a treatment; the Background row is a region too. The tile shows the live WYSIWYG result — the glow + label is a detection aid, not part of the written output.'
                    : 'Open-vocab names, outlined + labelled on the tile in each object\'s own colour. + include / − exclude. Treatments default to None here — the run tab shows real pixels; set one only to visualise. Click a camera tile to segment what is under it, or drag a box around it. Name edits apply ~1s after you stop typing.';
                els.modelBody.innerHTML = `
                    <label class="overlays-label">${esc(ctrl.label || 'Objects')}</label>
                    <div class="overlays-hint" title="${esc(hintFull)}">${hint}</div>
                    <div class="overlays-objrows"></div>
                    <button class="overlays-add-obj">+ Add object</button>
                    ${clickCapable() ? `
                    <label class="overlays-label" title="Which model reads a dragged box. Clicks are unaffected, and typed object names play no part either way.">box read by</label>
                    <span class="overlays-seg overlays-boxmethod">
                        <button class="overlays-seg-btn${boxMethod === 'tracker' ? ' sel' : ''}" data-boxm="tracker" title="Tracking model: cuts out whatever the box encloses. Predictable, but merges objects that touch — a ring against a dowel came back as both (IoU 0.445 against the ring alone).">Tracker</button>
                        <button class="overlays-seg-btn${boxMethod === 'exemplar' ? ' sel' : ''}" data-boxm="exemplar" title="Detection model — the one that finds objects from words — shown the box as an example, then searching the image for it. Separates touching objects in principle, but with only a box to go on it can lock onto the wrong one: in that same scene it took the dowel.">Detector</button>
                    </span>` : ''}
                    <label class="overlays-label" title="All: keep every instance of each object (e.g. both robot arms). Largest: keep only the single biggest match.">instances</label>
                    <span class="overlays-seg overlays-multi">
                        <button class="overlays-seg-btn${multiInstance ? ' sel' : ''}" data-multi="1">All</button>
                        <button class="overlays-seg-btn${multiInstance ? '' : ' sel'}" data-multi="0">Largest</button>
                    </span>
                    ${RESOLUTIONS.length ? `<label class="overlays-label" title="SAM inference resolution — lower is faster; Balanced measured equal-or-better masks than Full at ~1.8× the speed. Changing it reloads the model.">Quality</label>
                    <select class="overlays-select overlays-res">${RESOLUTIONS.map((r) => `<option value="${r.value}"${r.value === resolution ? ' selected' : ''}>${esc(r.label)}</option>`).join('')}</select>` : ''}
                    <label class="overlays-label">cameras</label>
                    <div class="overlays-cameras"></div>
                    ${mode === 'data' ? '<button class="overlays-process" title="Apply these per-region treatments to every episode as a new dataset">⚙ Process dataset…</button>' : ''}`;
                els.modelBody.querySelector('.overlays-add-obj').addEventListener('click', addObject);
                const procBtn = els.modelBody.querySelector('.overlays-process');
                if (procBtn) procBtn.addEventListener('click', openProcess);
                // Repaint the selection IN PLACE rather than re-rendering the body: a
                // re-render would blow away a half-typed object name (the same reason the
                // tint picker updates in place).
                els.modelBody.querySelectorAll('.overlays-boxmethod .overlays-seg-btn').forEach((b) => {
                    b.addEventListener('click', () => {
                        if (b.dataset.boxm === boxMethod) return;
                        boxMethod = b.dataset.boxm;
                        els.modelBody.querySelectorAll('.overlays-boxmethod .overlays-seg-btn').forEach((o) =>
                            o.classList.toggle('sel', o.dataset.boxm === boxMethod));
                        applyInstant();
                    });
                });
                els.modelBody.querySelectorAll('.overlays-multi .overlays-seg-btn').forEach((b) => {
                    b.addEventListener('click', () => {
                        const want = b.dataset.multi === '1';
                        if (want === multiInstance) return;
                        multiInstance = want;
                        els.modelBody.querySelectorAll('.overlays-multi .overlays-seg-btn').forEach((o) =>
                            o.classList.toggle('sel', (o.dataset.multi === '1') === multiInstance));
                        applyInstant();
                    });
                });
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
            const camTip = (o) => (o.clicked && o.cam
                ? ` title="Designated on ${esc(o.cam.split('.').pop())} by clicking. A click marks a place in ONE camera's image, so unlike a typed name it applies only to that camera — click it again on another tile to add it there."`
                : '');
            const nameInput = (o, i) => `<input class="overlays-obj-name" type="text" data-i="${i}"${camTip(o)} placeholder="${(i === 0 && !anyNamed) ? 'object' : 'object name (e.g. robot arm)'}" value="${esc(o.name)}">`;

                // One line per region: [+/− polarity] [name] [treatment icons] [× remove].
                // The polarity pill is a first-class per-concept filter: green + = add to the
                // foreground, red − = SUPPRESS (subtract from it — e.g. arm − gripper). The
                // Background row uses slot placeholders so its columns line up.
                const pol = (o, i) => `<button class="overlays-pol ${o.sign === '-' ? 'neg' : 'pos'}" data-i="${i}" title="${o.sign === '-' ? '− suppress: subtracted from the foreground — click to add' : '+ foreground: added — click to suppress'}">${o.sign === '-' ? '−' : '+'}</button>`;
                const rmBtn = (i) => `<span class="overlays-obj-rm" data-i="${i}" title="${objects.length > 1 ? 'remove' : 'clear'}">&times;</span>`;
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
                box.querySelectorAll('.overlays-obj-rm').forEach((b) => b.addEventListener('click', () => {
                    const i = +b.dataset.i;
                    const gone = objects[i];
                    // Removing the LAST row clears it in place instead of deleting it — a
                    // text row needs an input to type into. Every row is therefore always
                    // one click from gone, instead of the first row being immortal.
                    if (objects.length > 1) objects.splice(i, 1);
                    else objects[0] = { name: '', sign: '+', treatment: { key: 'none', params: {} } };
                    // A clicked object lives in the worker's tracker, not in the prompt —
                    // dropping its row must also tell the worker to stop tracking it.
                    if (gone && gone.clicked) {
                        pushClickOp({ clicks_remove: { [gone.cam]: [gone.workerName || gone.name] } }, mode);
                    }
                    renderObjects(); applyInstant();
                }));

            box.querySelectorAll('.overlays-obj-name').forEach((inp) => inp.addEventListener('input', () => {
                const o = objects[+inp.dataset.i];
                const was = o.name;
                o.name = inp.value;
                renderAction();
                // A clicked object is tracked, not searched for: relabel it. Pushing the
                // name as a prompt would ask the detector to find what has no word.
                if (o.clicked) { scheduleRename(o); return; }
                scheduleApply();
            }));

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

        // Debounced like any name edit, but travels on clicks_rename so the mask survives.
        let renameTimer = null;
        function scheduleRename(o) {
            clearTimeout(renameTimer);
            renameTimer = setTimeout(() => {
                const to = (o.name || '').trim();
                // Rename from the name the WORKER holds. Debouncing means intermediate
                // keystrokes were never sent, so they match nothing there.
                const from = o.workerName;
                if (!to || !from || to === from) return;
                pushClickOp({ clicks_rename: { [o.cam]: { [from]: to } } }, mode);
                o.workerName = to;
            }, 800);
        }

        // Name edits restart tracking, so debounce; sign/treatment/remove are display-only → instant.
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
                // One camera by default, on both tabs. Data mode used to select ALL of them while
                // the run tab selected one, so the same model cost 4x the inference and drew a
                // different thing depending on which tab you opened it from. Saliency is the
                // exception and says so: it colorizes the running policy's per-camera output, so
                // selecting one camera would leave every other tile blank.
                const allCams = modelSpec(current)?.load_cost === 'fast';
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
                // Cancel PARKS the worker (model stays in VRAM for the next start) and says so
                // in its response. Polling stops here, so renderStatus never runs while off —
                // this response is the only chance to show the truthful "warm" state.
                fetch('/api/overlays/data/cancel', { method: 'POST', headers: ovlHeaders() })
                    .then((r) => r.json())
                    .then((d) => { if (d && d.parked) setBadge('off · model warm', 'off'); })
                    .catch(() => {});
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
                body: JSON.stringify({ dataset_id: window.currentDataset, model: current, objects: payloadObjects(), background_treatment: bgPayload(), multi_instance: multiInstance, box_method: boxMethod, resolution, cameras: selectedCameras ? [...selectedCameras] : [], text_detection: textDetectionOn() }),
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
                    body: JSON.stringify({ model: current, objects: payloadObjects(), background_treatment: bgPayload(), cameras: camsArg(), style: ctl.style, smooth: ctl.smooth, method: ctl.method, resolution, multi_instance: multiInstance, box_method: boxMethod, text_detection: textDetectionOn() }),
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
                    body: JSON.stringify({ objects: payloadObjects(), background_treatment: bgPayload(), style: ctl.style, smooth: ctl.smooth, method: ctl.method, multi_instance: multiInstance, box_method: boxMethod, text_detection: textDetectionOn() }),
                }).catch(() => {});
            }
        }

        // Draw iff the backend machine says ACTIVE — single source of truth. `started` (below) is
        // only the frontend's launch *request*, not a second copy of "is it running". The decision
        // lives in OverlayGate.shouldDraw (overlay_gate.js) so it is unit-tested in isolation.
        function isLiveOn() { return !status.busy && OverlayGate.shouldDraw(mode, current, objectsReady(), status.state); }

        // ---- status polling + badge ----
        function startPoll() { stopPoll(); pollTimer = setInterval(refreshStatus, 500); refreshStatus(); if (mode === 'data') startOverlayStream(); }
        // Forget the cached status: the stop paths call this right after /live/stop, and
        // with no further poll a stale 'active' leaves the tiles wearing the crosshair.
        function stopPoll() {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
            stopOverlayStream(); lastDiag = '';
            status = { state: 'idle' };
            syncArmedCursor();
        }

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
                syncArmedCursor();  // the tiles ARE the control now — say so with the cursor
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
                // ACTIVE without a publisher = the worker is PARKED (canceled but kept warm so the
                // next start skips the model load). Saying "live" here would be a lie — nothing is
                // being overlaid; say what is actually true: off, with the model held in VRAM.
                if (!s.publishing) { setBadge('off · model warm', 'off'); return; }
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
        // Re-publish the current frame. The worker only reads controls when it processes
        // one, so on a paused episode a gesture has to hand it a frame or the op just sits
        // there until the next gesture overwrites it.
        this.nudge = () => {
            if (mode !== 'data' || !window.currentDataset || window.currentEpisode === null) return;
            fetch('/api/overlays/data/publish', {
                method: 'POST', headers: ovlHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({
                    dataset_id: window.currentDataset, episode: window.currentEpisode,
                    frame: window.currentFrame, force: true,
                }),
            }).then(() => { dataVersion++; onFrame(); }).catch(() => {});
        };

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

        // A clicked object is NOT a text concept, and the snapshot has to carry that. Dropping
        // the flag restored it as a plain named row, so the worker handed the label to the text
        // detector and hunted "object_3" forever — the reported "clicked objects go lost after
        // a while", which was really "lost on a dataset switch".
        const carry = (o) => ({
            name: o.name, sign: o.sign,
            treatment: { key: o.treatment.key, params: Object.assign({}, o.treatment.params) },
            ...(o.clicked ? { clicked: true, cam: o.cam, workerName: o.workerName } : {}),
        });
        function snapshotConfig() {
            return {
                objects: objects.map(carry),
                backgroundTreatment: {
                    key: backgroundTreatment.key, params: Object.assign({}, backgroundTreatment.params),
                },
                multiInstance,
                cameras: selectedCameras ? [...selectedCameras] : null,
            };
        }
        function restoreConfig(saved) {
            if (saved) {
                objects = saved.objects.map(carry);
                backgroundTreatment = {
                    key: saved.backgroundTreatment.key,
                    params: Object.assign({}, saved.backgroundTreatment.params),
                };
                multiInstance = saved.multiInstance;
                selectedCameras = saved.cameras ? new Set(saved.cameras) : null;
            } else {
                // A dataset never seen. Fully inert — every region None, background included —
                // which is also what the auto-opened process PREVIEW gets, so the source's
                // treatments are never re-applied on top of already-treated pixels (the
                // double-tint report this scoping exists to fix). Background was Random here,
                // which meant opening a dataset could bury it in static before you asked for
                // anything.
                objects = [{ name: '', sign: '+', treatment: { key: 'none', params: {} } }];
                backgroundTreatment = { key: 'none', params: {} };
                multiInstance = true;
                selectedCameras = null;  // resolved to the FIRST camera once they are known
            }
        }

        // A dataset switch (app.js calls this exactly then) swaps the panel's scoped
        // config: save the outgoing dataset's, restore the incoming one, then re-read
        // the new dataset's cameras. Only the data panel tracks a dataset.
        this.refreshCameras = () => {
            if (mode !== 'data') return;
            const id = window.currentDataset || null;
            if (id !== scopedDatasetId) {
                if (scopedDatasetId !== null) datasetConfigs.set(scopedDatasetId, snapshotConfig());
                if (datasetConfigs.has(id)) restoreConfig(datasetConfigs.get(id));
                else if (scopedDatasetId !== null) restoreConfig(null);
                // else: first dataset of the session ADOPTS whatever was configured
                // before it opened — resetting here would wipe a config typed moments
                // ago, and there is no previous dataset it could have belonged to.
                scopedDatasetId = id;
                if (current) renderBody();  // repaint rows/instances for the restored config
            }
            if (current) loadCameras();
        };
        this.onFrame = onFrame;
        this.isLiveOn = isLiveOn;
        this.addClickedObject = (cam) => {
            // Reuse the empty starter row rather than leaving a blank one above the result.
            const blank = objects.findIndex((o) => !(o.name || '').trim());
            // The same cap "+ Add object" enforces, over the same list. Counting only
            // clicked rows let the panel show more than the worker would admit.
            if (blank < 0 && objects.length >= MAX_OBJECTS) {
                setBadge(`click: at the ${MAX_OBJECTS}-object limit`, 'idle');
                return null;
            }
            // Never reuse a name: numbering by row count reuses one after a delete, and
            // the worker then overwrites the live object holding it.
            // Prefix with the camera. Clicks are per-camera but the worker keys treatments and
            // colours by NAME, so a bare object_N on two cameras silently shared both — and the
            // rows were indistinguishable in the panel.
            clickCount[cam] = (clickCount[cam] || 0) + 1;
            const name = `${cam.split('.').pop()}_${clickCount[cam]}`;
            const row = { name, workerName: name, sign: '+', treatment: { key: 'none', params: {} }, clicked: true, cam };
            if (blank >= 0) objects[blank] = row; else objects.push(row);
            renderObjects();
            // Push the config, not just the row: the first designation un-suppresses the
            // background treatment, so the payload changed.
            applyInstant();
            return name;
        };
        this.sync = sync;
        this.mode = mode;
        this.isClickCapable = clickCapable;
        // "Is the worker that would serve a gesture running?" — deliberately NOT isLiveOn():
        // that is the run tab's DRAW gate and is false for data mode by construction
        // (OverlayGate.shouldDraw hardcodes mode === 'live', because the data tab paints via
        // per-frame pulls instead). Gestures are accepted exactly when the shared worker is
        // ours and active, which is the same condition on both tabs.
        this.isOverlayLive = () => !!current && !status.busy && status.state === 'active';
        this.isCameraOn = (cam) => !!(selectedCameras && selectedCameras.has(cam));
    }

    // Click/box to segment. No arming control: a segmenter being live IS the
    // mode. The tiles own the gesture and the coordinate mapping (installTileGestures),
    // this owns "is it on" and the wire format. Prompts are FRAME pixels; the worker seeds
    // its tracker from them and the thing you pointed at becomes a concept named object_N,
    // tracked and drawn like any other.
    const panelFor = (mode) => panels.find((p) => p.mode === mode) || null;

    function isClickToSegmentOn(camKey, mode) {
        const p = panelFor(mode || 'live');
        return !!(p && p.isClickCapable() && p.isOverlayLive() && p.isCameraOn(camKey));
    }

    function sendClick(camKey, x, y, mode) {
        const p = panelFor(mode || 'live');
        const name = p && p.addClickedObject(camKey);
        if (!name) return;  // at the object cap — the panel already said so
        pushClickOp({ clicks: { [camKey]: [[x, y, 1]] }, click_name: { [camKey]: name } }, mode);
    }

    function sendBox(camKey, x0, y0, x1, y1, mode) {
        const p = panelFor(mode || 'live');
        const name = p && p.addClickedObject(camKey);
        if (!name) return;  // at the object cap — the panel already said so
        pushClickOp({ boxes: { [camKey]: [[x0, y0, x1, y1]] }, click_name: { [camKey]: name } }, mode);
    }

    // One delegated gesture handler per camera grid, shared by both tabs. Delegation (not a
    // listener per tile) because the data tab's grid outlives its tiles — innerHTML is
    // replaced each episode, so per-tile listeners would bind to dead elements. Installing
    // is idempotent, so callers need no bookkeeping.
    // The crosshair uses the same predicate the handler gates on, so look and behaviour
    // cannot drift.
    function syncArmedCursor() {
        const armed = panels.some((p) => p.isClickCapable && p.isClickCapable() && p.isOverlayLive());
        document.body.classList.toggle('ovl-click-armed', armed);
    }

    const DRAG_MIN = 5;  // display px; a release below this is a click, not a box
    function installTileGestures(container, mode) {
        if (!container || container.dataset.tileGestures) return;
        container.dataset.tileGestures = '1';
        container.addEventListener('mousedown', (e) => {
            if (e.button !== 0 || e.target.closest('button')) return;  // the zoom button owns its press
            const tile = e.target.closest('[data-cam-cell]');
            if (!tile) return;
            const key = tile.dataset.camCell;
            if (!isClickToSegmentOn(key, mode)) return;
            // Map through the FRAME img's displayed rect: tiles letterbox the frame
            // (`object-fit: contain`), so a raw offsetX/offsetY is wrong by both the bar
            // size and the scale.
            const img = tile.querySelector('img:not(.overlay-layer)');
            if (!img) return;
            const at = (cx, cy) => {
                const nw = img.naturalWidth, nh = img.naturalHeight;
                if (!nw || !nh) return null;
                const r = img.getBoundingClientRect();
                const scale = Math.min(r.width / nw, r.height / nh);
                const x = (cx - r.left - (r.width - nw * scale) / 2) / scale;
                const y = (cy - r.top - (r.height - nh * scale) / 2) / scale;
                return { x, y, nw, nh, inside: x >= 0 && y >= 0 && x < nw && y < nh };
            };
            const start = at(e.clientX, e.clientY);
            if (!start || !start.inside) return;  // pressed a letterbox bar
            e.preventDefault();  // suppress the browser's native image drag
            // Parent the band to the frame wrapper, which is position:relative on both
            // tabs. The tile is not: the data tab's .camera-panel is static, so a band
            // there escapes to a distant ancestor and draws far from the pointer.
            const box = img.parentElement || tile;
            const band = document.createElement('div');
            band.className = 'obs-click-band';
            box.appendChild(band);
            const onMove = (mv) => {
                const r = box.getBoundingClientRect();
                band.style.left = `${Math.min(e.clientX, mv.clientX) - r.left}px`;
                band.style.top = `${Math.min(e.clientY, mv.clientY) - r.top}px`;
                band.style.width = `${Math.abs(mv.clientX - e.clientX)}px`;
                band.style.height = `${Math.abs(mv.clientY - e.clientY)}px`;
            };
            // Move/up live on document so a drag that leaves the tile still completes, and
            // are removed on release — per gesture, so nothing accumulates.
            const onUp = (up) => {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                band.remove();
                if (Math.hypot(up.clientX - e.clientX, up.clientY - e.clientY) < DRAG_MIN) {
                    sendClick(key, Math.round(start.x), Math.round(start.y), mode);
                    return;
                }
                const end = at(up.clientX, up.clientY);
                if (!end) return;
                const cl = (v, hi) => Math.max(0, Math.min(hi - 1, v));  // release may be off-frame
                const x0 = cl(Math.min(start.x, end.x), start.nw), x1 = cl(Math.max(start.x, end.x), start.nw);
                const y0 = cl(Math.min(start.y, end.y), start.nh), y1 = cl(Math.max(start.y, end.y), start.nh);
                if (x1 - x0 < 3 || y1 - y0 < 3) return;  // degenerate sliver, not a real box
                sendBox(key, Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1), mode);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }
    const clickCount = {};  // cam -> how many objects have EVER been clicked (never reused)
    // POST once; the SERVER re-sends it on later control writes and drops it on teardown.
    // Replaying it here too would outlive the worker — a respawn resets click_seq to 0, so
    // a client-held op would re-apply to a different dataset at the old coordinates.
    function pushClickOp(op, mode) {
        fetch('/api/overlays/live/control', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(op),  // the SERVER assigns click_seq — a per-client clock drops ops
        })
            // Hand the worker a frame so it actually reads what we just wrote (see nudge).
            // After the response, so the control write is in place before the frame arrives.
            .then(() => { const p = panelFor(mode || 'live'); if (p && p.nudge) p.nudge(); })
            .catch(() => {});
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
        // app.js and run.js build the tile grids, so this is the only gesture symbol that
        // needs to be public. The rest stays private — exporting the panels would put every
        // tab's mutable state on window.
        installTileGestures,
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
