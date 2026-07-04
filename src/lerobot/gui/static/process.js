// Data editing — thin "Process dataset…" commit menu + job tray. The effect is
// configured (and previewed live) in the overlay panel; this menu just persists
// that same effect to every episode as a new dataset. The heavy work runs in a
// worker subprocess; this polls /api/process/jobs (same model as the Transfers tray).

(function () {
    let modal = null;
    let ctx = null;          // {datasetId, objects, cameras, effect, effectLabel} captured on open
    let jobs = [];
    let pollTimer = null;
    let onCountChange = null;  // overlays.js badge callback
    let openedPreviews = new Set();  // preview job_ids already auto-opened (open once)
    // Measured on an RTX 5090: ~90ms/frame/camera steady-state + ~6s SAM3 load.
    const MS_PER_FRAME_CAM = 90;
    const LOAD_S = 6;

    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    let _inited = false;
    function init(opts) {
        if (opts && opts.onCountChange) onCountChange = opts.onCountChange;  // overlays wires its badge
        if (!_inited) {
            _inited = true;
            // Click-outside closes the Processing popover (mirrors the Transfers tray).
            document.addEventListener('click', (e) => {
                const pop = document.getElementById('proc-popover');
                const ind = document.getElementById('proc-indicator');
                if (!pop || pop.hidden) return;
                if ((ind && ind.contains(e.target)) || pop.contains(e.target)) return;
                pop.hidden = true;
            });
        }
        // Render the indicator + resume polling if a job was left running.
        refreshJobs();
    }

    const activeCount = () => jobs.filter((j) => j.status === 'pending' || j.status === 'running').length;

    function startPoll() { if (!pollTimer) pollTimer = setInterval(refreshJobs, 1000); }
    function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

    function refreshJobs() {
        fetch('/api/process/jobs').then((r) => r.json()).then((d) => {
            jobs = d.jobs || [];
            // Auto-open AND navigate to a preview the moment it completes (once) —
            // the point of a preview is to look at it, so open it and jump to its
            // first episode instead of just adding it to the tree.
            for (const j of jobs) {
                if (j.preview && j.status === 'complete' && !openedPreviews.has(j.job_id)) {
                    openedPreviews.add(j.job_id);
                    if (j.out_root && typeof window.openDataset === 'function') {
                        Promise.resolve(window.openDataset(j.out_root)).then(() => {
                            if (typeof window.selectEpisode === 'function') window.selectEpisode(j.out_root, 0, 0);
                            if (typeof window.renderSources === 'function') window.renderSources();
                        }).catch(() => {});
                    }
                }
            }
            if (onCountChange) onCountChange(activeCount());
            renderJobs();
            // Keep polling while something is in flight, or a view showing jobs is open.
            const pop = document.getElementById('proc-popover');
            const viewing = (modal && modal.style.display !== 'none') || (pop && !pop.hidden);
            if (activeCount() > 0 || viewing) startPoll();
            else stopPoll();
        }).catch(() => {});
    }

    // ---- modal ----
    function ensureModal() {
        if (modal) return modal;
        modal = document.createElement('div');
        modal.className = 'proc-modal';
        modal.innerHTML = `
            <div class="proc-box">
                <div class="proc-head">
                    <span class="proc-title">Process dataset</span>
                    <button class="proc-close" title="close (Esc)">&times;</button>
                </div>
                <div class="proc-body">
                    <div class="proc-protected"></div>
                    <div class="proc-effect-summary"></div>
                    <div class="proc-row">
                        <div><label class="proc-label">Copies / episode</label>
                            <input class="proc-variants" type="number" min="1" max="10" value="1"></div>
                        <div class="proc-grow"><label class="proc-label">New dataset name</label>
                            <input class="proc-name" type="text" placeholder="myset_aug"></div>
                    </div>
                    <div class="proc-hint proc-est"></div>
                    <div class="proc-error"></div>
                    <div class="proc-actions-row">
                        <button class="proc-preview" title="Run on just the current episode (~seconds) so you can check the segmentation + effect before the full run">Preview this episode</button>
                        <button class="proc-start" title="Run on every episode and write the augmented dataset">Process all episodes</button>
                    </div>
                </div>
                <div class="proc-jobs-note proc-hint"></div>
            </div>`;
        document.body.appendChild(modal);
        const close = () => { modal.style.display = 'none'; refreshJobs(); };
        modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
        modal.querySelector('.proc-close').addEventListener('click', close);
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.style.display === 'flex') close(); });
        modal.querySelector('.proc-variants').addEventListener('input', updateEstimate);
        modal.querySelector('.proc-start').addEventListener('click', () => start(false));
        modal.querySelector('.proc-preview').addEventListener('click', () => start(true));
        return modal;
    }

    function open(c) {
        ctx = c;
        const m = ensureModal();
        // Protected-objects note (foreground that won't be altered).
        const names = (ctx.objects || []).map((o) => o.name).filter(Boolean);
        m.querySelector('.proc-protected').innerHTML =
            `<b>Protected (kept):</b> ${names.length ? esc(names.join(', ')) : '<i>name an object first</i>'}`
            + ` &middot; background gets the effect.`;
        // The effect is chosen + previewed in the overlay panel; here we just echo it.
        const label = ctx.effect ? (ctx.effectLabel || ctx.effect.key) : null;
        m.querySelector('.proc-effect-summary').innerHTML = label
            ? `<b>Effect:</b> ${esc(label)} <span class="proc-effect-hint">— change it in the Overlays panel; the tile previews it.</span>`
            : `<span class="proc-effect-none">Pick an effect in the Overlays panel first (the tile previews it live).</span>`;
        // Default output name from the dataset id.
        const base = (ctx.datasetId || 'dataset').split(/[\\/]/).filter(Boolean).pop();
        m.querySelector('.proc-name').value = `${base}_aug`;
        m.querySelector('.proc-error').textContent = '';
        updateEstimate();
        m.style.display = 'flex';
        refreshJobs();
    }

    const fmtDur = (s) => s < 90 ? `~${Math.max(1, Math.round(s))}s` : `~${Math.round(s / 60)} min`;

    // Rough wall-clock heads-up from the measured per-frame rate, so the user
    // knows a full run is minutes-to-an-hour before committing — and why the
    // per-episode preview exists.
    function updateEstimate() {
        const el = modal && modal.querySelector('.proc-est');
        if (!el) return;
        const eps = (window.episodes && window.episodes[ctx.datasetId]) || [];
        const ds = window.datasets && window.datasets[ctx.datasetId];
        const nCam = (ctx.cameras && ctx.cameras.length) || (ds && ds.camera_keys ? ds.camera_keys.length : 1);
        const variants = Math.max(1, Number(modal.querySelector('.proc-variants').value) || 1);
        const totalFrames = eps.reduce((a, e) => a + (e.length || 0), 0);
        const curLen = (window.currentEpisode != null && eps[window.currentEpisode]) ? eps[window.currentEpisode].length : 0;
        const cost = (frames) => LOAD_S + frames * nCam * MS_PER_FRAME_CAM / 1000;
        el.innerHTML = `Preview (this episode): <b>${fmtDur(cost(curLen))}</b>`
            + ` &middot; Full run (${eps.length} ep × ${nCam} cam${variants > 1 ? ` × ${variants}` : ''}): <b>${fmtDur(cost(totalFrames * variants))}</b>`;
    }

    // preview=true runs the pipeline on just the current episode (~seconds) and
    // auto-opens the result so the user can check tracking + effect before the
    // full commit. preview=false runs every episode. Randomization is always
    // per-episode (consistent within a trajectory — per-frame would flicker).
    function start(preview) {
        const errEl = modal.querySelector('.proc-error');
        errEl.textContent = '';
        if (!ctx.effect || !ctx.effect.key) { errEl.textContent = 'Pick an effect in the Overlays panel first.'; return; }
        const objects = (ctx.objects || []).filter((o) => (o.name || '').trim());
        if (!objects.length) { errEl.textContent = 'Name at least one object in the overlay panel first.'; return; }
        if (preview && window.currentEpisode == null) { errEl.textContent = 'Open an episode to preview.'; return; }
        const payload = {
            source_id: ctx.datasetId,
            objects,
            effect: ctx.effect.key,
            effect_params: ctx.effect.params || {},
            multi_instance: ctx.multiInstance !== false,
            apply_mode: 'per_episode',
            variants: preview ? 1 : Math.max(1, Number(modal.querySelector('.proc-variants').value) || 1),
            cameras: ctx.cameras && ctx.cameras.length ? ctx.cameras : null,
            out_name: modal.querySelector('.proc-name').value.trim() || null,
            preview,
            episodes: preview ? [Number(window.currentEpisode)] : null,
        };
        const btns = [modal.querySelector('.proc-preview'), modal.querySelector('.proc-start')];
        const btn = preview ? btns[0] : btns[1];
        const label = btn.textContent;
        btns.forEach((b) => { b.disabled = true; });
        btn.textContent = 'Starting…';
        fetch('/api/process/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        }).then(async (r) => {
            btns.forEach((b) => { b.disabled = false; });
            btn.textContent = label;
            if (!r.ok) { const d = await r.json().catch(() => ({})); errEl.textContent = (d && d.detail) || `Error ${r.status}`; return; }
            refreshJobs();
        }).catch((err) => { btns.forEach((b) => { b.disabled = false; }); btn.textContent = label; errEl.textContent = String(err); });
    }

    // ---- job cards ----
    const STATUS_CLS = { pending: 'run', running: 'run', complete: 'ok', failed: 'err', cancelled: 'err' };

    function jobCard(j) {
        const pct = j.frames_total ? Math.floor(100 * j.frames_done / j.frames_total) : 0;
        const terminal = ['complete', 'failed', 'cancelled'].includes(j.status);
        let detail;
        if (j.status === 'complete') detail = `done · ${j.episodes_done} episodes / ${j.frames_done} frames`;
        else if (j.status === 'failed') detail = `failed · ${esc(j.error || 'see log')}`;
        else if (j.status === 'cancelled') detail = `cancelled · ${j.episodes_done} episodes written`;
        else detail = `${esc(j.stage || 'starting')} · ${j.frames_done}/${j.frames_total || '?'} frames`
            + (j.current_episode != null ? ` · ep ${j.current_episode}` : '');
        const actions = [];
        if (!terminal) actions.push(`<button class="proc-job-btn" data-act="cancel" data-id="${j.job_id}">Cancel</button>`);
        if (j.status === 'complete') actions.push(`<button class="proc-job-btn primary" data-act="open" data-id="${j.job_id}">Open dataset</button>`);
        if (terminal) actions.push(`<button class="proc-job-btn" data-act="dismiss" data-id="${j.job_id}">Dismiss</button>`);
        const tag = j.preview ? '<span class="proc-job-tag">preview</span> ' : '';
        return `<div class="proc-job">
            <div class="proc-job-top"><span class="proc-job-name">${tag}${esc(j.out_repo_id)}</span>
                <span class="proc-job-status ${STATUS_CLS[j.status] || ''}">${esc(j.status)}</span></div>
            <div class="proc-bar"><div class="proc-bar-fill ${STATUS_CLS[j.status] || ''}" style="width:${j.status === 'complete' ? 100 : pct}%"></div></div>
            <div class="proc-job-detail">${detail}</div>
            <div class="proc-job-actions">${actions.join('')}</div>
        </div>`;
    }

    // ---- global tray (top bar, next to Transfers): always-visible progress ----
    // Jobs run in a detached worker; the config window is dismissable, so progress
    // lives here so it's never lost when the window is closed.
    function renderIndicator() {
        const ind = document.getElementById('proc-indicator');
        const label = document.getElementById('proc-indicator-label');
        if (!ind) return;
        const n = activeCount();  // always visible (like Transfers); highlight when active
        ind.classList.toggle('active', n > 0);
        if (label) label.textContent = n > 0 ? `Processing (${n})` : 'Processing';
    }

    function renderJobs() {
        renderIndicator();
        const note = modal && modal.querySelector('.proc-jobs-note');
        if (note) {
            const n = activeCount();
            note.textContent = n > 0 ? `${n} job(s) running — track them in the "Processing" tray (top bar).`
                : 'Jobs run in the background; track them in the "Processing" tray (top bar).';
        }
        const box = document.getElementById('proc-jobs-list');
        if (!box) return;
        box.innerHTML = jobs.length ? jobs.map(jobCard).join('') : '<div class="proc-hint">No processing jobs.</div>';
        box.querySelectorAll('.proc-job-btn').forEach((b) => b.addEventListener('click', () => jobAction(b.dataset.act, b.dataset.id)));
    }

    function togglePopover() {
        const pop = document.getElementById('proc-popover');
        if (!pop) return;
        pop.hidden = !pop.hidden;
        if (!pop.hidden) refreshJobs();
    }

    function jobAction(act, id) {
        if (act === 'open') {
            const j = jobs.find((x) => x.job_id === id);
            if (j && j.out_root && typeof window.openDataset === 'function') {
                Promise.resolve(window.openDataset(j.out_root)).then(() => {
                    if (typeof window.selectEpisode === 'function') window.selectEpisode(j.out_root, 0, 0);
                }).catch(() => {});
            }
            return;
        }
        fetch(`/api/process/${id}/${act}`, { method: 'POST' }).then(() => refreshJobs()).catch(() => {});
    }

    window.ProcessData = { init, open, refreshJobs, togglePopover };
    // Self-init at load so the top-bar indicator renders + polls regardless of which
    // tab is open; overlays.js calls init() again later just to wire onCountChange.
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => init());
    else init();
})();
