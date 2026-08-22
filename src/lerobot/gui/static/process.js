// Data editing — the dataset-wide commit menu + job tray. It applies the
// segmentation configured (and previewed live) in the overlay panel to EVERY
// episode, storing masks.
//
// It used to write a new dataset with the effects burned into the video. That
// is obsolete: masks are stored per frame and effects are dataset-level
// metadata applied when frames are read, so baking costs a full re-encode
// (lossy, slow) to produce something a recipe change can no longer alter. The
// old path still exists at POST /api/process/start for the case where an
// external consumer needs pixels it cannot composite itself; nothing in the UI
// reaches it. The heavy work runs in a worker subprocess; this polls
// /api/process/jobs (same model as the Transfers tray).

(function () {
    let modal = null;
    let ctx = null;          // {datasetId, objects, backgroundTreatment, cameras, multiInstance, model, resolution} captured on open
    let jobs = [];
    let pollTimer = null;
    let onCountChange = null;  // overlays.js badge callback
    let openedPreviews = new Set();  // preview job_ids already auto-opened (open once)

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
                            // Re-scan, not re-render: the job's own output is not in the cache.
                            if (typeof window.refreshExpandedSources === 'function') {
                                window.refreshExpandedSources();
                            }
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
                    <span class="proc-title">Apply segmentation to every episode</span>
                    <button class="proc-close" title="close (Esc)">&times;</button>
                </div>
                <div class="proc-body">
                    <div class="proc-protected"></div>
                    <div class="proc-effect-summary"></div>
                    <div class="proc-scope"></div>
                    <div class="proc-hint proc-est"></div>
                    <div class="proc-error"></div>
                    <div class="proc-actions-row">
                        <button class="proc-start" title="Segment every episode with these settings and store the masks. No video is rewritten.">Apply to all episodes</button>
                    </div>
                </div>
                <div class="proc-jobs-note proc-hint"></div>
            </div>`;
        document.body.appendChild(modal);
        const close = () => { modal.style.display = 'none'; refreshJobs(); };
        modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
        modal.querySelector('.proc-close').addEventListener('click', close);
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.style.display === 'flex') close(); });
        modal.querySelector('.proc-start').addEventListener('click', () => start(false));
        return modal;
    }

    // Human list of the per-region treatments set in the overlay panel ("background:
    // random", "robot arm: tint"), for the modal echo + the commit guard.
    function treatSummary(c) {
        const parts = [];
        const bg = c.backgroundTreatment;
        if (bg && bg.key && bg.key !== 'none') parts.push(`background: ${bg.key}`);
        (c.objects || []).forEach((o) => {
            const t = o.treatment;
            if (o.name && o.sign !== '-' && t && t.key && t.key !== 'none') parts.push(`${o.name}: ${t.key}`);
        });
        return parts;
    }

    function open(c) {
        ctx = c;
        const m = ensureModal();
        const names = (ctx.objects || []).map((o) => o.name).filter(Boolean);
        m.querySelector('.proc-protected').innerHTML =
            `<b>Objects:</b> ${names.length ? esc(names.join(', ')) : '<i>name an object first</i>'}`;
        const ds = window.datasets && window.datasets[ctx.datasetId];
        const cams = (ctx.cameras && ctx.cameras.length) ? ctx.cameras : ((ds && ds.camera_keys) || []);
        m.querySelector('.proc-effect-summary').innerHTML =
            `<b>Cameras:</b> ${cams.length ? esc(cams.map((k) => k.split('.').pop()).join(', ')) : '<i>none selected</i>'}`;
        const eps = (window.episodes && window.episodes[ctx.datasetId]) || [];
        // Say plainly what this writes and what it replaces. The old version of
        // this dialog wrote a second dataset; this one edits the open one, and
        // that difference has to be visible before the button is pressed.
        m.querySelector('.proc-scope').innerHTML =
            `Segments <b>${eps.length}</b> episode${eps.length === 1 ? '' : 's'} with these settings and stores the masks in `
            + `<b>${esc(ctx.datasetId || '')}</b>. Existing masks for these episodes are replaced. `
            + `No video is re-encoded — treatments stay a recipe you can change afterwards.`;
        m.querySelector('.proc-error').textContent = '';
        updateEstimate();
        m.style.display = 'flex';
        refreshJobs();
    }

    const fmtDur = (s) => s < 90 ? `~${Math.max(1, Math.round(s))}s` : `~${Math.round(s / 60)} min`;

    // Rough wall-clock heads-up from the measured per-frame rate, so the user
    // knows a full run is minutes-to-an-hour before committing.
    function updateEstimate() {
        const el = modal && modal.querySelector('.proc-est');
        if (!el) return;
        const eps = (window.episodes && window.episodes[ctx.datasetId]) || [];
        const ds = window.datasets && window.datasets[ctx.datasetId];
        const nCam = (ctx.cameras && ctx.cameras.length) || (ds && ds.camera_keys ? ds.camera_keys.length : 1);
        const totalFrames = eps.reduce((a, e) => a + (e.length || 0), 0);
        // Honest estimate policy: derive the rate from the live preview's MEASURED
        // per-frame compute (same model/resolution/cameras/objects as this job).
        // No measurement -> no number; never a hardcoded rate.
        if (!ctx.computeMs) {
            el.innerHTML = 'No time estimate yet — scrub or play a few frames with the overlay on and reopen; the estimate comes from that measurement.';
            return;
        }
        const cost = totalFrames * nCam * ctx.computeMs / 1000;
        el.innerHTML = `${eps.length} episodes × ${nCam} camera${nCam === 1 ? '' : 's'} `
            + `(${totalFrames.toLocaleString()} frames): <b>≈${fmtDur(cost)}</b>`
            + ` <span class="proc-effect-hint">(from the live preview's measured ${ctx.computeMs.toFixed(0)} ms/frame/cam; excludes model load)</span>`;
    }

    // Apply the panel's current segmentation to every episode. The request is
    // the same one the per-episode Save button makes -- same worker, same live
    // settings, same job tray -- with the full episode list instead of one, so
    // there is no second code path to keep in step.
    function start() {
        const errEl = modal.querySelector('.proc-error');
        errEl.textContent = '';
        const objects = (ctx.objects || []).filter((o) => (o.name || '').trim());
        if (!objects.length) { errEl.textContent = 'Name at least one object in the overlay panel first.'; return; }
        const eps = (window.episodes && window.episodes[ctx.datasetId]) || [];
        if (!eps.length) { errEl.textContent = 'No episodes loaded for this dataset.'; return; }
        if (!window.OverlayStream || !window.OverlayStream.applyToDataset) {
            errEl.textContent = 'The overlay panel is not ready.';
            return;
        }
        const btn = modal.querySelector('.proc-start');
        const episodes = eps.map((e, i) => (e.episode_index != null ? Number(e.episode_index) : i));
        window.OverlayStream.applyToDataset(btn, episodes);
        refreshJobs();
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
