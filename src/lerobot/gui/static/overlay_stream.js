// Playback of the live SAM3 preview as one server-composited H.264 stream.
//
// The per-frame path costs two round trips and a ~115 KB PNG per frame per
// camera; at a remote operator's real RTT that is ~1 fps against the worker's
// ~10. Here the server walks the episode at the worker's own pace, composites
// every selected camera into ONE atlas frame, and streams it as fragmented
// MP4; this module plays it with fetch + MediaSource (the run-tab preview's
// pattern) and slices the atlas onto a canvas per camera tile. One stream is
// what makes the cameras stay in sync: they share a frame.
//
// Play routes here only while the overlay is live (app.js asks eligible()).
// Pause, stream end, or a manual scrub tear it down onto the still path.

(function () {
    'use strict';

    const MIME = 'video/mp4; codecs="avc1.42C01E"';
    const state = {
        streaming: false,
        abort: null,
        video: null,
        canvases: {},   // cam -> canvas
        layout: null,
        raf: 0,
        started: false, // playback began (after the startup buffer)
        t0: undefined,  // wall clock at first append, for the arrival-rate estimate
    };

    function session() {
        try { return sessionStorage.getItem('ovlSession') || ''; } catch (e) { return ''; }
    }

    function badgeActive() {
        // The badge renders the backend lifecycle: 'ok' = active with fps,
        // 'idle' = active at 0 fps OR busy-held-by-another (text says which).
        const b = document.getElementById('overlays-badge');
        if (!b) return false;
        if (/\bok\b/.test(b.className)) return true;
        return /\bidle\b/.test(b.className) && !/^busy/.test(b.textContent || '');
    }

    function selectedCams() {
        const on = [...document.querySelectorAll('.overlays-cam-btn.on')].map((b) => b.dataset.cam);
        if (on.length) return on;
        const ds = window.datasets && window.datasets[window.currentDataset];
        return ds ? ds.camera_keys : [];
    }

    function eligible() { return state.streaming || badgeActive(); }

    // Runtime assertion for what must hold while the stream owns the tiles.
    // The rule lives in transport_invariants.js so this and its unit test
    // cannot drift; this only supplies the observed state and reports.
    let _lastAssert = 0;
    let _reported = new Set();
    function assertTransport(streamFrame) {
        const now = Date.now();
        if (now - _lastAssert < 500) return;      // a check per half second, not per frame
        _lastAssert = now;
        const check = window.transportViolations;
        if (!check) return;
        const btn = document.getElementById('play-btn');
        const violations = check({
            streaming: state.streaming,
            isPlaying: window.__streamIsPlaying ? window.__streamIsPlaying() : true,
            playBtnLabel: btn ? btn.textContent : '',
            liveActive: state.streaming,
            savedMasksDrawn: !!(window.MaskOverlay && window.MaskOverlay.isDrawing && window.MaskOverlay.isDrawing()),
            stillFetchInFlight: !!window.__stillFetchInFlight,
            streamFrame,
            playheadFrame: window.currentFrame,
        });
        for (const v of violations) {
            // Loud once per distinct violation per stream: a silent desync is
            // what made this hard to see -- the picture simply looked wrong.
            console.error('[overlay transport] ' + v);
            if (!_reported.has(v)) {
                _reported.add(v);
                window.showToast?.('Overlay preview is out of sync', v, 'error', 9000);
            }
        }
    }

    function setPlayBtn(playing) {
        // Through the app so ITS isPlaying moves too: the button is rendered
        // from that flag elsewhere, and setting only the label here left the
        // two disagreeing the moment anything re-rendered.
        if (window.__streamSetPlaying) { window.__streamSetPlaying(playing); return; }
        const btn = document.getElementById('play-btn');
        if (btn) btn.innerHTML = playing ? '&#9646;&#9646; Pause' : '&#9654; Play';
    }

    function tileCanvas(cam, rect) {
        const frame = document.getElementById(`frame-${cam.replace(/\./g, '-')}`);
        if (!frame || !frame.parentElement) return null;
        const c = document.createElement('canvas');
        c.className = 'overlay-layer stream-layer';
        c.width = rect[2]; c.height = rect[3];
        c.style.display = 'block';
        frame.parentElement.appendChild(c);
        return c;
    }

    function teardownTiles() {
        document.querySelectorAll('canvas.stream-layer').forEach((c) => c.remove());
        state.canvases = {};
    }

    async function start() {
        const dsId = window.currentDataset;
        if (!dsId || window.currentEpisode === null) return;
        // Pressing Play while parked on the last frame has to replay the
        // episode, not stream the one frame that is left: the stream ends
        // immediately, isPlaying resets, and every further press looks like a
        // dead button. The video path already restarts here; this is the same
        // rule for the live path.
        const total = window.totalFrames || 0;
        const at = window.currentFrame || 0;
        const from = (total && at >= total - 1) ? 0 : at;
        const cams = selectedCams();
        if (!cams.length) return;
        if (!window.MediaSource || !MediaSource.isTypeSupported(MIME)) {
            console.warn('[stream] MSE unavailable; falling back to still playback');
            return;
        }

        const url = `/api/overlays/data/stream.mp4?dataset_id=${encodeURIComponent(dsId)}` +
            `&episode=${window.currentEpisode}&from_frame=${from}` +
            `&cameras=${encodeURIComponent(cams.join(','))}`;
        const abort = new AbortController();
        let resp;
        try {
            resp = await fetch(url, { headers: { 'X-Overlay-Session': session() }, signal: abort.signal });
        } catch (e) { return; }
        if (!resp.ok) { console.warn('[stream] HTTP', resp.status); return; }
        const layout = JSON.parse(resp.headers.get('X-Overlay-Layout') || 'null');
        if (!layout) { abort.abort(); return; }

        state.streaming = true;
        state.abort = abort;
        state.layout = layout;
        state.started = false;
        _reported = new Set();
        setPlayBtn(true);
        // Apply is armed separately; playing is what makes it write.
        window.Overlays?.applyOnTransport?.(true);

        const video = document.createElement('video');
        video.muted = true;
        state.video = video;
        const ms = new MediaSource();
        // Held so stop() can revoke it: an object URL keeps its MediaSource —
        // and the decoder behind it — alive for the life of the document.
        // Leaking one per play is what made the preview weaken with each press
        // and then stop starting at all (measured: 101 frames, then 37, then
        // 15, then none), with the button still reading Pause.
        state.objectUrl = URL.createObjectURL(ms);
        state.ms = ms;
        video.src = state.objectUrl;
        await new Promise((r) => ms.addEventListener('sourceopen', r, { once: true }));
        const sb = ms.addSourceBuffer(MIME);

        for (const [cam, rect] of Object.entries(layout.cameras)) {
            const c = tileCanvas(cam, rect);
            if (c) state.canvases[cam] = c;
        }

        const reader = resp.body.getReader();
        state.reader = reader;
        (async () => {
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    await new Promise((res, rej) => {
                        sb.addEventListener('updateend', res, { once: true });
                        sb.addEventListener('error', rej, { once: true });
                        sb.appendBuffer(value);
                    });
                    // Soft real-time: play at the rate frames actually arrive,
                    // not the container's nominal fps. Production varies hugely
                    // with cameras and treatments (measured 9.7 fps for one
                    // camera outlined, 2.2 for three with background blur); a
                    // fixed 1x against nominal either stalls at the live edge
                    // forever or waits ~12s to buffer. Matching playbackRate to
                    // the measured arrival rate starts within ~1s and stays
                    // smooth: latency traded for smoothness, never for stalls.
                    const end = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0;
                    if (state.t0 === undefined) state.t0 = performance.now();
                    const wall = (performance.now() - state.t0) / 1000;
                    // Media now arrives at ~1x wall time (the server pads the
                    // timeline); play at exactly 1x once that is confirmed and
                    // only slow down if arrival genuinely lags.
                    const r = wall > 0.5 ? end / wall : 0.5;
                    const rate = r > 0.85 ? 1.0 : Math.max(0.1, 0.9 * r);
                    if (!state.started && end > 0.5) {
                        state.started = true;
                        video.currentTime = sb.buffered.start(0) + 0.05;
                        video.playbackRate = rate;
                        video.play().catch(() => {});
                    } else if (state.started && Math.abs(video.playbackRate - rate) > 0.05) {
                        video.playbackRate = rate;
                    }
                    // Bound live latency: if playback has fallen well behind
                    // the newest media, jump near the live edge rather than
                    // replaying the backlog.
                    if (state.started && sb.buffered.length && end - video.currentTime > 2.5) {
                        video.currentTime = end - 0.7;
                    }
                    // Keep memory bounded on long episodes.
                    if (sb.buffered.length && video.currentTime - sb.buffered.start(0) > 30 && !sb.updating) {
                        try { sb.remove(0, video.currentTime - 10); } catch (e) {}
                    }
                }
                if (ms.readyState === 'open') { try { ms.endOfStream(); } catch (e) {} }
                if (!state.started && sb.buffered.length) {
                    state.started = true;
                    video.currentTime = sb.buffered.start(0) + 0.05;
                    video.play().catch(() => {});
                }
            } catch (e) {
                // Aborts are routine (stop() cancels the reader); anything else
                // ended the preview and used to do it in total silence, which
                // is how a stream that died one second in looked exactly like a
                // frozen player.
                if (!(e && e.name === 'AbortError')) {
                    console.warn('[stream] reader stopped:', e && (e.message || e.name || e));
                }
            }
        })();

        state.loop = true;      // cleared by an explicit stop (the Pause button)
        video.addEventListener('ended', () => {
            const end = sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0;
            console.warn('[stream] element ended at', video.currentTime.toFixed(2),
                         'buffered to', end.toFixed(2), 'ms', ms.readyState);
            const again = state.loop;
            stop({ resume: true });
            // The preview is paced to wall time, so it ends when the episode's
            // duration elapses. Saved-mask playback loops there; this used to
            // stop, which reads as "it pauses at the end" and made the two
            // paths behave differently for no reason the operator can see.
            if (again) {
                window.currentFrame = 0;
                setTimeout(() => start(), 150);
            }
        });
        for (const ev of ['stalled', 'waiting', 'error', 'emptied']) {
            video.addEventListener(ev, () => console.warn('[stream] video', ev,
                'at', video.currentTime.toFixed(2), 'ready', video.readyState));
        }

        const draw = () => {
            if (!state.streaming) return;
            state.raf = requestAnimationFrame(draw);
            if (video.readyState < 2) {
                // The live-edge nudge from the run-tab preview: without it the
                // element can stall beside a full buffer (measured).
                if (state.started && sb.buffered.length) {
                    const end = sb.buffered.end(sb.buffered.length - 1);
                    if (video.currentTime < sb.buffered.start(0)) video.currentTime = sb.buffered.start(0) + 0.05;
                    else if (end - video.currentTime > 0.4) video.currentTime += 0.1;
                }
                return;
            }
            for (const [cam, c] of Object.entries(state.canvases)) {
                const [sx, sy, sw, sh] = state.layout.cameras[cam];
                c.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
            }
            const f = state.layout.from_frame + Math.floor(video.currentTime * state.layout.fps);
            if (f !== window.currentFrame && window.__streamSetPlayhead) window.__streamSetPlayhead(f);
            assertTransport(f);
        };
        state.raf = requestAnimationFrame(draw);
    }

    function stop(opts) {
        if (!state.streaming) return;
        state.streaming = false;
        // Pausing stops the run; what it already staged stays staged, and the
        // boundary flush picks up whatever the worker produced last.
        window.Overlays?.applyOnTransport?.(false);
        // An explicit stop ends the loop; the end-of-episode path reads this
        // flag before calling us, so its own restart still goes through.
        state.loop = false;
        cancelAnimationFrame(state.raf);
        // Order matters: cancel the reader before detaching the MediaSource,
        // or the loop wakes on a SourceBuffer that has already been removed and
        // reports a failure that is really just teardown.
        if (state.reader) { try { state.reader.cancel(); } catch (e) {} }
        if (state.abort) { try { state.abort.abort(); } catch (e) {} }
        if (state.video) {
            try {
                state.video.pause();
                // removeAttribute + load() is what actually detaches the
                // decoder; assigning '' leaves the element holding it.
                state.video.removeAttribute('src');
                state.video.load();
            } catch (e) {}
        }
        if (state.ms && state.ms.readyState === 'open') { try { state.ms.endOfStream(); } catch (e) {} }
        if (state.objectUrl) { try { URL.revokeObjectURL(state.objectUrl); } catch (e) {} }
        state.video = null; state.abort = null; state.layout = null; state.t0 = undefined;
        state.objectUrl = null; state.ms = null; state.reader = null;
        teardownTiles();
        setPlayBtn(false);
        // Land on the still path at the frame the stream reached, unless a
        // scrub is already fetching its own target frame.
        if (!opts || opts.resume !== false) {
            if (typeof window.loadAllFrames === 'function') window.loadAllFrames(window.currentFrame);
        }
    }

    function toggle() { state.streaming ? stop({ resume: true }) : start(); }

    // ---- the panel's mask coverage line ----
    // The panel is the live query and has no scope of its own, so it hosts no
    // write: the design names exactly two ways to add masks -- Apply while
    // playing, and the Inspector's dataset-wide filler -- and the buttons that
    // used to sit here were neither. What remains is the read: how much of this
    // episode already carries masks, which is context for the query above it.
    // Set by ensureCoverageHint once the element exists; a completed job calls
    // it so the line reflects the write immediately.
    let refreshHint = () => {};

    // Episodes THIS session saved: overwriting your own iteration is the
    // intended loop and skips the dialog; anything else asks first.
    const ownSaves = new Set();
    const epKey = () => `${window.currentDataset}::${window.currentEpisode}`;

    function ensureCoverageHint() {
        const body = document.getElementById('overlays-body');
        if (!body || document.getElementById('ovl-save-masks-hint')) return;
        const hint = document.createElement('div');
        hint.id = 'ovl-save-masks-hint';
        hint.style.cssText = 'font-size:11px; color:#8494a4; margin:8px 0 6px;';
        body.appendChild(hint);
        let lastEp = null;
        refreshHint = async (force) => {
            const k = epKey();
            if (!force && (k === lastEp || !window.currentDataset || window.currentEpisode === null)) return;
            lastEp = k;
            try {
                const s = await fetch(`/api/datasets/${encodeURIComponent(window.currentDataset)}` +
                                      `/episodes/${window.currentEpisode}/masks/status`).then((r) => r.json());
                if (!s.adopted) { hint.textContent = 'masks: feature not adopted yet'; return; }
                const parts = Object.entries(s.cameras).map(
                    ([key, c]) => `${key.split('.').pop()} ${c.with_masks}/${c.frames}`);
                const any = Object.values(s.cameras).some((c) => c.with_masks > 0);
                hint.textContent = any ? `masks saved: ${parts.join(' · ')}` : 'masks: none saved for this episode';
            } catch (e) { hint.textContent = ''; }
        };
        setInterval(() => refreshHint(false), 1000);
    }

    // ``episodes`` (an array) applies the CURRENT settings to that whole list
    // instead of the open episode. Same job, same worker, same live settings --
    // the only difference is how many episodes the worker walks, which is why
    // this is one function and not two.
    // The mask job, whoever asks for it. Kept when the panel's two buttons went
    // because the work is not the button: the 409 consent handshake, the job
    // polling, the "found nothing" report and the cache invalidation all live
    // here, and the Inspector's filler had none of them.
    //
    // ``btn`` is optional and only carries progress; ``episodes`` (an array)
    // walks that whole list instead of the open episode.
    async function saveMasks(btn, confirmed, overwriteOk, episodes, objects, onProgress) {
        const dsId = window.currentDataset;
        // Progress has to reach whoever asked for the run, and not every caller
        // has a button. The Inspector's filler passed null, so every update
        // below was skipped and the one path that can run for hours was the one
        // path with no progress at all -- indistinguishable from a hang.
        const report = (text) => {
            if (btn) btn.textContent = text;
            if (onProgress) { try { onProgress(text); } catch (err) { /* a reporter must not kill a run */ } }
        };
        if (!dsId || window.currentEpisode === null) return;
        const cams = selectedCams();
        // The preview and the save want the same GPU. A looping preview never
        // yields it, so the job sat queued for minutes with the button saying
        // "Saving…" and nothing happening. Stop the preview first: the save is
        // what the operator just asked for.
        if (state.streaming) stop({ resume: false });
        const was = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = 'Working…'; }
        try {
            const resp = await fetch('/api/process/episode-masks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Overlay-Session': session() },
                body: JSON.stringify({
                    source_id: dsId,
                    // `episode` is required by the endpoint even when a list is
                    // given -- it is the anchor the job reports against. Sending
                    // only `episodes` was rejected with a 422 the caller showed
                    // as "Save masks failed", so the whole-dataset path never ran.
                    episode: episodes && episodes.length ? episodes[0] : window.currentEpisode,
                    ...(episodes ? { episodes } : {}),
                    // Omitted, the server uses the live session's objects -- which is
                    // what the panel wanted. The filler names its own labels instead,
                    // because the ticks in its dialog are the request.
                    ...(objects ? { objects } : {}),
                    cameras: cams, confirm_adopt: confirmed,
                    confirm_overwrite: overwriteOk || (!episodes && ownSaves.has(epKey())),
                }),
            });
            const data = await resp.json().catch(() => ({}));
            if (resp.status === 409 && data.detail && data.detail.code === 'adopt_masks_feature') {
                const ok = window.confirm(
                    'This dataset has no masks feature yet.\n\n' + data.detail.message +
                    '\n\nAdd ' + (data.detail.features || []).join(', ') + '?');
                if (btn) btn.textContent = was;
                if (ok) return saveMasks(btn, true, overwriteOk, episodes, objects);
                if (btn) btn.disabled = false;
                return;
            }
            if (resp.status === 409 && data.detail && data.detail.code === 'masks_exist') {
                const cov = Object.entries(data.detail.coverage || {})
                    .map(([k, n]) => `${k.split('.').pop()} ${n}/${data.detail.frames}`).join(', ');
                const ok = window.confirm(data.detail.message + '\n\nCurrently saved: ' + cov);
                if (btn) btn.textContent = was;
                if (ok) return saveMasks(btn, confirmed, true, episodes, objects);
                if (btn) btn.disabled = false;
                return;
            }
            if (!resp.ok) {
                // A refusal is an explanation, and an explanation does not fit
                // in a button. The button goes back to saying what it does;
                // the reason goes where every other error in this app goes.
                const msg = (data.detail && (data.detail.message || data.detail.code)) || ('HTTP ' + resp.status);
                window.showToast?.('Save masks failed', msg, 'error', 9000);
                if (btn) { btn.textContent = was; btn.disabled = false; }
                return;
            }
            // Poll the shared jobs list until this job settles; the top-bar
            // Processing indicator shows the same job meanwhile.
            const jobId = data.job_id;
            // Wake the shared tray. It polls only while it believes something is
            // active, so a job started from here stayed invisible in the top-bar
            // Processing indicator until some unrelated action refreshed it.
            window.ProcessData?.refreshJobs?.();
            while (true) {
                await new Promise((r) => setTimeout(r, 2000));
                const jobs = await fetch('/api/process/jobs').then((r) => r.json()).catch(() => ({}));
                const j = (jobs.jobs || []).find((x) => x.job_id === jobId) || {};
                if (j.status === 'complete') {
                    // A pass that found nothing looks exactly like a good save
                    // unless we say otherwise: two episodes of a 274-episode
                    // dataset came back empty on every camera, and a re-run
                    // fixed them. Name the empty cameras so it is actionable.
                    const empty = Object.entries(j.coverage || {})
                        .filter(([, n]) => !n).map(([k]) => k.split('.').pop());
                    report('Saved ✓');
                    if (empty.length) {
                        // Worth a toast rather than a caption: it means those
                        // cameras' frames will render as pure background.
                        window.showToast?.('Saved, but nothing was found',
                            `No masks on ${empty.join(', ')} — those frames composite as all background. `
                            + 'Re-running the episode usually fixes a seed failure.', 'error', 12000);
                    }
                    ownSaves.add(epKey());
                    refreshHint();   // the counts just changed; do not wait for an episode switch
                    // The pass rewrote the episode's mask column, and on a first
                    // adopt added masks.* to the schema; the panel
                    // caches both and would otherwise need a page reload.
                    await window.FeatureEditing?.refreshFromServer?.(dsId);
                    break;
                }
                if (j.status === 'failed' || j.status === 'cancelled') {
                    if (j.status === 'failed') {
                        window.showToast?.('Save masks failed', j.error || 'see the server log', 'error', 9000);
                    }
                    if (btn) btn.textContent = was;
                    break;
                }
                report(episodes
                    ? 'Filling… episode ' + ((j.episodes_done || 0) + 1) + '/' + (j.episodes_total || episodes.length)
                      + (j.frames_total ? ' · frame ' + (j.frames_done || 0) + '/' + j.frames_total : '')
                    : 'Working… ' + (j.frames_done || 0) + '/' + (j.frames_total || '?'));
            }
            if (window.MaskOverlay) {
                window.MaskOverlay.invalidate(dsId);
                window.MaskOverlay.onPlayheadChanged();
            }
        } finally {
            if (btn) setTimeout(() => { btn.textContent = was; btn.disabled = false; }, 3000);
        }
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensureCoverageHint);
    else ensureCoverageHint();

    window.OverlayStream = {
        get streaming() { return state.streaming; },
        eligible, toggle, stop, start,
        // The mask job. Exported because the Inspector's dataset-wide filler is
        // the only caller now, and the session header and camera selection it
        // needs -- which decide WHAT gets segmented -- are this module's state.
        runMaskJob: (btn, episodes, opts) => saveMasks(btn, !!(opts && opts.confirmed),
            !!(opts && opts.overwriteOk), episodes, opts && opts.objects, opts && opts.onProgress),
        // Exposed so the test can prove the button goes through the app's
        // state rather than writing the label directly (the desync).
        _setPlayBtn: setPlayBtn,
        // Read-only introspection for diagnostics; nothing in the app uses it.
        _debug: () => ({
            streaming: state.streaming, started: state.started,
            ct: state.video ? state.video.currentTime : null,
            rs: state.video ? state.video.readyState : null,
            paused: state.video ? state.video.paused : null,
            buffered: (() => { try { const v = state.video; if (!v || !v.buffered.length) return null;
                return [v.buffered.start(0), v.buffered.end(v.buffered.length - 1)]; } catch (e) { return String(e); } })(),
            cams: Object.keys(state.canvases),
            layout: state.layout,
        }),
    };
})();
