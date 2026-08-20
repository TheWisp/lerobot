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

    function setPlayBtn(playing) {
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
        const from = window.currentFrame || 0;
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
        setPlayBtn(true);

        const video = document.createElement('video');
        video.muted = true;
        state.video = video;
        const ms = new MediaSource();
        video.src = URL.createObjectURL(ms);
        await new Promise((r) => ms.addEventListener('sourceopen', r, { once: true }));
        const sb = ms.addSourceBuffer(MIME);

        for (const [cam, rect] of Object.entries(layout.cameras)) {
            const c = tileCanvas(cam, rect);
            if (c) state.canvases[cam] = c;
        }

        const reader = resp.body.getReader();
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
            } catch (e) { /* aborted or SB error: stop() handles teardown */ }
        })();

        video.addEventListener('ended', () => stop({ resume: true }));

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
        };
        state.raf = requestAnimationFrame(draw);
    }

    function stop(opts) {
        if (!state.streaming) return;
        state.streaming = false;
        cancelAnimationFrame(state.raf);
        if (state.abort) { try { state.abort.abort(); } catch (e) {} }
        if (state.video) { try { state.video.pause(); state.video.src = ''; } catch (e) {} }
        state.video = null; state.abort = null; state.layout = null; state.t0 = undefined;
        teardownTiles();
        setPlayBtn(false);
        // Land on the still path at the frame the stream reached, unless a
        // scrub is already fetching its own target frame.
        if (!opts || opts.resume !== false) {
            if (typeof window.loadAllFrames === 'function') window.loadAllFrames(window.currentFrame);
        }
    }

    function toggle() { state.streaming ? stop({ resume: true }) : start(); }

    // ---- Save episode masks: the collaborative flow's commit button ----
    // Lives here rather than overlays.js because it consumes the same session
    // identity and eligibility signal the stream uses. What is saved is the
    // RECIPE — named masks per frame plus effect options in the feature
    // metadata — never baked pixels; the server refuses with a structured 409
    // until the user confirms adopting the dataset-wide feature.
    // Episodes THIS session saved: overwriting your own iteration is the
    // intended loop and skips the dialog; anything else asks first.
    const ownSaves = new Set();
    const epKey = () => `${window.currentDataset}::${window.currentEpisode}`;

    function ensureSaveButton() {
        const body = document.getElementById('overlays-body');
        if (!body || document.getElementById('ovl-save-masks')) return;
        const btn = document.createElement('button');
        btn.id = 'ovl-save-masks';
        btn.textContent = 'Save episode masks';
        btn.style.cssText = 'margin:8px 0 2px; width:100%; padding:6px;';
        const hint = document.createElement('div');
        hint.id = 'ovl-save-masks-hint';
        hint.style.cssText = 'font-size:11px; color:#8494a4; margin-bottom:6px;';
        btn.addEventListener('click', () => saveMasks(btn, false, false));
        body.appendChild(btn);
        body.appendChild(hint);
        let lastEp = null;
        setInterval(async () => {
            btn.disabled = !eligible() || state.streaming;
            const k = epKey();
            if (k === lastEp || !window.currentDataset || window.currentEpisode === null) return;
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
        }, 1000);
    }

    async function saveMasks(btn, confirmed, overwriteOk) {
        const dsId = window.currentDataset;
        if (!dsId || window.currentEpisode === null) return;
        const cams = selectedCams();
        btn.disabled = true;
        const was = btn.textContent;
        btn.textContent = 'Saving…';
        try {
            const resp = await fetch('/api/process/episode-masks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Overlay-Session': session() },
                body: JSON.stringify({
                    source_id: dsId, episode: window.currentEpisode,
                    cameras: cams, confirm_adopt: confirmed,
                    confirm_overwrite: overwriteOk || ownSaves.has(epKey()),
                }),
            });
            const data = await resp.json().catch(() => ({}));
            if (resp.status === 409 && data.detail && data.detail.code === 'adopt_masks_feature') {
                const ok = window.confirm(
                    'This dataset has no masks feature yet.

' + data.detail.message +
                    '

Add ' + (data.detail.features || []).join(', ') + '?');
                btn.textContent = was;
                if (ok) return saveMasks(btn, true, overwriteOk);
                btn.disabled = false;
                return;
            }
            if (resp.status === 409 && data.detail && data.detail.code === 'masks_exist') {
                const cov = Object.entries(data.detail.coverage || {})
                    .map(([k, n]) => `${k.split('.').pop()} ${n}/${data.detail.frames}`).join(', ');
                const ok = window.confirm(data.detail.message + '\n\nCurrently saved: ' + cov);
                btn.textContent = was;
                if (ok) return saveMasks(btn, confirmed, true);
                btn.disabled = false;
                return;
            }
            if (!resp.ok) {
                const msg = (data.detail && (data.detail.message || data.detail.code)) || ('HTTP ' + resp.status);
                btn.textContent = 'Save failed: ' + msg;
                setTimeout(() => { btn.textContent = was; btn.disabled = false; }, 4000);
                return;
            }
            // Poll the shared jobs list until this job settles; the top-bar
            // Processing indicator shows the same job meanwhile.
            const jobId = data.job_id;
            while (true) {
                await new Promise((r) => setTimeout(r, 2000));
                const jobs = await fetch('/api/process/jobs').then((r) => r.json()).catch(() => ({}));
                const j = (jobs.jobs || []).find((x) => x.job_id === jobId) || {};
                if (j.status === 'complete') { btn.textContent = 'Saved ✓'; ownSaves.add(epKey()); break; }
                if (j.status === 'failed' || j.status === 'cancelled') {
                    btn.textContent = 'Save ' + j.status; break;
                }
                btn.textContent = 'Saving… ' + (j.frames_done || 0) + '/' + (j.frames_total || '?');
            }
            if (window.MaskOverlay) {
                window.MaskOverlay.invalidate(dsId);
                window.MaskOverlay.onPlayheadChanged();
            }
        } finally {
            setTimeout(() => { btn.textContent = 'Save episode masks'; btn.disabled = false; }, 3000);
        }
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensureSaveButton);
    else ensureSaveButton();

    window.OverlayStream = {
        get streaming() { return state.streaming; },
        eligible, toggle, stop, start,
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
