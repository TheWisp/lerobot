/* Read-only tracker experiment browser. Video time owns overlay selection. */
const FLASH_POSE_METHODS = {
    ufm_chain: 'Pose from UFM matches',
    surface_corrected: 'Pose after surface correction',
    ufm_retained_reference: 'Pose with earlier-frame recovery',
};
const FLASH_POSE_EXPLANATIONS = {
    ufm_chain: 'RGB matches → depth at both ends → one 3D motion → accumulated pose.',
    surface_corrected: 'Move the saved 3D cloud using UFM, then shift/rotate the whole cloud to better fit current depth and mask. The second step does not preserve UFM’s point pairings.',
    ufm_retained_reference: 'If a pair fails, keep the last successfully linked image and try matching that older image to each new image until a motion can be fitted.',
};
const flashExperiments = { items: [], selected: null, data: null, frame: 0, request: null, listRequest: null, initialized: false };
const flashRebuild = { timer: null, request: null, starting: false, id: null, completed: null };

async function showFlashExperiments() {
    flashUiMode = 'experiments';
    if (flashReplayTimer) { clearTimeout(flashReplayTimer); flashReplayTimer = null; }
    stopFlashTelemetry();
    _flashUpdateWorkspace();
    initFlashExperimentPlayer();
    await loadFlashExperiments();
    pollFlashRebuild();
}

async function loadFlashExperiments() {
    const list = document.getElementById('flash-experiment-list');
    if (!list) return; // Run form may still be initializing during hash navigation.
    flashExperiments.listRequest?.abort();
    const request = new AbortController();
    flashExperiments.listRequest = request;
    try {
        const response = await fetch('/api/run/flash/experiments', { signal: request.signal });
        if (!response.ok) throw new Error(`Cannot list experiments (${response.status})`);
        const items = await response.json();
        if (request !== flashExperiments.listRequest) return;
        const oldItem = flashExperiments.items.find(item => item.id === flashExperiments.selected);
        const oldSlot = oldItem?.slot_id || oldItem?.id;
        flashExperiments.items = items;
        list.replaceChildren();
        for (const item of items) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'flash-experiment-item';
            button.dataset.id = item.id;
            button.textContent = item.name;
            button.title = `${item.name} · ${item.experiment}`;
            button.setAttribute('aria-pressed', String(item.id === flashExperiments.selected));
            button.onclick = () => selectFlashExperiment(item);
            list.append(button);
        }
        if (!items.length) list.textContent = 'No exported experiments yet.';
        if (items.length && !items.some(item => item.id === flashExperiments.selected) && flashUiMode === 'experiments') {
            await selectFlashExperiment(items.find(item => (item.slot_id || item.id) === oldSlot) || items[0]);
        }
    } catch (error) {
        if (error.name !== 'AbortError') list.textContent = error.message;
    }
}

async function rebuildFlashExamples() {
    if (flashRebuild.starting) return;
    flashRebuild.starting = true;
    document.getElementById('flash-experiment-rebuild').disabled = true;
    document.getElementById('flash-experiment-rebuild-status').textContent = 'Starting benchmarks…';
    try {
        const response = await fetch('/api/run/flash/experiments/rebuild', { method: 'POST' });
        if (!response.ok && response.status !== 409) {
            const error = await response.json();
            throw new Error(error.detail || `Cannot rebuild (${response.status})`);
        }
        flashRebuild.completed = null;
    } catch (error) {
        const status = document.getElementById('flash-experiment-rebuild-status');
        if (status) status.textContent = error.message;
        return;
    } finally {
        flashRebuild.starting = false;
        const button = document.getElementById('flash-experiment-rebuild');
        if (button) button.disabled = false;
    }
    await pollFlashRebuild();
}

async function pollFlashRebuild() {
    clearTimeout(flashRebuild.timer);
    flashRebuild.timer = null;
    if (flashRebuild.request || selectedWorkflow !== 'flash' || flashUiMode !== 'experiments') return;
    const request = new AbortController();
    flashRebuild.request = request;
    let active = false;
    try {
        const response = await fetch('/api/run/flash/experiments/rebuild', { signal: request.signal });
        if (!response.ok) throw new Error(`Cannot read rebuild progress (${response.status})`);
        const state = await response.json();
        if (selectedWorkflow !== 'flash' || flashUiMode !== 'experiments') return;
        active = ['starting', 'running'].includes(state.state);
        // Loading saved Run settings can replace the form while a request is
        // pending. The job survives; a later poll attaches to the new controls.
        if (!document.getElementById('flash-experiment-rebuild')) return;
        document.getElementById('flash-experiment-rebuild').disabled = active || flashRebuild.starting;
        const progress = document.getElementById('flash-experiment-rebuild-progress');
        progress.hidden = !active;
        progress.max = state.total || 1;
        progress.value = (state.completed || 0) + (state.frames ? state.frame / state.frames : 0);
        const text = document.getElementById('flash-experiment-rebuild-status');
        if (active) text.textContent = `${state.completed} / ${state.total} · ${state.current || state.phase}` +
            (state.current ? ` · ${state.phase}${state.frames ? ` ${state.frame}/${state.frames}` : ''}` : '');
        else if (state.state === 'idle') text.textContent = '';
        else if (state.state === 'failed') text.textContent = state.error || 'Rebuild stopped.';
        else text.textContent = `${state.succeeded} / ${state.total} benchmarked` + (state.errors.length ? ` · ${state.errors.length} failed` : '');
        const report = document.getElementById('flash-experiment-report');
        report.hidden = !state.report_ready;
        if (state.report_ready) {
            const base = `/api/run/flash/experiments/rebuild/${encodeURIComponent(state.id)}`;
            document.getElementById('flash-experiment-report-view').href = `${base}/report.html`;
            document.getElementById('flash-experiment-report-download').href = `${base}/report.json`;
        }
        const errors = document.getElementById('flash-experiment-rebuild-errors');
        errors.hidden = !state.errors?.length;
        const list = errors.querySelector('ul');
        list.replaceChildren(...(state.errors || []).map(error => {
            const li = document.createElement('li'); li.textContent = `${error.name}: ${error.error}`; return li;
        }));
        if (state.id && (flashRebuild.id !== state.id || flashRebuild.completed !== state.completed)) {
            const refresh = state.completed > 0;
            flashRebuild.id = state.id; flashRebuild.completed = state.completed;
            if (refresh) await loadFlashExperiments();
        }
    } catch (error) {
        const status = document.getElementById('flash-experiment-rebuild-status');
        if (status) status.textContent = error.message;
        active = true; // Retry status after a temporary server interruption.
    } finally {
        flashRebuild.request = null;
        if (selectedWorkflow === 'flash' && flashUiMode === 'experiments') {
            flashRebuild.timer = setTimeout(pollFlashRebuild, active ? 1000 : 5000);
        }
    }
}

async function selectFlashExperiment(item) {
    flashExperiments.request?.abort();
    const request = new AbortController();
    flashExperiments.request = request;
    const video = document.getElementById('flash-experiment-video');
    video.pause();
    video.removeAttribute('src');
    video.load();
    flashExperiments.data = null;
    flashExperiments.selected = item.id;
    document.getElementById('flash-experiment-title').textContent = item.name;
    document.getElementById('flash-experiment-detail').textContent =
        `${item.experiment} · ${item.mask_source} mask · ${item.frames} frames`;
    document.getElementById('flash-experiment-status').textContent = 'Loading recording…';
    document.querySelectorAll('.flash-experiment-item').forEach(button =>
        button.setAttribute('aria-pressed', String(button.dataset.id === item.id)));
    drawFlashExperiment();
    try {
        const response = await fetch(`/api/run/flash/experiments/${item.id}/overlays.json`, { signal: request.signal });
        if (!response.ok) throw new Error(`Cannot load overlays (${response.status})`);
        const data = await response.json();
        if (request !== flashExperiments.request) return;
        flashExperiments.data = data;
        flashExperiments.frame = 0;
        const canvas = document.getElementById('flash-experiment-overlay');
        canvas.width = data.width;
        canvas.height = data.height;
        const select = document.getElementById('flash-experiment-method');
        const previous = select.value;
        select.replaceChildren(...Object.entries(data.methods).map(([value, label]) => new Option(FLASH_POSE_METHODS[value] || label, value)));
        if (Object.hasOwn(data.methods, previous)) select.value = previous;
        const matches = document.getElementById('flash-experiment-matches');
        matches.disabled = !data.has_correspondences;
        matches.title = data.has_correspondences ? 'Paths follow UFM predictions between images. Endpoints outside the current mask are retained.' : 'Raw matches were not recorded for this older result.';
        video.src = `/api/run/flash/experiments/${item.id}/video.mp4`;
        video.load();
    } catch (error) {
        if (error.name !== 'AbortError') document.getElementById('flash-experiment-status').textContent = error.message;
    }
}

function flashExperimentFrameAt(frames, time) {
    let low = 0, high = frames.length;
    while (low < high) {
        const mid = (low + high) >>> 1;
        if (frames[mid].t <= time + 1e-6) low = mid + 1;
        else high = mid;
    }
    return Math.max(0, low - 1);
}

function initFlashExperimentPlayer() {
    if (flashExperiments.initialized) return;
    flashExperiments.initialized = true;
    const video = document.getElementById('flash-experiment-video');
    const update = time => {
        if (!flashExperiments.data) return;
        flashExperiments.frame = flashExperimentFrameAt(flashExperiments.data.frames, time);
        drawFlashExperiment();
    };
    if (video.requestVideoFrameCallback) {
        const decoded = (_now, metadata) => {
            update(metadata.mediaTime);
            video.requestVideoFrameCallback(decoded);
        };
        video.requestVideoFrameCallback(decoded);
    } else {
        video.addEventListener('timeupdate', () => update(video.currentTime));
        // Standard HTMLMediaElement event name.
        video.addEventListener('seeked', () => update(video.currentTime)); // spellchecker:disable-line
    }
    video.addEventListener('error', () => {
        if (video.getAttribute('src')) document.getElementById('flash-experiment-status').textContent = 'Video could not be loaded.';
    });
    new ResizeObserver(() => drawFlashExperiment()).observe(document.getElementById('flash-experiment-player'));
    document.addEventListener('visibilitychange', () => { if (document.hidden) video.pause(); });
}

function drawFlashExperiment() {
    const canvas = document.getElementById('flash-experiment-overlay');
    if (!canvas) return;
    const context = canvas.getContext('2d');
    context.resetTransform();
    context.clearRect(0, 0, canvas.width, canvas.height);
    const data = flashExperiments.data;
    if (!data) return;
    // Render vectors at display resolution, keeping the same camera coordinates
    // and contain/letterbox layout as the video. No bitmap upscaling blur.
    const bounds = canvas.getBoundingClientRect();
    const displayScale = Math.min(bounds.width / data.width, bounds.height / data.height) || 1;
    const pixelScale = displayScale * (window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(data.width * pixelScale));
    const height = Math.max(1, Math.round(data.height * pixelScale));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width; canvas.height = height;
    }
    context.setTransform(width / data.width, 0, 0, height / data.height, 0, 0);
    const row = data.frames[flashExperiments.frame];
    const method = document.getElementById('flash-experiment-method').value;
    const showMask = document.getElementById('flash-experiment-mask').checked;
    const showPose = document.getElementById('flash-experiment-pose').checked;
    const matchMode = document.getElementById('flash-experiment-matches').value;
    const trailFrames = Number(document.getElementById('flash-experiment-flow-trail').value);
    document.getElementById('flash-experiment-player').classList.toggle('flow-visible', matchMode !== 'off');
    const matchInfo = [];
    document.getElementById('flash-experiment-explanation').textContent = FLASH_POSE_EXPLANATIONS[method] +
        (showPose ? ' White points: depth measured in the first usable image, moved by the estimated pose.' : '');
    const lost = [];
    for (const [role, object] of Object.entries(row.objects)) {
        if (showMask) {
            context.beginPath();
            for (const contour of object.mask) {
                contour.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
                context.closePath();
            }
            context.fillStyle = 'rgba(0, 210, 200, 0.16)';
            context.strokeStyle = '#00d2c8';
            context.lineWidth = 1.3;
            context.fill('evenodd'); context.stroke();
        }
        const pair = object.matches?.[method === 'ufm_retained_reference' ? 'retry' : 'pair'];
        if (matchMode !== 'off') {
            if (pair) {
                const pixels = pair[matchMode === 'used' ? 'used' : 'raw'];
                const total = pair.counts.reduce((a, b) => a + b, 0);
                const gap = pair.target_frame - pair.source_frame;
                matchInfo.push(`${role}: frame ${pair.source_frame + 1} → ${pair.target_frame + 1} · ${(pair.target_t - pair.source_t).toFixed(3)} s${gap > 1 ? ` · ${gap - 1} skipped frames` : ''} · ${pixels.length} samples available / ${matchMode === 'used' ? pair.counts[4] : total} matches`);
                matchInfo.push(pair.path_sampling === 'full_ufm_output' ? 'Paths sampled from full UFM output.' : 'Older recording: paths interpolated from a sparse saved grid.');
            } else matchInfo.push(`${role}: ${data.has_correspondences ? 'no matching call on this frame' : '2D matches not recorded'}`);
        }
        const points = object.poses[method];
        if (!points) { lost.push(role); continue; }
        if (!showPose) continue;
        // Same white point-cloud + coloured axes convention as flash_surface_cloud.js.
        // This is the moved reference cloud, never a current measurement or a UFM match.
        context.fillStyle = 'rgba(255,255,255,0.85)';
        context.shadowColor = '#07131f';
        context.shadowBlur = 2;
        context.beginPath();
        for (const [x, y] of points.slice(4)) {
            context.moveTo(x + 1, y);
            context.arc(x, y, 1, 0, Math.PI * 2);
        }
        context.fill();
        context.shadowBlur = 0;
        const [origin, ...ends] = points.slice(0, 4);
        context.lineWidth = 2;
        ends.forEach(([x, y], index) => {
            context.strokeStyle = ['#ff6969', '#6fe08a', '#75a8ff'][index];
            context.beginPath(); context.moveTo(...origin); context.lineTo(x, y); context.stroke();
        });
        context.font = '12px sans-serif';
        context.lineWidth = 3;
        context.strokeStyle = '#101827';
        context.fillStyle = '#fff';
        context.strokeText(role, origin[0] + 5, origin[1] - 7);
        context.fillText(role, origin[0] + 5, origin[1] - 7);
    }
    // Matches remain visible above surface and mask overlays, including other objects.
    if (matchMode !== 'off') {
        drawFlashPointPaths(context, data, flashExperiments.frame, method, matchMode, trailFrames, displayScale);
    }
    // Keep detailed evidence available on hover, without a changing paragraph
    // below the player or a second label covering the video.
    document.getElementById('flash-experiment-matches').title = matchMode === 'off' ? '' :
        'Connected paths follow UFM flow. Hollow dots: low UFM visibility; dim dots: little image motion. ' +
        (matchMode === 'raw' ? 'Includes predictions outside the current mask. ' : 'Only matches agreeing with the rigid fit. ') + matchInfo.join(' | ');
    canvas.dataset.frame = String(row.frame);
    canvas.dataset.method = method;
    const next = data.frames[flashExperiments.frame + 1];
    document.getElementById('flash-experiment-status').textContent =
        `Frame ${row.frame + 1} / ${data.frames.length} · ${row.t.toFixed(2)} s` +
        (next ? ` · Next recorded image: +${(next.t - row.t).toFixed(2)} s` : '') +
        (lost.length ? ` · No pose: ${lost.join(', ')}` : '');
}

function drawFlashPointPaths(context, data, index, method, mode, trailFrames, displayScale) {
    // TrackEverything-style heads and fading lines, with identities supplied by
    // composed UFM flow (not by joining nearby independent samples in the GUI).
    const pairKey = method === 'ufm_retained_reference' ? 'retry' : 'pair';
    const first = Math.max(0, index - trailFrames);
    context.save();
    context.shadowBlur = 0;
    context.lineCap = 'round'; context.lineJoin = 'round';
    context.lineWidth = 1.5 / displayScale;
    for (const [role, object] of Object.entries(data.frames[index].objects)) {
        const heads = (object.matches?.[pairKey]?.paths || []).filter(p => mode !== 'used' || p[6] === 4);
        const histories = new Map(heads.map(p => [p[0], { previous: null, segments: [], distance: 0 }]));
        for (let frame = first; frame <= index; frame++) {
            const pair = data.frames[frame].objects[role]?.matches?.[pairKey];
            for (const p of pair?.paths || []) {
                const history = histories.get(p[0]);
                if (!history || (mode === 'used' && p[6] !== 4)) continue;
                const previous = history.previous;
                // Do not draw across a frame where this particle was absent.
                let from = previous?.frame === frame - 1 ? previous.xy : null;
                if (!from && frame > first && pair.source_frame === data.frames[frame - 1].frame) from = p.slice(2, 4);
                const to = p.slice(4, 6);
                if (from) {
                    history.segments.push({ from, to, frame });
                    history.distance += Math.hypot(to[0] - from[0], to[1] - from[1]);
                }
                history.previous = { frame, xy: to };
            }
        }
        for (const p of heads) {
            const history = histories.get(p[0]);
            const moving = history.distance * displayScale > 1;
            const color = `hsl(${p[1]}, 95%, 62%)`;
            context.strokeStyle = color; context.fillStyle = color;
            if (moving && trailFrames) {
                for (const segment of history.segments) {
                    context.globalAlpha = 1 - (index - segment.frame) / (trailFrames + 1);
                    context.beginPath(); context.moveTo(...segment.from); context.lineTo(...segment.to); context.stroke();
                }
            }
            context.globalAlpha = moving || !trailFrames ? 1 : .4;
            context.beginPath(); context.arc(p[4], p[5], 2.5 / displayScale, 0, Math.PI * 2);
            if (p[6] === 0) context.stroke(); else context.fill();
        }
    }
    context.restore();
}

function stepFlashExperiment(delta) {
    const data = flashExperiments.data;
    if (!data) return;
    const video = document.getElementById('flash-experiment-video');
    video.pause();
    const index = Math.max(0, Math.min(data.frames.length - 1, flashExperiments.frame + delta));
    video.currentTime = data.frames[index].t + 1e-6;
}

function toggleFlashExperimentFullscreen() {
    const player = document.getElementById('flash-experiment-player');
    if (document.fullscreenElement) document.exitFullscreen();
    else player.requestFullscreen();
}
