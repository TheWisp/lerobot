/* Read-only tracker experiment browser. Video time owns overlay selection. */
const FLASH_POSE_METHODS = {
    ufm_chain: 'UFM matching → Pose',
    surface_corrected: 'UFM matching → Pose → Depth + mask correction',
    ufm_retained_reference: 'UFM matching → Pose (last matched frame fallback)',
    conditioned_fixed: 'Depth + mask fit, UFM-assisted',
    conditioned_growing: 'Depth + mask fit, UFM-assisted — growing surface',
};
const FLASH_POSE_EXPLANATIONS = {
    ufm_chain: 'RGB matches → depth at both ends → one 3D motion → accumulated pose.',
    surface_corrected: 'Start from UFM’s pose, then fit the surface to depth and mask without retaining the image matches.',
    ufm_retained_reference: 'After matching fails, retry from the last successfully matched frame.',
    conditioned_fixed: 'Fit depth and mask together with usable UFM matches. UFM is queried when a source image exists; the current weights do not enforce strict depth-first priority.',
    conditioned_growing: 'The same UFM-assisted fit, adding newly measured surface over time. The current weights do not enforce strict depth-first priority.',
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
            const name = document.createElement('span'); name.textContent = item.name;
            const score = document.createElement('span'); score.className = 'flash-experiment-list-score';
            button.append(name, score);
            button.title = `${item.name} · ${item.experiment}`;
            button.setAttribute('aria-pressed', String(item.id === flashExperiments.selected));
            button.onclick = () => { showFlashComparison(false); selectFlashExperiment(item); };
            list.append(button);
        }
        refreshFlashMetrics();
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
        const force = document.getElementById('flash-experiment-ignore-cache')?.checked;
        const response = await fetch('/api/run/flash/experiments/rebuild' + (force ? '?force=true' : ''), { method: 'POST' });
        if (!response.ok && response.status !== 409) {
            const error = await response.json();
            throw new Error(error.detail || `Cannot rerun (${response.status})`);
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
        if (!response.ok) throw new Error(`Cannot read rerun progress (${response.status})`);
        const state = await response.json();
        if (selectedWorkflow !== 'flash' || flashUiMode !== 'experiments') return;
        active = ['starting', 'running', 'cancelling'].includes(state.state);
        // Loading saved Run settings can replace the form while a request is
        // pending. The job survives; a later poll attaches to the new controls.
        if (!document.getElementById('flash-experiment-rebuild')) return;
        document.getElementById('flash-experiment-rebuild').disabled = active || flashRebuild.starting;
        const cancel = document.getElementById('flash-experiment-cancel');
        cancel.hidden = !active; cancel.disabled = state.state === 'cancelling';
        cancel.textContent = state.state === 'cancelling' ? 'Cancelling…' : 'Cancel rerun';
        const progress = document.getElementById('flash-experiment-rebuild-progress');
        progress.hidden = !active;
        progress.max = state.total || 1;
        progress.value = (state.completed || 0) + (state.frames ? state.frame / state.frames : 0);
        const text = document.getElementById('flash-experiment-rebuild-status');
        if (active) text.textContent = `${state.completed} / ${state.total} · ${state.current || state.phase}` +
            (state.current ? ` · ${state.phase}${state.frames ? ` ${state.frame}/${state.frames}` : ''}` : '');
        else if (state.state === 'idle') text.textContent = '';
        else if (state.state === 'cancelled') text.textContent = `Cancelled · ${state.succeeded} / ${state.total} completed. Previous results retained.`;
        else if (state.state === 'failed') text.textContent = state.error || 'Rerun stopped.';
        else text.textContent = `${state.succeeded} / ${state.total} benchmarked` +
            (state.reused ? ` · ${state.reused} reused` : '') + (state.errors.length ? ` · ${state.errors.length} failed` : '');
        const download = document.getElementById('flash-experiment-report-download');
        download.hidden = !state.report_ready;
        if (state.report_ready) {
            const base = `/api/run/flash/experiments/rebuild/${encodeURIComponent(state.id)}`;
            download.href = `${base}/report.json`;
        } else download.removeAttribute('href');
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
    resetFlashExperimentScene();
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
    refreshFlashMetrics();
    drawFlashExperiment();
    try {
        const response = await fetch(`/api/run/flash/experiments/${item.id}/overlays.json`, { signal: request.signal });
        if (!response.ok) throw new Error(`Cannot load overlays (${response.status})`);
        const data = await response.json();
        if (request !== flashExperiments.request) return;
        flashExperiments.data = data;
        flashExperiments.frame = 0;
        document.getElementById("flash-experiment-scrub").max = data.frames.length - 1;
        const canvas = document.getElementById('flash-experiment-overlay');
        canvas.width = data.width;
        canvas.height = data.height;
        const select = document.getElementById('flash-experiment-method');
        const previous = select.value;
        select.replaceChildren(...Object.entries(data.methods).map(([value, label]) => new Option(FLASH_POSE_METHODS[value] || label, value)));
        if (Object.hasOwn(data.methods, previous)) select.value = previous;
        refreshFlashMetrics();
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
        // Browsers may stop video-frame callbacks when the video is hidden by
        // the 3D canvas. Its media clock still drives RGB-D playback and seeks.
        for (const event of ['timeupdate', 'seeked']) video.addEventListener(event, () => { // spellchecker:disable-line
            if (flashSceneState.enabled) update(video.currentTime);
        });
    } else {
        video.addEventListener('timeupdate', () => update(video.currentTime));
        // Standard HTMLMediaElement event name.
        video.addEventListener('seeked', () => update(video.currentTime)); // spellchecker:disable-line
    }
    video.addEventListener('loadeddata', () => update(video.currentTime));
    for (const event of ['play', 'pause']) video.addEventListener(event, () => {
        document.getElementById('flash-experiment-play').textContent = video.paused ? 'Play' : 'Pause';
    });
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
    document.getElementById('flash-experiment-scrub').value = flashExperiments.frame;
    document.getElementById('flash-experiment-time').textContent = `${data.frames[flashExperiments.frame].t.toFixed(2)} s`;
    if (flashSceneState.enabled) requestFlashExperimentScene();
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
    const surfaceDescription = method === 'conditioned_growing' ?
        ' White points: accumulated measured surface at this frame; axes keep their original reference.' :
        ' White points: depth measured in the first usable image, moved by the estimated pose.';
    document.getElementById('flash-experiment-explanation').textContent = (FLASH_POSE_EXPLANATIONS[method] || '') +
        (showPose ? surfaceDescription : '');
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
        const pair = flashExperimentPair(object, method);
        if (matchMode !== 'off') {
            if (pair) {
                const pixels = pair[matchMode === 'used' ? 'used' : 'raw'];
                const total = pair.counts.reduce((a, b) => a + b, 0);
                const gap = pair.target_frame - pair.source_frame;
                matchInfo.push(`${role}: frame ${pair.source_frame + 1} → ${pair.target_frame + 1} · ${(pair.target_t - pair.source_t).toFixed(3)} s${gap > 1 ? ` · ${gap - 1} skipped frames` : ''} · ${pixels.length} samples available / ${matchMode === 'used' ? pair.counts[4] : total} matches`);
                matchInfo.push(pair.path_sampling === 'full_ufm_output' ? 'Paths sampled from full UFM output.' : 'Older recording: paths interpolated from a sparse saved grid.');
            } else matchInfo.push(`${role}: 2D matches unavailable for this estimate`);
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
        (matchMode === 'raw' ? 'Includes predictions outside the current mask. ' : 'Only matches agreeing with the initial UFM rigid fit. ') + matchInfo.join(' | ');
    canvas.dataset.frame = String(row.frame);
    canvas.dataset.method = method;
    const next = data.frames[flashExperiments.frame + 1];
    document.getElementById('flash-experiment-status').textContent =
        `Frame ${row.frame + 1} / ${data.frames.length} · ${row.t.toFixed(2)} s` +
        (next ? ` · Next recorded image: +${(next.t - row.t).toFixed(2)} s` : '') +
        (lost.length ? ` · No pose: ${lost.join(', ')}` : '');
}

function flashExperimentPair(object, method) {
    const pair = object?.matches?.[method === 'ufm_retained_reference' ? 'retry' : 'pair'];
    // A recovery may compare a different earlier image. Never show unrelated
    // previous-frame matches as if they were that estimator's recorded input.
    if (method.startsWith('conditioned_') && object?.match_sources?.[method] !== pair?.source_frame) return null;
    return pair;
}

function drawFlashPointPaths(context, data, index, method, mode, trailFrames, displayScale) {
    // TrackEverything-style heads and fading lines, with identities supplied by
    // composed UFM flow (not by joining nearby independent samples in the GUI).
    const first = Math.max(0, index - trailFrames);
    context.save();
    context.shadowBlur = 0;
    context.lineCap = 'round'; context.lineJoin = 'round';
    context.lineWidth = 1.5 / displayScale;
    for (const [role, object] of Object.entries(data.frames[index].objects)) {
        const heads = (flashExperimentPair(object, method)?.paths || []).filter(p => mode !== 'used' || p[6] === 4);
        const histories = new Map(heads.map(p => [p[0], { previous: null, segments: [], distance: 0 }]));
        for (let frame = first; frame <= index; frame++) {
            const pair = flashExperimentPair(data.frames[frame].objects[role], method);
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

// Whole-recording scores are separate from per-frame playback and fit confidence.
const FLASH_METHOD_COLORS = ['#72b7ff', '#ffbd69', '#b8a0ff', '#61d4c3', '#f48fa7'];
const FLASH_METHOD_SHORT = { ufm_chain: 'UFM', surface_corrected: 'Corrected', ufm_retained_reference: 'Retry', conditioned_fixed: 'Joint fit', conditioned_growing: 'Growing' };

function flashText(parent, tag, text, className = '') {
    const element = document.createElement(tag);
    element.textContent = text; element.className = className; parent.append(element);
    return element;
}
function flashNumber(value, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function flashLimit() { return document.getElementById('flash-experiment-limits')?.value || '10'; }
function flashHasAccuracyLabels(item, role) {
    return item.metrics?.roles?.[role]?.accuracy_labels ?? item.metrics?.accuracy_labels ?? false;
}
function flashMethodMetrics(item, role, method) {
    const result = item.metrics?.roles?.[role];
    const data = result?.methods?.[method];
    if (!data) return null;
    const total = data.total_after_initialization;
    const available = data.available_after_initialization;
    const errors = data.pose_error || {};
    const count = errors[`within_${flashLimit()}mm_${flashLimit()}deg`] ?? (available === 0 ? 0 : null);
    const timing = result.timing_ms || {};
    return {
        pass: flashHasAccuracyLabels(item, role) && total > 0 && count !== null ? count / total * 100 : null,
        count, total, available,
        median: errors.cloud_mm_median, p90: errors.cloud_mm_p90,
        angle: errors.angle_deg_median,
        availability: total > 0 ? available / total * 100 : null,
        prepare: timing.prepare_ms?.median,
        fit: timing[method === 'surface_corrected' ? 'finish_ms' : method]?.median,
    };
}
function changeFlashExperimentMethod() { refreshFlashMetrics(); updateFlashSceneSurfaceControl(); drawFlashExperiment(); }
function refreshFlashMetrics() {
    const method = document.getElementById('flash-experiment-method')?.value;
    document.querySelectorAll('.flash-experiment-item').forEach(button => {
        const item = flashExperiments.items.find(x => x.id === button.dataset.id);
        const score = button.querySelector('.flash-experiment-list-score');
        if (!score || !item) return;
        const roles = Object.keys(item.metrics?.roles || {});
        const values = roles.map(role => [role, flashMethodMetrics(item, role, method)?.pass]);
        score.textContent = values.length && values.every(([,v]) => Number.isFinite(v))
            ? values.map(([role,v]) => `${roles.length > 1 ? role + ': ' : ''}${v.toFixed(0)}%`).join(' · ')
            : !item.metrics ? 'No saved scores' : 'Accuracy unscored';
        score.title = `Selected method · within ${flashLimit()} mm / ${flashLimit()}°. Missing poses fail. Initialization excluded.`;
    });
    renderFlashResults();
    if (!document.getElementById('flash-experiment-comparison')?.hidden) renderFlashComparison();
}
function renderFlashResults() {
    const panel = document.getElementById('flash-experiment-results');
    if (!panel) return;
    panel.replaceChildren();
    const item = flashExperiments.items.find(x => x.id === flashExperiments.selected);
    if (!item) return;
    flashText(panel, 'h3', 'Whole-clip results');
    const method = document.getElementById('flash-experiment-method').value;
    flashText(panel, 'p', FLASH_POSE_METHODS[method] || method, 'flash-metric-muted');
    if (!item.metrics?.roles) { flashText(panel, 'p', 'No saved scores for this recording.'); return; }
    for (const role of Object.keys(item.metrics.roles)) {
        const card = flashText(panel, 'section', '', 'flash-metric-card');
        flashText(card, 'h4', role);
        const value = flashMethodMetrics(item, role, method);
        if (!value) { flashText(card, 'p', 'No result / no initialization'); continue; }
        flashText(card, 'strong', Number.isFinite(value.pass) ? `${value.pass.toFixed(1)}%` : 'Accuracy unmeasured', 'flash-metric-primary');
        if (Number.isFinite(value.pass)) flashText(card, 'p', `${value.count} / ${value.total} frames within ${flashLimit()} mm / ${flashLimit()}°`);
        else if (!flashHasAccuracyLabels(item, role)) flashText(card, 'p', 'No reference poses connected to this result. Tracking availability and timing are measured below.');
        const fields = [
            ['Point error, median / p90', flashHasAccuracyLabels(item, role) ? `${flashNumber(value.median)} / ${flashNumber(value.p90)} mm` : 'Unmeasured'],
            ['Rotation error, median', flashHasAccuracyLabels(item, role) ? `${flashNumber(value.angle)}°` : 'Unmeasured'],
            ['Frames with a pose', `${value.available} / ${value.total} (${flashNumber(value.availability)}%)`],
            ['Shared preparation', `${flashNumber(value.prepare)} ms`],
            ['Additional processing', `${flashNumber(value.fit)} ms`],
        ];
        const dl = flashText(card, 'dl', '');
        for (const [label, text] of fields) { flashText(dl, 'dt', label); flashText(dl, 'dd', text); }
    }
    const help = flashText(panel, 'details', '');
    flashText(help, 'summary', 'How to read these scores');
    flashText(help, 'p', 'Missing poses fail the percentage score. Initialization is excluded. Error medians use returned poses only. Point error measures displacement of the same initial surface points against annotated motion; rotation is not adjusted for symmetry.');
    flashText(help, 'p', 'These are accuracy limits, not confidence. Saved masks are reused, so segmentation is not evaluated. Times are median per-object stages, not end-to-end FPS; — means not separately measured.');
}
function showFlashComparison(show) {
    document.getElementById('flash-experiment-playback').hidden = show;
    document.getElementById('flash-experiment-comparison').hidden = !show;
    document.getElementById('flash-experiment-report-view')?.setAttribute('aria-pressed', String(show));
    if (show) { document.getElementById('flash-experiment-video').pause(); renderFlashComparison(); }
    else {
        const item = flashExperiments.items.find(x => x.id === flashExperiments.selected);
        if (item) {
            document.getElementById('flash-experiment-title').textContent = item.name;
            document.getElementById('flash-experiment-detail').textContent = `${item.experiment} · ${item.mask_source} mask · ${item.frames} frames`;
        }
        drawFlashExperiment();
    }
}
function renderFlashComparison() {
    const charts = document.getElementById('flash-experiment-charts');
    if (!charts) return;
    const metric = document.getElementById('flash-experiment-metric').value;
    document.getElementById('flash-experiment-title').textContent = 'Benchmark results';
    document.getElementById('flash-experiment-detail').textContent = `${flashExperiments.items.length} scenes · Latest completed results; reruns replace each scene when finished.`;
    const methods = [...new Set(flashExperiments.items.flatMap(x => Object.keys(x.methods)))];
    const legend = document.getElementById('flash-experiment-legend'); legend.replaceChildren();
    methods.forEach((method, index) => {
        const label = flashText(legend, 'span', FLASH_POSE_METHODS[method] || method);
        label.style.setProperty('--method-color', FLASH_METHOD_COLORS[index % FLASH_METHOD_COLORS.length]);
    });
    const notes = {
        pass: `Higher is better. Within ${flashLimit()} mm / ${flashLimit()}°. Missing poses fail; initialization excluded. Unlabelled scenes have no accuracy score.`,
        median: 'Lower is better. Error against annotated motion, on returned poses only. Compare availability too: a method may return fewer, easier frames.',
        p90: 'Lower is better. 90% of returned poses have point error below this value. Missing poses are not included; compare availability too.',
        availability: 'Higher means more frames with a pose, not more correct poses. Includes unlabelled scenes; initialization excluded.',
        fit: 'Lower is faster. Median additional processing per object after shared preparation. Not end-to-end latency or FPS. UFM-only stages were not separately timed.',
    };
    const accuracyMetric = ['pass', 'median', 'p90'].includes(metric);
    const scored = flashExperiments.items.filter(item => item.metrics?.roles);
    const labelled = scored.filter(item => item.metrics.accuracy_labels).length;
    const coverage = accuracyMetric ? ` ${labelled} of ${scored.length} scenes with results have reference poses loaded for accuracy measurement.` : '';
    document.getElementById('flash-experiment-comparison-note').textContent = `${notes[metric]}${coverage} Same scale on every scene. Click a bar to view that result.`;
    const groups = flashExperiments.items.flatMap(item => Object.keys(item.metrics?.roles || {}).map(role => ({item, role})))
        .sort((a, b) => Number(flashHasAccuracyLabels(b.item, b.role)) - Number(flashHasAccuracyLabels(a.item, a.role)));
    const values = groups.flatMap(({item, role}) => methods.map(method => flashMethodMetrics(item, role, method)?.[metric])).filter(Number.isFinite);
    const scale = ['pass', 'availability'].includes(metric) ? 100 : Math.max(1, ...values);
    charts.replaceChildren();
    for (const {item, role} of groups) {
        const card = flashText(charts, 'section', '', 'flash-comparison-card');
        card.dataset.id = item.id; card.dataset.role = role;
        flashText(card, 'h3', `${item.name} · ${role}`);
        if (accuracyMetric && !flashHasAccuracyLabels(item, role)) {
            flashText(card, 'p', 'Accuracy unscored: no reference poses connected to this result.', 'flash-metric-muted');
            const availability = flashText(card, 'button', 'Show tracking availability', 'flash-comparison-availability');
            availability.type = 'button';
            availability.onclick = () => {
                document.getElementById('flash-experiment-metric').value = 'availability';
                renderFlashComparison();
                // Keep this scene in view after the chart heights change.
                charts.querySelector(`[data-id="${item.id}"][data-role="${CSS.escape(role)}"]`)?.scrollIntoView({block: 'nearest'});
            };
            continue;
        }
        flashText(card, 'p', `0 — ${flashNumber(scale)} ${['pass', 'availability'].includes(metric) ? '%' : metric === 'fit' ? 'ms' : 'mm'}`, 'flash-chart-scale');
        methods.forEach((method, index) => {
            const data = flashMethodMetrics(item, role, method);
            const value = data?.[metric];
            const row = flashText(card, 'button', '', 'flash-chart-row');
            row.type = 'button'; row.dataset.method = method; row.disabled = !item.methods[method];
            row.style.setProperty('--method-color', FLASH_METHOD_COLORS[index % FLASH_METHOD_COLORS.length]);
            flashText(row, 'span', FLASH_METHOD_SHORT[method] || method);
            const track = flashText(row, 'span', '', 'flash-chart-track');
            const bar = flashText(track, 'span', '', 'flash-chart-bar');
            bar.style.width = `${Number.isFinite(value) ? Math.max(0, Math.min(100, value / scale * 100)) : 0}%`;
            const label = Number.isFinite(value) ? flashNumber(value) : '—';
            flashText(row, 'span', label, 'flash-chart-value');
            row.title = `${FLASH_POSE_METHODS[method] || method}: ${label}${data ? ` · ${data.available}/${data.total} poses returned` : ' · no result'}`;
            row.onclick = async () => {
                await selectFlashExperiment(item);
                if (flashExperiments.selected !== item.id) return;
                document.getElementById('flash-experiment-method').value = method;
                changeFlashExperimentMethod(); showFlashComparison(false);
            };
        });
    }
    for (const item of flashExperiments.items.filter(x => !Object.keys(x.metrics?.roles || {}).length)) {
        const card = flashText(charts, 'section', '', 'flash-comparison-card');
        flashText(card, 'h3', item.name); flashText(card, 'p', 'No saved scores.');
    }
}
async function cancelFlashBenchmarks() {
    const button = document.getElementById('flash-experiment-cancel'); button.disabled = true;
    try {
        const response = await fetch('/api/run/flash/experiments/rebuild/cancel', {method: 'POST'});
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Could not cancel rerun'); }
        await pollFlashRebuild();
    } catch (error) {
        document.getElementById('flash-experiment-rebuild-status').textContent = error.message;
        if (button.isConnected) button.disabled = false;
    }
}


// The 3D frame contains its own RGB, depth and poses; it never paints a late
// result onto a newer video frame. One request at a time, latest scrub wins.
const flashSceneState = {enabled: false, viewer: null, generation: 0, loading: false,
    packet: null, bitmap: null, key: null, retry: null};

function toggleFlashExperimentPlay() {
    const video = document.getElementById('flash-experiment-video');
    if (video.paused) video.play().catch(error => { document.getElementById('flash-experiment-status').textContent = error.message; });
    else video.pause();
}

function seekFlashExperiment(index) {
    const data = flashExperiments.data;
    if (!data) return;
    const video = document.getElementById('flash-experiment-video');
    video.pause();
    video.currentTime = data.frames[index].t + 1e-6;
}

function resetFlashExperimentScene() {
    flashSceneState.generation++;
    clearTimeout(flashSceneState.retry);
    flashSceneState.viewer?.dispose();
    flashSceneState.viewer = null;
    flashSceneState.bitmap?.close();
    flashSceneState.bitmap = null;
    flashSceneState.packet = null;
    flashSceneState.key = null;
    const canvas = document.getElementById('flash-experiment-scene');
    for (const name of ['frame', 'method', 'identity']) delete canvas.dataset[name];
    document.getElementById('flash-scene-note').textContent = '';
}

function toggleFlashExperiment3D() {
    flashSceneState.enabled = !flashSceneState.enabled;
    document.getElementById('flash-experiment-3d-toggle').setAttribute('aria-pressed', String(flashSceneState.enabled));
    document.getElementById('flash-experiment-player').classList.toggle('scene-visible', flashSceneState.enabled);
    document.getElementById('flash-experiment-playback').classList.toggle('scene-visible', flashSceneState.enabled);
    document.getElementById('flash-experiment-scene').hidden = !flashSceneState.enabled;
    document.getElementById('flash-experiment-scene-tools').hidden = !flashSceneState.enabled;
    updateFlashSceneSurfaceControl();
    if (!flashSceneState.enabled) resetFlashExperimentScene();
    drawFlashExperiment();
}

function updateFlashSceneSurfaceControl() {
    document.getElementById('flash-scene-surface-control').hidden =
        document.getElementById('flash-experiment-method').value !== 'conditioned_growing';
}

function drawFlashExperimentScene(rebuild = false) {
    const state = flashSceneState;
    if (!state.viewer || !state.packet) return;
    if (rebuild) {
        const descriptions = [];
        updateFlashSceneSurfaceControl();
        const expanded = state.packet.method === 'conditioned_growing' &&
            document.getElementById('flash-scene-surface').value === 'expanded';
        for (const object of Object.values(state.packet.objects)) {
            descriptions.push({layer: 'estimate', color: 0xffa84b, pose: object.pose,
                points: expanded ? object.surface : object.reference, reference: object.reference});
            descriptions.push({layer: 'labelled', color: 0x34e5a1, pose: object.labelled_pose,
                points: object.labelled_surface, reference: object.reference});
        }
        state.viewer.update(state.packet.depth, state.bitmap, state.packet.camera, descriptions);
    }
    state.viewer.setOptions({scene: document.getElementById('flash-scene-context').checked,
        estimate: document.getElementById('flash-scene-estimate').checked,
        labelled: document.getElementById('flash-scene-label').checked,
        axes: document.getElementById('flash-scene-axes').checked});
}

async function requestFlashExperimentScene() {
    const state = flashSceneState;
    if (!state.enabled || state.loading || !flashExperiments.data || document.hidden) return;
    if (!document.getElementById('flash-experiment-player').getClientRects().length) return;
    const identity = flashExperiments.selected;
    const frame = flashExperiments.frame;
    const method = document.getElementById('flash-experiment-method').value;
    const key = `${identity}:${frame}:${method}`;
    if (key === state.key) { state.viewer?.draw(); return; }
    const generation = state.generation;
    state.loading = true;
    let retry = false;
    const note = document.getElementById('flash-scene-note');
    note.textContent = state.packet ? `Showing 3D frame ${state.packet.frame + 1} · loading…` : 'Loading RGB-D…';
    try {
        const response = await fetch(`/api/run/flash/experiments/${identity}/scene/${frame}?method=${encodeURIComponent(method)}`);
        if (!response.ok) {
            if (response.status === 429) { retry = true; return; }
            const error = await response.json(); throw new Error(error.detail || 'Cannot load this 3D frame');
        }
        const buffer = await response.arrayBuffer();
        const headerLength = new DataView(buffer).getUint32(0, true);
        const packet = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
        if (packet.frame !== frame || packet.method !== method)
            throw new Error('3D response belongs to another frame or tracking method');
        const depthStart = 4 + headerLength;
        packet.depth = new Float32Array(buffer.slice(depthStart, depthStart + packet.depth_bytes));
        const bitmap = await createImageBitmap(new Blob([buffer.slice(depthStart + packet.depth_bytes)], {type: 'image/png'}));
        if (generation !== state.generation || !state.enabled || identity !== flashExperiments.selected ||
            method !== document.getElementById('flash-experiment-method').value) { bitmap.close(); return; }
        const module = await import('/static/rgbd_scene.js?v=3');
        if (generation !== state.generation || !state.enabled || identity !== flashExperiments.selected ||
            method !== document.getElementById('flash-experiment-method').value) { bitmap.close(); return; }
        if (!state.viewer) state.viewer = module.createRgbdScene(document.getElementById('flash-experiment-scene'));
        const previous = state.bitmap;
        state.bitmap = bitmap; state.packet = packet; state.key = key;
        drawFlashExperimentScene(true);
        previous?.close();
        const labelled = Object.values(packet.objects).some(object => object.labelled_pose);
        const hasModel = Object.values(packet.objects).some(object => object.label_kind === 'dataset_model');
        document.getElementById('flash-scene-label').parentElement.title = hasModel
            ? 'Includes a supplied dataset model positioned by its pose labels. The accuracy score uses the initial measured points, independent of this drawing.'
            : 'Initial measured surface moved using the dataset’s labelled poses. A partial observation, not a complete ground-truth mesh.';
        document.getElementById('flash-scene-label').disabled = !labelled;
        const errors = Object.entries(packet.objects).filter(([, obj]) => obj.error).map(([role, obj]) =>
            `${Object.keys(packet.objects).length > 1 ? role + ': ' : ''}${obj.error.point_mm.toFixed(1)} mm / ${obj.error.angle_deg.toFixed(1)}°`);
        const missing = Object.entries(packet.objects).filter(([, obj]) => !obj.pose).map(([role]) => role);
        note.textContent = `3D frame ${frame + 1}` + (errors.length ? ` · Error: ${errors.join(' · ')}` : '') +
            (missing.length ? ` · No tracker pose: ${missing.join(', ')}` : '') +
            (labelled ? '' : packet.label_error ? ' · Labels unavailable' : ' · no pose labels');
        note.title = 'Left-drag: rotate. Right-drag: pan. Scroll: zoom. Error compares the same original points, using the same calculation as the benchmark.' +
            (packet.label_error ? ` ${packet.label_error}` : '');
        const canvas = document.getElementById('flash-experiment-scene');
        canvas.dataset.frame = String(frame); canvas.dataset.method = method; canvas.dataset.identity = identity;
    } catch (error) {
        if (generation === state.generation) { note.textContent = error.message; state.key = key; }
    } finally {
        state.loading = false;
        if (state.enabled && flashExperiments.data) {
            const current = `${flashExperiments.selected}:${flashExperiments.frame}:${document.getElementById('flash-experiment-method').value}`;
            if (retry || current !== key || generation !== state.generation)
                state.retry = setTimeout(requestFlashExperimentScene, retry ? 250 : 0);
        }
    }
}
