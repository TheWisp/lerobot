// RGB-D-only object relationship editing in the shared FLASH playback.
let flashObservationEditor = null;

function closeFlashObservationEditor() {
    if (!flashObservationEditor) return;
    flashObservationEditor = null;
    flashSceneView?.setTurnPreview(null);
    document.getElementById('flash-observation-editor')?.remove();
    document.getElementById('rerun-viewer')?.classList.remove('flash-demo-editing');
    document.getElementById('tab-run')?.classList.remove('flash-demo-editing');
    document.getElementById('flash-demo-edit-toggle').textContent = 'Edit relationships';
}

function _flashObservationFrame() {
    return flashReplay?.frames[Number(document.getElementById('flash-tracker-scrub').value)];
}

function _flashObservationRoles(replay) {
    return [...new Set(replay.frames.flatMap(frame => Object.keys(frame.snapshot.observed_masks || {})))].sort();
}

function _flashObservationCurrentKey() {
    const state = flashObservationEditor;
    const index = _flashObservationFrame()?.frame_index;
    if (state?.addingNew) return null;
    return state?.annotation.keyframes.find(item => item.frame_index === index &&
        item.moving_role === state.selected?.moving_role && item.reference_role === state.selected?.reference_role)
        || state?.annotation.keyframes.find(item => item.frame_index === index);
}

function _flashObservationSameKey(a, b) {
    return a.frame_index === b.frame_index && a.moving_role === b.moving_role &&
        a.reference_role === b.reference_role;
}

function _flashObservationSelectKey(key) {
    const state = flashObservationEditor;
    if (!state) return;
    state.selected = key ? {...key, allowed_turn: key.allowed_turn ? {...key.allowed_turn} : null} : null;
    state.pickingTurn = false;
    _flashObservationRender();
}

function _flashObservationUpdateFrame() {
    const state = flashObservationEditor;
    if (!state) return;
    const frame = _flashObservationFrame();
    if (frame?.frame_index !== state.lastFrameIndex) {
        state.lastFrameIndex = frame?.frame_index;
        state.addingNew = false;
    }
    const status = document.getElementById('flash-observation-frame');
    if (status && frame) {
        const observed = Object.entries(frame.snapshot.accepted || {}).filter(([, estimate]) => estimate.evidence === 'observed').map(([role]) => role);
        status.textContent = `Frame ${frame.frame_index} · camera-observed: ${observed.join(', ') || 'none'}`;
    }
    const key = _flashObservationCurrentKey();
    if (key !== state.selected && (key?.frame_index !== state.selected?.frame_index ||
        key?.moving_role !== state.selected?.moving_role ||
        key?.reference_role !== state.selected?.reference_role)) _flashObservationSelectKey(key);
}

async function toggleFlashObservationEditor() {
    if (flashObservationEditor) { closeFlashObservationEditor(); return; }
    const replay = flashReplay;
    if (replay?.kind !== 'observation_demo') return;
    closeFlashObservationTracking();
    if (flashReplayTimer) toggleFlashReplay();
    const roles = _flashObservationRoles(replay);
    if (!replay.evidence_available || roles.length < 2 || !replay.annotations) {
        showToast('Cannot edit relationships', replay.annotation_warning ||
            'Track at least two objects in this RGB-D episode first.', 'error');
        return;
    }
    flashObservationEditor = {
        annotation: structuredClone(replay.annotations), roles,
        selected: null, addingNew: false, lastFrameIndex: null,
        pickingTurn: false, saving: false, message: '',
    };
    document.getElementById('rerun-viewer').classList.add('flash-demo-editing');
    document.getElementById('tab-run').classList.add('flash-demo-editing');
    const panel = document.createElement('aside');
    panel.id = 'flash-observation-editor';
    panel.className = 'flash-demo-editor';
    panel.setAttribute('aria-label', 'Edit object relationships in RGB-D episode');
    document.getElementById('rerun-viewer').append(panel);
    document.getElementById('flash-demo-edit-toggle').textContent = 'Close editor';
    _flashObservationRender();
}

function _flashObservationRender() {
    const state = flashObservationEditor;
    if (!state) return;
    const frame = _flashObservationFrame();
    const selected = state.selected;
    const roles = state.roles;
    const moving = selected?.moving_role || roles[0];
    const reference = selected?.reference_role || roles.find(role => role !== moving);
    const required = selected?.required ?? false;
    const turn = selected?.allowed_turn;
    const evidence = frame?.snapshot.accepted || {};
    const evidenceText = [moving, reference].map(role =>
        `${role}: ${evidence[role]?.evidence || frame?.snapshot.pose_status?.[role]?.evidence || 'unavailable'}`
    ).join(' · ');
    const relationStatus = selected
        ? (evidence[moving]?.evidence === 'observed' && evidence[reference]?.evidence === 'observed'
            ? 'Both poses measured in this frame.' : 'Relative arrangement is unverified in this frame.')
        : '';
    const options = value => roles.map(role => `<option value="${_esc(role)}" ${role === value ? 'selected' : ''}>${_esc(role)}</option>`).join('');
    const items = state.annotation.keyframes.map((key, position) =>
        `<button type="button" onclick="flashObservationJumpByIndex(${position})">` +
        `Frame ${key.frame_index} · ${_esc(key.moving_role)} → ${_esc(key.reference_role)}` +
        `${key.required ? ' · required' : ' · review'}${key.allowed_turn ? ' · turn allowed' : ''}</button>`).join('');
    document.getElementById('flash-observation-editor').innerHTML =
        '<h3>Object relationships <button type="button" class="flash-recorded-close" onclick="closeFlashObservationEditor()" aria-label="Close editor">×</button></h3>' +
        '<p>Mark the relationship at this frame. Robot actions are unavailable in this RGB-D recording.</p>' +
        `<p id="flash-observation-frame">Frame ${frame?.frame_index ?? 0}</p>` +
        `<p>${_esc(evidenceText)}. A predicted pose is not a fresh camera measurement.</p>` +
        (relationStatus ? `<p ${relationStatus.startsWith('Both poses') ? '' : 'class="flash-recorded-warning"'}>${relationStatus}</p>` : '') +
        `<label>Moving object<select id="flash-observation-moving">${options(moving)}</select></label>` +
        `<label>Reference object<select id="flash-observation-reference">${options(reference)}</select></label>` +
        `<label>Relationship<select id="flash-observation-required"><option value="false" ${!required ? 'selected' : ''}>Review only</option>` +
        `<option value="true" ${required ? 'selected' : ''}>Required at this key frame</option></select></label>` +
        '<button type="button" onclick="flashObservationMark()">' + (selected ? 'Update key frame' : 'Mark key frame') + '</button>' +
        (selected ? '<button type="button" onclick="flashObservationNew()">Add another relation here</button>' : '') +
        (selected ? '<button type="button" onclick="flashObservationRemove()">Remove</button>' : '') +
        '<section><strong>Allowed adaptation</strong><p>Optional permission for the selected relationship, not an automatic motion.</p>' +
        (turn ? `<p>Turn around a point on ${_esc(reference)} · ${_esc(turn.source_evidence)} pivot</p>` : '<p>No turn permitted.</p>') +
        `<button type="button" onclick="flashObservationPickTurn()" ${selected ? '' : 'disabled'}>Pick turn point in 3D</button>` +
        (turn ? `<label>Minimum angle (degrees)<input id="flash-observation-min" type="number" min="-180" max="180" value="${Math.round(turn.min_angle_rad * 180 / Math.PI)}" onchange="flashObservationTurnRange()"></label>` +
            `<label>Maximum angle (degrees)<input id="flash-observation-max" type="number" min="-180" max="180" value="${Math.round(turn.max_angle_rad * 180 / Math.PI)}" onchange="flashObservationTurnRange()"></label>` +
            '<button type="button" onclick="flashObservationClearTurn()">Remove turn permission</button>' : '') +
        '</section><section><strong>Key frames</strong><div class="flash-recorded-sections">' + (items || '<p>None yet.</p>') + '</div></section>' +
        `<section class="flash-editor-footer"><button type="button" onclick="flashObservationSave()" ${state.saving ? 'disabled' : ''}>Save annotations</button>` +
        `<p class="flash-editor-status" id="flash-observation-message">${_esc(state.message)}</p></section>`;
    flashSceneView?.setTurnPreview(turn && selected ? {reference_role: selected.reference_role, ...turn} : null);
    _flashObservationUpdateFrame();
}

function flashObservationMark() {
    const state = flashObservationEditor;
    const index = _flashObservationFrame()?.frame_index;
    const moving = document.getElementById('flash-observation-moving').value;
    const reference = document.getElementById('flash-observation-reference').value;
    if (moving === reference) { state.message = 'Select two different objects.'; _flashObservationRender(); return; }
    const previous = state.selected;
    const key = {frame_index: index, moving_role: moving, reference_role: reference,
        required: document.getElementById('flash-observation-required').value === 'true',
        allowed_turn: previous?.moving_role === moving && previous.reference_role === reference
            ? previous.allowed_turn : null};
    state.annotation.keyframes = [...state.annotation.keyframes.filter(item =>
        !(previous && _flashObservationSameKey(item, previous)) && !_flashObservationSameKey(item, key)), key]
        .sort((a, b) => a.frame_index - b.frame_index);
    state.selected = key;
    state.addingNew = false;
    state.message = 'Key frame marked as a draft. Save annotations to keep it.';
    _flashObservationRender();
    _flashRenderTimeline();
}

function flashObservationRemove() {
    const state = flashObservationEditor;
    state.annotation.keyframes = state.annotation.keyframes.filter(item =>
        !_flashObservationSameKey(item, state.selected));
    state.selected = null;
    state.addingNew = true;
    state.message = 'Key frame removed from draft.';
    _flashObservationRender();
    _flashRenderTimeline();
}

function flashObservationNew() {
    flashObservationEditor.selected = null;
    flashObservationEditor.addingNew = true;
    _flashObservationRender();
}

function flashObservationJump(index, moving, reference) {
    const position = flashReplay.frames.findIndex(frame => frame.frame_index === index);
    if (position >= 0) {
        if (flashObservationEditor) flashObservationEditor.addingNew = false;
        seekFlashReplay(position);
        const key = flashObservationEditor?.annotation.keyframes.find(item =>
            item.frame_index === index && item.moving_role === moving && item.reference_role === reference);
        if (key) _flashObservationSelectKey(key);
    }
}

function flashObservationJumpByIndex(position) {
    const key = flashObservationEditor?.annotation.keyframes[position];
    if (key) flashObservationJump(key.frame_index, key.moving_role, key.reference_role);
}

async function flashObservationPickTurn() {
    const state = flashObservationEditor;
    if (!state?.selected) return;
    if (flashSceneMode !== '3d') await toggleFlashSceneView();
    state.pickingTurn = true;
    state.message = `Click the measured surface of ${state.selected.reference_role} in the 3D scene.`;
    _flashObservationRender();
}

function flashObservationPickedTurn(picked) {
    const state = flashObservationEditor;
    if (!state?.pickingTurn) return;
    if (picked.role !== state.selected.reference_role || !picked.reference_xyz_m) {
        state.message = `Pick the measured surface of ${state.selected.reference_role}.`;
        _flashObservationRender();
        return;
    }
    state.selected.allowed_turn = {
        pivot_reference_m: picked.reference_xyz_m,
        axis_reference: picked.reference_normal,
        min_angle_rad: -Math.PI,
        max_angle_rad: Math.PI,
        source_evidence: picked.reference_pose_evidence,
    };
    state.annotation.keyframes = state.annotation.keyframes.map(item =>
        _flashObservationSameKey(item, state.selected) ? state.selected : item);
    state.pickingTurn = false;
    state.message = picked.reference_pose_evidence === 'observed'
        ? 'Turn permission is based on this camera-observed surface.'
        : 'Turn permission is a draft: its local point depends on a predicted pose.';
    _flashObservationRender();
}

function flashObservationTurnRange() {
    const state = flashObservationEditor;
    const turn = state?.selected?.allowed_turn;
    if (!turn) return;
    const min = Number(document.getElementById('flash-observation-min').value);
    const max = Number(document.getElementById('flash-observation-max').value);
    if (!Number.isFinite(min) || !Number.isFinite(max) || min < -180 || max > 180 || min >= max) {
        state.message = 'Choose an increasing angle range within −180° to 180°.';
        document.getElementById('flash-observation-message').textContent = state.message;
        return;
    }
    turn.min_angle_rad = min * Math.PI / 180;
    turn.max_angle_rad = max * Math.PI / 180;
    state.annotation.keyframes = state.annotation.keyframes.map(item =>
        _flashObservationSameKey(item, state.selected) ? state.selected : item);
    state.message = 'Turn range changed in draft.';
    _flashObservationRender();
}

function flashObservationClearTurn() {
    const state = flashObservationEditor;
    state.selected.allowed_turn = null;
    state.annotation.keyframes = state.annotation.keyframes.map(item =>
        _flashObservationSameKey(item, state.selected) ? state.selected : item);
    state.message = 'Turn permission removed from draft.';
    _flashObservationRender();
}

async function flashObservationSave() {
    const state = flashObservationEditor;
    if (!state || state.saving) return;
    state.saving = true;
    state.message = 'Saving…';
    _flashObservationRender();
    try {
        const response = await fetch('/api/run/flash/observation/annotations', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: flashReplay.datasetRoot, episode: flashReplay.episode,
                annotation: state.annotation}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot save annotations');
        if (flashObservationEditor !== state) return;
        state.annotation = result;
        flashReplay.annotations = structuredClone(result);
        state.message = 'Annotations saved. They describe object relationships; this episode has no robot actions.';
        _flashRenderTimeline();
    } catch (error) {
        state.message = error.message;
    } finally {
        state.saving = false;
        if (flashObservationEditor === state) _flashObservationRender();
    }
}

// Reprocess observation-only episodes in the same camera view used for review.
let flashObservationTracking = null;
let flashObservationProcessing = null;

function closeFlashObservationTracking() {
    flashObservationTracking = null;
    document.getElementById('flash-observation-tracking')?.remove();
    document.getElementById('rerun-viewer')?.classList.remove('flash-demo-editing');
    document.getElementById('tab-run')?.classList.remove('flash-demo-editing');
}

async function toggleFlashObservationTracking() {
    if (flashObservationTracking) { closeFlashObservationTracking(); return; }
    const replay = flashReplay;
    if (replay?.kind !== 'observation_demo') return;
    closeFlashObservationEditor();
    if (flashReplayTimer) toggleFlashReplay();
    seekFlashReplay(0);
    try {
        const response = await fetch('/api/run/flash/observation/defaults');
        const defaults = await response.json();
        if (!response.ok) throw new Error(defaults.detail || 'Cannot load tracking parameters');
        const prompts = replay.processing?.prompts_by_role || {};
        const seeds = replay.processing?.seed_points_by_role || {};
        const objects = Object.entries(prompts).map(([role, prompt]) => ({role, prompt, seed: seeds[role] || null}));
        flashObservationTracking = {
            replay, objects: objects.length ? objects : [{role: '', prompt: '', seed: null}],
            config: replay.processing_config || defaults, picking: null, busy: false, message: '',
        };
        const panel = document.createElement('aside');
        panel.id = 'flash-observation-tracking';
        panel.className = 'flash-demo-editor';
        panel.setAttribute('aria-label', 'Track objects in RGB-D observation');
        document.getElementById('rerun-viewer').append(panel);
        document.getElementById('rerun-viewer').classList.add('flash-demo-editing');
        document.getElementById('tab-run').classList.add('flash-demo-editing');
        _flashObservationTrackingRender();
    } catch (error) { showToast('Cannot open object tracking', error.message, 'error'); }
}

function _flashObservationTrackingRender() {
    const state = flashObservationTracking;
    if (!state) return;
    const attribute = value => _esc(value).replaceAll('"', '&quot;').replaceAll("'", '&#39;');
    const objects = state.objects.map((object, index) =>
        `<section class="flash-observation-object"><strong>Object ${index + 1}</strong>` +
        `<label>Name<input aria-label="Object ${index + 1} name" value="${attribute(object.role)}" oninput="flashObservationTrackingSet(${index}, 'role', this.value)"></label>` +
        `<label>What it looks like<input aria-label="Object ${index + 1} description" value="${attribute(object.prompt)}" oninput="flashObservationTrackingSet(${index}, 'prompt', this.value)"></label>` +
        `<button type="button" onclick="flashObservationTrackingPick(${index})">${state.picking === index ? 'Click the object in the image…' : 'Click object in first frame'}</button>` +
        (object.seed ? `<span class="flash-observation-seed">Seed: ${object.seed.map(value => Math.round(value)).join(', ')}</span>` : '') +
        `<button type="button" onclick="flashObservationTrackingRemove(${index})" ${state.objects.length === 1 ? 'disabled' : ''}>Remove</button></section>`
    ).join('');
    document.getElementById('flash-observation-tracking').innerHTML =
        '<h3>Track objects <button type="button" class="flash-recorded-close" onclick="closeFlashObservationTracking()" aria-label="Close tracking setup">×</button></h3>' +
        '<p>Name what to follow. A click helps identify it in the first frame; tracking then follows the RGB-D video. This episode has no robot actions.</p>' +
        objects + '<button type="button" onclick="flashObservationTrackingAdd()">Add object</button>' +
        '<details><summary>Tracking parameters</summary><p>Shared surface and scene-tracking settings, saved with this processing pass.</p>' +
        `<textarea id="flash-observation-config" aria-label="Tracking parameters JSON" rows="12" spellcheck="false" onchange="flashObservationTrackingConfig(this.value)">${_esc(JSON.stringify(state.config, null, 2))}</textarea></details>` +
        `<section class="flash-editor-footer"><button type="button" id="flash-observation-process" onclick="flashObservationTrackingStart()" ${state.busy || _isRunning ? 'disabled' : ''}>Process tracking</button>` +
        `<p class="flash-editor-status" id="flash-observation-tracking-message" role="status">${_esc(state.message)}</p></section>`;
}

function flashObservationTrackingSet(index, field, value) {
    const object = flashObservationTracking?.objects[index];
    if (object) object[field] = value;
}

function flashObservationTrackingConfig(value) {
    const state = flashObservationTracking;
    if (!state) return;
    try { state.config = JSON.parse(value); state.message = 'Parameters updated.'; }
    catch (error) { state.message = `Invalid tracking parameters: ${error.message}`; }
    document.getElementById('flash-observation-tracking-message').textContent = state.message;
}

function flashObservationTrackingAdd() {
    flashObservationTracking.objects.push({role: '', prompt: '', seed: null});
    _flashObservationTrackingRender();
}

function flashObservationTrackingRemove(index) {
    if (flashObservationTracking?.objects.length > 1) {
        flashObservationTracking.objects.splice(index, 1);
        flashObservationTracking.picking = null;
        _flashObservationTrackingRender();
    }
}

function flashObservationTrackingPick(index) {
    const state = flashObservationTracking;
    if (!state) return;
    state.picking = index;
    if (flashSceneMode === '3d') void toggleFlashSceneView();
    seekFlashReplay(0);
    state.message = 'Click the object in the camera image, not in the 3D scene.';
    _flashObservationTrackingRender();
}

function flashObservationTrackingClick(event) {
    const state = flashObservationTracking;
    if (!state || state.picking === null ||
        (event.target !== event.currentTarget && !event.target.matches('.flash-replay-grid > img, .flash-pose-canvas'))) return false;
    const camera = state.replay.frames[0]?.snapshot.camera;
    const image = document.querySelector('#rerun-viewer .flash-replay-grid > img');
    if (!camera || !image) return false;
    const bounds = image.getBoundingClientRect();
    const scale = Math.min(bounds.width / camera.width, bounds.height / camera.height);
    const left = bounds.left + (bounds.width - camera.width * scale) / 2;
    const top = bounds.top + (bounds.height - camera.height * scale) / 2;
    const x = (event.clientX - left) / scale;
    const y = (event.clientY - top) / scale;
    if (x < 0 || y < 0 || x >= camera.width || y >= camera.height) return false;
    state.objects[state.picking].seed = [x, y];
    state.picking = null;
    state.message = 'Object click saved for the first frame.';
    _flashObservationTrackingRender();
    return true;
}

async function flashObservationTrackingStart() {
    const state = flashObservationTracking;
    if (!state || state.busy) return;
    let config;
    try { config = JSON.parse(document.getElementById('flash-observation-config').value); }
    catch (error) { state.message = `Invalid tracking parameters: ${error.message}`; _flashObservationTrackingRender(); return; }
    const prompts = {}, seeds = {};
    for (const object of state.objects) {
        const role = object.role.trim(), prompt = object.prompt.trim();
        if (!role || !prompt || prompts[role]) {
            state.message = 'Give each object a distinct name and a visual description.';
            _flashObservationTrackingRender();
            return;
        }
        prompts[role] = prompt;
        if (object.seed) seeds[role] = object.seed;
    }
    state.busy = true;
    state.message = 'Starting RGB-D tracking…';
    _flashObservationTrackingRender();
    try {
        const response = await fetch('/api/run/flash/observation/process', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: state.replay.datasetRoot, episode: state.replay.episode,
                prompts_by_role: prompts, seed_points_by_role: seeds, ...config}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot start RGB-D tracking');
        flashObservationProcessing = {
            datasetRoot: state.replay.datasetRoot, episode: state.replay.episode, jobId: result.job_id,
        };
        state.message = 'Tracking is running. The previous replay stays available until this pass completes.';
        document.getElementById('flash-observation-tracking-message').textContent = state.message;
        await pollRunStatus();
    } catch (error) {
        state.busy = false;
        state.message = error.message;
        _flashObservationTrackingRender();
    }
}

async function flashObservationProcessComplete() {
    const job = flashObservationProcessing;
    if (!job) return;
    const response = await fetch(`/api/run/flash/observation/process-status?job_id=${encodeURIComponent(job.jobId)}`);
    const status = await response.json();
    if (!response.ok) {
        flashObservationProcessing = null;
        showToast('Tracking status unavailable', status.detail || 'Reload the recording to inspect the saved evidence.', 'error');
        return;
    }
    if (status.status === 'running') return;
    flashObservationProcessing = null;
    if (status.status !== 'complete') {
        const message = `RGB-D tracking stopped without replacing the previous replay: ${status.reason || 'unknown error'}`;
        if (flashObservationTracking) {
            flashObservationTracking.busy = false;
            flashObservationTracking.message = message;
            _flashObservationTrackingRender();
        }
        showToast('Tracking did not finish', message, 'error');
        return;
    }
    await _loadFlashRecordings();
    if (flashReplay?.datasetRoot === job.datasetRoot && flashReplay?.episode === job.episode) {
        await loadFlashObservationReplay(job.datasetRoot, job.episode);
    }
}
