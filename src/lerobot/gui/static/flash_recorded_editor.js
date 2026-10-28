// Timeline editing for regular FLASH recordings. Uses the same Run viewer as playback.
let flashRecordedEditor = null;
function closeFlashRecordedEditor() {
    if (!flashRecordedEditor) return;
    flashRecordedEditor = null;
    document.getElementById('flash-recorded-editor')?.remove();
    document.getElementById('rerun-viewer')?.classList.remove('flash-demo-editing');
    document.getElementById('tab-run')?.classList.remove('flash-demo-editing');
    const toggle = document.getElementById('flash-demo-edit-toggle');
    if (toggle) toggle.textContent = 'Mark motion';
}

async function toggleFlashReplayEditor() {
    if (['rollout', 'recorded_demo'].includes(flashReplay?.kind)) return toggleFlashRecordedEditor();
    if (flashReplay?.kind === 'observation_demo') return toggleFlashObservationEditor();
    return toggleFlashDemoEditor();
}

async function toggleFlashRecordedEditor() {
    if (flashRecordedEditor) { closeFlashRecordedEditor(); return; }
    if (!['rollout', 'recorded_demo'].includes(flashReplay?.kind)) return;
    if (flashReplayTimer) toggleFlashReplay();
    const replay = flashReplay;
    const toggle = document.getElementById('flash-demo-edit-toggle');
    toggle.disabled = true;
    try {
        const query = new URLSearchParams({dataset_root: replay.datasetRoot, episode: String(replay.episode)});
        const response = await fetch(`/api/run/flash/recorded/source?${query}`);
        const source = await response.json();
        if (!response.ok) throw new Error(source.detail || 'Cannot inspect recording');
        if (flashReplay !== replay) return;
        flashRecordedEditor = {
            source, intervals: (replay.prepared_intervals || []).map(item => ({...item})),
            startFrame: replay.prepared_intervals?.[0]?.first_frame ?? 0,
            selectedIndex: 0,
            message: '', preparing: false,
        };
        document.getElementById('rerun-viewer').classList.add('flash-demo-editing');
        document.getElementById('tab-run').classList.add('flash-demo-editing');
        const panel = document.createElement('aside');
        panel.id = 'flash-recorded-editor';
        panel.className = 'flash-demo-editor';
        document.getElementById('rerun-viewer').append(panel);
        toggle.textContent = 'Close editor';
        _flashRecordedRender();
    } catch (error) {
        showToast('Cannot mark recording', error.message, 'error');
    } finally {
        toggle.disabled = false;
    }
}

function _flashRecordedPlayhead() {
    return flashReplay?.frames[Number(document.getElementById('flash-tracker-scrub').value)]?.frame_index ?? 0;
}

function _flashRecordedOptions(values, selected) {
    return values.map(([value, label]) => `<option value="${_esc(value)}" ${selected === value ? 'selected' : ''}>${_esc(label)}</option>`).join('');
}

function _flashRecordedUpdateFrame() {
    const state = flashRecordedEditor;
    if (!state) return;
    const playhead = _flashRecordedPlayhead();
    const last = state.source.frame_count - 1;
    const nextStart = state.intervals.length ? state.intervals.at(-1).last_frame : state.startFrame;
    const label = document.getElementById('flash-recorded-playhead');
    if (label) label.textContent = `Frame ${playhead} of ${last}. Move the playback cursor to a change in motion, then end a section there.`;
    const here = document.getElementById('flash-recorded-add-here');
    if (here) here.disabled = playhead <= nextStart;
}

function _flashRecordedRender() {
    const state = flashRecordedEditor;
    if (!state) return;
    const {source, intervals} = state;
    const panel = document.getElementById('flash-recorded-editor');
    const playhead = _flashRecordedPlayhead();
    const last = source.frame_count - 1;
    const nextStart = intervals.length ? intervals[intervals.length - 1].last_frame : state.startFrame;
    const roles = source.roles || [];
    let html = '<h3>Mark motion <button type="button" class="flash-recorded-close" onclick="closeFlashRecordedEditor()" aria-label="Close editor">×</button></h3>';
    html += `<p id="flash-recorded-playhead">Frame ${playhead} of ${last}. Move the playback cursor to a change in motion, then end a section there.</p>`;
    if (!source.ready) html += `<p class="flash-recorded-warning">${_esc(source.reason)}</p>`;
    const evidence = Object.entries(source.pose_counts || {}).map(([role, counts]) =>
        `${_esc(role)}: ${counts.observed || 0} camera-observed, ${counts.predicted || 0} predicted`);
    if (evidence.length) html += `<p>${evidence.join(' · ')}</p>`;
    if (!intervals.length) html += `<p>Motion starts at frame ${state.startFrame}. <button type="button" onclick="flashRecordedSetStart()">Start here</button></p>`;
    html += '<div class="flash-recorded-add">';
    html += `<button type="button" id="flash-recorded-add-here" onclick="flashRecordedAddSection(false)" ${playhead <= nextStart ? 'disabled' : ''}>End section here</button>`;
    html += `<button type="button" onclick="flashRecordedAddSection(true)" ${last <= nextStart ? 'disabled' : ''}>End at last frame</button>`;
    html += '</div>';
    if (intervals.length) {
        html += '<nav class="flash-recorded-sections" aria-label="Motion sections">';
        for (const [index, item] of intervals.entries()) {
            const label = item.mode === 'connection' ? 'Route' : 'Recorded';
            html += `<button type="button" class="${index === state.selectedIndex ? 'is-selected' : ''}" onclick="flashRecordedFocus(${index})">${index + 1}. ${label} · ${item.first_frame}–${item.last_frame}${item.verify_relation ? ' ✓' : ''}</button>`;
        }
        html += '</nav>';
    }
    for (const [index, item] of intervals.entries()) {
        if (index !== state.selectedIndex) continue;
        html += `<section class="flash-recorded-section"><div class="flash-recorded-section-head"><strong>${index + 1}. Frames ${item.first_frame}–${item.last_frame}</strong><button type="button" onclick="flashRecordedRemoveFrom(${index})">Remove</button></div>`;
        html += `<label>How to travel<select data-index="${index}" data-field="mode">${_flashRecordedOptions([
            ['demonstrated', 'Follow recorded motion'], ['connection', 'Plan a route to the end']
        ], item.mode)}</select></label>`;
        html += `<label>What stays aligned<select data-index="${index}" data-field="relation">${_flashRecordedOptions([
            ['ee_to_start', 'Starting position'], ['ee_to_object', 'Gripper to an object'],
            ['held_to_target', 'Held object to a target']
        ], item.relation)}</select></label>`;
        if (item.relation !== 'ee_to_start') html += `<label>Target object<select data-index="${index}" data-field="reference_role">${_flashRecordedOptions(roles.map(role => [role, role]), item.reference_role)}</select></label>`;
        if (item.relation === 'held_to_target') html += `<label>Held object<select data-index="${index}" data-field="held_role">${_flashRecordedOptions(roles.filter(role => role !== item.reference_role).map(role => [role, role]), item.held_role)}</select></label>`;
        if (item.relation !== 'ee_to_start' && source.ready) {
            const route = item.mode === 'connection';
            const total = route ? 2 : item.last_frame - item.first_frame + 1;
            const evidenceFor = role => {
                const count = (source.observed_frames?.[role] || []).filter(frame =>
                    route ? frame === item.first_frame || frame === item.last_frame
                        : frame >= item.first_frame && frame <= item.last_frame).length;
                return `${count}/${total} frames with measured full pose`;
            };
            html += `<p>Target: ${evidenceFor(item.reference_role)}${item.held_role && item.relation === 'held_to_target' ? ` · held: ${evidenceFor(item.held_role)}` : ''}</p>`;
        }
        if (item.mode === 'connection') html += `<label>End point<select data-index="${index}" data-field="endpoint_requirement">${_flashRecordedOptions([
            ['required', 'Must reach'], ['preferred', 'Best effort']
        ], item.endpoint_requirement)}</select></label>`;
        html += `<label>At the end<select data-index="${index}" data-field="attachment_action">${_flashRecordedOptions([
            ['', 'No hold change'], ['attach', 'Gripper takes an object'], ['detach', 'Gripper releases an object']
        ], item.attachment_action)}</select></label>`;
        if (item.attachment_action) html += `<label>Object<select data-index="${index}" data-field="attachment_role">${_flashRecordedOptions(roles.map(role => [role, role]), item.attachment_role)}</select></label>`;
        if (roles.length >= 2) {
            html += `<label class="flash-recorded-check"><input type="checkbox" data-index="${index}" data-field="verify_relation" ${item.verify_relation ? 'checked' : ''}> Check final object relationship</label>`;
            if (item.verify_relation) {
                const check = item.verify_relation;
                html += '<p>Compare the two objects with their relative position in this demo frame.</p>';
                html += `<label>Moving object<select data-index="${index}" data-check-field="moving_role">${_flashRecordedOptions(roles.map(role => [role, role]), check.moving_role)}</select></label>`;
                html += `<label>Target object<select data-index="${index}" data-check-field="reference_role">${_flashRecordedOptions(roles.filter(role => role !== check.moving_role).map(role => [role, role]), check.reference_role)}</select></label>`;
                html += `<label>Allowed position difference from demo (mm)<input type="number" min="0" step="any" placeholder="Choose for this task" data-index="${index}" data-check-field="translation_tolerance_mm" value="${_esc(check.translation_tolerance_mm ?? '')}"></label>`;
                html += `<label>Allowed rotation difference (degrees; optional)<input type="number" min="0" step="any" placeholder="Any rotation" data-index="${index}" data-check-field="rotation_tolerance_deg" value="${_esc(check.rotation_tolerance_deg ?? '')}"></label>`;
                const observed = [check.moving_role, check.reference_role].filter(role =>
                    !(source.observed_frames?.[role] || []).includes(item.last_frame));
                html += observed.length
                    ? `<p class="flash-recorded-warning">Frame ${item.last_frame} has no measured full pose for ${_esc(observed.join(' and '))}. Choose an observed end frame or improve tracking before preparing this check.</p>`
                    : '<p>The run waits for fresh measured poses of both objects before this section counts as complete.</p>';
            }
        }
        if (item.relation !== 'ee_to_start') html += `<label class="flash-recorded-check"><input type="checkbox" data-index="${index}" data-field="allow_predicted_demo_pose" ${item.allow_predicted_demo_pose ? 'checked' : ''}> Accept predicted pose where camera observation is missing</label>`;
        html += '</section>';
    }
    for (const role of source.symmetry_roles || []) {
        if (!intervals.some(item => item.reference_role === role || item.held_role === role)) continue;
        const observed = source.observed_frames?.[role] || [];
        const current = state.symmetryBaselines?.[role];
        html += `<label>Observed frame for ${_esc(role)} rotation rule<select data-symmetry="${_esc(role)}">${_flashRecordedOptions([['', 'Select observed frame'], ...observed.map(frame => [String(frame), `Frame ${frame}`])], current == null ? '' : String(current))}</select></label>`;
    }
    html += `<p>${intervals.length ? `Selected motion: frames ${intervals[0].first_frame}–${intervals.at(-1).last_frame}.` : 'Mark at least one section of motion to prepare a candidate.'}</p>`;
    html += `<button type="button" id="flash-recorded-prepare" onclick="flashRecordedPrepare()" ${!source.ready || !intervals.length || state.preparing ? 'disabled' : ''}>Prepare candidate</button>`;
    html += `<p class="flash-editor-status" id="flash-recorded-status">${_esc(state.message)}</p>`;
    panel.innerHTML = html;
    for (const input of panel.querySelectorAll('[data-field]')) {
        input.addEventListener('change', () => {
            const item = intervals[Number(input.dataset.index)];
            if (input.dataset.field === 'verify_relation') {
                item.verify_relation = input.checked ? {
                    moving_role: roles[0], reference_role: roles[1], translation_tolerance_mm: '',
                    rotation_tolerance_deg: '',
                } : null;
            } else item[input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value;
            if (input.dataset.field === 'reference_role' && item.held_role === item.reference_role)
                item.held_role = roles.find(role => role !== item.reference_role) || '';
            if (input.dataset.field === 'attachment_action' && item.attachment_action && !item.attachment_role)
                item.attachment_role = roles[0] || '';
            state.message = '';
            _flashRecordedRender();
        });
    }
    for (const input of panel.querySelectorAll('[data-check-field]')) {
        input.addEventListener('change', () => {
            const check = intervals[Number(input.dataset.index)].verify_relation;
            check[input.dataset.checkField] = input.value;
            if (input.dataset.checkField === 'moving_role' && check.reference_role === check.moving_role)
                check.reference_role = roles.find(role => role !== check.moving_role) || '';
            state.message = '';
            _flashRecordedRender();
        });
    }
    for (const input of panel.querySelectorAll('[data-symmetry]')) {
        input.addEventListener('change', () => {
            state.symmetryBaselines ||= {};
            if (input.value) state.symmetryBaselines[input.dataset.symmetry] = Number(input.value);
            else delete state.symmetryBaselines[input.dataset.symmetry];
        });
    }
}

function flashRecordedFocus(index) {
    const state = flashRecordedEditor;
    if (!state || index < 0 || index >= state.intervals.length) return;
    state.selectedIndex = index;
    const frame = state.intervals[index].last_frame;
    const replayIndex = flashReplay?.frames.findIndex(item => item.frame_index === frame) ?? -1;
    if (replayIndex >= 0) seekFlashReplay(replayIndex);
    _flashRecordedRender();
}

function flashRecordedAddSection(toEnd) {
    const state = flashRecordedEditor;
    if (!state) return;
    const first = state.intervals.length ? state.intervals[state.intervals.length - 1].last_frame : state.startFrame;
    const last = toEnd ? state.source.frame_count - 1 : _flashRecordedPlayhead();
    if (last <= first) return;
    const roles = state.source.roles || [];
    state.intervals.push({
        first_frame: first, last_frame: last, mode: 'demonstrated', relation: 'ee_to_start',
        reference_role: roles[0] || '', held_role: roles[1] || '', endpoint_requirement: 'required',
        attachment_action: '', attachment_role: '', allow_predicted_demo_pose: false,
    });
    state.selectedIndex = state.intervals.length - 1;
    state.message = '';
    _flashRecordedRender();
}

function flashRecordedSetStart() {
    if (!flashRecordedEditor || flashRecordedEditor.intervals.length) return;
    flashRecordedEditor.startFrame = _flashRecordedPlayhead();
    _flashRecordedRender();
}

function flashRecordedRemoveFrom(index) {
    flashRecordedEditor?.intervals.splice(index);
    if (flashRecordedEditor)
        flashRecordedEditor.selectedIndex = Math.min(index, flashRecordedEditor.intervals.length - 1);
    _flashRecordedRender();
}

async function flashRecordedPrepare() {
    const state = flashRecordedEditor;
    if (!state || state.preparing) return;
    state.preparing = true;
    state.message = 'Checking recorded poses and preparing…';
    _flashRecordedRender();
    try {
        const response = await fetch('/api/run/flash/recorded/prepare', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                dataset_root: flashReplay.datasetRoot, episode: flashReplay.episode,
                name: `recorded-${flashReplay.episode}-${Date.now()}`,
                intervals: state.intervals, symmetry_baseline_frames: state.symmetryBaselines || {},
            }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Preparation failed');
        state.message = `Candidate saved: ${result.bundle_path.split('/').slice(-2, -1)[0]}`;
        await _loadFlashRecordings();
    } catch (error) {
        state.message = error.message;
    } finally {
        state.preparing = false;
        if (flashRecordedEditor === state) _flashRecordedRender();
    }
}
