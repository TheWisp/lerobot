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
    const nextStart = state.intervals.length ? state.intervals.at(-1).last_frame : 0;
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
    const nextStart = intervals.length ? intervals[intervals.length - 1].last_frame : 0;
    const roles = source.roles || [];
    let html = '<h3>Mark motion <button type="button" class="flash-recorded-close" onclick="closeFlashRecordedEditor()" aria-label="Close editor">×</button></h3>';
    html += `<p id="flash-recorded-playhead">Frame ${playhead} of ${last}. Move the playback cursor to a change in motion, then end a section there.</p>`;
    if (!source.ready) html += `<p class="flash-recorded-warning">${_esc(source.reason)}</p>`;
    html += '<div class="flash-recorded-add">';
    html += `<button type="button" id="flash-recorded-add-here" onclick="flashRecordedAddSection(false)" ${playhead <= nextStart ? 'disabled' : ''}>End section here</button>`;
    html += `<button type="button" onclick="flashRecordedAddSection(true)" ${last <= nextStart ? 'disabled' : ''}>End at last frame</button>`;
    html += '</div>';
    for (const [index, item] of intervals.entries()) {
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
        if (item.mode === 'connection') html += `<label>End point<select data-index="${index}" data-field="endpoint_requirement">${_flashRecordedOptions([
            ['required', 'Must reach'], ['preferred', 'Best effort']
        ], item.endpoint_requirement)}</select></label>`;
        html += `<label>At the end<select data-index="${index}" data-field="attachment_action">${_flashRecordedOptions([
            ['', 'No hold change'], ['attach', 'Gripper takes an object'], ['detach', 'Gripper releases an object']
        ], item.attachment_action)}</select></label>`;
        if (item.attachment_action) html += `<label>Object<select data-index="${index}" data-field="attachment_role">${_flashRecordedOptions(roles.map(role => [role, role]), item.attachment_role)}</select></label>`;
        if (item.relation !== 'ee_to_start') html += `<label class="flash-recorded-check"><input type="checkbox" data-index="${index}" data-field="allow_predicted_demo_pose" ${item.allow_predicted_demo_pose ? 'checked' : ''}> Accept predicted pose where camera observation is missing</label>`;
        html += '</section>';
    }
    for (const role of source.symmetry_roles || []) {
        if (!intervals.some(item => item.reference_role === role || item.held_role === role)) continue;
        const observed = source.observed_frames?.[role] || [];
        const current = state.symmetryBaselines?.[role];
        html += `<label>Observed frame for ${_esc(role)} rotation rule<select data-symmetry="${_esc(role)}">${_flashRecordedOptions([['', 'Select observed frame'], ...observed.map(frame => [String(frame), `Frame ${frame}`])], current == null ? '' : String(current))}</select></label>`;
    }
    const complete = intervals.length && intervals[0].first_frame === 0 && intervals[intervals.length - 1].last_frame === last;
    html += `<p>${complete ? 'Timeline covers the whole recording.' : `Continue until frame ${last} to prepare a candidate.`}</p>`;
    html += `<button type="button" id="flash-recorded-prepare" onclick="flashRecordedPrepare()" ${!source.ready || !complete || state.preparing ? 'disabled' : ''}>Prepare candidate</button>`;
    html += `<p class="flash-editor-status" id="flash-recorded-status">${_esc(state.message)}</p>`;
    panel.innerHTML = html;
    for (const input of panel.querySelectorAll('[data-field]')) {
        input.addEventListener('change', () => {
            const item = intervals[Number(input.dataset.index)];
            item[input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value;
            if (input.dataset.field === 'reference_role' && item.held_role === item.reference_role)
                item.held_role = roles.find(role => role !== item.reference_role) || '';
            if (input.dataset.field === 'attachment_action' && item.attachment_action && !item.attachment_role)
                item.attachment_role = roles[0] || '';
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

function flashRecordedAddSection(toEnd) {
    const state = flashRecordedEditor;
    if (!state) return;
    const first = state.intervals.length ? state.intervals[state.intervals.length - 1].last_frame : 0;
    const last = toEnd ? state.source.frame_count - 1 : _flashRecordedPlayhead();
    if (last <= first) return;
    const roles = state.source.roles || [];
    state.intervals.push({
        first_frame: first, last_frame: last, mode: 'demonstrated', relation: 'ee_to_start',
        reference_role: roles[0] || '', held_role: roles[1] || '', endpoint_requirement: 'required',
        attachment_action: '', attachment_role: '', allow_predicted_demo_pose: false,
    });
    state.message = '';
    _flashRecordedRender();
}

function flashRecordedRemoveFrom(index) {
    flashRecordedEditor?.intervals.splice(index);
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
