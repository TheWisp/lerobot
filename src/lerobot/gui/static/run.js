/* LeRobot GUI - Run Tab Logic */

let runTabInitialized = false;
let runEventSource = null;
let selectedWorkflow = 'teleop'; // 'teleop' | 'replay' | 'policy'
let rerunPorts = null; // {available, web_port, grpc_port}
let _runFormRendered = false; // true once all three workflow sections are in the DOM

// ============================================================================
// Initialization
// ============================================================================

async function runTabInit() {
    if (runTabInitialized) {
        // Re-check status and reconnect SSE if needed
        await pollRunStatus();
        return;
    }
    runTabInitialized = true;

    // Ensure profiles are loaded (Robot tab may not have been visited yet)
    if (typeof robotProfiles !== 'undefined' && !robotProfiles.length) {
        try {
            const res = await fetch('/api/robot/profiles');
            robotProfiles = await res.json();
        } catch (e) { /* ignore */ }
    }
    if (typeof teleopProfiles !== 'undefined' && !teleopProfiles.length) {
        try {
            const res = await fetch('/api/robot/teleop-profiles');
            teleopProfiles = await res.json();
        } catch (e) { /* ignore */ }
    }

    // Fetch Rerun ports
    try {
        const res = await fetch('/api/run/rerun-ports');
        rerunPorts = await res.json();
    } catch (e) {
        console.warn('Failed to fetch Rerun ports:', e);
    }

    initSplitHandle();
    renderRunForm();
    await pollRunStatus();
}

// ============================================================================
// Workflow selection
// ============================================================================

function selectWorkflow(workflow) {
    selectedWorkflow = workflow;
    document.querySelectorAll('.workflow-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.workflow === workflow);
    });
    // Toggle visibility of workflow sections (no re-render)
    for (const wf of ['teleop', 'replay', 'policy']) {
        const section = document.getElementById(`run-section-${wf}`);
        if (section) section.style.display = (wf === workflow) ? '' : 'none';
    }
    if (workflow === 'policy') {
        _ensureModelDataLoaded();
    }
}

// ============================================================================
// Form rendering
// ============================================================================

function _esc(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function _robotProfileOptions() {
    // robotProfiles is global from robot.js
    if (typeof robotProfiles === 'undefined' || !robotProfiles.length) {
        return '<option value="">No robot profiles</option>';
    }
    return robotProfiles.map(p =>
        `<option value="${_esc(p.name)}">${_esc(p.name)} (${_esc(p.type || '?')})</option>`
    ).join('');
}

function _teleopProfileOptions() {
    if (typeof teleopProfiles === 'undefined' || !teleopProfiles.length) {
        return '<option value="">No teleop profiles</option>';
    }
    return teleopProfiles.map(p =>
        `<option value="${_esc(p.name)}">${_esc(p.name)} (${_esc(p.type || '?')})</option>`
    ).join('');
}

// ---- Dataset / episode option helpers ----

function _episodeOptions() {
    // Build <optgroup> per opened dataset, with episodes as options
    const ds = window.datasets || {};
    const ep = window.episodes || {};
    const ids = Object.keys(ds);
    if (!ids.length) {
        return '<option value="" disabled selected>Open a dataset in the Data tab first</option>';
    }
    let html = '<option value="" disabled selected>Select episode</option>';
    for (const id of ids) {
        const d = ds[id];
        const epList = ep[id] || [];
        const label = _esc(d.repo_id || id);
        html += `<optgroup label="${label} (${epList.length} ep)">`;
        for (const e of epList) {
            const val = `${id}:${e.episode_index}`;
            html += `<option value="${_esc(val)}">${label} / Episode ${e.episode_index} (${e.length} frames)</option>`;
        }
        html += '</optgroup>';
    }
    return html;
}

function _recordDatasetOptions() {
    // None (pure teleop) + opened datasets (resume) + New dataset
    const ds = window.datasets || {};
    let html = '<option value="" selected>None (pure teleop)</option>';
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        const label = d.repo_id || id;
        html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
    }
    html += '<option value="__new__">+ New dataset...</option>';
    return html;
}

function _getDatasetTasks(dsId) {
    // Collect unique tasks from episodes
    const epList = (window.episodes || {})[dsId] || [];
    const tasks = [];
    const seen = new Set();
    for (const e of epList) {
        if (e.task && !seen.has(e.task)) {
            seen.add(e.task);
            tasks.push(e.task);
        }
    }
    return tasks;
}

function _updateTaskSelect(dsId) {
    const sel = document.getElementById('run-teleop-task-select');
    const customInput = document.getElementById('run-teleop-task-custom');
    if (!sel) return;

    const tasks = dsId ? _getDatasetTasks(dsId) : [];
    let html = '';
    if (tasks.length) {
        for (const t of tasks) {
            html += `<option value="${_esc(t)}">${_esc(t)}</option>`;
        }
        html += '<option value="__new_task__">+ New task...</option>';
    } else {
        // No existing tasks — show placeholder, user types freely
        html = '<option value="__new_task__" selected>+ New task...</option>';
    }
    sel.innerHTML = html;

    // Show/hide custom input based on selection
    if (sel.value === '__new_task__') {
        if (customInput) { customInput.style.display = ''; customInput.value = ''; }
    } else {
        if (customInput) customInput.style.display = 'none';
    }
}

function _onTaskSelectChange() {
    const sel = document.getElementById('run-teleop-task-select');
    const customInput = document.getElementById('run-teleop-task-custom');
    if (!sel || !customInput) return;
    if (sel.value === '__new_task__') {
        customInput.style.display = '';
        customInput.focus();
    } else {
        customInput.style.display = 'none';
    }
}

function _onRecordDatasetChange() {
    const sel = document.getElementById('run-teleop-record-dataset');
    if (!sel) return;
    const val = sel.value;

    const newNameRow = document.getElementById('run-teleop-new-dataset-row');
    const recordFields = document.getElementById('run-teleop-record-fields');

    if (val === '') {
        // None — pure teleop
        if (newNameRow) newNameRow.style.display = 'none';
        if (recordFields) recordFields.style.display = 'none';
    } else if (val === '__new__') {
        // New dataset
        if (newNameRow) newNameRow.style.display = '';
        if (recordFields) recordFields.style.display = '';
        _updateTaskSelect(null);
        _checkNewDatasetConflict();
    } else {
        // Existing dataset — resume
        if (newNameRow) newNameRow.style.display = 'none';
        if (recordFields) recordFields.style.display = '';
        const dsId = val.replace('existing:', '');
        // Pre-fill FPS from dataset metadata
        const d = (window.datasets || {})[dsId];
        if (d) {
            const fpsInput = document.getElementById('run-teleop-fps');
            if (fpsInput) fpsInput.value = d.fps;
        }
        // Populate task selector from existing episodes
        _updateTaskSelect(dsId);
    }
}

function _checkNewDatasetConflict() {
    const nameInput = document.getElementById('run-teleop-new-dataset-name');
    const warning = document.getElementById('run-teleop-dataset-conflict');
    if (!nameInput || !warning) return;

    const name = nameInput.value.trim();
    if (!name) { warning.style.display = 'none'; return; }

    // Check across all sources
    const sd = window.sourceDatasets || {};
    for (const datasets of Object.values(sd)) {
        for (const ds of datasets) {
            if (ds.name === name) {
                warning.textContent = `"${name}" already exists in a source folder`;
                warning.style.display = '';
                return;
            }
        }
    }
    warning.style.display = 'none';
}

function _onReplayEpisodeChange() {
    const sel = document.getElementById('run-replay-episode');
    if (!sel || !sel.value) return;
    // Parse "datasetId:episodeIndex" and auto-fill FPS
    const [dsId] = sel.value.split(':');
    const d = (window.datasets || {})[dsId];
    if (d) {
        const fpsInput = document.getElementById('run-replay-fps');
        if (fpsInput) fpsInput.value = d.fps;
    }
}

// ---- Model / checkpoint option helpers ----

function _modelCheckpointOptions() {
    // Build options from the Model tab's cached scan data
    const data = (typeof modelSourceData !== 'undefined') ? modelSourceData : {};
    const entries = Object.values(data).flat();
    if (!entries.length) {
        return '<option value="" disabled selected>Loading models...</option>';
    }
    let html = '<option value="" disabled selected>Select checkpoint</option>';
    for (const m of entries) {
        // Use the last checkpoint's pretrained_model path
        const ckptPath = m.path + '/checkpoints/last/pretrained_model';
        const stepText = m.current_step != null ? ` (step ${m.current_step.toLocaleString()})` : '';
        html += `<option value="${_esc(ckptPath)}" title="${_esc(m.path)}">${_esc(m.name)} — ${_esc(m.policy_type)}${stepText}</option>`;
    }
    return html;
}

function _onPolicyCheckpointChange() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.value) return;
    // Extract the run path from the checkpoint path (strip /checkpoints/last/pretrained_model)
    const runPath = sel.value.replace(/\/checkpoints\/last\/pretrained_model$/, '');
    if (typeof _prefillPolicyFields === 'function') {
        _prefillPolicyFields(runPath);
    }
}

async function _ensureModelDataLoaded() {
    // If Model tab hasn't been visited, load model sources and scan them
    if (typeof modelSourceData === 'undefined' || !Object.keys(modelSourceData).length) {
        if (typeof modelTabInit === 'function') {
            await modelTabInit();
        }
        // Re-render the checkpoint selector after data is loaded
        const sel = document.getElementById('run-policy-checkpoint');
        if (sel) sel.innerHTML = _modelCheckpointOptions();
    }
}

// ---- Refresh helpers (called from other tabs when data changes) ----

function refreshRunProfileSelects() {
    if (!_runFormRendered) return;
    const robotOpts = _robotProfileOptions();
    const teleopOpts = _teleopProfileOptions();
    for (const id of ['run-teleop-robot', 'run-replay-robot', 'run-policy-robot']) {
        const sel = document.getElementById(id);
        if (sel) { const prev = sel.value; sel.innerHTML = robotOpts; sel.value = prev; }
    }
    const teleopSel = document.getElementById('run-teleop-teleop');
    if (teleopSel) { const prev = teleopSel.value; teleopSel.innerHTML = teleopOpts; teleopSel.value = prev; }
    const policyTeleopSel = document.getElementById('run-policy-teleop');
    if (policyTeleopSel) {
        const prev = policyTeleopSel.value;
        policyTeleopSel.innerHTML = '<option value="">None (policy only)</option>' + teleopOpts;
        policyTeleopSel.value = prev;
    }
}

function refreshRunDatasetSelects() {
    if (!_runFormRendered) return;
    // Refresh replay episode selector
    const replaySel = document.getElementById('run-replay-episode');
    if (replaySel) { const prev = replaySel.value; replaySel.innerHTML = _episodeOptions(); replaySel.value = prev; }
    // Refresh teleop record dataset selector
    const recordSel = document.getElementById('run-teleop-record-dataset');
    if (recordSel) { const prev = recordSel.value; recordSel.innerHTML = _recordDatasetOptions(); recordSel.value = prev; }
}

// ---- Form rendering ----

function renderRunForm() {
    if (_runFormRendered) return; // Only build once
    _runFormRendered = true;

    const form = document.getElementById('run-form');
    if (!form) return;

    let html = '';

    // ---- Teleop workflow section ----
    html += `<div id="run-section-teleop" style="${selectedWorkflow === 'teleop' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot</label>`;
    html += `<select id="run-teleop-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Teleop</label>`;
    html += `<select id="run-teleop-teleop">${_teleopProfileOptions()}</select>`;
    html += `<label>FPS</label>`;
    html += `<input type="number" id="run-teleop-fps" value="60" min="1" max="200">`;
    html += '</div>';
    // Dataset selector (optional recording)
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Record dataset</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-teleop-record-dataset" onchange="_onRecordDatasetChange()">${_recordDatasetOptions()}</select>`;
    html += `</div>`;
    // New dataset name (hidden by default)
    html += `<div id="run-teleop-new-dataset-row" class="form-grid" style="display:none;">`;
    html += `<label>Name</label>`;
    html += `<div><input type="text" id="run-teleop-new-dataset-name" placeholder="my_new_dataset" oninput="_checkNewDatasetConflict()">`;
    html += `<div class="dataset-conflict-warning" id="run-teleop-dataset-conflict" style="display:none;"></div></div>`;
    html += `</div>`;
    // Record fields (hidden when None selected)
    html += `<div id="run-teleop-record-fields" class="form-grid" style="display:none;">`;
    html += `<label>Task</label>`;
    html += `<div>`;
    html += `<select id="run-teleop-task-select" onchange="_onTaskSelectChange()"><option value="" selected>Pick up the cube</option></select>`;
    html += `<input type="text" id="run-teleop-task-custom" placeholder="Describe the task" style="display:none; margin-top:4px;">`;
    html += `</div>`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-teleop-num-episodes" value="50" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-teleop-episode-time" value="60" min="1">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-teleop-reset-time" value="60" min="0">`;
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end teleop section

    // ---- Replay workflow section ----
    html += `<div id="run-section-replay" style="${selectedWorkflow === 'replay' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot</label>`;
    html += `<select id="run-replay-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Episode</label>`;
    html += `<select id="run-replay-episode" onchange="_onReplayEpisodeChange()">${_episodeOptions()}</select>`;
    html += `<label>FPS</label>`;
    html += `<input type="number" id="run-replay-fps" value="30" min="1" max="200">`;
    html += '</div>';
    html += '</div>'; // end replay section

    // ---- Policy workflow section ----
    html += `<div id="run-section-policy" style="${selectedWorkflow === 'policy' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot</label>`;
    html += `<select id="run-policy-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Checkpoint</label>`;
    html += `<select id="run-policy-checkpoint" onchange="_onPolicyCheckpointChange()">${_modelCheckpointOptions()}</select>`;
    // Teleop profile (optional — for manual resets between episodes)
    html += `<label>Teleop</label>`;
    html += `<div><select id="run-policy-teleop">`;
    html += `<option value="">None (policy only)</option>`;
    html += _teleopProfileOptions();
    html += `</select>`;
    html += `<div class="form-hint">Optional: for manual resets between episodes</div></div>`;
    html += `<label>FPS</label>`;
    html += `<input type="number" id="run-policy-fps" value="30" min="1" max="200">`;
    html += '</div>';
    // Recording settings
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Record evaluation</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<input type="text" id="run-policy-repo-id" placeholder="user/eval_my_policy" value="eval/eval_policy">`;
    html += `<label>Task</label>`;
    html += `<input type="text" id="run-policy-task" placeholder="Describe the task" value="">`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-policy-episodes" value="10" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-policy-episode-time" value="60" min="1">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-policy-reset-time" value="60" min="0">`;
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end policy section

    form.innerHTML = html;
}

// ============================================================================
// Profile data retrieval
// ============================================================================

async function _getProfileData(kind, selectedName) {
    // If the Robot tab has this profile currently loaded, collect fresh form values
    // (the user may have unsaved edits that should be respected)
    if (typeof currentProfile !== 'undefined' && currentProfile &&
        currentProfile.kind === kind && currentProfile.name === selectedName) {
        if (typeof _collectFormFields === 'function') {
            return { ...currentProfile.data, fields: _collectFormFields() };
        }
        return currentProfile.data;
    }
    // Fetch full profile from API (list only has name + type)
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(selectedName)}`);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        return null;
    }
}

// ============================================================================
// Launch / Stop
// ============================================================================

function _getActiveRobotSelectId() {
    return `run-${selectedWorkflow}-robot`;
}

function _getActiveFpsInputId() {
    return `run-${selectedWorkflow}-fps`;
}

async function launchRun() {
    const robotSelect = document.getElementById(_getActiveRobotSelectId());
    if (!robotSelect || !robotSelect.value) {
        showToast('Error', 'Select a robot profile', 'error');
        return;
    }

    const robotData = await _getProfileData('robot', robotSelect.value);
    if (!robotData) {
        showToast('Error', `Robot profile "${robotSelect.value}" not found`, 'error');
        return;
    }

    // Cap FPS to the slowest camera's capability
    const fpsInput = document.getElementById(_getActiveFpsInputId());
    let userFps = parseInt(fpsInput?.value) || 30;
    const cams = robotData.cameras || {};
    const camFpsList = Object.values(cams).map(c => c.fps).filter(Boolean);
    if (camFpsList.length > 0) {
        const minCamFps = Math.min(...camFpsList);
        if (userFps > minCamFps) {
            fpsInput.value = minCamFps;
            userFps = minCamFps;
            showToast('FPS adjusted', `FPS capped to ${minCamFps} (camera limit)`, 'warning');
        }
    }

    let endpoint, body;

    if (selectedWorkflow === 'teleop') {
        const teleopSelect = document.getElementById('run-teleop-teleop');
        const teleopData = await _getProfileData('teleop', teleopSelect?.value);
        if (!teleopData) {
            showToast('Error', 'Select a teleop profile', 'error');
            return;
        }

        const datasetSel = document.getElementById('run-teleop-record-dataset');
        const datasetVal = datasetSel?.value || '';

        if (datasetVal === '') {
            // Pure teleoperate — no recording
            endpoint = '/api/run/teleoperate';
            body = {
                robot: robotData,
                teleop: teleopData,
                fps: parseInt(document.getElementById('run-teleop-fps')?.value) || 60,
            };
        } else {
            // Record mode — existing or new dataset
            const taskSel = document.getElementById('run-teleop-task-select');
            const taskCustom = document.getElementById('run-teleop-task-custom');
            const singleTask = (taskSel?.value === '__new_task__')
                ? taskCustom?.value?.trim()
                : taskSel?.value?.trim();
            if (!singleTask) {
                showToast('Error', 'Task description is required for recording', 'error');
                return;
            }

            let repoId, root = null, resume = false;
            if (datasetVal === '__new__') {
                const name = document.getElementById('run-teleop-new-dataset-name')?.value?.trim();
                if (!name) {
                    showToast('Error', 'Enter a name for the new dataset', 'error');
                    return;
                }
                repoId = name;
            } else {
                // existing:{datasetId}
                const dsId = datasetVal.replace('existing:', '');
                const d = (window.datasets || {})[dsId];
                if (!d) {
                    showToast('Error', 'Selected dataset not found — was it closed?', 'error');
                    return;
                }
                repoId = d.repo_id;
                root = d.root;
                resume = true;

                // Warn on FPS mismatch — dataset FPS is immutable across episodes
                const currentFps = parseInt(document.getElementById('run-teleop-fps')?.value) || 30;
                if (d.fps && currentFps !== d.fps) {
                    const ok = confirm(
                        `FPS mismatch: dataset "${d.repo_id}" uses ${d.fps} FPS ` +
                        `but you selected ${currentFps} FPS.\n\n` +
                        `A dataset cannot have different FPS across episodes. ` +
                        `The recording will use ${d.fps} FPS.\n\nContinue?`
                    );
                    if (!ok) return;
                }

                // Warn on robot type mismatch
                if (d.robot_type && robotData.type && d.robot_type !== robotData.type) {
                    const ok = confirm(
                        `Robot mismatch: dataset was recorded with "${d.robot_type}" ` +
                        `but selected robot is "${robotData.type}".\n\n` +
                        `Recording with a different robot may produce incompatible data.\n\nContinue anyway?`
                    );
                    if (!ok) return;
                }
            }

            endpoint = '/api/run/record';
            body = {
                robot: robotData,
                teleop: teleopData,
                repo_id: repoId,
                root: root,
                single_task: singleTask,
                fps: parseInt(document.getElementById('run-teleop-fps')?.value) || 30,
                episode_time_s: parseFloat(document.getElementById('run-teleop-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-teleop-reset-time')?.value) || 60,
                num_episodes: parseInt(document.getElementById('run-teleop-num-episodes')?.value) || 50,
                resume: resume,
            };
        }
    } else if (selectedWorkflow === 'replay') {
        const episodeSel = document.getElementById('run-replay-episode');
        if (!episodeSel?.value) {
            showToast('Error', 'Select an episode to replay', 'error');
            return;
        }
        const [dsId, epIdx] = episodeSel.value.split(':');
        const d = (window.datasets || {})[dsId];
        if (!d) {
            showToast('Error', 'Selected dataset not found — was it closed?', 'error');
            return;
        }
        // Warn if robot type doesn't match dataset
        if (d.robot_type && robotData.type && d.robot_type !== robotData.type) {
            const ok = confirm(
                `Robot mismatch: dataset was recorded with "${d.robot_type}" ` +
                `but selected robot is "${robotData.type}".\n\n` +
                `Replaying on the wrong robot can send incorrect motor commands.\n\nContinue anyway?`
            );
            if (!ok) return;
        }
        endpoint = '/api/run/replay';
        body = {
            robot: robotData,
            repo_id: d.repo_id,
            root: d.root,
            episode: parseInt(epIdx),
            fps: parseInt(document.getElementById('run-replay-fps')?.value) || 30,
        };
    } else if (selectedWorkflow === 'policy') {
        const checkpointSel = document.getElementById('run-policy-checkpoint');
        if (!checkpointSel?.value) {
            showToast('Error', 'Select a model checkpoint', 'error');
            return;
        }

        const repoId = document.getElementById('run-policy-repo-id')?.value?.trim();
        if (!repoId) {
            showToast('Error', 'Enter a dataset name for evaluation recordings', 'error');
            return;
        }

        const task = document.getElementById('run-policy-task')?.value?.trim();
        if (!task) {
            showToast('Error', 'Task description is required', 'error');
            return;
        }

        // Optional teleop for manual resets
        const teleopSelect = document.getElementById('run-policy-teleop');
        let teleopData = null;
        if (teleopSelect?.value) {
            teleopData = await _getProfileData('teleop', teleopSelect.value);
        }

        endpoint = '/api/run/record';
        body = {
            robot: robotData,
            teleop: teleopData,
            policy_path: checkpointSel.value,
            repo_id: repoId,
            single_task: task,
            fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
            episode_time_s: parseFloat(document.getElementById('run-policy-episode-time')?.value) || 60,
            reset_time_s: parseFloat(document.getElementById('run-policy-reset-time')?.value) || 60,
            num_episodes: parseInt(document.getElementById('run-policy-episodes')?.value) || 10,
        };
    } else {
        showToast('Error', 'Unknown workflow', 'error');
        return;
    }

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            showToast('Launch failed', err.detail || 'Unknown error', 'error');
            return;
        }
        const data = await res.json();
        showToast('Started', `${data.command} started (PID ${data.pid})`, 'success');
        updateRunUI(true);
        connectOutputSSE();
        // Delay so the subprocess has time to start its gRPC server
        setTimeout(showRerunViewer, 5000);
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function stopRun() {
    try {
        const res = await fetch('/api/run/stop', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            showToast('Error', err.detail || 'Failed to stop', 'error');
            return;
        }
        showToast('Stopped', 'Process stopped', 'info');
        updateRunUI(false);
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

// ============================================================================
// SSE output streaming
// ============================================================================

function connectOutputSSE() {
    disconnectOutputSSE();

    const terminal = document.getElementById('run-terminal');
    if (terminal) terminal.innerHTML = '';

    // Reset table deduplication state
    _lastTableBlock = null;
    _tableBlockNode = null;

    runEventSource = new EventSource('/api/run/output');

    runEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.done) {
            disconnectOutputSSE();
            pollRunStatus();
            // Re-scan dataset sources to pick up newly created datasets
            if (typeof window.refreshExpandedSources === 'function') {
                window.refreshExpandedSources();
            }
            return;
        }
        if (data.line !== undefined) {
            appendTerminalLine(data.line);
        }
    };

    runEventSource.onerror = () => {
        disconnectOutputSSE();
    };
}

function disconnectOutputSSE() {
    if (runEventSource) {
        runEventSource.close();
        runEventSource = null;
    }
}

// Strip ANSI escape codes (cursor movement, colors, etc.)
function _stripAnsi(text) {
    return text.replace(/\x1b\[[0-9;]*[A-Za-z]|\[\d+[A-Za-z]/g, '');
}

// Classify a terminal line for styling
function _classifyLine(text, isStderr) {
    // System messages from our wrapper
    if (text.startsWith('---')) return 'system';
    if (text.startsWith('$')) return 'system';
    if (!isStderr) return '';
    // Classify stderr by log level
    if (/\bERROR\b/i.test(text)) return 'stderr-error';
    if (/\bWARN(?:ING)?\b/i.test(text)) return 'stderr-warn';
    // Most stderr is just INFO logging — use subdued style
    return 'stderr-info';
}

// Detect repeating table blocks (teleop uses cursor-up ANSI to overwrite)
let _lastTableBlock = null;
let _tableBlockNode = null;

function appendTerminalLine(rawText) {
    const terminal = document.getElementById('run-terminal');
    if (!terminal) return;

    const isStderr = rawText.startsWith('[stderr] ');
    const text = _stripAnsi(isStderr ? rawText.slice(9) : rawText);

    // Skip empty lines from ANSI stripping
    if (!text.trim()) return;

    // Detect repeating table output (teleop prints the same table repeatedly)
    // The table starts with a long "---" separator and ends with a "loop time" line.
    // We collapse repeated table blocks into a single DOM node updated in-place.
    if (!isStderr && text.match(/^-{10,}$/)) {
        _lastTableBlock = [text];
        return;
    }
    if (_lastTableBlock !== null) {
        _lastTableBlock.push(text);
        if (/loop.time/i.test(text)) {
            const blockText = _lastTableBlock.join('\n');
            if (_tableBlockNode && _tableBlockNode.parentNode === terminal) {
                _tableBlockNode.textContent = blockText;
            } else {
                _tableBlockNode = document.createElement('div');
                _tableBlockNode.className = 'terminal-line';
                _tableBlockNode.textContent = blockText;
                terminal.appendChild(_tableBlockNode);
            }
            _lastTableBlock = null;
            terminal.scrollTop = terminal.scrollHeight;
            return;
        }
        // Safety: flush if accumulator grows too large (missed terminator)
        if (_lastTableBlock.length > 50) {
            console.warn('Table block accumulator overflow, flushing', _lastTableBlock.length, 'lines');
            _lastTableBlock = null;
        } else {
            return;
        }
    }

    const line = document.createElement('div');
    line.className = 'terminal-line';
    const cls = _classifyLine(text, isStderr);
    if (cls) line.classList.add(cls);
    line.textContent = text;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;

    // Cap DOM nodes
    while (terminal.children.length > 2000) {
        terminal.removeChild(terminal.firstChild);
    }
}

// ============================================================================
// Status polling
// ============================================================================

async function pollRunStatus() {
    try {
        const res = await fetch('/api/run/status');
        const status = await res.json();
        updateRunUI(status.running);

        // If running but no SSE, reconnect
        if (status.running && !runEventSource) {
            connectOutputSSE();
            showRerunViewer();
        }
    } catch (e) {
        console.error('Status poll failed:', e);
    }
}

// ============================================================================
// UI state management
// ============================================================================

function updateRunUI(isRunning) {
    const launchBtn = document.getElementById('run-launch-btn');
    const stopBtn = document.getElementById('run-stop-btn');
    const formInputs = document.querySelectorAll('#run-form input, #run-form select');
    const workflowBtns = document.querySelectorAll('.workflow-btn');

    if (launchBtn) launchBtn.disabled = isRunning;
    if (stopBtn) stopBtn.disabled = !isRunning;

    formInputs.forEach(el => el.disabled = isRunning);
    workflowBtns.forEach(el => el.disabled = isRunning);

    const indicator = document.getElementById('run-status-indicator');
    if (indicator) {
        indicator.textContent = isRunning ? 'Running' : 'Idle';
        indicator.className = isRunning ? 'run-status running' : 'run-status idle';
    }
}

// ============================================================================
// Resizable split between Rerun viewer and terminal
// ============================================================================

function initSplitHandle() {
    const handle = document.getElementById('run-split-handle');
    const viewer = document.getElementById('rerun-viewer');
    const termContainer = document.getElementById('run-terminal-container');
    if (!handle || !viewer || !termContainer) return;

    let startY, startViewerH, startTermH;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        startY = e.clientY;
        startViewerH = viewer.offsetHeight;
        startTermH = termContainer.offsetHeight;
        handle.classList.add('dragging');

        function onMove(e) {
            const dy = e.clientY - startY;
            const newViewerH = Math.max(100, startViewerH + dy);
            const newTermH = Math.max(60, startTermH - dy);
            viewer.style.flex = 'none';
            viewer.style.height = newViewerH + 'px';
            termContainer.style.flex = 'none';
            termContainer.style.height = newTermH + 'px';
        }

        function onUp() {
            handle.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}

// ============================================================================
// Rerun viewer
// ============================================================================

function showRerunViewer() {
    const container = document.getElementById('rerun-viewer');
    if (!container || !rerunPorts?.available) return;

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    const grpcUrl = encodeURIComponent(`rerun+http://localhost:${rerunPorts.grpc_port}/proxy`);
    const src = `http://localhost:${rerunPorts.web_port}?url=${grpcUrl}&hide_welcome_screen`;

    let iframe = container.querySelector('iframe');
    if (!iframe) {
        iframe = document.createElement('iframe');
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.border = 'none';
        container.appendChild(iframe);
    }
    // Always (re)set src so the viewer reconnects to the subprocess gRPC server
    iframe.src = src;
}
