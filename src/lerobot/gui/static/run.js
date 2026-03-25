/* LeRobot GUI - Run Tab Logic */

let runTabInitialized = false;
let runEventSource = null;
let selectedWorkflow = 'teleop'; // 'teleop' | 'replay' | 'policy'
let obsStreamMeta = null; // {available, obs_scalar_keys, action_keys, image_keys}
let obsStreamTimer = null; // interval ID for camera polling
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
    if (workflow === 'policy' || workflow === 'teleop') {
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

// ---- Dataset name resolution ----

/**
 * Resolve a dataset's correct repo_id by looking up its root path in sourceDatasets.
 * The source scanner's `name` field is the relative path from the source root (e.g., "thewisp/aaa"),
 * which is what the CLI expects. Falls back to the dataset's own repo_id if not found in sources.
 */
function _resolveDatasetRepoId(dataset) {
    const sd = window.sourceDatasets || {};
    for (const datasets of Object.values(sd)) {
        for (const ds of datasets) {
            if (ds.root === dataset.root) return ds.name;
        }
    }
    return dataset.repo_id;
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

function _findDatasetByRepoId(repoId) {
    // Find an opened dataset matching a repo_id (e.g. from train_config)
    if (!repoId) return null;
    const ds = window.datasets || {};
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        if (d.repo_id === repoId || d.repo_id?.endsWith('/' + repoId)) {
            return id;
        }
    }
    return null;
}

// Track the currently selected model's training dataset info
let _policyTrainingDataset = null;  // { repo_id, root }

function _updatePolicyTaskSelector() {
    const container = document.getElementById('run-policy-task-container');
    if (!container) return;

    const info = _policyTrainingDataset;
    if (!info || !info.repo_id) {
        // No model selected or no training dataset — plain text input
        container.innerHTML = `<input type="text" id="run-policy-task" placeholder="Select a checkpoint first" value="">`;
        return;
    }

    const dsId = _findDatasetByRepoId(info.repo_id);
    if (!dsId) {
        // Training dataset not opened — show prompt to open it
        const displayName = info.repo_id.length > 40 ? '...' + info.repo_id.slice(-37) : info.repo_id;
        container.innerHTML =
            `<div class="policy-task-open-prompt">` +
            `<span class="form-hint">Training dataset not opened</span>` +
            `<button class="btn-tiny btn-accent" onclick="_openTrainingDataset()">Open ${_esc(displayName)}</button>` +
            `</div>` +
            `<input type="text" id="run-policy-task" placeholder="Or type task manually" value="">`;
        return;
    }

    // Training dataset is opened — show task dropdown + custom option
    const tasks = _getDatasetTasks(dsId);
    let html = '<select id="run-policy-task-select" onchange="_onPolicyTaskSelectChange()">';
    if (tasks.length) {
        for (const t of tasks) {
            html += `<option value="${_esc(t)}">${_esc(t)}</option>`;
        }
        html += '<option value="__new_task__">+ New task...</option>';
    } else {
        html += '<option value="__new_task__" selected>+ New task...</option>';
    }
    html += '</select>';
    html += `<input type="text" id="run-policy-task-custom" placeholder="Describe the task" style="${tasks.length ? 'display:none' : ''}" value="">`;
    container.innerHTML = html;
}

function _onPolicyTaskSelectChange() {
    const sel = document.getElementById('run-policy-task-select');
    const customInput = document.getElementById('run-policy-task-custom');
    if (!sel || !customInput) return;
    if (sel.value === '__new_task__') {
        customInput.style.display = '';
        customInput.focus();
    } else {
        customInput.style.display = 'none';
    }
}

async function _openTrainingDataset() {
    const info = _policyTrainingDataset;
    if (!info) return;
    // Open by root path if available, otherwise by repo_id
    const openPath = info.root || info.repo_id;
    if (typeof openDataset === 'function') {
        await openDataset(openPath);
        // After opening, refresh the task selector
        _updatePolicyTaskSelector();
    }
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

// ---- Policy dataset helpers ----

function _policyDatasetOptions() {
    const ds = window.datasets || {};
    let html = '<option value="__new__" selected>+ New dataset...</option>';
    for (const id of Object.keys(ds)) {
        const d = ds[id];
        const label = d.repo_id || id;
        html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
    }
    return html;
}

function _onPolicyDatasetChange() {
    const sel = document.getElementById('run-policy-dataset');
    if (!sel) return;
    const val = sel.value;
    const newNameRow = document.getElementById('run-policy-new-dataset-row');

    if (val === '__new__') {
        if (newNameRow) newNameRow.style.display = '';
    } else {
        if (newNameRow) newNameRow.style.display = 'none';
        // Pre-fill FPS from existing dataset
        const dsId = val.replace('existing:', '');
        const d = (window.datasets || {})[dsId];
        if (d) {
            const fpsInput = document.getElementById('run-policy-fps');
            if (fpsInput) fpsInput.value = d.fps;
        }
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
        // Standard checkpoints: path/checkpoints/last/pretrained_model
        // Flat checkpoints (converted S2 etc.): path directly contains model.safetensors
        const isFlat = m.num_checkpoints === 1 && m.current_step == null;
        const ckptPath = isFlat ? m.path : m.path + '/checkpoints/last/pretrained_model';
        const stepText = m.current_step != null ? ` (step ${m.current_step.toLocaleString()})` : '';
        html += `<option value="${_esc(ckptPath)}" data-policy-type="${_esc(m.policy_type)}" title="${_esc(m.path)}">${_esc(m.name)} — ${_esc(m.policy_type)}${stepText}</option>`;
    }
    return html;
}

function _getDebugModelConfig() {
    // Returns debug model config or null if none selected.
    const sel = document.getElementById('run-teleop-debug-model');
    if (!sel?.value) return null;
    const opt = sel.selectedOptions[0];
    const policyType = opt?.dataset?.policyType || '';
    const config = {
        checkpoint: sel.value,
        policy_type: policyType,
    };
    if (policyType === 'hvla_s2_vlm') {
        config.task = document.getElementById('run-teleop-debug-s2-task')?.value?.trim() || '';
        config.decode_subtask = document.getElementById('run-teleop-debug-s2-decode')?.checked || false;
    }
    return config;
}

let _debugModelLoading = false;  // prevent double-click

async function _loadDebugModel() {
    if (_debugModelLoading) return;
    const config = _getDebugModelConfig();
    if (!config) {
        showToast('Error', 'Select a model first', 'error');
        return;
    }
    const status = document.getElementById('run-debug-model-status');
    _debugModelLoading = true;
    _updateDebugButtons();
    if (status) status.textContent = 'Loading...';
    // Clear model terminal
    const modelTerminal = document.getElementById('run-model-terminal');
    if (modelTerminal) modelTerminal.innerHTML = '<div class="terminal-line" style="color: #666;">Loading model...</div>';
    try {
        const res = await fetch('/api/run/debug/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config),
        });
        const data = await res.json();
        if (res.ok) {
            if (status) status.textContent = `Loaded (PID ${data.pid})`;
            _debugModelLoaded = true;
            showToast('Debug model', 'Model loaded', 'success');
        } else {
            if (status) status.textContent = 'Not loaded';
            showToast('Error', data.detail || 'Failed to load', 'error');
        }
    } catch (e) {
        if (status) status.textContent = 'Error';
        showToast('Error', e.message, 'error');
    } finally {
        _debugModelLoading = false;
        _updateDebugButtons();
    }
}

async function _unloadDebugModel() {
    if (_debugModelLoading) return;
    const status = document.getElementById('run-debug-model-status');
    _debugModelLoading = true;
    _updateDebugButtons();
    try {
        const res = await fetch('/api/run/debug/unload', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            if (status) status.textContent = 'Not loaded';
            _debugModelLoaded = false;
            if (data.status !== 'not_loaded') showToast('Debug model', 'Model unloaded', 'info');
        }
    } catch (e) {
        showToast('Error', e.message, 'error');
    } finally {
        _debugModelLoading = false;
        _updateDebugButtons();
    }
}

let _debugModelLoaded = false;

function _updateDebugButtons() {
    const loadBtn = document.querySelector('[onclick="_loadDebugModel()"]');
    const unloadBtn = document.querySelector('[onclick="_unloadDebugModel()"]');
    if (loadBtn) loadBtn.disabled = _debugModelLoading || _debugModelLoaded;
    if (unloadBtn) unloadBtn.disabled = _debugModelLoading || !_debugModelLoaded;
    _showModelPanel(_debugModelLoaded);
}

async function _refreshDebugModelStatus() {
    const status = document.getElementById('run-debug-model-status');
    if (!status) return;
    try {
        const res = await fetch('/api/run/debug/status');
        const data = await res.json();
        _debugModelLoaded = data.loaded;
        status.textContent = data.loaded ? `Loaded (PID ${data.pid})` : 'Not loaded';
        _updateDebugButtons();
    } catch (e) {
        status.textContent = 'Unknown';
    }
}

function _onDebugModelChange() {
    const sel = document.getElementById('run-teleop-debug-model');
    const s2Fields = document.getElementById('run-teleop-debug-s2-fields');
    if (!sel || !s2Fields) return;
    const opt = sel.selectedOptions[0];
    const policyType = opt?.dataset?.policyType || '';
    s2Fields.style.display = policyType === 'hvla_s2_vlm' ? '' : 'none';
}

function _getSelectedPolicyType() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.selectedOptions?.length) return '';
    return sel.selectedOptions[0].dataset.policyType || '';
}

function _updateHVLAFieldsVisibility() {
    const isHVLA = _getSelectedPolicyType() === 'hvla_flow_s1';
    const hvlaSection = document.getElementById('run-policy-hvla-fields');
    const standardSection = document.getElementById('run-policy-standard-fields');
    if (hvlaSection) hvlaSection.style.display = isHVLA ? '' : 'none';
    if (standardSection) standardSection.style.display = isHVLA ? 'none' : '';
}

function _onPolicyCheckpointChange() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.value) return;
    // Extract the run path from the checkpoint path (strip /checkpoints/last/pretrained_model)
    const runPath = sel.value.replace(/\/checkpoints\/last\/pretrained_model$/, '');
    if (typeof _prefillPolicyFields === 'function') {
        _prefillPolicyFields(runPath);
    }
    _updateHVLAFieldsVisibility();
}

async function _ensureModelDataLoaded() {
    // If Model tab hasn't been visited, load model sources and scan them
    if (typeof modelSourceData === 'undefined' || !Object.keys(modelSourceData).length) {
        if (typeof modelTabInit === 'function') {
            await modelTabInit();
        }
        // Re-render checkpoint selectors after data is loaded
        const sel = document.getElementById('run-policy-checkpoint');
        if (sel) sel.innerHTML = _modelCheckpointOptions();
        const debugSel = document.getElementById('run-teleop-debug-model');
        if (debugSel) {
            const prev = debugSel.value;
            debugSel.innerHTML = '<option value="">None</option>' + _modelCheckpointOptions();
            debugSel.value = prev;
        }
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
    if (recordSel) {
        const prev = recordSel.value;
        recordSel.innerHTML = _recordDatasetOptions();
        // If user was creating a new dataset, the recording is done — check if it
        // now exists as an opened dataset, otherwise reset to "None"
        if (prev === '__new__') {
            const newName = document.getElementById('run-teleop-new-dataset-name')?.value?.trim();
            let matched = false;
            if (newName) {
                const ds = window.datasets || {};
                for (const id of Object.keys(ds)) {
                    if (ds[id].repo_id === newName || ds[id].repo_id?.endsWith('/' + newName)) {
                        recordSel.value = `existing:${id}`;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                recordSel.value = '';  // Reset to "None (pure teleop)"
            }
            _onRecordDatasetChange();
            return;
        }
        recordSel.value = prev;
    }
    // Refresh policy dataset selector
    const policySel = document.getElementById('run-policy-dataset');
    if (policySel) { const prev = policySel.value; policySel.innerHTML = _policyDatasetOptions(); policySel.value = prev; }
    // Refresh task selector (training dataset may have been opened/closed)
    _updatePolicyTaskSelector();
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
    // Debug model (optional — runs alongside teleop for live prediction display)
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Debug model</div>';
    html += '<div class="form-grid">';
    html += `<label>Model</label>`;
    html += `<select id="run-teleop-debug-model" onchange="_onDebugModelChange()">`;
    html += `<option value="">None</option>`;
    html += _modelCheckpointOptions();
    html += `</select>`;
    html += `<label></label>`;
    html += `<div><button class="btn-tiny btn-accent" onclick="_loadDebugModel()">Load</button> `;
    html += `<button class="btn-tiny" onclick="_unloadDebugModel()">Unload</button> `;
    html += `<span id="run-debug-model-status" class="form-hint">Not loaded</span></div>`;
    html += `</div>`;
    // HVLA S2 specific fields (shown when S2 checkpoint selected)
    html += `<div id="run-teleop-debug-s2-fields" style="display:none">`;
    html += '<div class="form-grid">';
    html += `<label>Task Prompt</label>`;
    html += `<input type="text" id="run-teleop-debug-s2-task" placeholder="assemble cylinder into ring" value="assemble cylinder into ring">`;
    html += `<label>Decode Subtask</label>`;
    html += `<div style="text-align:left"><input type="checkbox" id="run-teleop-debug-s2-decode" checked></div>`;
    html += '</div>';
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
    // ---- HVLA-specific fields (shown when HVLA checkpoint selected) ----
    html += `<div id="run-policy-hvla-fields" style="display:none">`;
    html += '<div class="form-section">';
    html += '<div class="form-section-title">HVLA Settings</div>';
    html += '<div class="form-grid">';
    html += `<label>S2 Checkpoint</label>`;
    html += `<input type="text" id="run-hvla-s2-checkpoint" placeholder="/path/to/s2/model.safetensors" value="">`;
    html += `<label>Task Prompt</label>`;
    html += `<input type="text" id="run-hvla-task" placeholder="assemble cylinder into ring" value="">`;
    html += `<label>Decode Subtask</label>`;
    html += `<label class="toggle-label"><input type="checkbox" id="run-hvla-decode-subtask"> Enable subtask decoding</label>`;
    html += `<label>Record Dataset</label>`;
    html += `<input type="text" id="run-hvla-record-dataset" placeholder="eval/hvla_eval (optional)" value="">`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-hvla-episodes" value="1" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-hvla-episode-time" value="60" min="0">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-hvla-reset-time" value="20" min="0">`;
    html += '</div>';
    html += '</div>';
    html += '</div>';
    // ---- Standard policy fields (hidden when HVLA selected) ----
    html += `<div id="run-policy-standard-fields">`;
    // Recording settings
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Record evaluation</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-policy-dataset" onchange="_onPolicyDatasetChange()">${_policyDatasetOptions()}</select>`;
    html += `</div>`;
    // New dataset name (shown by default since __new__ is default)
    html += `<div id="run-policy-new-dataset-row" class="form-grid">`;
    html += `<label>Name</label>`;
    html += `<input type="text" id="run-policy-repo-id" placeholder="user/eval_my_policy" value="eval/eval_policy">`;
    html += `</div>`;
    html += `<div class="form-grid">`;
    html += `<label>Task</label>`;
    html += `<div id="run-policy-task-container"><input type="text" id="run-policy-task" placeholder="Select a checkpoint first" value=""></div>`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-policy-episodes" value="10" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-policy-episode-time" value="60" min="1">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-policy-reset-time" value="60" min="0">`;
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end standard fields
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

        const debugModel = _getDebugModelConfig();

        if (datasetVal === '') {
            // Pure teleoperate — no recording
            endpoint = '/api/run/teleoperate';
            body = {
                robot: robotData,
                teleop: teleopData,
                fps: parseInt(document.getElementById('run-teleop-fps')?.value) || 60,
                debug_model: debugModel,
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
                repoId = _resolveDatasetRepoId(d);
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
            repo_id: _resolveDatasetRepoId(d),
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

        const policyType = _getSelectedPolicyType();

        if (policyType === 'hvla_flow_s1') {
            // ---- HVLA dispatch ----
            const s2Ckpt = document.getElementById('run-hvla-s2-checkpoint')?.value?.trim();
            if (!s2Ckpt) {
                showToast('Error', 'S2 checkpoint path is required for HVLA', 'error');
                return;
            }
            const hvlaTask = document.getElementById('run-hvla-task')?.value?.trim();
            if (!hvlaTask) {
                showToast('Error', 'Task prompt is required for HVLA', 'error');
                return;
            }
            const recordDs = document.getElementById('run-hvla-record-dataset')?.value?.trim() || null;
            endpoint = '/api/run/hvla';
            body = {
                robot: robotData,
                s1_checkpoint: checkpointSel.value,
                s2_checkpoint: s2Ckpt,
                task: hvlaTask,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                decode_subtask: document.getElementById('run-hvla-decode-subtask')?.checked || false,
                record_dataset: recordDs,
                num_episodes: parseInt(document.getElementById('run-hvla-episodes')?.value) || 1,
                episode_time_s: parseFloat(document.getElementById('run-hvla-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-hvla-reset-time')?.value) || 20,
            };
        } else {
            // ---- Standard policy dispatch ----
            // Get task from dropdown or custom input
            const taskSel = document.getElementById('run-policy-task-select');
            const taskCustom = document.getElementById('run-policy-task-custom');
            const taskPlain = document.getElementById('run-policy-task');
            let task;
            if (taskSel) {
                task = (taskSel.value === '__new_task__')
                    ? taskCustom?.value?.trim()
                    : taskSel.value?.trim();
            } else {
                task = taskPlain?.value?.trim();
            }
            if (!task) {
                showToast('Error', 'Task description is required', 'error');
                return;
            }

            // Resolve dataset: new or existing
            const datasetSel = document.getElementById('run-policy-dataset');
            const datasetVal = datasetSel?.value || '__new__';
            let repoId, root = null, resume = false;

            if (datasetVal === '__new__') {
                repoId = document.getElementById('run-policy-repo-id')?.value?.trim();
                if (!repoId) {
                    showToast('Error', 'Enter a dataset name for evaluation recordings', 'error');
                    return;
                }
            } else {
                const dsId = datasetVal.replace('existing:', '');
                const d = (window.datasets || {})[dsId];
                if (!d) {
                    showToast('Error', 'Selected dataset not found — was it closed?', 'error');
                    return;
                }
                repoId = _resolveDatasetRepoId(d);
                root = d.root;
                resume = true;
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
                root: root,
                single_task: task,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                episode_time_s: parseFloat(document.getElementById('run-policy-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-policy-reset-time')?.value) || 60,
                num_episodes: parseInt(document.getElementById('run-policy-episodes')?.value) || 10,
                resume: resume,
            };
        }
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
        // Start live camera viewer (polls until obs stream is available)
        startObsStreamViewer();
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
        stopObsStreamViewer();
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
            // Re-scan dataset sources and refresh opened datasets
            if (typeof window.refreshExpandedSources === 'function') {
                window.refreshExpandedSources();
            }
            if (typeof window.refreshOpenedDatasets === 'function') {
                window.refreshOpenedDatasets();
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

function _showModelPanel(show) {
    const panel = document.getElementById('run-terminal-panel-model');
    if (panel) panel.style.display = show ? '' : 'none';
}

function appendTerminalLine(rawText) {
    // Route [S2] lines to model output panel
    if (rawText.startsWith('[S2] ') || rawText.startsWith('[S2 err] ')) {
        const modelTerminal = document.getElementById('run-model-terminal');
        if (modelTerminal) {
            const text = _stripAnsi(rawText.startsWith('[S2 err] ') ? rawText.slice(9) : rawText.slice(5));
            const line = document.createElement('div');
            line.className = 'terminal-line';
            line.textContent = text;
            modelTerminal.appendChild(line);
            modelTerminal.scrollTop = modelTerminal.scrollHeight;
            while (modelTerminal.children.length > 500) {
                modelTerminal.removeChild(modelTerminal.firstChild);
            }
        }
        return;
    }

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
            startObsStreamViewer();
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
// Resizable split between camera viewer and terminal
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
// Live camera viewer (obs-stream via shared memory)
// ============================================================================

async function startObsStreamViewer() {
    stopObsStreamViewer();

    const container = document.getElementById('rerun-viewer');
    if (!container) return;

    // Poll until the stream becomes available (robot needs time to connect)
    let attempts = 0;
    const maxAttempts = 30; // 30 × 500ms = 15s
    while (attempts < maxAttempts) {
        try {
            const res = await fetch('/api/run/obs-stream/meta');
            obsStreamMeta = await res.json();
            if (obsStreamMeta?.available) break;
        } catch (e) { /* not ready yet */ }
        await new Promise(r => setTimeout(r, 500));
        attempts++;
    }

    if (!obsStreamMeta?.available) {
        console.warn('Observation stream not available after timeout');
        return;
    }

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    // Remove any old content (iframe or previous grid)
    const oldIframe = container.querySelector('iframe');
    if (oldIframe) oldIframe.remove();
    let grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    // Build camera grid
    const camKeys = Object.keys(obsStreamMeta.image_keys);
    if (camKeys.length === 0) return;

    grid = document.createElement('div');
    grid.className = 'obs-cam-grid';
    const cols = camKeys.length <= 2 ? camKeys.length : Math.min(camKeys.length, 3);
    grid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(${cols}, 1fr);
        gap: 4px;
        width: 100%; height: 100%;
        padding: 4px;
        box-sizing: border-box;
    `;

    const imgElements = {};
    for (const key of camKeys) {
        const cell = document.createElement('div');
        cell.style.cssText = 'position: relative; overflow: hidden; background: #111; border-radius: 4px;';

        const img = document.createElement('img');
        img.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
        img.alt = key;
        cell.appendChild(img);

        const label = document.createElement('div');
        label.textContent = key;
        label.style.cssText = `
            position: absolute; top: 4px; left: 6px;
            color: #ccc; font-size: 11px; font-family: monospace;
            background: rgba(0,0,0,0.5); padding: 1px 5px; border-radius: 3px;
        `;
        cell.appendChild(label);

        grid.appendChild(cell);
        imgElements[key] = img;
    }
    container.appendChild(grid);

    // Poll camera frames at ~10fps
    let frameSeq = 0;
    obsStreamTimer = setInterval(() => {
        const seq = ++frameSeq;
        for (const key of camKeys) {
            const img = imgElements[key];
            if (!img) continue;
            // Append seq to bust browser cache
            img.src = `/api/run/obs-stream/image/${encodeURIComponent(key)}?_=${seq}`;
        }
    }, 100);
}

function stopObsStreamViewer() {
    if (obsStreamTimer) {
        clearInterval(obsStreamTimer);
        obsStreamTimer = null;
    }
    obsStreamMeta = null;

    const container = document.getElementById('rerun-viewer');
    if (!container) return;
    const grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = '';
}
