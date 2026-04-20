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
        _toggleHvlaRecordFields();
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
    _initSelectTitleSync();
    await _ensureModelDataLoaded();
    await pollRunStatus();
}

/** Sync select title attribute to selected option text (for ellipsis tooltip). */
function _initSelectTitleSync() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    function _syncTitle(sel) {
        const opt = sel.options[sel.selectedIndex];
        sel.title = opt ? opt.textContent : '';
    }
    sidebar.addEventListener('change', (e) => {
        if (e.target.tagName === 'SELECT') _syncTitle(e.target);
    });
    // Also observe innerHTML changes so programmatic updates get titles too
    new MutationObserver(() => {
        sidebar.querySelectorAll('select').forEach(_syncTitle);
    }).observe(sidebar, { childList: true, subtree: true });
}

// ============================================================================
// Workflow selection
// ============================================================================

async function selectWorkflow(workflow) {
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
        await _ensureModelDataLoaded();
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
    const mainIsExisting = val.startsWith('existing:');

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

    // Update intervention dataset options: existing intervention datasets only available
    // when main dataset is also existing (both share the resume flag in the CLI)
    const intSel = document.getElementById('run-policy-intervention-dataset');
    if (intSel) {
        const prev = intSel.value;
        intSel.innerHTML = _interventionDatasetOptions(mainIsExisting);
        intSel.value = prev;
        // If previous selection is no longer valid, reset to None
        if (intSel.value !== prev) {
            intSel.value = '';
            _onInterventionDatasetChange();
        }
    }
}

async function _onPolicyTeleopChange() {
    const sel = document.getElementById('run-policy-teleop');
    const section = document.getElementById('run-policy-intervention-section');
    if (!section) return;

    if (!sel?.value) {
        section.style.display = 'none';
        return;
    }

    // Show the section but check if intervention_enabled is set in the profile
    section.style.display = '';
    const profileData = await _getProfileData('teleop', sel.value);
    const interventionEnabled = profileData?.fields?.intervention_enabled === true;
    const datasetSel = document.getElementById('run-policy-intervention-dataset');
    const nameInput = document.getElementById('run-policy-intervention-repo-id');
    if (datasetSel) {
        datasetSel.disabled = !interventionEnabled;
        if (!interventionEnabled) {
            datasetSel.value = '';
            _onInterventionDatasetChange();
        }
    }
    if (nameInput) nameInput.disabled = !interventionEnabled;
    // Show warning when intervention is not enabled
    const warning = document.getElementById('run-policy-intervention-warning');
    if (warning) warning.style.display = interventionEnabled ? 'none' : '';
}

function _interventionDatasetOptions(mainIsExisting) {
    const ds = window.datasets || {};
    let html = '<option value="" selected>None</option>';
    // Intervention resume is tied to the main dataset's resume flag in the CLI,
    // so existing intervention requires existing main, new intervention requires new main.
    if (mainIsExisting) {
        for (const id of Object.keys(ds)) {
            const d = ds[id];
            const label = d.repo_id || id;
            html += `<option value="existing:${_esc(id)}">${_esc(label)} (${d.total_episodes} ep)</option>`;
        }
    } else {
        html += '<option value="__new__">+ New dataset...</option>';
    }
    return html;
}

function _onInterventionDatasetChange() {
    const sel = document.getElementById('run-policy-intervention-dataset');
    if (!sel) return;
    const val = sel.value;
    const newNameRow = document.getElementById('run-policy-intervention-new-row');
    if (val === '__new__') {
        if (newNameRow) newNameRow.style.display = '';
    } else {
        if (newNameRow) newNameRow.style.display = 'none';
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
    _showModelPanel(true);  // show panel immediately during loading
    if (status) status.textContent = 'Loading...';
    // Clear model terminal
    const modelTerminal = document.getElementById('run-model-terminal');
    if (modelTerminal) modelTerminal.innerHTML = '<div class="terminal-line" style="color: #666;">Loading model...</div>';
    // Connect SSE immediately so output appears as model loads
    _connectDebugOutputSSE();
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
            console.log('Debug model loaded, showing panel');
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
            _disconnectDebugOutputSSE();
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
        _debugModelLoading = false;  // reset stuck loading flag on status refresh
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
    if (isHVLA) {
        _toggleHvlaRecordFields();
        _refreshRltCheckpoints();
    }
}

function _toggleHvlaRecordFields() {
    const hasDataset = !!document.getElementById('run-hvla-record-dataset')?.value?.trim();
    const rltVal = document.getElementById('run-hvla-rlt-select')?.value || '';
    const rltEnabled = rltVal !== '';
    const enableEpFields = hasDataset || rltEnabled;
    for (const id of ['run-hvla-episodes', 'run-hvla-episode-time', 'run-hvla-reset-time']) {
        const el = document.getElementById(id);
        if (el) el.disabled = !enableEpFields;
    }
}

function _onRltSelectChange() {
    const sel = document.getElementById('run-hvla-rlt-select');
    const val = sel?.value || '';
    const isNew = val === '__new__';
    const isExisting = val && !isNew;

    // Token encoder: shown for both New and Existing (auto-discovery may fail for old checkpoints)
    for (const id of ['run-hvla-rlt-token-label', 'run-hvla-rlt-token-ckpt']) {
        const el = document.getElementById(id);
        if (el) el.style.display = (isNew || isExisting) ? '' : 'none';
    }
    // Output dir: only for New training
    for (const id of ['run-hvla-rlt-outdir-label', 'run-hvla-rlt-output-dir']) {
        const el = document.getElementById(id);
        if (el) el.style.display = isNew ? '' : 'none';
    }
    // Existing checkpoint fields
    for (const id of ['run-hvla-rlt-mode-label', 'run-hvla-rlt-run-mode']) {
        const el = document.getElementById(id);
        if (el) el.style.display = isExisting ? '' : 'none';
    }
    // Common fields (chunk length, start engaged)
    for (const id of ['run-hvla-rlt-chunk-label', 'run-hvla-rlt-chunk-length', 'run-hvla-rlt-start-label', 'run-hvla-rlt-start-engaged']) {
        const el = document.getElementById(id);
        if (el) el.style.display = (isNew || isExisting) ? '' : 'none';
    }
    _toggleHvlaRecordFields();
}

async function _refreshRltCheckpoints() {
    try {
        const res = await fetch('/api/models/rlt-checkpoints');
        if (!res.ok) return;
        const checkpoints = await res.json();
        const sel = document.getElementById('run-hvla-rlt-select');
        if (!sel) return;
        // Preserve current selection
        const currentVal = sel.value;
        // Keep None and New options, replace the rest
        sel.innerHTML = '<option value="">None (no RL)</option><option value="__new__">+ New training...</option>';
        for (const ckpt of checkpoints) {
            const label = ckpt.name + (ckpt.episode ? ` (ep ${ckpt.episode})` : '');
            const opt = document.createElement('option');
            opt.value = ckpt.path;
            opt.textContent = label;
            opt.dataset.hasReplayBuffer = ckpt.has_replay_buffer || false;
            opt.dataset.hasCritic = ckpt.has_critic || false;
            sel.appendChild(opt);
        }
        // Restore selection if still valid
        if (currentVal && [...sel.options].some(o => o.value === currentVal)) {
            sel.value = currentVal;
        }
    } catch (e) {
        console.warn('Failed to fetch RLT checkpoints:', e);
    }
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
    // Already have data — nothing to do
    if (typeof modelSourceData !== 'undefined' && Object.keys(modelSourceData).length) return;
    // Load sources and scan (bypass modelTabInit's one-shot guard which
    // blocks retries when the first attempt finished with empty data).
    if (typeof loadModelSources === 'function') {
        await loadModelSources();
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
    // Refresh intervention dataset selector
    const intSel = document.getElementById('run-policy-intervention-dataset');
    if (intSel) {
        const mainVal = document.getElementById('run-policy-dataset')?.value || '__new__';
        const prev = intSel.value;
        intSel.innerHTML = _interventionDatasetOptions(mainVal.startsWith('existing:'));
        intSel.value = prev;
    }
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
    html += `<div><select id="run-policy-teleop" onchange="_onPolicyTeleopChange()">`;
    html += `<option value="">None (policy only)</option>`;
    html += _teleopProfileOptions();
    html += `</select>`;
    html += `<div class="form-hint">For leader inverse following and manual resets between episodes</div></div>`;
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
    html += `<input type="checkbox" id="run-hvla-decode-subtask">`;
    html += `<label>Query Interval (steps between S1 inference, default 2)</label>`;
    html += `<input type="number" id="run-hvla-query-interval" placeholder="2" min="0">`;
    html += `<label>Denoise Steps (default 10)</label>`;
    html += `<input type="number" id="run-hvla-denoise-steps" placeholder="10" min="1">`;
    html += `<label>Record Dataset</label>`;
    html += `<input type="text" id="run-hvla-record-dataset" placeholder="eval/hvla_eval (optional)" value="" oninput="_toggleHvlaRecordFields()">`;
    html += `<label>Episodes</label>`;
    html += `<input type="number" id="run-hvla-episodes" value="1" min="1">`;
    html += `<label>Episode Duration</label>`;
    html += `<input type="number" id="run-hvla-episode-time" value="60" min="0">`;
    html += `<label>Reset Duration</label>`;
    html += `<input type="number" id="run-hvla-reset-time" value="20" min="0">`;
    html += `<label>Intervention Dataset</label>`;
    html += `<input type="text" id="run-hvla-intervention-dataset" placeholder="eval/hvla_interventions (optional)" value="">`;
    html += `<div class="form-hint" style="grid-column: 1 / -1;">When the human takes over via SPACE, intervention fragments are saved to this dataset.</div>`;
    html += '</div>';
    html += '</div>';
    // ---- RLT section ----
    html += '<div class="form-section">';
    html += '<div class="form-section-title">RLT Online RL</div>';
    html += '<div class="form-grid">';
    html += `<label>RL Checkpoint</label>`;
    html += `<select id="run-hvla-rlt-select" onchange="_onRltSelectChange()">`;
    html += `<option value="">None (no RL)</option>`;
    html += `<option value="__new__">+ New training...</option>`;
    html += `</select>`;
    // Fields shown when New is selected
    html += `<label id="run-hvla-rlt-token-label" style="display:none">RL Token Encoder</label>`;
    html += `<input type="text" id="run-hvla-rlt-token-ckpt" placeholder="outputs/rlt_token_v2/checkpoint-10000" style="display:none">`;
    html += `<label id="run-hvla-rlt-outdir-label" style="display:none">Output Directory</label>`;
    html += `<input type="text" id="run-hvla-rlt-output-dir" value="outputs/rlt_online" style="display:none">`;
    // Fields shown when existing checkpoint selected
    html += `<label id="run-hvla-rlt-mode-label" style="display:none">Mode</label>`;
    html += `<select id="run-hvla-rlt-run-mode" style="display:none">`;
    html += `<option value="train">Train (continue learning)</option>`;
    html += `<option value="deploy">Deploy (inference only)</option>`;
    html += `</select>`;
    // Common fields
    html += `<label id="run-hvla-rlt-chunk-label" style="display:none">Chunk Length</label>`;
    html += `<input type="number" id="run-hvla-rlt-chunk-length" value="10" min="1" max="50" style="display:none">`;
    html += `<label id="run-hvla-rlt-start-label" style="display:none">Start with RL active</label>`;
    html += `<input type="checkbox" id="run-hvla-rlt-start-engaged" checked style="display:none">`;
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
    // Intervention dataset (for leader inverse following — human correction fragments)
    html += '<div class="form-section" id="run-policy-intervention-section" style="display:none;">';
    html += '<div class="form-section-title">Intervention dataset</div>';
    html += '<div class="form-hint" style="margin-bottom:8px;">When the human takes over the leader arm during policy evaluation, intervention fragments are saved to a separate dataset.</div>';
    html += '<div class="form-hint dataset-conflict-warning" id="run-policy-intervention-warning" style="display:none; margin-bottom:8px;">Enable intervention_enabled in the teleop profile to use this feature.</div>';
    html += '<div class="form-grid">';
    html += `<label>Dataset</label>`;
    html += `<select id="run-policy-intervention-dataset" onchange="_onInterventionDatasetChange()">${_interventionDatasetOptions()}</select>`;
    html += '</div>';
    html += '<div id="run-policy-intervention-new-row" class="form-grid" style="display:none;">';
    html += `<label>Name</label>`;
    html += `<input type="text" id="run-policy-intervention-repo-id" placeholder="e.g. eval/interventions" value="">`;
    html += '</div>';
    html += '</div>';
    html += '</div>'; // end standard fields
    html += '</div>'; // end policy section

    form.innerHTML = html;
    _toggleHvlaRecordFields();
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
                debug_model: _getDebugModelConfig(),
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
            const s2Ckpt = document.getElementById('run-hvla-s2-checkpoint')?.value?.trim() || null;
            const hvlaTask = document.getElementById('run-hvla-task')?.value?.trim();
            if (!hvlaTask) {
                showToast('Error', 'Task prompt is required for HVLA', 'error');
                return;
            }
            const recordDs = document.getElementById('run-hvla-record-dataset')?.value?.trim() || null;
            const intDs = document.getElementById('run-hvla-intervention-dataset')?.value?.trim() || null;

            // Optional teleop for intervention / inverse follow
            const teleopSelect = document.getElementById('run-policy-teleop');
            let hvlaTeleopData = null;
            if (teleopSelect?.value) {
                hvlaTeleopData = await _getProfileData('teleop', teleopSelect.value);
            }

            endpoint = '/api/run/hvla';
            body = {
                robot: robotData,
                s1_checkpoint: checkpointSel.value,
                s2_checkpoint: s2Ckpt,
                task: hvlaTask,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                s1_query_interval: parseInt(document.getElementById('run-hvla-query-interval')?.value) || null,
                denoise_steps: parseInt(document.getElementById('run-hvla-denoise-steps')?.value) || null,
                decode_subtask: document.getElementById('run-hvla-decode-subtask')?.checked || false,
                record_dataset: recordDs,
                num_episodes: parseInt(document.getElementById('run-hvla-episodes')?.value) || 1,
                episode_time_s: parseFloat(document.getElementById('run-hvla-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-hvla-reset-time')?.value) || 20,
                teleop: hvlaTeleopData,
                intervention_dataset: intDs,
                ...(() => {
                    const rltSel = document.getElementById('run-hvla-rlt-select');
                    const rltVal = rltSel?.value || '';
                    if (!rltVal) return {};  // None selected
                    const rltStartEngaged = document.getElementById('run-hvla-rlt-start-engaged')?.checked !== false;
                    if (rltVal === '__new__') {
                        return {
                            rlt_mode: true,
                            rlt_token_checkpoint: document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim() || null,
                            rl_chunk_length: parseInt(document.getElementById('run-hvla-rlt-chunk-length')?.value) || 10,
                            rlt_output_dir: document.getElementById('run-hvla-rlt-output-dir')?.value?.trim() || 'outputs/rlt_online',
                            rlt_start_engaged: rltStartEngaged,
                        };
                    }
                    // Existing checkpoint
                    const runMode = document.getElementById('run-hvla-rlt-run-mode')?.value || 'train';
                    return {
                        rlt_mode: true,
                        rlt_checkpoint: rltVal,  // path to existing actor.pt directory
                        rlt_token_checkpoint: document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim() || null,
                        rlt_deploy: runMode === 'deploy',
                        rl_chunk_length: parseInt(document.getElementById('run-hvla-rlt-chunk-length')?.value) || 10,
                        rlt_output_dir: rltVal,  // same dir for continued training
                        rlt_start_engaged: rltStartEngaged,
                    };
                })(),
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

            // Intervention dataset (for leader inverse following correction data)
            let interventionRepoId = null;
            const interventionSel = document.getElementById('run-policy-intervention-dataset');
            const interventionVal = interventionSel?.value || '';
            if (interventionVal === '__new__') {
                interventionRepoId = document.getElementById('run-policy-intervention-repo-id')?.value?.trim() || null;
                if (!interventionRepoId) {
                    showToast('Error', 'Enter a name for the intervention dataset', 'error');
                    return;
                }
            } else if (interventionVal.startsWith('existing:')) {
                const intDsId = interventionVal.replace('existing:', '');
                const intDs = (window.datasets || {})[intDsId];
                if (intDs) interventionRepoId = _resolveDatasetRepoId(intDs);
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
                intervention_repo_id: interventionRepoId,
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
        if (body?.rlt_mode) _startRLTPoll();
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
            _stopRLTPoll();
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
        if (data.overlay) {
            const el = document.getElementById('debug-model-overlay');
            if (el) {
                el.textContent = data.overlay.text || '';
                el.style.color = data.overlay.color || '#fff';
                el.style.display = data.overlay.text ? 'block' : 'none';
            }
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

let _debugOutputSSE = null;

function _showModelPanel(show) {
    const panel = document.getElementById('run-terminal-panel-model');
    if (!panel) { console.warn('Model panel element not found'); return; }
    panel.style.display = show ? '' : 'none';
    console.log('Model panel display:', show ? 'visible' : 'hidden');
    if (show && !_debugOutputSSE) {
        _connectDebugOutputSSE();
    } else if (!show && _debugOutputSSE) {
        _disconnectDebugOutputSSE();
    }
}

function _connectDebugOutputSSE() {
    _disconnectDebugOutputSSE();
    _debugOutputSSE = new EventSource('/api/run/debug/output');
    _debugOutputSSE.onmessage = (event) => {
        const modelTerminal = document.getElementById('run-model-terminal');
        if (!modelTerminal) return;
        const text = _stripAnsi(event.data);
        const line = document.createElement('div');
        line.className = 'terminal-line';
        line.textContent = text;
        modelTerminal.appendChild(line);
        modelTerminal.scrollTop = modelTerminal.scrollHeight;
        while (modelTerminal.children.length > 500) {
            modelTerminal.removeChild(modelTerminal.firstChild);
        }

    };
}

// TODO: generalize — currently hardcoded for HVLA S2 subtask text.
// When multiple debug model types exist, the model should declare its
// output schema and the user picks which field to overlay.

// Called from the camera poll loop (startObsStreamViewer) — no separate timer.
async function _pollSubtaskOverlay() {
    if (!_debugModelLoaded) return;
    try {
        const res = await fetch('/api/run/debug/subtask');
        if (!res.ok) return;
        const data = await res.json();
        const overlay = document.getElementById('debug-model-overlay');
        if (!overlay) return;
        if (data.subtask) {
            const conf = data.confidence != null ? ` (${(data.confidence * 100).toFixed(0)}%)` : '';
            overlay.textContent = data.subtask + conf;
            overlay.style.display = 'block';
        }
    } catch (e) { /* ignore */ }
}

function _hideSubtaskOverlay() {
    const overlay = document.getElementById('debug-model-overlay');
    if (overlay) {
        overlay.style.display = 'none';
        overlay.textContent = '';
    }
}

function _disconnectDebugOutputSSE() {
    if (_debugOutputSSE) {
        _debugOutputSSE.close();
        _debugOutputSSE = null;
    }
    _hideSubtaskOverlay();
}

function copyTerminal(terminalId) {
    const terminal = document.getElementById(terminalId);
    if (!terminal) return;
    const lines = Array.from(terminal.querySelectorAll('.terminal-line'))
        .map(el => el.textContent).join('\n');
    navigator.clipboard.writeText(lines).then(() => {
        showToast('Terminal', 'Copied to clipboard', 'info');
    }).catch(() => {});
}

function clearTerminal(terminalId) {
    const terminal = document.getElementById(terminalId);
    if (!terminal) return;
    terminal.innerHTML = '';
}

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
        _pollSubtaskOverlay();
    }, 50);
}

function stopObsStreamViewer() {
    if (obsStreamTimer) {
        clearInterval(obsStreamTimer);
        obsStreamTimer = null;
    }
    obsStreamMeta = null;
    _hideSubtaskOverlay();

    const container = document.getElementById('rerun-viewer');
    if (!container) return;
    const grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = '';
}


// =============================================================================
// RLT Training Dashboard
// =============================================================================

let _rltPollTimer = null;

function _switchBottomTab(tab) {
    document.querySelectorAll('.run-bottom-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tab);
    });
    const outputPanel = document.getElementById('run-bottom-panel-output');
    const rlPanel = document.getElementById('run-bottom-panel-rl');
    if (outputPanel) outputPanel.style.display = (tab === 'output') ? '' : 'none';
    if (rlPanel) rlPanel.style.display = (tab === 'rl') ? '' : 'none';
}

function _showRLTab(show) {
    const tab = document.getElementById('run-bottom-tab-rl');
    if (tab) tab.style.display = show ? '' : 'none';
}

function _startRLTPoll() {
    _stopRLTPoll();
    _showRLTab(true);
    _rltPollTimer = setInterval(_fetchRLTMetrics, 2000);
    _fetchRLTMetrics();
    _initRLTSliders();
}

function _stopRLTPoll() {
    if (_rltPollTimer) { clearInterval(_rltPollTimer); _rltPollTimer = null; }
}

async function _fetchRLTMetrics() {
    try {
        const res = await fetch('/api/run/rlt-metrics');
        if (!res.ok) return;
        const data = await res.json();
        _updateRLTDashboard(data);
    } catch (e) { /* ignore */ }
}

function _updateRLTDashboard(data) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('rlt-stat-episode', data.episode);
    set('rlt-stat-mode', data.mode);
    set('rlt-stat-rate', Math.round((data.autonomous_rate || 0) * 100) + '%');
    set('rlt-stat-throughput', (data.throughput_10min || 0) + '/10min');
    set('rlt-stat-buffer', data.buffer_size?.toLocaleString() || '0');
    set('rlt-stat-updates', data.total_updates?.toLocaleString() || '0');

    const rawS = data.series || {};

    // Clip all per-step series to the user-selected history length
    const nMax = parseInt(document.getElementById('rlt-stat-history')?.value) || 1000;
    const clip = arr => (arr && arr.length > nMax) ? arr.slice(-nMax) : (arr || []);
    const s = {
        ...rawS,
        critic_losses: clip(rawS.critic_losses),
        actor_losses: clip(rawS.actor_losses),
        actor_deltas: clip(rawS.actor_deltas),
        q_values_mean: clip(rawS.q_values_mean),
        q_values_min: clip(rawS.q_values_min),
        q_values_max: clip(rawS.q_values_max),
        actor_q_terms: clip(rawS.actor_q_terms),
        actor_bc_terms: clip(rawS.actor_bc_terms),
        step_timestamps: clip(rawS.step_timestamps),
    };

    // Success rate uses episode axis, not synced with per-step charts
    if (s.autonomous_rate_rolling && s.autonomous_rate_rolling.length > 0) {
        _drawSparkline('rlt-chart-success', s.autonomous_rate_rolling, '#4fc3f7', 0, 1, true);
    }

    // Per-step synced charts — all share the same step index as X
    _syncedTimestamps = s.step_timestamps || [];
    _syncedLatestStep = data.total_updates || 0;

    if (s.q_values_mean && s.q_values_mean.length > 0) {
        _registerSyncedChart('rlt-chart-qvalues', [
            {data: s.q_values_min || [], color: '#2ecc71', label: 'min',
             bandPair: 'min', bandColor: '#2ecc71', hideLine: true},
            {data: s.q_values_max || [], color: '#2ecc71', label: 'max',
             bandPair: 'max', hideLine: true},
            {data: s.q_values_mean || [], color: '#2ecc71', label: 'mean'},
        ]);
    }
    if (s.actor_deltas && s.actor_deltas.length > 0) {
        _registerSyncedChart('rlt-chart-delta', [
            {data: s.actor_deltas, color: '#e5c07b', label: 'delta'},
        ]);
    }
    if (s.critic_losses && s.critic_losses.length > 0) {
        _registerSyncedChart('rlt-chart-critic', [
            {data: s.critic_losses, color: '#f39c12', label: 'critic'},
        ]);
    }
    if ((s.actor_q_terms && s.actor_q_terms.length > 0) ||
        (s.actor_bc_terms && s.actor_bc_terms.length > 0)) {
        _registerSyncedChart('rlt-chart-actor-components', [
            {data: s.actor_q_terms || [], color: '#e06c75', label: 'q'},
            {data: s.actor_bc_terms || [], color: '#4fc3f7', label: 'bc'},
        ]);
    }
}

let _rltConfigTimer = null;
function _initRLTSliders() {
    const controls = document.getElementById('rlt-controls');
    if (controls) controls.style.display = '';

    const betaSlider = document.getElementById('rlt-slider-beta');
    const sigmaSlider = document.getElementById('rlt-slider-sigma');
    const betaVal = document.getElementById('rlt-slider-beta-val');
    const sigmaVal = document.getElementById('rlt-slider-sigma-val');

    if (betaSlider) {
        betaSlider.oninput = () => {
            betaVal.textContent = parseFloat(betaSlider.value).toFixed(2);
            _sendRLTConfig();
        };
    }
    if (sigmaSlider) {
        sigmaSlider.oninput = () => {
            sigmaVal.textContent = parseFloat(sigmaSlider.value).toFixed(3);
            _sendRLTConfig();
        };
    }

    const historySelect = document.getElementById('rlt-stat-history');
    if (historySelect) {
        historySelect.onchange = () => _fetchRLTMetrics();
    }
}

function _sendRLTConfig() {
    // Debounce: wait 300ms after last change before sending
    if (_rltConfigTimer) clearTimeout(_rltConfigTimer);
    _rltConfigTimer = setTimeout(async () => {
        const beta = parseFloat(document.getElementById('rlt-slider-beta')?.value);
        const sigma = parseFloat(document.getElementById('rlt-slider-sigma')?.value);
        if (isNaN(beta) || isNaN(sigma)) return;
        try {
            await fetch('/api/run/rlt-config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({beta, actor_sigma: sigma}),
            });
        } catch (e) { /* ignore */ }
    }, 300);
}

// Registry of per-step synced charts. When the cursor hovers over any of
// them, a vertical crosshair and value readouts appear on all.
// Each entry: { canvas, series: [{data, color, label, percentage}], min, max, pad, W, H }
const _syncedCharts = {};
let _syncedTimestamps = [];  // shared wall-clock timestamps for per-step charts
let _syncedLatestStep = 0;   // global step count of the rightmost (newest) data point
let _hoverIndex = -1;        // -1 = no hover, else index into series

function _fmtTime(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
}

function _fmtAgo(ts) {
    if (!ts) return '';
    const now = Date.now() / 1000;
    const ago = Math.max(0, now - ts);
    if (ago < 60) return Math.round(ago) + 's ago';
    if (ago < 3600) return Math.round(ago / 60) + 'm ago';
    return Math.round(ago / 3600) + 'h ago';
}

function _renderSyncedChart(id) {
    const c = _syncedCharts[id];
    if (!c) return;
    const ctx = c.canvas.getContext('2d');
    const {W, H, pad, min, max, series} = c;
    const range = (max - min) || 1;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#1a1a3e'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad + (H - 2 * pad) * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    // Zero line if range crosses zero
    if (min < 0 && max > 0) {
        const zeroY = pad + (H - 2 * pad) * (1 - (0 - min) / range);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(W, zeroY); ctx.stroke();
    }

    const N = Math.max(...series.map(s => s.data.length), 1);
    const toX = i => (i / Math.max(N - 1, 1)) * W;
    const toY = v => pad + (H - 2 * pad) * (1 - (v - min) / range);

    // Optional min/max band fill (pair of series with `bandPair` key)
    const bandMin = series.find(s => s.bandPair === 'min');
    const bandMax = series.find(s => s.bandPair === 'max');
    if (bandMin && bandMax && bandMin.data.length > 0 && bandMax.data.length > 0) {
        ctx.fillStyle = (bandMin.bandColor || bandMin.color) + '33';  // ~20% opacity
        ctx.beginPath();
        for (let i = 0; i < bandMax.data.length; i++) ctx.lineTo(toX(i), toY(bandMax.data[i]));
        for (let i = bandMin.data.length - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(bandMin.data[i]));
        ctx.closePath();
        ctx.fill();
    }

    // Series lines
    for (const s of series) {
        if (s.data.length === 0) continue;
        if (s.hideLine) continue;
        ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < s.data.length; i++) {
            const x = toX(i);
            const y = toY(s.data[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Crosshair
    if (_hoverIndex >= 0 && _hoverIndex < N) {
        const x = (_hoverIndex / Math.max(N - 1, 1)) * W;
        ctx.strokeStyle = '#888'; ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        ctx.setLineDash([]);
    }

    // Value labels (top-right) — at hover index if hovering, else last value
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    const showIdx = _hoverIndex >= 0 ? _hoverIndex : -1;
    for (let i = 0; i < series.length; i++) {
        const s = series[i];
        if (s.data.length === 0) continue;
        const idx = showIdx >= 0 && showIdx < s.data.length ? showIdx : s.data.length - 1;
        const v = s.data[idx];
        const label = s.percentage ? (v * 100).toFixed(0) + '%' : v.toFixed(4);
        ctx.fillStyle = s.color;
        ctx.fillText(label, W - 4, 12 + i * 12);
    }

    // Step/time label (bottom-left)
    ctx.fillStyle = '#444';
    ctx.textAlign = 'left';
    if (_hoverIndex >= 0) {
        // Global step: rightmost data point is at step _syncedLatestStep
        const globalStep = _syncedLatestStep - (N - 1 - _hoverIndex);
        // Align timestamps from the right (shorter arrays right-aligned)
        const tsOffset = N - _syncedTimestamps.length;
        const tsIdx = _hoverIndex - tsOffset;
        const ts = tsIdx >= 0 ? _syncedTimestamps[tsIdx] : null;
        const label = ts
            ? `step ${globalStep}  ${_fmtTime(ts)} (${_fmtAgo(ts)})`
            : `step ${globalStep}`;
        ctx.fillText(label, 4, H - 4);
    } else {
        ctx.fillText(N + ' pts', 4, H - 4);
    }
}

function _registerSyncedChart(canvasId, seriesList) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const valid = seriesList.filter(s => s.data && s.data.length > 0);
    if (valid.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.getContext('2d').scale(dpr, dpr);

    let allVals = [];
    for (const s of valid) allVals = allVals.concat(s.data);
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);

    _syncedCharts[canvasId] = {
        canvas, series: seriesList, min, max, pad: 4,
        W: rect.width, H: rect.height,
    };

    // Attach hover handlers once
    if (!canvas._syncedAttached) {
        canvas.addEventListener('mousemove', (e) => {
            const r = canvas.getBoundingClientRect();
            const x = e.clientX - r.left;
            const c = _syncedCharts[canvasId];
            if (!c) return;
            const N = Math.max(...c.series.map(s => s.data.length), 1);
            _hoverIndex = Math.min(N - 1, Math.max(0, Math.round((x / c.W) * (N - 1))));
            for (const id of Object.keys(_syncedCharts)) _renderSyncedChart(id);
        });
        canvas.addEventListener('mouseleave', () => {
            _hoverIndex = -1;
            for (const id of Object.keys(_syncedCharts)) _renderSyncedChart(id);
        });
        canvas._syncedAttached = true;
    }

    _renderSyncedChart(canvasId);
}

function _drawSparklineMulti(canvasId, seriesList) {
    // seriesList: [{data, color, label}, ...] — all on shared Y axis computed from combined range
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const validSeries = seriesList.filter(s => s.data && s.data.length > 0);
    if (validSeries.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    ctx.clearRect(0, 0, W, H);

    let allVals = [];
    for (const s of validSeries) allVals = allVals.concat(s.data);
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);
    const range = (max - min) || 1;
    const pad = 4;

    ctx.strokeStyle = '#1a1a3e';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad + (H - 2 * pad) * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Zero line if range crosses zero
    if (min < 0 && max > 0) {
        const zeroY = pad + (H - 2 * pad) * (1 - (0 - min) / range);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(W, zeroY); ctx.stroke();
    }

    for (const s of validSeries) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < s.data.length; i++) {
            const x = (i / Math.max(s.data.length - 1, 1)) * W;
            const y = pad + (H - 2 * pad) * (1 - (s.data[i] - min) / range);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Latest values on right
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i < validSeries.length; i++) {
        const s = validSeries[i];
        ctx.fillStyle = s.color;
        ctx.fillText(s.data[s.data.length - 1].toFixed(4), W - 4, 12 + i * 12);
    }
    // Data point count on left
    ctx.fillStyle = '#444';
    ctx.textAlign = 'left';
    ctx.fillText(validSeries[0].data.length + ' pts', 4, H - 4);
}

function _drawSparkline(canvasId, data, color, fixedMin, fixedMax, percentage) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    ctx.clearRect(0, 0, W, H);

    const min = fixedMin !== undefined ? fixedMin : Math.min(...data);
    const max = fixedMax !== undefined ? fixedMax : Math.max(...data);
    const range = (max - min) || 1;
    const pad = 4;

    ctx.strokeStyle = '#1a1a3e';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad + (H - 2 * pad) * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = (i / Math.max(data.length - 1, 1)) * W;
        const y = pad + (H - 2 * pad) * (1 - (data[i] - min) / range);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const last = data[data.length - 1];
    const label = percentage ? (last * 100).toFixed(0) + '%' : last.toFixed(4);
    ctx.fillStyle = color;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(label, W - 4, 12);
    // Show data point count on left (x-axis scale)
    ctx.fillStyle = '#444';
    ctx.textAlign = 'left';
    ctx.fillText(data.length + ' pts', 4, H - 4);
}

function _drawSparkBand(canvasId, dataMin, dataMax, dataMean, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || dataMean.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const allVals = [...dataMin, ...dataMax];
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);
    const range = (max - min) || 1;
    const pad = 4;
    const n = dataMean.length;

    const toY = (v) => pad + (H - 2 * pad) * (1 - (v - min) / range);
    const toX = (i) => (i / Math.max(n - 1, 1)) * W;

    // Grid
    ctx.strokeStyle = '#1a1a3e'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad + (H - 2 * pad) * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Min-max band (filled)
    ctx.fillStyle = color + '20';  // 12% opacity
    ctx.beginPath();
    for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(dataMax[i] || 0));
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(dataMin[i] || 0));
    ctx.closePath();
    ctx.fill();

    // Mean line
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = toX(i), y = toY(dataMean[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Labels
    ctx.fillStyle = color; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    const lastMean = dataMean[n - 1];
    const lastMax = (dataMax[n - 1] || 0);
    const lastMin = (dataMin[n - 1] || 0);
    ctx.fillText(`${lastMin.toFixed(2)}/${lastMean.toFixed(2)}/${lastMax.toFixed(2)}`, W - 4, 12);
}
