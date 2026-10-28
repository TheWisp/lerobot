/* LeRobot GUI - Run Tab Logic */

let runTabInitialized = false;
let runEventSource = null;
let selectedWorkflow = 'teleop'; // 'teleop' | 'replay' | 'policy' | 'flash'
let obsStreamMeta = null; // {available, obs_scalar_keys, action_keys, image_keys}
let obsStreamTimer = null; // interval ID for camera polling
let liveVideo = null;      // the Low Bandwidth stream client, when that is the path
let obsStreamGen = 0; // bumped on stop/restart to cancel an in-flight "wait for the obs-stream" loop
let _runFormRendered = false; // true once all workflow sections are in the DOM
let flashDatasets = []; // opened ordinary LeRobot datasets, refreshed on FLASH selection
let flashPreparedSkills = [];
let flashInspectGeneration = 0;
let flashRecordingsGeneration = 0;
let flashUiMode = 'review';
let flashLibraryTab = 'demos';
let flashSceneView = null;
let flashSceneRequest = 0;
let flashSceneMode = 'camera';
let flashSceneFrameCache = null;
let flashSceneLoadingKey = null;

function _flashPaintObservedPhoto(frame, camera) {
    const photo = document.querySelector('#rerun-viewer .flash-scene-photo');
    if (!photo) return;
    const runsByRole = frame.evidence?.masks_rle;
    photo.style.visibility = _flashSceneOptions().showSurface && runsByRole ? 'visible' : 'hidden';
    if (!runsByRole || photo.style.visibility === 'hidden') return;
    if (photo.width !== camera.width || photo.height !== camera.height) {
        photo.width = camera.width;
        photo.height = camera.height;
    }
    const pixels = new Uint8ClampedArray(frame.rgba.length);
    for (const runs of Object.values(runsByRole)) {
        for (const [start, count] of runs) {
            if (start < 0 || start + count > camera.width * camera.height)
                throw new Error('Object mask exceeds camera frame');
            pixels.set(frame.rgba.subarray(start * 4, (start + count) * 4), start * 4);
        }
    }
    photo.getContext('2d').putImageData(new ImageData(pixels, camera.width, camera.height), 0, 0);
}

function _flashSceneOptions() {
    return {
        showSurface: document.getElementById('flash-scene-surface')?.checked ?? false,
        showPoints: document.getElementById('flash-scene-points')?.checked ?? false,
        pointSpacingPx: Math.max(2, Math.min(16, Number(document.getElementById('flash-scene-point-spacing')?.value || 5))),
        showProjection: document.getElementById('flash-scene-projection')?.checked,
        showDepthScene: document.getElementById('flash-scene-depth')?.checked,
        showAxes: document.getElementById('flash-scene-axes')?.checked,
        showGroup: document.getElementById('flash-scene-group')?.checked,
        showGripper: document.getElementById('flash-scene-gripper')?.checked,
    };
}

async function _flashUpdateSceneFrame() {
    const kind = flashReplay?.kind || (flashTelemetryTimer ? 'live' : null);
    if (flashSceneMode !== '3d' || !flashSceneView || !['demo', 'rollout', 'live'].includes(kind)) return;
    const image = document.querySelector('#rerun-viewer .flash-replay-grid img');
    const index = Number(document.getElementById('flash-tracker-scrub').value);
    const selected = kind === 'live'
        ? {frame_index: flashSnapshot?.camera?.frame_id, snapshot: flashSnapshot}
        : flashReplay.frames[index];
    const camera = selected?.snapshot?.camera;
    if (!camera) {
        document.getElementById('flash-scene-status').textContent = kind === 'live'
            ? 'Waiting for a calibrated live frame' : 'No camera calibration was recorded at this tick';
        document.querySelector('#rerun-viewer .flash-replay-grid')?.classList.remove('flash-frame-loading');
        return;
    }
    if (kind === 'live' && camera.rgbd_available === false) {
        document.getElementById('flash-scene-status').textContent = 'Live RGB-D display frame is unavailable';
        return;
    }
    if (!image?.complete || image.naturalWidth !== camera.width ||
        image.dataset.loadedFrame !== String(selected.frame_index)) return;
    const loadingKey = `${kind}:${selected.frame_index}`;
    if (flashSceneFrameCache?.frameIndex !== selected.frame_index && flashSceneLoadingKey === loadingKey) return;
    const generation = ++flashSceneRequest;
    const replay = flashReplay;
    const status = document.getElementById('flash-scene-status');
    try {
        let frame = flashSceneFrameCache;
        if (frame?.replay !== replay || frame.frameIndex !== selected.frame_index) {
            flashSceneLoadingKey = loadingKey;
            const demo = kind === 'demo';
            const query = demo
                ? new URLSearchParams({dataset_root: replay.datasetRoot, frame_index: String(selected.frame_index)})
                : kind === 'rollout' ? new URLSearchParams({camera: replay.camera_features.depth}) : null;
            const evidenceReady = demo && replay.sceneEvidence?.available &&
                selected.frame_index < replay.sceneEvidence.frames_completed;
            const evidenceQuery = evidenceReady ? new URLSearchParams({dataset_root: replay.datasetRoot,
                bundle_path: replay.bundle_path, frame_index: String(selected.frame_index)}) : null;
            const depthUrl = demo ? `/api/run/flash/demo/depth-frame?${query}`
                : kind === 'rollout' ? `/api/datasets/${encodeURIComponent(replay.datasetId)}/episodes/${replay.episode}` +
                    `/depth-frame/${selected.frame_index}?${query}`
                    : `/api/run/flash/telemetry/frame/${selected.frame_index}/depth`;
            const [response, evidenceResponse] = await Promise.all([
                fetch(depthUrl),
                evidenceQuery ? fetch(`/api/run/flash/demo/scene-evidence/frame?${evidenceQuery}`) : null,
            ]);
            if (!response.ok) throw new Error(`Depth frame ${selected.frame_index} is unavailable`);
            if (evidenceResponse && !evidenceResponse.ok) throw new Error('Rebuilt scene evidence is unavailable');
            const [bytes, evidence] = await Promise.all([response.arrayBuffer(),
                evidenceResponse ? evidenceResponse.json() : null]);
            if (generation !== flashSceneRequest || replay !== flashReplay || flashSceneMode !== '3d') return;
            if (bytes.byteLength !== camera.width * camera.height * 2)
                throw new Error('Recorded depth dimensions do not match this camera frame');
            const paint = document.createElement('canvas');
            paint.width = camera.width; paint.height = camera.height;
            const context = paint.getContext('2d', {willReadFrequently: true});
            context.drawImage(image, 0, 0, camera.width, camera.height);
            frame = {replay, frameIndex: selected.frame_index, depth: new Uint16Array(bytes),
                rgba: context.getImageData(0, 0, camera.width, camera.height).data, evidence};
            flashSceneFrameCache = frame;
        }
        _flashPaintObservedPhoto(frame, camera);
        flashSceneView.update(frame.depth, frame.rgba, {...selected.snapshot, ...(replay?.geometry || {})},
            _flashSceneOptions(), frame.evidence);
        flashSceneView.highlight(flashHighlightedOverlay);
        if (generation === flashSceneRequest) document.querySelector('#rerun-viewer .flash-replay-grid')
            ?.classList.remove('flash-frame-loading');
        const groups = Object.entries(frame.evidence?.groups || {}).map(([role, group]) =>
            `${role} context: ${group.ids.length} ${group.verified ? 'verified' : 'candidate'} support points${group.measured_this_frame === false ? ' (fit unavailable)' : ''}`).join(' · ');
        status.textContent = frame.evidence?.available
            ? `Rebuilt RGB-D evidence · frame ${selected.frame_index}${groups ? ` · ${groups}` : ''}`
            : kind === 'demo'
                ? `Recorded RGB-D · frame ${selected.frame_index} · no rebuilt per-frame masks`
                : kind === 'live'
                    ? `Live RGB-D · frame ${selected.frame_index} · mask outlines are shown only when frame-aligned`
                    : `Recorded RGB-D · frame ${selected.frame_index} · saved pose; measured 3D surface unavailable` +
                        (selected.alignment === 'nearest_camera_time' ? ' · camera match is approximate' : '');
    } catch (error) {
        if (generation === flashSceneRequest && status) status.textContent = `3D view: ${error.message}`;
    } finally {
        if (flashSceneLoadingKey === loadingKey) flashSceneLoadingKey = null;
    }
}

async function toggleFlashSceneView() {
    const kind = flashReplay?.kind || (flashTelemetryTimer ? 'live' : null);
    if (!['demo', 'rollout', 'live'].includes(kind)) return;
    const grid = document.querySelector('#rerun-viewer .flash-replay-grid');
    const toggle = document.getElementById('flash-scene-toggle');
    const tools = document.getElementById('flash-scene-tools');
    if (flashSceneMode === '3d') {
        flashSceneMode = 'camera';
        flashSceneRequest++;
        flashSceneLoadingKey = null;
        grid.classList.remove('flash-scene-active');
        toggle.textContent = '3D scene';
        tools.hidden = true;
        return;
    }
    try {
        if (!flashSceneView) {
            let surfaces;
            if (kind === 'demo') {
                const query = new URLSearchParams({dataset_root: flashReplay.datasetRoot,
                    bundle_path: flashReplay.bundle_path});
                const response = await fetch(`/api/run/flash/demo/reference-surfaces?${query}`);
                surfaces = await response.json();
                if (!response.ok) throw new Error(surfaces.detail || 'Cannot load reference surfaces');
                const evidenceResponse = await fetch(`/api/run/flash/demo/scene-evidence/status?${query}`);
                if (evidenceResponse.ok) flashReplay.sceneEvidence = await evidenceResponse.json();
                if (flashReplay.sceneEvidence?.available) {
                    const index = Number(document.getElementById('flash-tracker-scrub').value);
                    renderFlashDemoSnapshot({...flashReplay.frames[index].snapshot, ...flashReplay.geometry});
                }
            } else {
                const pointsByRole = kind === 'live' ? flashSnapshot?.points_object_m
                    : flashReplay.frames.find(frame => Object.keys(frame.snapshot?.points_object_m || {}).length)?.snapshot.points_object_m;
                surfaces = Object.fromEntries(Object.entries(pointsByRole || {}).map(([role, points]) => [
                    role, {positions_object_m: points.flat(), colors_rgb: points.map(() => [180, 190, 200]), indices: []},
                ]));
            }
            const select = document.getElementById('flash-scene-object');
            for (const role of Object.keys(surfaces)) {
                const option = document.createElement('option');
                option.value = role;
                option.textContent = role;
                select.append(option);
            }
            const {createFlashSceneView} = await import('/static/flash_scene_view.js?v=16');
            flashSceneView = createFlashSceneView(grid.querySelector('.flash-scene-canvas'), surfaces);
            if (flashDemoEditor) {
                const point = flashDemoEditor.keypointsDocument.keypoints[flashDemoEditor.keypointIndex];
                flashSceneView.setTurnPreview(point ? _flashEditorTurnForPoint(flashDemoEditor, point) || null : null);
            }
        }
        flashSceneMode = '3d';
        grid.classList.add('flash-scene-active');
        grid.classList.add('flash-frame-loading');
        toggle.textContent = 'RGB camera';
        tools.hidden = false;
        flashSceneView.draw();
        void _flashUpdateSceneFrame();
    } catch (error) {
        const status = document.getElementById('flash-scene-status');
        if (status) status.textContent = `3D view: ${error.message}`;
    }
}

async function openFlashDemoSceneEditor() {
    if (!flashDemoEditor) await toggleFlashDemoEditor();
    if (flashDemoEditor) {
        const points = flashDemoEditor.keypointsDocument.keypoints;
        const time = _flashEditorCurrentFrame().snapshot.demo.time_s;
        if (points.length) {
            flashDemoEditor.keypointIndex = points.reduce((best, point, index) =>
                Math.abs(point.t - time) < Math.abs(points[best].t - time) ? index : best, 0);
            _flashEditorJumpTime(points[flashDemoEditor.keypointIndex].t);
        }
        _flashEditorRender();
    }
}

function _flashScenePointerUp(event) {
    if (flashSceneMode !== '3d' || !flashSceneView) return;
    const start = event.currentTarget._flashPickStart;
    if (!start || Math.hypot(start.x - event.clientX, start.y - event.clientY) > 5) return;
    const selected = flashReplay?.frames[Number(document.getElementById('flash-tracker-scrub')?.value)];
    if (flashSceneFrameCache?.frameIndex !== selected?.frame_index) return;
    const picked = flashSceneView.pickObserved(event.clientX, event.clientY);
    if (!picked) return;
    document.getElementById('flash-scene-object').value = picked.role;
    if (flashDemoEditor?.pickingTurn) {
        flashDemoEditorPickTurn(picked);
    } else if (flashDemoEditor?.pickingReference) {
        flashDemoEditorPickReference(picked.role);
    } else if (flashDemoEditor) {
        flashDemoEditor.role = picked.role;
        flashDemoEditor.section = 'objects';
        _flashEditorRender();
    }
}

function _flashUpdateWorkspace() {
    const reviewing = selectedWorkflow === 'flash' && flashUiMode === 'review' &&
        !(_isRunning && _runCommand === 'flash');
    document.getElementById('tab-run').classList.toggle('flash-reviewing', reviewing);
    document.getElementById('tab-run').classList.toggle('flash-tracking', selectedWorkflow === 'flash' && flashUiMode === 'tracking');
    const section = document.getElementById('run-section-flash');
    if (section) {
        section.dataset.mode = flashUiMode;
        section.dataset.libraryTab = flashLibraryTab;
    }
    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder && reviewing && !flashReplay) placeholder.textContent = 'Choose a demo or rollout to inspect its camera frames and spatial estimates.';
}

function selectFlashLibraryTab(tab) {
    flashLibraryTab = tab;
    _flashUpdateWorkspace();
    document.querySelectorAll('.flash-library-tab').forEach(button => {
        button.classList.toggle('active', button.dataset.tab === tab);
        button.setAttribute('aria-selected', String(button.dataset.tab === tab));
    });
}

function showFlashLiveSetup() {
    flashUiMode = 'live';
    _flashUpdateWorkspace();
    document.getElementById('run-flash-manual-setup').open = true;
}

async function showFlashReview() {
    if (_isRunning && _runCommand === 'flash') return;
    if (_isRunning && _runCommand === 'flash_tracking') await flashTrackingStop();
    flashUiMode = 'review';
    _flashUpdateWorkspace();
}

// ============================================================================
// Initialization
// ============================================================================

async function runTabInit() {
    // Before anything else, and before the run has frames: the connection
    // the pictures will arrive on. Its setup is several round trips, and
    // this is where they are spent rather than in front of the first
    // picture (design R2).
    _openLiveVideoEarly();
    if (runTabInitialized) {
        // Re-check status and reconnect SSE if needed
        _toggleHvlaRecordFields();
        await pollRunStatus();
        return;
    }
    runTabInitialized = true;
    _bindRunControlHotkeys();

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
    initFlashTrackerPanelDrag();
    renderRunForm();
    _flashUpdateWorkspace();
    if (selectedWorkflow === 'flash') await Promise.all([_loadFlashDatasets(), _loadFlashRecordings()]);
    _initSelectTitleSync();
    await _ensureModelDataLoaded();
    await pollRunStatus();

    // Backstop reconciliation: SSE events are the primary trigger for
    // re-polling, but if the stream drops at the wrong moment (server-side
    // close racing with the `done` event, browser tab backgrounded, etc.)
    // the frontend can be left thinking a long-dead subprocess is still
    // running. A low-frequency poll closes that gap without user
    // intervention; cost is one tiny GET every second.
    if (!window._runStatusPollTimer) {
        window._runStatusPollTimer = setInterval(pollRunStatus, 1000);
    }
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
    _flashUpdateWorkspace();
    document.querySelectorAll('.workflow-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.workflow === workflow);
    });
    // Toggle visibility of workflow sections (no re-render)
    for (const wf of ['teleop', 'replay', 'policy', 'flash']) {
        const section = document.getElementById(`run-section-${wf}`);
        if (section) section.style.display = (wf === workflow) ? '' : 'none';
    }
    const episodeControl = document.getElementById('run-episode-control');
    if (episodeControl) episodeControl.style.display = workflow === 'flash' ? 'none' : '';
    if (workflow === 'policy' || workflow === 'teleop') {
        await _ensureModelDataLoaded();
    }
    if (workflow === 'flash') await Promise.all([_loadFlashDatasets(), _loadFlashRecordings()]);
    const trackerPanel = document.getElementById('flash-tracker-panel');
    if (trackerPanel) trackerPanel.hidden = workflow !== 'flash';
    document.getElementById('flash-tracker-replay-controls').hidden = workflow !== 'flash' || !flashReplay;
    if (workflow === 'flash' && _isRunning && _runCommand === 'flash') startFlashTelemetry();
    _updateLaunchButton();
}

// ============================================================================
// Launch-button validation
// ============================================================================
//
// Each workflow / policy type has its own required-field rules. We keep
// them as small per-{workflow,policy} validator functions in registries
// so adding a new policy or workflow stays local — no editing of one big
// `if/else if` chain. Each validator returns either `null` (ready) or a
// string reason (not ready, shown as the Launch button tooltip).

// Helper for validators: read trimmed value from a form input/select.
function _val(id) {
    return document.getElementById(id)?.value?.trim() || '';
}

// Per-policy-type validators. Add a new entry here when adding a new
// policy with its own required fields.
const _POLICY_VALIDATORS = {
    hvla_flow_s1: () => {
        if (!_val('run-hvla-task')) {
            return 'HVLA needs a task prompt — describe what the robot should do.';
        }
        // RLT mode (when selected) needs the Phase-1 token encoder dir;
        // without it actor.pt loads with a state-dict size mismatch.
        if (_val('run-hvla-rlt-select') && !_val('run-hvla-rlt-token-ckpt')) {
            return 'RLT mode needs an RL Token Encoder checkpoint dir.';
        }
        return null;
    },
    // Vanilla (non-HVLA) policies have no extra required fields beyond the
    // checkpoint that's already validated up-stream — no entry needed.
};

// Per-workflow validators. Each gets called only after the workflow's
// universal prerequisites (robot profile, plus workflow's own primary
// selector — episode for replay, checkpoint for policy, teleop for teleop)
// pass.
const _WORKFLOW_VALIDATORS = {
    teleop: () => {
        if (!_val('run-teleop-teleop')) return 'Select a teleop profile.';
        // Recording sub-fields apply only when a dataset is selected;
        // pure teleop runs without them.
        const datasetVal = _val('run-teleop-record-dataset');
        if (datasetVal === '') return null;
        const taskSelVal = document.getElementById('run-teleop-task-select')?.value;
        const task = (taskSelVal === '__new_task__')
            ? _val('run-teleop-task-custom')
            : (taskSelVal || '').trim();
        if (!task) return 'Recording requires a task description.';
        if (datasetVal === '__new__' && !_val('run-teleop-new-dataset-name')) {
            return 'Enter a name for the new dataset.';
        }
        return null;
    },
    replay: () => {
        if (!_val('run-replay-episode')) {
            return 'Select an episode to replay (open a dataset first if the list is empty).';
        }
        return null;
    },
    policy: () => {
        if (!_val('run-policy-checkpoint')) return 'Select a model checkpoint.';
        // Delegate to the per-policy-type registry. Picks up whatever the
        // currently selected checkpoint declares its policy_type to be.
        const policyType = _getSelectedPolicyType();
        const policyValidator = _POLICY_VALIDATORS[policyType];
        if (policyValidator) {
            const reason = policyValidator();
            if (reason) return reason;
        }
        return null;
    },
    flash: () => {
        if (!_val('run-flash-root')) return 'Select an opened dataset or enter a local dataset directory.';
        const episode = _val('run-flash-episode');
        if (!/^\d+$/.test(episode)) return 'Enter a nonnegative demo episode number.';
        if (!_val('run-flash-skill')) return 'Inspect the dataset and select a prepared skill before running.';
        if (!_val('run-flash-config')) return 'Enter the local robot/camera trial config before running.';
        return null;
    },
};

let _isRunning = false;
// Kind of the active subprocess ('record', 'teleoperate', ...); null when idle.
// Episode controls and their hotkeys are record-only.
let _runCommand = null;

function _validateLaunch() {
    if (_isRunning) {
        return { ready: false, reason: 'A process is already running — Stop it first.' };
    }
    // Robot profile is the universal prerequisite for every workflow.
    if (selectedWorkflow !== 'flash' && !_val(_getActiveRobotSelectId())) {
        return { ready: false, reason: 'Select a robot profile.' };
    }
    const workflowValidator = _WORKFLOW_VALIDATORS[selectedWorkflow];
    if (!workflowValidator) {
        // Unknown workflow — let it through; backend will reject if invalid.
        return { ready: true, reason: '' };
    }
    const reason = workflowValidator();
    return reason ? { ready: false, reason } : { ready: true, reason: '' };
}

function _updateLaunchButton() {
    const btn = document.getElementById('run-launch-btn');
    if (!btn) return;
    const { ready, reason } = _validateLaunch();
    btn.disabled = !ready;
    btn.textContent = selectedWorkflow === 'flash' ? 'Run FLASH' : 'Launch';
    // Native `title` shows on hover; gives the missing-field hint without
    // needing extra inline text below the button.
    btn.title = reason;
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

async function _loadFlashDatasets() {
    const select = document.getElementById('run-flash-dataset');
    if (!select) return;
    const previous = select.value;
    try {
        const res = await fetch('/api/datasets');
        if (!res.ok) throw new Error(`Dataset list returned ${res.status}`);
        flashDatasets = await res.json();
        select.innerHTML = '<option value="">Choose an opened dataset...</option>' +
            flashDatasets.map(d => `<option value="${_esc(d.root)}">${_esc(d.repo_id)} (${d.total_episodes} episodes)</option>`).join('');
        if (flashDatasets.some(d => d.root === previous)) select.value = previous;
    } catch (e) {
        select.innerHTML = '<option value="">Dataset list unavailable; enter a local directory below</option>';
    }
}

function _onFlashDatasetChange() {
    const root = _val('run-flash-dataset');
    const input = document.getElementById('run-flash-root');
    if (input && root) input.value = root;
    const episode = document.getElementById('run-flash-episode');
    if (episode) episode.value = '0';
    _clearFlashPreparedSkills();
    _updateLaunchButton();
}

function _clearFlashPreparedSkills() {
    flashInspectGeneration += 1;
    flashPreparedSkills = [];
    const select = document.getElementById('run-flash-skill');
    if (select) select.replaceChildren(new Option('Inspect skills to choose one', ''));
    const detail = document.getElementById('run-flash-skill-detail');
    if (detail) detail.replaceChildren();
    const status = document.getElementById('run-flash-status');
    if (status) status.textContent = 'View the episode in Data, or inspect its FLASH skills here.';
    _updateLaunchButton();
}

function _flashDatasetInputError() {
    if (!_val('run-flash-root')) return 'Select a dataset or enter its local directory.';
    if (!/^\d+$/.test(_val('run-flash-episode'))) return 'Enter a nonnegative demo episode number.';
    return null;
}

async function viewFlashDemo(datasetRoot = null, episodeNumber = null, play = false) {
    if (datasetRoot !== null) {
        document.getElementById('run-flash-root').value = datasetRoot;
        document.getElementById('run-flash-episode').value = String(episodeNumber);
        const select = document.getElementById('run-flash-dataset');
        if (select && [...select.options].some(option => option.value === datasetRoot)) select.value = datasetRoot;
        _clearFlashPreparedSkills();
    }
    const error = _flashDatasetInputError();
    if (error) { showToast('Cannot view demo', error, 'warning'); return; }
    const root = _val('run-flash-root');
    const episodeIndex = Number(_val('run-flash-episode'));
    let dataset = Object.values(datasets).find(item => item.root === root);
    if (!dataset) {
        await openDataset(root);
        dataset = Object.values(datasets).find(item => item.root === root);
    }
    if (!dataset) return; // openDataset already showed the API error.
    const episode = (episodes[dataset.id] || []).find(item => item.episode_index === episodeIndex);
    if (!episode) {
        showToast('Cannot view demo', `Episode ${episodeIndex} is not in this dataset.`, 'error');
        return;
    }
    switchTab('data');
    selectEpisode(dataset.id, episodeIndex, episode.video_length || episode.length);
    if (play && !isPlaying) togglePlay();
}

let flashTelemetryTimer = null;
let flashTelemetryGeneration = 0;
let flashSnapshot = null;
let flashReplay = null;
let flashReplayTimer = null;
let flashOverlayRows = [];
let flashHighlightedOverlay = null;
let flashTrackPoints = null;
let flashSelectedTrack = null;
let flashDemoEditor = null;

function closeFlashDemoEditor() {
    flashDemoEditor = null;
    flashSceneView?.setTurnPreview(null);
    const trackerPanel = document.getElementById('flash-tracker-panel');
    const viewer = document.getElementById('rerun-viewer');
    if (trackerPanel && viewer && trackerPanel.parentElement !== viewer) {
        trackerPanel.style.removeProperty('top');
        trackerPanel.style.removeProperty('left');
        trackerPanel.style.removeProperty('right');
        trackerPanel.style.removeProperty('max-height');
        viewer.append(trackerPanel);
    }
    document.getElementById('flash-demo-editor')?.remove();
    document.querySelector('#rerun-viewer .flash-demo-mask')?.remove();
    document.querySelector('#rerun-viewer .flash-demo-polygon')?.remove();
    document.getElementById('rerun-viewer')?.classList.remove('flash-demo-editing');
    document.getElementById('tab-run')?.classList.remove('flash-demo-editing');
    const toggle = document.getElementById('flash-demo-edit-toggle');
    if (toggle) toggle.textContent = flashReplay?.kind === 'raw_demo' ? 'Prepare demo' : 'Edit demo';
}

function toggleFlashTrackerPanel() {
    const panel = document.getElementById('flash-tracker-panel');
    setFlashTrackerPanelCollapsed(!panel.classList.contains('is-collapsed'));
}

function setFlashTrackerPanelCollapsed(collapsed) {
    const panel = document.getElementById('flash-tracker-panel');
    const button = document.getElementById('flash-tracker-collapse');
    panel.classList.toggle('is-collapsed', collapsed);
    button.textContent = collapsed ? 'Pose details' : 'Hide details';
    button.setAttribute('aria-expanded', String(!collapsed));
}

function initFlashTrackerPanelDrag() {
    const viewer = document.getElementById('rerun-viewer');
    const panel = document.getElementById('flash-tracker-panel');
    const heading = document.getElementById('flash-tracker-heading');
    let drag = null;
    heading.addEventListener('pointerdown', event => {
        if (event.button !== 0 || event.target.closest('button') || flashDemoEditor || flashUiMode === 'review') return;
        const panelRect = panel.getBoundingClientRect();
        const viewerRect = viewer.getBoundingClientRect();
        drag = {
            x: event.clientX,
            y: event.clientY,
            left: panelRect.left - viewerRect.left,
            top: panelRect.top - viewerRect.top,
        };
        heading.setPointerCapture(event.pointerId);
    });
    heading.addEventListener('pointermove', event => {
        if (!drag) return;
        const left = Math.max(0, Math.min(viewer.clientWidth - panel.offsetWidth,
            drag.left + event.clientX - drag.x));
        const top = Math.max(0, Math.min(viewer.clientHeight - panel.offsetHeight,
            drag.top + event.clientY - drag.y));
        panel.style.left = `${left}px`;
        panel.style.right = 'auto';
        panel.style.top = `${top}px`;
        panel.style.maxHeight = `${Math.max(32, viewer.clientHeight - top - 8)}px`;
    });
    const endDrag = () => { drag = null; };
    heading.addEventListener('pointerup', endDrag);
    heading.addEventListener('pointercancel', endDrag);
}

function _flashSelectOverlay(next) {
    const changed = flashHighlightedOverlay?.kind !== next?.kind ||
        flashHighlightedOverlay?.role !== next?.role ||
        flashHighlightedOverlay?.estimator !== next?.estimator;
    flashHighlightedOverlay = next;
    document.querySelectorAll('#flash-tracker-content [data-flash-index]').forEach(row => {
        const selected = flashOverlayRows[Number(row.dataset.flashIndex)];
        row.classList.toggle('flash-tracker-hover', !!next && selected?.kind === next.kind &&
            selected?.role === next.role && selected?.estimator === next.estimator);
    });
    if (changed) _flashDrawOverlay();
    if (changed && flashSceneMode === '3d') flashSceneView?.highlight(next);
}

function _flashOverlayRow(kind, role, estimator, className, body) {
    const index = flashOverlayRows.push({ kind, role, estimator }) - 1;
    return `<div class="${className}" data-flash-index="${index}" tabindex="0">${body}</div>`;
}

function _flashLayerName(estimator) {
    return ({
        demo_extraction: 'Demo extraction',
        candidate_tracker: 'Tracker candidate',
        appearance_rgbd: 'Direct RGB-D',
        visible_shape_rgbd: 'Visible shape RGB-D',
        object_point_motion: 'Object points',
        environment_group_motion: 'Verified group',
        environment_support_hypothesis: 'Group hypothesis',
        stationary_prior: 'Stationary prior',
        grasp_attachment_prediction: 'Gripper attachment prediction',
    })[estimator] || estimator.replaceAll('_', ' ');
}

function _flashLayerColor(estimator) {
    if (estimator === 'candidate_tracker') return '#f472b6';
    if (estimator === 'raw_proposal') return '#f472b6';
    if (estimator === 'demo_reference') return '#34d399';
    if (estimator === 'demo_held' || estimator === 'demo_held_recorded_gripper') return '#fb923c';
    if (estimator === 'demo_ee') return '#58a6ff';
    if (estimator === 'object_point_motion') return '#fbbf24';
    if (estimator === 'environment_group_motion') return '#34d399';
    if (estimator === 'environment_support_hypothesis') return '#a3e635';
    if (estimator === 'stationary_prior') return '#94a3b8';
    if (estimator === 'grasp_attachment_prediction') return '#fb923c';
    return '#58a6ff';
}

function _flashPoseText(pose, snapshot) {
    if (!pose) return 'No estimate';
    const age = Number(snapshot.recorded_at_s) - Number(pose.source_time_s);
    const freshness = Number.isFinite(age) ? ` · ${Math.max(0, age).toFixed(2)} s old` : '';
    const xyz = pose.pose ? ` · ${pose.pose.slice(0, 3).map((row, i) => (row[3] * 1000).toFixed(0)).join(', ')} mm` : '';
    return `${pose.evidence || 'unknown'}${freshness}${xyz}`;
}

function renderFlashSnapshot(snapshot, mode = 'Live') {
    flashSnapshot = snapshot;
    const panel = document.getElementById('flash-tracker-panel');
    const content = document.getElementById('flash-tracker-content');
    const modeEl = document.getElementById('flash-tracker-mode');
    if (!panel || !content || !snapshot) return;
    panel.hidden = selectedWorkflow !== 'flash';
    panel.querySelector('.flash-tracker-heading strong').textContent = 'FLASH tracker inspection';
    modeEl.textContent = mode;
    const roles = [...new Set([
        ...Object.keys(snapshot.proposals || {}),
        ...Object.keys(snapshot.raw_proposals || {}),
        ...Object.keys(snapshot.layers || {}),
        ...Object.keys(snapshot.accepted || {}),
        ...Object.keys(snapshot.observed_masks || {}),
    ])].sort();
    if (!roles.length) {
        content.textContent = 'Waiting for a pose proposal. The camera and motor loops can already be running.';
        _flashDrawOverlay();
        return;
    }
    flashOverlayRows = [];
    let html = '<div class="flash-tracker-legend">Hover a row to highlight it in the camera view. Solid outline: control pose · thin outlines: other estimates · purple: visible mask when its camera frame matches.</div>';
    for (const role of roles) {
        html += `<div class="flash-tracker-role"><strong>${_esc(role)}</strong>`;
        const mask = snapshot.observed_masks?.[role];
        if (mask && snapshot.camera && mask.frame_id === snapshot.camera.frame_id) {
            html += _flashOverlayRow('mask', role, '', 'flash-tracker-layer',
                '<span class="flash-tracker-dot" style="background:#d48bff"></span>Visible mask <small>Measured pixels in this camera frame</small>');
        } else {
            html += '<div class="flash-tracker-layer flash-tracker-unavailable">Visible mask <small>No mask for this camera frame</small></div>';
        }
        const proposal = snapshot.proposals?.[role];
        if (proposal) html += _flashOverlayRow('proposal', role, '', 'flash-tracker-layer',
            `<span class="flash-tracker-dot" style="background:#58a6ff"></span>Direct proposal <small>${_esc(_flashPoseText(proposal, snapshot))}</small>`);
        const raw = snapshot.raw_proposals?.[role];
        if (raw?.pose && proposal?.evidence === 'lost') {
            const shown = raw.frame_id === snapshot.camera?.frame_id &&
                (!snapshot.camera.calibration_version || raw.calibration_version === snapshot.camera.calibration_version);
            html += _flashOverlayRow('raw_proposal', role, '', 'flash-tracker-layer',
                `<span class="flash-tracker-dot" style="background:#f472b6"></span>Rejected raw fit` +
                `<small>${_esc(raw.failure || 'Not accepted by vision')} · camera frame ${_esc(String(raw.frame_id))}` +
                `${shown ? '' : ' · open its camera frame to see the outline'}</small>`);
        }
        for (const [name, pose] of Object.entries(snapshot.layers?.[role] || {})) {
            html += _flashOverlayRow('layer', role, name, 'flash-tracker-layer',
                `<span class="flash-tracker-dot" style="background:${_flashLayerColor(name)}"></span>${_esc(_flashLayerName(name))}<small>${_esc(_flashPoseText(pose, snapshot))}</small>`);
        }
        const accepted = snapshot.accepted?.[role];
        const held = snapshot.held_prediction?.role === role ? snapshot.held_prediction : null;
        if (held && accepted) html += _flashOverlayRow('accepted', role, '', 'flash-tracker-layer',
            `Independent estimate <small>${_esc(_flashLayerName(accepted.estimator || 'unspecified'))}: ${_esc(_flashPoseText(accepted, snapshot))}</small>`);
        const planningPose = held || accepted;
        const selectedText = `${held ? 'Planner prediction (rigid grasp)' : 'Accepted estimator pose'}: ${planningPose ? _esc(_flashLayerName(planningPose.estimator || 'unspecified')) + '<small>' + _esc(_flashPoseText(planningPose, snapshot)) + (held && !held.last_direct_observation_s ? ' · unverified after grasp' : '') + '</small>' : 'none'}`;
        html += planningPose
            ? _flashOverlayRow(held ? 'held' : 'accepted', role, '', 'flash-tracker-selected', selectedText)
            : `<div class="flash-tracker-selected">${selectedText}</div>`;
        const decision = snapshot.anchor_decisions?.[role];
        if (decision) {
            const reasons = {
                accepted: 'Pose candidate accepted',
                accepted_historical_group: 'Older pose verified scene motion; newer pose retained',
                missing_mask: 'No matching object mask',
                lost_pose: 'No usable direct pose',
                time_mismatch: 'Pose and camera times differ',
                source_time_mismatch: 'Result and camera times differ',
                calibration_mismatch: 'Camera calibration differs',
                frame_history_missing: 'Camera frame expired before the pose arrived',
                scene_history_missing: 'Scene frame expired before the pose arrived',
                direct_symmetry_invalid: 'Pose could not be aligned with the object rule',
                superseded_by_direct: 'A newer direct pose was already used',
                direct_group_unresolved: 'Direct pose disagrees with tracked scene motion',
                direct_point_unresolved: 'Direct pose disagrees with tracked object points',
            };
            const label = reasons[decision.outcome] || decision.outcome.replaceAll('_', ' ');
            html += `<div class="flash-tracker-comparison">${_esc(label)}` +
                `<small>Candidate from camera frame ${_esc(String(decision.frame_id))}` +
                ` · ${_esc(decision.candidate?.estimator || 'unknown source')}</small></div>`;
        }
        const comparison = snapshot.direct_group_comparisons?.[role];
        if (comparison) {
            const age = Math.max(0, Number(snapshot.recorded_at_s) - Number(comparison.available_at_s));
            const separation = Number(comparison.object_center_difference_m) * 1000;
            html += `<div class="flash-tracker-comparison">Latest direct/group check: ${_esc(comparison.verdict.replaceAll('_', ' '))}`
                + `<small>${separation.toFixed(1)} mm between centres · ${age.toFixed(1)} s ago</small></div>`;
        }
        const pointComparison = snapshot.direct_point_comparisons?.[role];
        if (pointComparison) {
            const age = Math.max(0, Number(snapshot.recorded_at_s) - Number(pointComparison.available_at_s));
            const separation = Number(pointComparison.object_center_difference_m) * 1000;
            html += `<div class="flash-tracker-comparison">Latest direct/points check: ${_esc(pointComparison.verdict.replaceAll('_', ' '))}`
                + `<small>${separation.toFixed(1)} mm between centres · ${age.toFixed(1)} s ago</small></div>`;
        }
        const depthCheck = snapshot.direct_free_space_comparisons?.[role];
        if (depthCheck) {
            const age = Math.max(0, Number(snapshot.recorded_at_s) - Number(depthCheck.available_at_s));
            const signedGap = depthCheck.support?.median_signed_gap_m;
            const gapText = signedGap == null ? 'no valid depth' : `${(Number(signedGap) * 1000).toFixed(1)} mm median signed gap`;
            html += `<div class="flash-tracker-comparison">Latest direct/depth diagnostic: ${_esc(depthCheck.verdict.replaceAll('_', ' '))}`
                + `<small>${_esc(gapText)} · ${Number(depthCheck.support?.valid_depth_pixels || 0)} rays · ${age.toFixed(1)} s ago · diagnostic only</small></div>`;
        }
        html += '</div>';
    }
    content.innerHTML = html;
    if (flashHighlightedOverlay && !flashOverlayRows.some(row => row.kind === flashHighlightedOverlay.kind &&
        row.role === flashHighlightedOverlay.role && row.estimator === flashHighlightedOverlay.estimator)) {
        flashHighlightedOverlay = null;
    }
    _flashSelectOverlay(flashHighlightedOverlay);
    _flashBindOverlayRows(content);
    _flashDrawOverlay();
}

function _flashBindOverlayRows(content) {
    const hoveredRow = event => {
        const element = event.target.closest('[data-flash-index]');
        _flashSelectOverlay(element && content.contains(element)
            ? flashOverlayRows[Number(element.dataset.flashIndex)] : null);
    };
    content.onpointermove = hoveredRow;
    content.onpointerleave = () => _flashSelectOverlay(null);
    content.onfocusin = hoveredRow;
    content.onfocusout = event => {
        if (!content.contains(event.relatedTarget)) _flashSelectOverlay(null);
    };
}

function renderFlashDemoSnapshot(snapshot) {
    flashSnapshot = snapshot;
    flashOverlayRows = [];
    const panel = document.getElementById('flash-tracker-panel');
    const content = document.getElementById('flash-tracker-content');
    panel.hidden = selectedWorkflow !== 'flash';
    panel.querySelector('.flash-tracker-heading strong').textContent = 'FLASH demo extraction';
    document.getElementById('flash-tracker-mode').textContent = 'Demo';
    const demo = snapshot.demo;
    const rebuilt = flashReplay.sceneEvidence?.available;
    let html = `<div class="flash-tracker-legend">Demo time ${demo.time_s.toFixed(2)} s · arm reading ${(demo.camera_arm_delta_s * 1000).toFixed(0)} ms from this RGB frame. ${rebuilt ? 'The 3D view shows object surfaces and group points rebuilt from this recording.' : 'The 3D view can project partial reference patches; per-frame masks were not saved.'} Prepared skill poses and extraction estimates are separate.</div>`;
    if (flashReplay.extraction_history !== 'recorded_for_prepared_version' && !rebuilt)
        html += '<div class="flash-tracker-unavailable">Per-frame extraction and group support: not recorded for this prepared version; no mask here does not mean segmentation was lost.</div>';
    if (flashReplay.tracker_candidate) {
        if (flashTrackPoints?.available && flashTrackPoints.candidate_id === flashReplay.tracker_candidate.candidate_id) {
            html += flashTrackPoints.surface_filtered
                ? '<div class="flash-tracker-legend">Dots and trails are tracked points visible on the SAM3 mask and agreeing with measured RGB-D depth at this camera frame. All points use the same color. Filled dots agree with the pose propagated from their birth-frame anchor; hollow dots disagree. Click a dot to follow its visible path, object and birth frame.</div>'
                : '<div class="flash-tracker-legend">Dots and trails are the model’s raw point tracks, projected into this RGB frame without the rigid fit. Filled dots agree with the fitted pose within its registration tolerance; hollow dots disagree. Click a dot to follow that point’s full visible path. Fit agreement is not evidence that a point is on the object.</div>';
        }
        for (const role of flashReplay.tracker_candidate.roles || [flashReplay.tracker_candidate.role]) {
            const proposal = snapshot.layers?.[role]?.candidate_tracker;
            html += `<div class="flash-tracker-role"><strong>${_esc(flashReplay.tracker_candidate.tracker)} proposal · ${_esc(role)}</strong>`;
            html += proposal
                ? _flashOverlayRow('layer', role, 'candidate_tracker', 'flash-tracker-layer',
                    `<span class="flash-tracker-dot" style="background:#f472b6"></span>Candidate pose on this camera frame <small>Preview only; hover to highlight the pink outline</small>`)
                : '<div class="flash-tracker-unavailable">No candidate pose at this camera frame</div>';
            const pointRole = flashTrackPoints?.roles?.find(item => item.role === role);
            const pointFrame = flashTrackPoints?.frameLookup?.get(flashReplay.frames[Number(document.getElementById('flash-tracker-scrub').value)]?.frame_index);
            if (pointRole && pointFrame !== undefined) {
                const current = pointRole.positions[pointFrame];
                const visible = current.filter(Boolean).length;
                const selected = flashSelectedTrack?.role === role
                    ? ` · selected point ${flashSelectedTrack.index}` : '';
                html += _flashOverlayRow('raw_track', role, '', 'flash-tracker-layer',
                    `<span class="flash-tracker-dot" style="background:${_flashTrackColor()}"></span>` +
                    `${flashTrackPoints.surface_filtered ? 'Visible surface point paths' : 'Raw 3D point paths'} <small>${visible}/${pointRole.count} ${flashTrackPoints.surface_filtered ? 'mask + depth supported' : 'model-visible'}${selected}; hover to isolate</small>`);
            } else if (flashReplay.tracker_candidate.tracker === 'trackeverything') {
                html += `<div class="flash-tracker-unavailable">${flashTrackPoints?.available
                    ? 'No raw point tracks at this camera frame' : 'Loading raw point tracks…'}</div>`;
            }
            html += '</div>';
        }
    }
    for (const [role, layers] of Object.entries(snapshot.layers || {})) {
        const pose = layers.demo_extraction;
        if (!pose) continue;
        const lastDirect = pose.last_direct_observation_s == null ? ''
            : ` · last direct observation ${Math.max(0, demo.time_s - pose.last_direct_observation_s).toFixed(1)} s ago`;
        const used = snapshot.held_prediction?.role === role ? snapshot.held_prediction : snapshot.accepted?.[role];
        const center = snapshot.centers_object_m?.[role];
        const worldCenter = estimate => [0, 1, 2].map(i => estimate.pose[i][3] +
            [0, 1, 2].reduce((sum, j) => sum + estimate.pose[i][j] * center[j], 0));
        const separation = pose.pose && used?.pose && center
            ? ` · ${(Math.hypot(...worldCenter(pose).map((value, i) => value - worldCenter(used)[i])) * 1000).toFixed(1)} mm from skill pose`
            : '';
        const detail = pose.evidence === 'lost' ? `Lost${lastDirect}`
            : `${_flashLayerName(pose.estimator)} · ${_flashPoseText(pose, snapshot)}${lastDirect}${separation}`;
        const predictionNote = pose.evidence === 'predicted' && pose.estimator === 'appearance_rgbd'
            ? 'Visual evidence contributed, but the full pose remained predicted. ' : '';
        const originNote = rebuilt
            ? 'Current masks and group supports were rebuilt with the same perception pipeline; the saved skill was unchanged.'
            : 'Per-frame masks and supporting group points were not stored.';
        html += `<div class="flash-tracker-role"><strong>${_esc(role)} · extraction</strong>`;
        html += _flashOverlayRow('layer', role, 'demo_extraction', 'flash-tracker-layer',
            `<span class="flash-tracker-dot" style="background:${_flashLayerColor(pose.estimator)}"></span>${_esc(detail)}<small>${_esc(predictionNote)}${_esc(originNote)}</small>`);
        html += '</div>';
    }
    if (demo.phase === 'outside_skill') {
        html += '<div class="flash-tracker-role">No spatial skill sample applies to this part of the recording.</div>';
    } else {
        const mode = demo.phase === 'connection_endpoint'
            ? 'Generated connection: its target pose is the saved endpoint goal, not a measurement at this frame. The held outline follows the recorded gripper with the predicted grasp attachment.'
            : 'Demonstrated motion: this sample and its object relation were retained.';
        html += `<div class="flash-tracker-role"><strong>${_esc(demo.interval_name)}</strong><small>${_esc(mode)}</small>`;
        html += `<div>${_esc(demo.relation.replaceAll('_', ' '))} · arm sample ${demo.source_arm_frame_index} at ${demo.sample_time_s.toFixed(2)} s</div>`;
        if (demo.event_id) html += `<div>Event: ${_esc(demo.event_id)}</div>`;
        html += '</div>';
        if (demo.reference_role === '__start__') {
            html += '<div class="flash-tracker-role">Motion is anchored to the achieved grasp or interval entry; this interval uses no object reference pose.</div>';
        } else {
            const role = demo.reference_role;
            const pose = snapshot.accepted[role];
            const constraint = flashReplay.pose_constraints?.[role];
            const graspReference = flashReplay.grasp_reference;
            const evidenceNote = constraint?.orientation_locked_to_baseline
                ? `${pose.evidence}; orientation locked to a baseline. Fresh versus carried translation is not saved here.`
                : `${pose.evidence}; ${graspReference?.role === role && demo.time_s <= graspReference.grasp_time_s
                    ? `pre-grasp seed from camera frame ${graspReference.camera_frame_index}` : 'source proposal history not saved'}`;
            html += `<div class="flash-tracker-role"><strong>${_esc(role)}</strong>`;
            html += _flashOverlayRow('accepted', role, '', 'flash-tracker-selected',
                `<span class="flash-tracker-dot" style="background:#34d399"></span>Reference pose used by skill <small>${_esc(evidenceNote)} · sampled at ${pose.source_time_s.toFixed(2)} s</small>`);
            html += '</div>';
        }
        if (snapshot.held_prediction) {
            const held = snapshot.held_prediction;
            const heldNote = held.estimator === 'demo_held_recorded_gripper'
                ? 'predicted grasp attachment projected from this frame’s recorded gripper; not independently measured'
                : `${held.evidence} from grasp attachment; not independently measured here`;
            html += `<div class="flash-tracker-role"><strong>${_esc(held.role)}</strong>`;
            html += _flashOverlayRow('held', held.role, '', 'flash-tracker-selected',
                `<span class="flash-tracker-dot" style="background:#fb923c"></span>Held pose shown in demo <small>${_esc(heldNote)}</small>`);
            html += '</div>';
        }
    }
    html += _flashOverlayRow('accepted', '__gripper__', '', 'flash-tracker-role',
        '<span class="flash-tracker-dot" style="background:#58a6ff"></span>Recorded gripper position');
    content.innerHTML = html;
    if (flashHighlightedOverlay && !flashOverlayRows.some(row => row.kind === flashHighlightedOverlay.kind &&
        row.role === flashHighlightedOverlay.role && row.estimator === flashHighlightedOverlay.estimator)) {
        flashHighlightedOverlay = null;
    }
    _flashSelectOverlay(flashHighlightedOverlay);
    _flashBindOverlayRows(content);
    _flashDrawOverlay();
}

function renderFlashRawSnapshot(snapshot) {
    flashSnapshot = snapshot;
    flashOverlayRows = [];
    flashHighlightedOverlay = null;
    const panel = document.getElementById('flash-tracker-panel');
    panel.hidden = selectedWorkflow !== 'flash';
    panel.querySelector('.flash-tracker-heading strong').textContent = 'Raw demo';
    document.getElementById('flash-tracker-mode').textContent = 'Unprepared';
    document.getElementById('flash-tracker-content').textContent =
        'No object poses have been extracted. Use Prepare demo to name the objects, mark the action and create a reviewable version.';
    _flashDrawOverlay();
}

function _flashProject(point, pose, camera) {
    if (!pose?.pose || !camera?.base_from_camera || !camera?.intrinsics) return null;
    const t = pose.pose;
    const world = [0, 1, 2].map(i => t[i][3] + t[i][0] * point[0] + t[i][1] * point[1] + t[i][2] * point[2]);
    const base = camera.base_from_camera;
    const delta = world.map((v, i) => v - base[i][3]);
    const cam = [0, 1, 2].map(i => base[0][i] * delta[0] + base[1][i] * delta[1] + base[2][i] * delta[2]);
    if (cam[2] <= 0) return null;
    const k = camera.intrinsics;
    return [k[0][0] * cam[0] / cam[2] + k[0][2], k[1][1] * cam[1] / cam[2] + k[1][2]];
}

function _flashHull(points) {
    const sorted = points.filter(point => point && point.every(Number.isFinite))
        .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (sorted.length < 3) return sorted;
    const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const lower = [], upper = [];
    for (const point of sorted) {
        while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
        lower.push(point);
    }
    for (const point of [...sorted].reverse()) {
        while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
        upper.push(point);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
}

function _flashTrackColor() {
    return '#22d3ee';
}

function _flashRawTrackFrame() {
    if (!flashTrackPoints?.available || !flashReplay?.frames?.length) return null;
    const replayFrame = flashReplay.frames[Number(document.getElementById('flash-tracker-scrub').value)];
    return flashTrackPoints.frameLookup.get(replayFrame?.frame_index);
}

function _flashDrawPointTracks(context, pixel) {
    if (!document.getElementById('flash-track-visible')?.checked) return;
    const at = _flashRawTrackFrame();
    if (at === undefined || at === null) return;
    const seconds = Number(document.getElementById('flash-track-trail')?.value || 0);
    const trailFrames = Math.round(seconds * (flashReplay.fps || 30));
    for (const role of flashTrackPoints.roles) {
        const selectedRole = flashSelectedTrack?.role === role.role;
        const isolated = flashHighlightedOverlay?.kind === 'raw_track' &&
            (flashHighlightedOverlay.role === role.role || flashHighlightedOverlay.role === null);
        const dimmed = !!flashHighlightedOverlay && !isolated;
        const current = role.positions[at];
        for (let id = 0; id < role.count; id++) {
            const pointColor = _flashTrackColor();
            const selected = selectedRole && flashSelectedTrack.index === id;
            const alpha = dimmed || (flashSelectedTrack && !selected) ? 0.1 : 1;
            const start = selected ? 0 : Math.max(0, at - trailFrames);
            if (selected || trailFrames) {
                context.strokeStyle = pointColor;
                context.lineWidth = selected ? 2.5 : 1;
                context.globalAlpha = alpha * (selected ? 0.9 : 0.25);
                context.beginPath();
                let connected = false;
                for (let frame = start; frame <= at; frame++) {
                    const point = role.positions[frame][id];
                    if (!point) { connected = false; continue; }
                    const xy = pixel(point);
                    if (connected) context.lineTo(...xy);
                    else context.moveTo(...xy);
                    connected = true;
                }
                context.stroke();
            }
            const point = current[id];
            if (!point) continue;
            const [x, y] = pixel(point);
            const agreesWithFit = point[2] !== null && point[2] <= flashTrackPoints.fit_threshold_mm;
            context.globalAlpha = alpha;
            context.beginPath();
            context.arc(x, y, selected ? 5 : 2.5, 0, 2 * Math.PI);
            if (agreesWithFit) {
                context.fillStyle = pointColor;
                context.fill();
            } else {
                context.strokeStyle = pointColor;
                context.lineWidth = selected ? 2.5 : 1.5;
                context.stroke();
            }
            if (selected) {
                context.font = 'bold 12px sans-serif';
                context.fillStyle = '#fff';
                context.fillText(`${role.role} point ${id} · born frame ${role.birth_frames?.[id] ?? '?'} · ${point[2] === null ? 'no fit' : `${point[2]} mm from fit`}`, x + 9, y - 9);
            }
        }
    }
    context.globalAlpha = 1;
}

function _flashSelectPointAt(event) {
    if (flashDemoEditor?.drawing || !flashTrackPoints?.available || !document.getElementById('flash-track-visible')?.checked) return;
    const at = _flashRawTrackFrame();
    if (at === undefined || at === null) return;
    const canvas = document.querySelector('#rerun-viewer .flash-replay-grid .flash-pose-canvas');
    const camera = flashSnapshot?.camera;
    if (!canvas || !camera?.width || !camera?.height) return;
    const bounds = canvas.getBoundingClientRect();
    const scale = Math.min(bounds.width / camera.width, bounds.height / camera.height);
    const left = (bounds.width - camera.width * scale) / 2;
    const top = (bounds.height - camera.height * scale) / 2;
    const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    let closest = null, distance = 12;
    for (const role of flashTrackPoints.roles) {
        for (const [index, point] of role.positions[at].entries()) {
            if (!point) continue;
            const gap = Math.hypot(x - left - point[0] * scale, y - top - point[1] * scale);
            if (gap < distance) { closest = {role: role.role, index}; distance = gap; }
        }
    }
    flashSelectedTrack = closest;
    if (closest && flashReplayTimer) toggleFlashReplay();
    const selected = flashReplay.frames[Number(document.getElementById('flash-tracker-scrub').value)];
    renderFlashDemoSnapshot({ ...selected.snapshot, ...flashReplay.geometry });
}

function _flashDrawOverlay() {
    const snapshot = flashSnapshot;
    const camera = snapshot?.camera;
    document.querySelectorAll('.flash-pose-canvas').forEach(canvas => {
        const context = canvas.getContext('2d');
        const width = canvas.clientWidth, height = canvas.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        context.clearRect(0, 0, width, height);
        if (!camera?.width || !camera?.height) return;
        const scale = Math.min(width / camera.width, height / camera.height);
        const left = (width - camera.width * scale) / 2;
        const top = (height - camera.height * scale) / 2;
        const pixel = point => [left + point[0] * scale, top + point[1] * scale];
        const outline = (points, color, width, dashed = false) => {
            if (points.length < 3) return;
            context.beginPath();
            context.moveTo(...pixel(points[0]));
            for (const point of points.slice(1)) context.lineTo(...pixel(point));
            context.closePath();
            context.strokeStyle = color;
            context.lineWidth = width;
            context.setLineDash(dashed ? [5, 4] : []);
            context.stroke();
            context.setLineDash([]);
        };
        const emphasized = (kind, role, estimator = '') => !flashHighlightedOverlay ||
            (flashHighlightedOverlay.kind === kind &&
                (flashHighlightedOverlay.role === null || flashHighlightedOverlay.role === role) &&
                flashHighlightedOverlay.estimator === estimator);
        const draw = (role, name, pose, accepted, kind, estimator = '') => {
            const active = emphasized(kind, role, estimator);
            context.globalAlpha = active ? 1 : 0.12;
            const points = snapshot.points_object_m?.[role] || [];
            const projectedPoints = points.map(point => _flashProject(point, pose, camera)).filter(Boolean);
            if (projectedPoints.length >= 3) {
                outline(_flashHull(projectedPoints), _flashLayerColor(name),
                    flashHighlightedOverlay && active ? 5 : accepted ? 2.5 : 1.5, !accepted);
                if (accepted) {
                    const previousAlpha = context.globalAlpha;
                    context.globalAlpha = flashReplay?.kind === 'demo' ? previousAlpha * 0.45 : previousAlpha;
                    context.fillStyle = flashReplay?.kind === 'demo' ? _flashLayerColor(name) : '#d8f8ff';
                    for (const point of projectedPoints) {
                        const [x, y] = pixel(point);
                        context.fillRect(x - 1, y - 1, 2, 2);
                    }
                    context.globalAlpha = previousAlpha;
                }
            }
            const center = snapshot.centers_object_m?.[role] || [0, 0, 0];
            const projected = _flashProject(center, pose, camera);
            if (!projected) { context.globalAlpha = 1; return; }
            const [x, y] = pixel(projected);
            if (!projectedPoints.length) {
                context.strokeStyle = _flashLayerColor(name);
                context.lineWidth = accepted ? 2.5 : 1.5;
                context.beginPath(); context.arc(x, y, accepted ? 9 : 5, 0, 2 * Math.PI); context.stroke();
            }
            if (accepted) {
                context.font = 'bold 13px sans-serif';
                context.fillStyle = '#fff';
                context.fillText(role === '__gripper__' ? 'gripper' : role, x + 8, y - 8);
            }
            context.globalAlpha = 1;
        };
        for (const [role, layers] of Object.entries(snapshot.layers || {})) {
            Object.entries(layers).forEach(([name, pose]) => {
                if (name !== 'candidate_tracker') draw(role, name === 'demo_extraction' ? pose.estimator : name,
                    pose, false, 'layer', name);
            });
        }
        for (const [role, pose] of Object.entries(snapshot.accepted || {})) {
            if (snapshot.held_prediction?.role === role) draw(role, pose.estimator || 'accepted', pose, false, 'accepted');
            else draw(role, pose.estimator || 'accepted', pose, true, 'accepted');
        }
        if (snapshot.held_prediction) {
            const held = snapshot.held_prediction;
            draw(held.role, held.estimator, held, true, 'held');
        }
        for (const [role, pose] of Object.entries(snapshot.proposals || {})) {
            const aligned = snapshot.proposal_frame_id != null
                ? snapshot.proposal_frame_id === camera.frame_id
                : pose.source_time_s === camera.source_time_s;
            if (aligned) draw(role, 'appearance_rgbd', pose, false, 'proposal');
        }
        for (const [role, raw] of Object.entries(snapshot.raw_proposals || {})) {
            if (snapshot.proposals?.[role]?.evidence !== 'lost' || !raw.pose ||
                raw.frame_id !== camera.frame_id ||
                (camera.calibration_version && raw.calibration_version !== camera.calibration_version)) continue;
            draw(role, 'raw_proposal', {pose: raw.pose}, false, 'raw_proposal');
        }
        for (const [role, mask] of Object.entries(snapshot.observed_masks || {})) {
            if (mask.frame_id !== camera.frame_id) continue;
            const active = emphasized('mask', role);
            context.globalAlpha = active ? 1 : 0.12;
            outline(mask.boundary_xy || [], '#d48bff', flashHighlightedOverlay && active ? 5 : 2);
            context.globalAlpha = 1;
        }
        // Review candidates stay visible when they closely overlap the saved
        // surface. Hovering their row still isolates them from all other layers.
        for (const [role, layers] of Object.entries(snapshot.layers || {})) {
            if (layers.candidate_tracker) draw(role, 'candidate_tracker', layers.candidate_tracker,
                false, 'layer', 'candidate_tracker');
        }
        _flashDrawPointTracks(context, pixel);
    });
}

function startFlashTelemetry() {
    if (flashTelemetryTimer) return;
    const generation = ++flashTelemetryGeneration;
    closeFlashDemoEditor();
    flashSceneRequest++;
    flashSceneFrameCache = null;
    flashSceneLoadingKey = null;
    flashSceneView?.dispose();
    flashSceneView = null;
    flashSceneMode = 'camera';
    flashReplay = null;
    flashSnapshot = null;
    clearInterval(flashReplayTimer); flashReplayTimer = null;
    stopObsStreamViewer();
    flashUiMode = 'live';
    _flashUpdateWorkspace();
    setFlashTrackerPanelCollapsed(true);
    const grid = _flashCreateSceneGrid('live', 'Live aligned RGB-D camera');
    const image = grid.querySelector('img');
    document.getElementById('flash-tracker-replay-controls').hidden = true;
    let busy = false;
    const poll = async () => {
        if (busy) return;
        busy = true;
        try {
            const response = await fetch('/api/run/flash/telemetry/live');
            const data = await response.json();
            if (generation !== flashTelemetryGeneration || !flashTelemetryTimer) return;
            if (!data.available) return;
            const snapshot = data.snapshot;
            const camera = snapshot.camera;
            const frameId = camera?.frame_id;
            const ready = camera?.rgbd_available === true;
            document.getElementById('flash-scene-toggle').disabled = !ready ||
                !Object.keys(snapshot.points_object_m || {}).length;
            if (!ready) {
                document.getElementById('flash-scene-status').textContent =
                    'Waiting for an aligned live RGB-D display frame';
                return;
            }
            if (image.dataset.loadedFrame === String(frameId)) {
                renderFlashSnapshot(snapshot, 'Live');
                if (flashSceneMode === '3d') void _flashUpdateSceneFrame();
                return;
            }
            image._flashPendingSnapshot = snapshot;
            if (image.dataset.requestedFrame === String(frameId)) return;
            flashSceneRequest++;
            grid.classList.add('flash-frame-loading');
            image.dataset.requestedFrame = String(frameId);
            image.onload = () => {
                if (image.dataset.requestedFrame !== String(frameId) ||
                    generation !== flashTelemetryGeneration || !flashTelemetryTimer) return;
                image.dataset.loadedFrame = String(frameId);
                renderFlashSnapshot(image._flashPendingSnapshot, 'Live');
                if (flashSceneMode === '3d') void _flashUpdateSceneFrame();
                else grid.classList.remove('flash-frame-loading');
            };
            image.onerror = () => {
                if (image.dataset.requestedFrame !== String(frameId) ||
                    generation !== flashTelemetryGeneration) return;
                image.dataset.requestedFrame = '';
                grid.classList.remove('flash-frame-loading');
                document.getElementById('flash-scene-status').textContent =
                    `Live display frame ${frameId} expired; waiting for the next frame`;
            };
            image.src = `/api/run/flash/telemetry/frame/${frameId}/rgb`;
        } catch (error) { console.warn('FLASH tracker stream:', error); }
        finally { busy = false; }
    };
    flashTelemetryTimer = setInterval(poll, 200);
    void poll();
}

function stopFlashTelemetry() {
    flashTelemetryGeneration++;
    if (flashTelemetryTimer) clearInterval(flashTelemetryTimer);
    flashTelemetryTimer = null;
}

async function loadFlashTrackerReplay(datasetRoot, episodeIndex) {
    const root = datasetRoot || _val('run-flash-root');
    const episode = episodeIndex ?? Number(_val('run-flash-episode'));
    if (!root || !Number.isInteger(episode) || episode < 0) {
        showToast('Cannot inspect FLASH recording', 'Select a local dataset and episode.', 'warning'); return;
    }
    try {
        const response = await fetch(`/api/run/flash/telemetry/replay?dataset_root=${encodeURIComponent(root)}&episode=${episode}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot read tracker trace');
        let dataset = Object.values(datasets).find(item => item.root === root);
        if (!dataset) {
            await openDataset(root);
            dataset = Object.values(datasets).find(item => item.root === root);
        }
        if (!dataset) return;
        _flashInstallReplay({ ...result, datasetId: dataset.id, episode }, 'Recorded overhead camera');
        const firstTrackedFrame = result.frames.findIndex(frame =>
            frame.snapshot?.camera && Object.keys(frame.snapshot?.accepted || {}).length);
        seekFlashReplay(Math.max(0, firstTrackedFrame));
        if (!result.layer_history) showToast('Older FLASH recording', 'This trace contains planner poses, but predates per-tracker layer recording.', 'warning');
    } catch (error) { showToast('Cannot inspect FLASH recording', error.message, 'error'); }
}

function _flashCreateSceneGrid(kind, imageAlt, depthAvailable = true) {
    const viewer = document.getElementById('rerun-viewer');
    viewer.querySelector('.flash-replay-grid')?.remove();
    const grid = document.createElement('div');
    grid.className = 'flash-replay-grid';
    const image = document.createElement('img');
    image.alt = imageAlt;
    const canvas = document.createElement('canvas');
    canvas.className = 'flash-pose-canvas';
    grid.append(image, canvas);
    if (kind === 'demo' || kind === 'live' || (kind === 'rollout' && depthAvailable)) {
        const photo = document.createElement('canvas');
        photo.className = 'flash-scene-photo';
        const scene = document.createElement('canvas');
        scene.className = 'flash-scene-canvas';
        scene.addEventListener('pointerdown', event => {
            scene._flashPickStart = {x: event.clientX, y: event.clientY};
        });
        scene.addEventListener('pointerup', _flashScenePointerUp);
        const toolbar = document.createElement('div');
        toolbar.className = 'flash-scene-toolbar';
        toolbar.innerHTML = `<button type="button" id="flash-scene-toggle" onclick="toggleFlashSceneView()"${kind === 'live' ? ' disabled' : ''}>3D scene</button>` +
            '<div id="flash-scene-tools" hidden>' +
            '<select id="flash-scene-object" aria-label="Focus object" onchange="flashSceneView?.focus(this.value)"><option value="">Focus object</option></select>' +
            '<button type="button" onclick="flashSceneView?.reset()">Reset view</button>' +
            (kind === 'demo'
                ? '<button type="button" id="flash-scene-edit" onclick="openFlashDemoSceneEditor()">Edit key frames</button>'
                : '') +
            '<details class="flash-scene-layer-menu"><summary>Layers</summary><div class="flash-scene-layer-options">' +
            (kind === 'demo'
                ? '<label title="Current frame, using this frame’s mask and measured depth"><input type="checkbox" id="flash-scene-surface" checked onchange="_flashUpdateSceneFrame()">Measured surface</label>' +
                    '<label title="Current mask-and-depth samples for display; these are not tracked landmarks"><input type="checkbox" id="flash-scene-points" checked onchange="_flashUpdateSceneFrame()">Surface points</label>' +
                    '<label title="Display spacing in recorded camera pixels">Spacing <input type="number" id="flash-scene-point-spacing" min="2" max="16" step="1" value="5" onchange="_flashUpdateSceneFrame()"></label>'
                : '') +
            '<label title="Reference surface projected by the used estimate"><input type="checkbox" id="flash-scene-projection" checked onchange="_flashUpdateSceneFrame()">Pose estimate</label>' +
            '<label title="Coarse depth mesh for spatial context; incomplete at missing-depth pixels"><input type="checkbox" id="flash-scene-depth" onchange="_flashUpdateSceneFrame()">Scene context</label>' +
            '<label><input type="checkbox" id="flash-scene-axes" checked onchange="_flashUpdateSceneFrame()">Axes</label>' +
            (kind === 'demo'
                ? '<label title="Tracked scene context used by the group predictor"><input type="checkbox" id="flash-scene-group" onchange="_flashUpdateSceneFrame()">Scene support</label>'
                : '') +
            '<label><input type="checkbox" id="flash-scene-gripper" onchange="_flashUpdateSceneFrame()">Gripper</label></div></details></div>';
        const status = document.createElement('div');
        status.id = 'flash-scene-status';
        status.textContent = kind === 'live' ? 'Waiting for aligned live RGB-D' : 'Loading recorded RGB-D';
        grid.append(photo, scene, toolbar, status);
    }
    grid.addEventListener('click', _flashSelectPointAt);
    viewer.prepend(grid);
    document.getElementById('rerun-placeholder').style.display = 'none';
    return grid;
}

function _flashInstallReplay(replay, imageAlt) {
    flashSceneRequest++;
    flashSceneFrameCache = null;
    flashSceneLoadingKey = null;
    flashSceneView?.dispose();
    flashSceneView = null;
    flashSceneMode = 'camera';
    closeFlashDemoEditor();
    stopFlashTelemetry();
    stopObsStreamViewer();
    clearInterval(flashReplayTimer);
    flashReplayTimer = null;
    flashReplay = replay;
    flashTrackPoints = null;
    flashSelectedTrack = null;
    document.getElementById('flash-track-tools').hidden = true;
    flashUiMode = 'review';
    _flashUpdateWorkspace();
    setFlashTrackerPanelCollapsed(true);
    flashHighlightedOverlay = null;
    _flashCreateSceneGrid(replay.kind, imageAlt, !!replay.camera_features?.depth);
    document.getElementById('flash-tracker-replay-controls').hidden = false;
    document.getElementById('flash-tracker-scrub').max = Math.max(0, replay.frames.length - 1);
    document.getElementById('flash-tracker-play').textContent = 'Play';
    const editToggle = document.getElementById('flash-demo-edit-toggle');
    editToggle.hidden = replay.kind !== 'demo' && replay.kind !== 'raw_demo';
    editToggle.textContent = replay.kind === 'raw_demo' ? 'Prepare demo' : 'Edit demo';
    _flashRenderTimeline();
}

function stepFlashReplay(delta) {
    if (!flashReplay?.frames?.length) return;
    if (flashReplayTimer) toggleFlashReplay();
    seekFlashReplay(Number(document.getElementById('flash-tracker-scrub').value) + delta);
}

function _flashRenderTimeline() {
    const markers = document.getElementById('flash-timeline-markers');
    markers.replaceChildren();
    const replay = flashReplay;
    document.querySelector('.flash-timeline-legend').hidden = !['demo', 'raw_demo'].includes(replay?.kind);
    if (!['demo', 'raw_demo'].includes(replay?.kind) || !replay.frames?.length) return;
    const annotation = flashDemoEditor?.annotation || replay.preparation?.annotation;
    const keypoints = flashDemoEditor?.keypointsDocument?.keypoints || replay.preparation?.keypoints_document?.keypoints || [];
    const duration = replay.frames.at(-1).snapshot.demo.time_s;
    if (!(duration > 0)) return;
    const events = [
        ...keypoints.map((point, index) => ({time: point.t, label: `Key frame ${index + 1} · ${point.move}`, kind: 'key', index})),
    ];
    if (annotation) {
        events.push({time: annotation.grasp_time_s, label: 'Grasp', kind: 'grasp'});
        events.push({time: annotation.release_time_s, label: 'Release', kind: 'release'});
    }
    for (const event of events) {
        if (event.time == null || !Number.isFinite(Number(event.time)) || event.time < 0 || event.time > duration) continue;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `flash-timeline-marker flash-timeline-${event.kind}`;
        if (event.kind === 'key' && flashDemoEditor?.keypointIndex === event.index)
            button.classList.add('active');
        button.style.left = `${100 * event.time / duration}%`;
        button.title = `${event.label} · ${Number(event.time).toFixed(2)} s`;
        button.setAttribute('aria-label', button.title);
        button.onclick = () => {
            if (flashReplayTimer) toggleFlashReplay();
            const nearest = replay.frames.reduce((best, frame, index) =>
                Math.abs(frame.snapshot.demo.time_s - event.time) < Math.abs(replay.frames[best].snapshot.demo.time_s - event.time)
                    ? index : best, 0);
            seekFlashReplay(nearest);
            if (flashDemoEditor) {
                flashDemoEditor.section = event.kind === 'key' ? 'keyframes' : 'events';
                if (event.kind === 'key') flashDemoEditor.keypointIndex = event.index;
                _flashEditorRender();
                _flashRenderTimeline();
            }
        };
        markers.append(button);
    }
}

async function _flashLoadPreparationForTimeline(replay, query) {
    try {
        const response = await fetch(`/api/run/flash/demo/preparation?${query}`);
        if (!response.ok) return;
        const preparation = await response.json();
        if (flashReplay !== replay) return;
        replay.preparation = preparation;
        _flashRenderTimeline();
    } catch (_) { /* Playback remains available if editable inputs cannot be read. */ }
}

async function loadFlashDemoReplay(datasetRoot, episodeIndex, bundlePath, candidateId = null) {
    if (!bundlePath) {
        showToast('Cannot inspect demo', 'This demo has no prepared FLASH skill.', 'warning');
        return;
    }
    const status = document.getElementById('run-flash-status');
    if (status) status.textContent = 'Loading aligned demo frames and prepared spatial poses…';
    try {
        const query = new URLSearchParams({ dataset_root: datasetRoot, episode: String(episodeIndex), bundle_path: bundlePath });
        if (candidateId) query.set('candidate_id', candidateId);
        const response = await fetch(`/api/run/flash/demo/replay?${query}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot read prepared demo');
        document.getElementById('run-flash-root').value = datasetRoot;
        document.getElementById('run-flash-episode').value = String(episodeIndex);
        _clearFlashPreparedSkills();
        _flashInstallReplay({ ...result, datasetRoot, episode: episodeIndex }, 'Original aligned demo RGB frame');
        const replay = flashReplay;
        for (const card of document.querySelectorAll('#run-flash-demos .flash-recording-card')) {
            if (card.dataset.datasetRoot !== datasetRoot || Number(card.dataset.episode) !== episodeIndex) continue;
            const select = card.querySelector('.flash-recording-version');
            if (select && [...select.options].some(option => option.value === bundlePath)) select.value = bundlePath;
            const meta = card.querySelector('.flash-recording-meta');
            if (meta) meta.textContent = `Viewing: ${bundlePath.split('/').slice(-2, -1)[0]} · Play with extracted poses`;
        }
        seekFlashReplay(0);
        if (candidateId && result.tracker_candidate?.tracker === 'trackeverything')
            void _flashLoadCandidatePoints(replay, candidateId);
        toggleFlashReplay();
        if (status) status.textContent = `${result.skill_sample_count} spatial samples from ${bundlePath.split('/').slice(-2, -1)[0]} are shown on the original camera frames.`;
        void _flashLoadPreparationForTimeline(replay, query);
    } catch (error) {
        if (status) status.textContent = `Cannot inspect demo: ${error.message}`;
        showToast('Cannot inspect demo', error.message, 'error');
    }
}

async function previewFlashDemoTrackerCandidate(candidateId) {
    if (flashReplay?.kind !== 'demo') return;
    const replay = flashReplay;
    const current = Number(document.getElementById('flash-tracker-scrub').value);
    const query = new URLSearchParams({dataset_root: replay.datasetRoot,
        episode: String(replay.episode), bundle_path: replay.bundle_path});
    if (candidateId) query.set('candidate_id', candidateId);
    const response = await fetch(`/api/run/flash/demo/replay?${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Cannot preview tracker candidate');
    if (flashReplay !== replay) return;
    replay.frames = result.frames;
    replay.geometry = result.geometry;
    replay.tracker_candidate = result.tracker_candidate || null;
    flashTrackPoints = null;
    flashSelectedTrack = null;
    document.getElementById('flash-track-tools').hidden = true;
    seekFlashReplay(current);
    if (result.tracker_candidate?.tracker === 'trackeverything')
        void _flashLoadCandidatePoints(replay, candidateId);
}

async function _flashLoadCandidatePoints(replay, candidateId) {
    try {
        const query = new URLSearchParams({dataset_root: replay.datasetRoot,
            bundle_path: replay.bundle_path, candidate_id: candidateId});
        const response = await fetch(`/api/run/flash/demo/track-points?${query}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot read raw point tracks');
        if (flashReplay !== replay || flashReplay.tracker_candidate?.candidate_id !== candidateId) return;
        if (!result.available) throw new Error(result.reason);
        result.frameLookup = new Map(result.frames.map((frame, index) => [frame, index]));
        flashTrackPoints = result;
        document.getElementById('flash-track-tools').hidden = false;
        const selected = replay.frames[Number(document.getElementById('flash-tracker-scrub').value)];
        renderFlashDemoSnapshot({ ...selected.snapshot, ...replay.geometry });
    } catch (error) {
        if (flashReplay === replay && flashReplay.tracker_candidate?.candidate_id === candidateId) {
            const status = document.getElementById('run-flash-status');
            if (status) status.textContent = `Cannot inspect raw point tracks: ${error.message}`;
        }
    }
}

async function loadFlashRawDemoReplay(datasetRoot, episodeIndex) {
    const status = document.getElementById('run-flash-status');
    if (status) status.textContent = 'Loading raw RGB-D demo…';
    try {
        const query = new URLSearchParams({dataset_root: datasetRoot, episode: String(episodeIndex)});
        const response = await fetch(`/api/run/flash/demo/raw-replay?${query}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot read raw demo');
        document.getElementById('run-flash-root').value = datasetRoot;
        document.getElementById('run-flash-episode').value = String(episodeIndex);
        _flashInstallReplay({...result, datasetRoot, episode: episodeIndex}, 'Original raw aligned demo RGB frame');
        seekFlashReplay(0);
        toggleFlashReplay();
        if (status) status.textContent = 'Raw demo: no objects or spatial skill prepared yet.';
    } catch (error) {
        if (status) status.textContent = `Cannot inspect raw demo: ${error.message}`;
        showToast('Cannot inspect raw demo', error.message, 'error');
    }
}

async function _loadFlashRecordings() {
    const list = document.getElementById('run-flash-recordings');
    const demosList = document.getElementById('run-flash-demos');
    if (!list || !demosList) return;
    const generation = ++flashRecordingsGeneration;
    list.textContent = 'Loading rollouts…';
    demosList.textContent = 'Loading demos…';
    try {
        const response = await fetch('/api/run/flash/recordings');
        if (!response.ok) throw new Error(`Recordings returned ${response.status}`);
        const recordings = await response.json();
        if (generation !== flashRecordingsGeneration) return;
        list.replaceChildren();
        demosList.replaceChildren();
        const demos = recordings.filter(recording => recording.kind === 'demo');
        const rollouts = recordings.filter(recording => recording.kind !== 'demo');
        if (!demos.length) demosList.textContent = 'No FLASH demos in your dataset sources.';
        if (!rollouts.length) list.textContent = 'No FLASH rollouts in your dataset sources.';
        for (const recording of demos) {
            const card = document.createElement('div');
            card.className = 'flash-recording-card';
            card.dataset.datasetRoot = recording.dataset_root;
            card.dataset.episode = String(recording.episode);
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'flash-recording';
            button.title = recording.dataset_root;
            const label = document.createElement('strong');
            label.textContent = `${recording.dataset_root.split('/').pop()} · episode ${recording.episode}`;
            const meta = document.createElement('span');
            meta.className = 'flash-recording-meta';
            const skills = recording.skills || [];
            meta.textContent = recording.bundle_path
                ? `${skills.length} FLASH skill(s) · viewing: ${recording.prepared_skill} · Play with extracted poses`
                : 'Raw SO-107 RGB-D demo · prepare objects and action in playback';
            button.append(label, meta);
            button.onclick = () => recording.bundle_path
                ? loadFlashDemoReplay(recording.dataset_root, recording.episode,
                    card.querySelector('.flash-recording-version')?.value || recording.bundle_path)
                : loadFlashRawDemoReplay(recording.dataset_root, recording.episode);
            card.append(button);
            const versions = recording.prepared_versions || [];
            if (versions.length > 1) {
                const select = document.createElement('select');
                select.className = 'flash-recording-version';
                select.setAttribute('aria-label', `Prepared version for ${label.textContent}`);
                for (const version of [...versions].reverse()) {
                    const option = document.createElement('option');
                    option.value = version.bundle_path;
                    option.textContent = `${version.name}${version.candidate ? ' · review candidate' : ''}`;
                    select.append(option);
                }
                select.value = flashReplay?.kind === 'demo' && flashReplay.datasetRoot === recording.dataset_root &&
                    flashReplay.episode === recording.episode
                    ? flashReplay.bundle_path : recording.bundle_path;
                if (!select.value) select.value = recording.bundle_path;
                select.onchange = () => loadFlashDemoReplay(recording.dataset_root, recording.episode, select.value);
                card.append(select);
            }
            demosList.append(card);
        }
        for (const recording of rollouts) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'flash-recording';
            button.title = `${recording.dataset_root}\n${recording.reason || ''}`;
            const label = document.createElement('strong');
            label.textContent = recording.dataset_root.split('/').pop();
            const meta = document.createElement('span');
            const when = new Date(recording.modified_at_s * 1000).toLocaleString();
            meta.textContent = `${when} · episode ${recording.episode} · ${recording.outcome}` +
                (recording.frames == null ? '' : ` · ${recording.frames} frames`);
            button.append(label, meta);
            if (recording.reason && recording.outcome !== 'success') {
                const reason = document.createElement('span');
                reason.className = 'flash-recording-reason';
                reason.textContent = recording.reason;
                button.append(reason);
            }
            button.onclick = () => loadFlashTrackerReplay(recording.dataset_root, recording.episode);
            list.append(button);
        }
    } catch (error) {
        if (generation === flashRecordingsGeneration) {
            list.textContent = `Cannot list rollouts: ${error.message}`;
            demosList.textContent = `Cannot list demos: ${error.message}`;
        }
    }
}

function seekFlashReplay(index) {
    if (!flashReplay?.frames?.length) return;
    flashSceneRequest++;
    const selected = flashReplay.frames[Math.max(0, Math.min(index, flashReplay.frames.length - 1))];
    const scrub = document.getElementById('flash-tracker-scrub');
    scrub.value = String(index);
    const demo = flashReplay.kind === 'demo' || flashReplay.kind === 'raw_demo';
    document.getElementById('flash-tracker-frame').textContent = demo
        ? `${selected.snapshot.demo.time_s.toFixed(1)} s · ${index + 1}/${flashReplay.frames.length}`
        : `${index + 1} / ${flashReplay.frames.length}`;
    const image = document.querySelector('#rerun-viewer .flash-replay-grid img');
    const grid = image.closest('.flash-replay-grid');
    grid.classList.add('flash-frame-loading');
    image.dataset.loadedFrame = '';
    const url = demo
        ? `/api/run/flash/demo/frame?${new URLSearchParams({ dataset_root: flashReplay.datasetRoot, frame_index: String(selected.frame_index) })}`
        : `/api/datasets/${encodeURIComponent(flashReplay.datasetId)}/episodes/${flashReplay.episode}` +
            `/frame/${selected.frame_index}?camera=${encodeURIComponent(flashReplay.camera_features?.rgb || 'observation.images.overhead')}`;
    image.onload = () => {
        if (flashReplay.frames[Number(scrub.value)] !== selected) return;
        image.dataset.loadedFrame = String(selected.frame_index);
        _flashDrawOverlay();
        if (flashSceneMode === '3d') void _flashUpdateSceneFrame();
        else grid.classList.remove('flash-frame-loading');
    };
    image.onerror = () => {
        if (flashReplay.frames[Number(scrub.value)] !== selected) return;
        const status = document.getElementById('flash-scene-status');
        if (status) status.textContent = `Frame ${selected.frame_index} could not be loaded`;
    };
    image.src = url;
    if (flashReplay.kind === 'demo') renderFlashDemoSnapshot({ ...selected.snapshot, ...flashReplay.geometry });
    else if (flashReplay.kind === 'raw_demo') renderFlashRawSnapshot(selected.snapshot);
    else renderFlashSnapshot(selected.snapshot, 'Recording');
    if (flashDemoEditor) _flashEditorUpdateFrame();
}

function toggleFlashReplay() {
    if (!flashReplay) return;
    const button = document.getElementById('flash-tracker-play');
    if (flashReplayTimer) {
        clearInterval(flashReplayTimer); flashReplayTimer = null; button.textContent = 'Play'; return;
    }
    button.textContent = 'Pause';
    flashReplayTimer = setInterval(() => {
        const current = flashReplay.frames[Number(document.getElementById('flash-tracker-scrub').value)];
        const grid = document.querySelector('#rerun-viewer .flash-replay-grid');
        if (grid?.classList.contains('flash-frame-loading') ||
            (flashSceneMode === '3d' && flashSceneFrameCache?.frameIndex !== current?.frame_index)) return;
        const next = Number(document.getElementById('flash-tracker-scrub').value) + 1;
        if (next >= flashReplay.frames.length) { toggleFlashReplay(); return; }
        seekFlashReplay(next);
    }, ['demo', 'raw_demo'].includes(flashReplay.kind) ? 1000 / flashReplay.fps : 100);
}

// The demo editor is loaded from flash_demo_editor.js.

function _flashSkillDetail() {
    const container = document.getElementById('run-flash-skill-detail');
    if (!container) return;
    container.replaceChildren();
    const selected = flashPreparedSkills.find(item => item.path === _val('run-flash-skill'));
    if (!selected) { _updateLaunchButton(); return; }
    const skill = selected.inspection;
    const heading = document.createElement('div');
    heading.className = 'form-section-title';
    heading.textContent = `Selected: ${selected.path.split('/').slice(-2, -1)[0]}`;
    container.append(heading);
    const summary = document.createElement('div');
    summary.textContent = `${skill.intervals.length} motion intervals · ${skill.prepared_references.length} object references · ${skill.milestones.length} milestones`;
    container.append(summary);
    const intervals = document.createElement('details');
    const intervalHeading = document.createElement('summary');
    intervalHeading.textContent = 'Motion intervals';
    intervals.append(intervalHeading);
    const intervalList = document.createElement('ul');
    for (const interval of skill.intervals) {
        const row = document.createElement('li');
        const endpoint = interval.mode === 'connection'
            ? `, ${interval.endpoint_requirement || 'required'} endpoint`
            : '';
        row.textContent = `${interval.name}: ${interval.mode}${endpoint}, ${interval.relation} to ${interval.reference_role}, ${interval.samples} samples`;
        intervalList.append(row);
    }
    intervals.append(intervalList);
    container.append(intervals);
    for (const milestone of skill.milestones) {
        const row = document.createElement('div');
        const rule = milestone.pose_relation;
        row.textContent = rule
            ? `Milestone ${milestone.name}: verify ${rule.moving_role} relative to ${rule.reference_role} within ${rule.translation_tolerance_m} m`
            : `Milestone ${milestone.name}: external verifier needed`;
        container.append(row);
    }
    if (skill.warnings.length) {
        const warnings = document.createElement('div');
        warnings.className = 'form-hint';
        warnings.textContent = `${skill.warnings.length} evidence warning(s): ${skill.warnings.join('; ')}`;
        container.append(warnings);
    }
    _updateLaunchButton();
}

async function inspectFlashDataset() {
    const error = _flashDatasetInputError();
    if (error) { showToast('Cannot inspect skills', error, 'warning'); return; }
    const button = document.getElementById('run-flash-inspect-btn');
    const status = document.getElementById('run-flash-status');
    const root = _val('run-flash-root');
    const episode = Number(_val('run-flash-episode'));
    _clearFlashPreparedSkills();
    const generation = flashInspectGeneration;
    if (button) button.disabled = true;
    if (status) status.textContent = 'Inspecting skills stored with this episode…';
    try {
        const res = await fetch('/api/run/flash/dataset-inspect', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_root: root, episode }),
        });
        const data = await res.json();
        if (generation !== flashInspectGeneration) return;
        if (!res.ok) throw new Error(data.detail || 'Inspection failed');
        flashPreparedSkills = data.skills.filter(item =>
            item.kind === 'prepared' && item.inspection.runtime_bindings_complete);
        const select = document.getElementById('run-flash-skill');
        if (select) {
            select.replaceChildren(new Option('Select a prepared skill', ''));
            for (const item of flashPreparedSkills) {
                const skill = item.inspection;
                const name = item.path.split('/').slice(-2, -1)[0];
                select.add(new Option(`${name} (${skill.intervals.length} intervals, ${skill.prepared_references.length} references)`, item.path));
            }
            if (flashPreparedSkills.length === 1) select.value = flashPreparedSkills[0].path;
        }
        if (status) {
            status.textContent = flashPreparedSkills.length
                ? `${flashPreparedSkills.length} prepared skill(s) found. Select one to inspect its summary. A calibrated robot/camera config is needed only for a live run.`
                : `No runnable prepared skill found. ${data.missing.join(' ')}`;
        }
        _flashSkillDetail();
    } catch (e) {
        if (generation !== flashInspectGeneration) return;
        _clearFlashPreparedSkills();
        if (status) status.textContent = `Inspection failed: ${e.message}`;
        showToast('FLASH inspection failed', e.message, 'error');
    } finally {
        if (button) button.disabled = false;
    }
}

async function inspectFlashTrialConfig() {
    const target = document.getElementById('run-flash-config-detail');
    if (!target) return;
    target.textContent = 'Reading trial parameters…';
    try {
        const res = await fetch('/api/run/flash/config-inspect', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_path: _val('run-flash-config') }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Cannot read trial config');
        target.replaceChildren();
        const note = document.createElement('div');
        note.className = 'form-hint';
        note.textContent = `Parameters from ${data.path}. Edit that JSON file to change them, then inspect again. Each run records its effective config beside the episode.`;
        target.append(note);
        const highlights = [
            ['attachment.max_observation_shift_m', 'Held-pose correction gate', value => `${value * 1000} mm`],
            ['attachment.max_unverified_age_s', 'Held prediction expiry', value => value == null ? 'disabled' : `${value} s`],
            ['first_pose_timeout_s', 'Initial pose wait', value => `${value} s`],
            ['causal_tracking.max_pose_disagreement_translation_m', 'Group/direct translation disagreement', value => `${value * 1000} mm`],
            ['causal_tracking.max_pose_disagreement_rotation_rad', 'Group/direct rotation disagreement', value => `${(value * 180 / Math.PI).toFixed(1)}°`],
        ];
        for (const [path, label, format] of highlights) {
            const value = path.split('.').reduce((part, key) => part?.[key], data.settings);
            if (value === undefined) continue;
            const row = document.createElement('div');
            row.textContent = `${label}: ${format(value)} (${path})`;
            target.append(row);
        }
        const all = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'All trial parameters';
        const contents = document.createElement('pre');
        contents.textContent = JSON.stringify(data.settings, null, 2);
        all.append(summary, contents);
        target.append(all);
    } catch (error) {
        target.textContent = `Cannot inspect trial parameters: ${error.message}`;
    }
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
    // No-op for now: replay paces at the dataset's recorded fps, which the
    // backend reads from the dataset metadata. Kept as a hook for future
    // per-episode UI (e.g. preview thumbnail when an episode is picked).
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
    // Build options from the Model tab's cached scan data. The server
    // emits `default_policy_path` per run (api/models.py:_scan_training_run)
    // and `policy_path` per checkpoint (api/models.py:_read_checkpoint_meta);
    // we use those directly — DO NOT reconstruct the path from `m.path`.
    // Why: the layout under `m.path` differs between vanilla lerobot-train
    // (`<run>/checkpoints/<step>/pretrained_model`) and the GUI-managed
    // docker recipe (`<run>/output/checkpoints/<step>/pretrained_model`),
    // and the flat HVLA-S2-VLM layout has no `pretrained_model/` subdir
    // at all. Server-side synthesis handles all three; client-side
    // reconstruction used to silently break for the latter two.
    const data = (typeof modelSourceData !== 'undefined') ? modelSourceData : {};
    const entries = Object.values(data).flat();
    if (!entries.length) {
        return '<option value="" disabled selected>Loading models...</option>';
    }
    let html = '<option value="" disabled selected>Select checkpoint</option>';
    for (const m of entries) {
        // Defensive: an entry with no usable policy path (RLT, future
        // variants) is skipped rather than serialized as `value="undefined"`.
        if (!m.default_policy_path) continue;
        const stepText = m.current_step != null ? ` (step ${m.current_step.toLocaleString()})` : '';
        // `data-run-path` lets _onPolicyCheckpointChange recover the run
        // dir without regex-stripping the policy path (which would fail
        // for flat layouts where policy_path === run path).
        html += `<option value="${_esc(m.default_policy_path)}" data-policy-type="${_esc(m.policy_type)}" data-run-path="${_esc(m.path)}" title="${_esc(m.path)}">${_esc(m.name)} — ${_esc(m.policy_type)}${stepText}</option>`;
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
        checkpoint: document.getElementById('run-teleop-debug-step')?.value || sel.value,
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
    const loadBtn = document.getElementById('run-debug-load-btn');
    const unloadBtn = document.getElementById('run-debug-unload-btn');
    const sel = document.getElementById('run-teleop-debug-model');
    const hasSelection = !!sel?.value;

    // Load: enabled only when (a) not currently loading, (b) not already
    // loaded, and (c) a model is selected. Without (c) the button used to be
    // clickable and would just throw a toast — disable upfront so the
    // affordance matches the actual enabled state.
    const loadEnabled = !_debugModelLoading && !_debugModelLoaded && hasSelection;
    // Unload: enabled only when something is currently loaded (and not
    // currently loading/unloading).
    const unloadEnabled = !_debugModelLoading && _debugModelLoaded;

    if (loadBtn) {
        loadBtn.disabled = !loadEnabled;
        loadBtn.classList.toggle('btn-accent', loadEnabled);
    }
    if (unloadBtn) {
        unloadBtn.disabled = !unloadEnabled;
        unloadBtn.classList.toggle('btn-accent', unloadEnabled);
    }
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
    _refreshDebugStepOptions();
    // Selection drives the Load button's enabled state.
    _updateDebugButtons();
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
    // Experimental launch-time knobs: visible for New AND for resumed
    // training (so you can A/B the flag against your current checkpoint),
    // but NOT for deploy — there's no exploration noise to share in
    // deterministic deploy mode. Mid-training toggle does mix two noise
    // distributions in the replay buffer; the tooltip flags that.
    const isResumeTrain = isExisting &&
        (document.getElementById('run-hvla-rlt-run-mode')?.value || 'train') === 'train';
    const showShared = isNew || isResumeTrain;
    for (const id of ['run-hvla-rlt-shared-noise-label', 'run-hvla-rlt-shared-noise']) {
        const el = document.getElementById(id);
        if (el) el.style.display = showShared ? '' : 'none';
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
            // run_dir is the parent run directory — the dir new checkpoints
            // should land in. For latest/ entries it equals path; for ep_N
            // snapshots it's the snapshot's parent so further training
            // continues into the original run dir, not into the snapshot.
            opt.dataset.runDir = ckpt.run_dir || ckpt.path;
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

// Per-run checkpoint steps, fetched lazily and cached by run path.
// /api/models/sources/<src>/models omits the per-checkpoint array (it reports
// only num_checkpoints), so the cached scan data cannot answer this.
const _policyStepCache = {};

async function _fetchRunCheckpoints(runPath) {
    if (_policyStepCache[runPath]) return _policyStepCache[runPath];
    try {
        const resp = await fetch(`/api/models/run/${encodeURIComponent(runPath)}/checkpoints`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const rows = await resp.json();
        _policyStepCache[runPath] = Array.isArray(rows) ? rows : (rows.checkpoints || []);
    } catch (e) {
        console.warn('Failed to fetch checkpoints for', runPath, e);
        _policyStepCache[runPath] = [];
    }
    return _policyStepCache[runPath];
}

// Newest first, latest preselected — so not touching this control reproduces
// the old behaviour exactly.
async function _refreshStepOptions(modelSelId, stepSelId) {
    const sel = document.getElementById(modelSelId);
    const stepSel = document.getElementById(stepSelId);
    if (!sel || !stepSel) return;
    const runPath = sel.selectedOptions[0]?.dataset.runPath || sel.value;
    // Captured up here, before the placeholder below wipes it. This function is
    // async, so the value is long gone by the time the new options are built.
    const prevStep = stepSel.value;
    if (!runPath) {
        stepSel.innerHTML = '<option value="" disabled selected>Step</option>';
        return;
    }
    stepSel.innerHTML = '<option value="" disabled selected>…</option>';
    const rows = (await _fetchRunCheckpoints(runPath)).filter(c => c.policy_path);
    if (!rows.length) {
        // Flat layouts (HVLA-S2-VLM) have no step dirs; the model dropdown's
        // own value is already the policy path, so leave the control inert.
        stepSel.innerHTML = '<option value="" disabled selected>n/a</option>';
        return;
    }
    const sorted = [...rows].sort((a, b) => (b.step ?? 0) - (a.step ?? 0));
    stepSel.innerHTML = sorted.map((c, i) => {
        const step = c.step != null ? c.step.toLocaleString() : (c.name || '?');
        return `<option value="${_esc(c.policy_path)}"${i === 0 ? ' selected' : ''}>`
            + `${_esc(step)}${c.is_last ? ' — latest' : ''}</option>`;
    }).join('');
    // Still offered? keep it. Absent means the run changed, and defaulting to
    // that run's latest is correct.
    if (prevStep && [...stepSel.options].some(o => o.value === prevStep)) {
        stepSel.value = prevStep;
    }
}

async function _refreshPolicyStepOptions() {
    return _refreshStepOptions('run-policy-checkpoint', 'run-policy-step');
}

async function _refreshDebugStepOptions() {
    return _refreshStepOptions('run-teleop-debug-model', 'run-teleop-debug-step');
}

// The path to launch: the step dropdown when it has one, else the model
// dropdown's value (flat layouts, or before the steps have loaded).
function _selectedPolicyPath() {
    return document.getElementById('run-policy-step')?.value
        || document.getElementById('run-policy-checkpoint')?.value
        || '';
}

function _onPolicyCheckpointChange() {
    const sel = document.getElementById('run-policy-checkpoint');
    if (!sel?.value) return;
    // Recover the run dir from the option's dataset (stashed by
    // _modelCheckpointOptions). Falls through to sel.value for flat
    // layouts where policy_path IS the run path. The old regex-strip
    // assumed `<run>/checkpoints/last/pretrained_model`, which broke
    // for both the docker recipe (extra `output/` segment) and flat
    // checkpoints (no `pretrained_model/` suffix at all).
    const runPath = sel.selectedOptions[0]?.dataset.runPath || sel.value;
    _refreshPolicyStepOptions();
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
    // loadModelSources only scans sources whose `expanded` flag is true.
    // The policy tab needs models regardless of UI expansion state, so if
    // we're still empty after loadModelSources, scan every source.
    if (
        typeof modelSourceData !== 'undefined' &&
        !Object.keys(modelSourceData).length &&
        typeof modelSources !== 'undefined' &&
        modelSources.length &&
        typeof scanModelSource === 'function'
    ) {
        await Promise.all(modelSources.map(s => scanModelSource(s.path)));
    }
    // Re-render checkpoint selectors after data is loaded
    const sel = document.getElementById('run-policy-checkpoint');
    if (sel) {
        // Mirrors the debug-model select below, which has always done this.
        // Without it, a checkpoint picked before this async load finished was
        // wiped and the launch failed validation with "Select a model
        // checkpoint" — or, worse, silently reverted to a different run.
        const prevModel = sel.value;
        sel.innerHTML = _modelCheckpointOptions();
        if (prevModel && [...sel.options].some(o => o.value === prevModel)) {
            sel.value = prevModel;
        }
    }
    _refreshPolicyStepOptions();
    const debugSel = document.getElementById('run-teleop-debug-model');
    if (debugSel) {
        const prev = debugSel.value;
        debugSel.innerHTML = '<option value="">None</option>' + _modelCheckpointOptions();
        debugSel.value = prev;
        _refreshDebugStepOptions();
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
    if (selectedWorkflow === 'flash') void _loadFlashDatasets();
    // Refresh teleop record dataset selector
    const recordSel = document.getElementById('run-teleop-record-dataset');
    if (recordSel) {
        // Restore the user's selection verbatim, like every other select here.
        //
        // This used to special-case "__new__": after a recording it tried to
        // re-point the selector at the dataset that now exists, falling back to
        // "None (pure teleop)" when it couldn't find it. Two things made that
        // wrong. The match was dead code — repo_ids are date-stamped at
        // creation, so the typed name never matches what got created. And the
        // fallback silently changed the NEXT launch from record to teleop: a
        // user who aborted a take, fixed their robot selection and hit Launch
        // again got a teleop session with no recording, no phases and no audio,
        // with nothing telling them the mode had changed. The selector is the
        // record/teleop mode switch, so only the user moves it.
        const prev = recordSel.value;
        recordSel.innerHTML = _recordDatasetOptions();
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

// Marker for required-field labels. Visually consistent across the whole
// form — pair with the corresponding entry in `_WORKFLOW_VALIDATORS` /
// `_POLICY_VALIDATORS` so the asterisk and the Launch-button gating tell
// the same story. For conditional fields (RL Token Encoder, recording
// task / new-dataset name) the asterisk is part of the label; the
// label's `display:none` toggle naturally hides the marker too.
const _REQ = ' <span class="required-marker">*</span>';

function renderRunForm() {
    if (_runFormRendered) return; // Only build once
    _runFormRendered = true;

    const form = document.getElementById('run-form');
    if (!form) return;

    let html = '';

    // ---- Teleop workflow section ----
    html += `<div id="run-section-teleop" style="${selectedWorkflow === 'teleop' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-teleop-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Teleop${_REQ}</label>`;
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
    html += `<label>Name${_REQ}</label>`;
    html += `<div><input type="text" id="run-teleop-new-dataset-name" placeholder="my_new_dataset" oninput="_checkNewDatasetConflict()">`;
    html += `<div class="dataset-conflict-warning" id="run-teleop-dataset-conflict" style="display:none;"></div></div>`;
    html += `</div>`;
    // Record fields (hidden when None selected)
    html += `<div id="run-teleop-record-fields" class="form-grid" style="display:none;">`;
    html += `<label>Task${_REQ}</label>`;
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
    // Debug model (optional — runs alongside teleop for live inspection of
    // model output while the human drives. Distinct from a Policy run, which
    // hands the arm to the model.)
    html += '<div class="form-section">';
    html += '<div class="form-section-title">Debug model</div>';
    html += '<div class="form-hint" style="margin-bottom:8px;">';
    html += 'Mount a model alongside teleop so its output (text, images, ';
    html += 'subtask predictions, latents, etc.) is logged for inspection ';
    html += 'while <em>you</em> drive the arm. Unlike Policy, the model ';
    html += 'never sends actions — this is for verifying what a model ';
    html += '<em>sees</em> on real human-driven trajectories.';
    html += '</div>';
    html += '<div class="form-grid">';
    html += `<label>Model</label>`;
    html += `<select id="run-teleop-debug-model" onchange="_onDebugModelChange()">`;
    html += `<option value="">None</option>`;
    html += _modelCheckpointOptions();
    html += `</select>`;
    html += `<label>Step</label>`;
    html += `<select id="run-teleop-debug-step" title="Checkpoint step"><option value="" disabled selected>Step</option></select>`;
    html += `<label></label>`;
    // Buttons start disabled; _updateDebugButtons() flips state based on
    // selection + load status. Keeps consistent with the Launch/Stop pattern:
    // exactly one is enabled+highlighted at any time.
    html += `<div><button id="run-debug-load-btn" class="btn-tiny" onclick="_loadDebugModel()" disabled>Load</button> `;
    html += `<button id="run-debug-unload-btn" class="btn-tiny" onclick="_unloadDebugModel()" disabled>Unload</button> `;
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
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-replay-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Episode${_REQ}</label>`;
    html += `<select id="run-replay-episode" onchange="_onReplayEpisodeChange()">${_episodeOptions()}</select>`;
    html += '</div>';
    html += '</div>'; // end replay section

    // ---- Policy workflow section ----
    html += `<div id="run-section-policy" style="${selectedWorkflow === 'policy' ? '' : 'display:none'}">`;
    html += '<div class="form-grid">';
    html += `<label>Robot${_REQ}</label>`;
    html += `<select id="run-policy-robot">${_robotProfileOptions()}</select>`;
    html += `<label>Model${_REQ}</label>`;
    html += `<select id="run-policy-checkpoint" onchange="_onPolicyCheckpointChange()">${_modelCheckpointOptions()}</select>`;
    // Its own row: run names are long enough that sharing one row made the
    // model unreadable. Defaults to the latest step, so leaving this alone
    // reproduces the previous behaviour exactly.
    html += `<label>Step</label>`;
    html += `<select id="run-policy-step" title="Checkpoint step"><option value="" disabled selected>Step</option></select>`;
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
    html += `<label>Task Prompt${_REQ}</label>`;
    html += `<input type="text" id="run-hvla-task" placeholder="assemble cylinder into ring" value="">`;
    const _hvlaDecodeSubtaskDesc = "Have S2 also emit a natural-language subtask label alongside its latent → action chunk. Useful for VLA debugging / interpretability; off by default to keep S2 inference latency minimal.";
    html += `<label title="${_hvlaDecodeSubtaskDesc}">Decode Subtask</label>`;
    html += `<input type="checkbox" id="run-hvla-decode-subtask" title="${_hvlaDecodeSubtaskDesc}">`;
    const _hvlaQueryIntervalDesc = "How often S1 (chunk policy) runs new inference, measured in policy steps. Default 2 = inference every 2 steps (15 Hz at fps=30) — small enough that motion stays responsive but big enough that S1 latency doesn't bottleneck the loop. Set to 1 for max responsiveness at higher compute cost.";
    html += `<label title="${_hvlaQueryIntervalDesc}">Query Interval (steps between S1 inference, default 2)</label>`;
    html += `<input type="number" id="run-hvla-query-interval" placeholder="2" min="0" title="${_hvlaQueryIntervalDesc}">`;
    const _hvlaDenoiseStepsDesc = "Optional runtime override for the flow-matching ODE solver steps per S1 inference. Higher values cost more latency. Leave empty to use the value saved in the checkpoint.";
    html += `<label title="${_hvlaDenoiseStepsDesc}">Denoise Steps (checkpoint default)</label>`;
    html += `<input type="number" id="run-hvla-denoise-steps" placeholder="checkpoint" min="1" title="${_hvlaDenoiseStepsDesc}">`;
    html += `<label title="How the policy frame is sent to the robot each tick. 'chunk' (default) packs the remaining chunk frames as an ActionChunk → predictive robots use the exact-lookup path for zero-estimation-error lookahead. 'dict' sends only the current frame → predictive robots fall back to velocity-LSQ extrapolation (rate-agnostic post-fix but still has EMA residual). Use 'dict' for A/B comparison or as rollback. Non-predictive robots are unaffected.">Action Send Shape</label>`;
    html += `<select id="run-hvla-send-action-shape" title="Default 'chunk' is the recommended best-perf path. Switch to 'dict' for A/B comparison against pre-fix behaviour.">`;
    html += `<option value="chunk">chunk (default — predictive exact-lookup)</option>`;
    html += `<option value="dict">dict (single frame — A/B / rollback)</option>`;
    html += `</select>`;
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
    const _rltSelectDesc = "Online RL fine-tuning of the frozen S1 chunk policy via a 1×768 'RL token' bottleneck. None = pure HVLA inference. + New training = start a fresh actor/critic against the loaded S1. Existing checkpoint = resume / deploy a previously-trained actor. See project_rlt_design memory note for the broader architecture.";
    html += `<label title="${_rltSelectDesc}">RL Checkpoint</label>`;
    html += `<select id="run-hvla-rlt-select" onchange="_onRltSelectChange()" title="${_rltSelectDesc}">`;
    html += `<option value="" title="Pure HVLA inference — no RL actor in the loop.">None (no RL)</option>`;
    html += `<option value="__new__" title="Start a fresh RLT training run. Requires a Phase-1 RL-token encoder checkpoint.">+ New training...</option>`;
    html += `</select>`;
    // Fields shown when New is selected
    html += `<label id="run-hvla-rlt-token-label" style="display:none">RL Token Encoder${_REQ}</label>`;
    html += `<input type="text" id="run-hvla-rlt-token-ckpt" placeholder="outputs/rlt_token_v4_4layer_d2048/checkpoint-10000" style="display:none" title="Required: path to the trained Phase-1 RL token encoder checkpoint dir (containing encoder.pt + config.json). The actor's input dim depends on this — wrong / missing means a state_dict size mismatch crash on load.">`;
    html += `<label id="run-hvla-rlt-outdir-label" style="display:none">Output Directory</label>`;
    html += `<input type="text" id="run-hvla-rlt-output-dir" value="outputs/rlt_online" style="display:none">`;
    // Experimental knobs (NEW training only — launch-time, not runtime-tunable).
    // Architectural choices baked into the run rather than mid-training tuning.
    html += `<label id="run-hvla-rlt-shared-noise-label" style="display:none" title="Sample exploration noise once per chunk and broadcast across C frames. Smoother joint commands than per-frame iid noise. Default ON after v2_widened ep101→120 A/B (autonomous +10pp). Uncheck to A/B back to per-frame iid noise — flipping mid-training mixes noise distributions in the replay buffer for ~10 episodes.">Shared noise per chunk</label>`;
    html += `<input type="checkbox" id="run-hvla-rlt-shared-noise" checked style="display:none" title="Sample exploration noise once per chunk and broadcast across C frames. Smoother joint commands than per-frame iid noise. Default ON after v2_widened ep101→120 A/B (autonomous +10pp). Uncheck to A/B back to per-frame iid noise — flipping mid-training mixes noise distributions in the replay buffer for ~10 episodes.">`;
    // Fields shown when existing checkpoint selected
    const _rltModeDesc = "Train continues the RL gradient updates on the loaded checkpoint — actor/critic keep learning during the run. Deploy runs the actor in pure inference (no optimizer, no replay buffer writes) — use to evaluate a trained policy without polluting it with new transitions.";
    html += `<label id="run-hvla-rlt-mode-label" style="display:none" title="${_rltModeDesc}">Mode</label>`;
    html += `<select id="run-hvla-rlt-run-mode" style="display:none" onchange="_onRltSelectChange()" title="${_rltModeDesc}">`;
    html += `<option value="train" title="Continue RL training: actor + critic + replay buffer all active.">Train (continue learning)</option>`;
    html += `<option value="deploy" title="Inference only: actor weights frozen, no gradient updates, no replay buffer writes. Safe way to evaluate a trained policy.">Deploy (inference only)</option>`;
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

    // Demo viewing and skill inspection are read-only. The shared Run button
    // only starts a robot trial after a prepared bundle and rig are selected.
    html += `<div id="run-section-flash" style="${selectedWorkflow === 'flash' ? '' : 'display:none'}">`;
    html += '<div class="flash-library"><div class="flash-library-tabs" role="tablist" aria-label="FLASH recordings">';
    html += `<button type="button" class="flash-library-tab ${flashLibraryTab === 'demos' ? 'active' : ''}" data-tab="demos" role="tab" aria-selected="${flashLibraryTab === 'demos'}" onclick="selectFlashLibraryTab('demos')">Demos</button>`;
    html += `<button type="button" class="flash-library-tab ${flashLibraryTab === 'rollouts' ? 'active' : ''}" data-tab="rollouts" role="tab" aria-selected="${flashLibraryTab === 'rollouts'}" onclick="selectFlashLibraryTab('rollouts')">Rollouts</button>`;
    html += '<button type="button" class="flash-library-refresh" onclick="_loadFlashRecordings()" aria-label="Refresh FLASH recordings" title="Refresh recordings">↻</button></div>';
    html += '<div id="run-flash-demos" class="flash-recordings">Loading demos…</div>';
    html += '<div id="run-flash-recordings" class="flash-recordings">Loading rollouts…</div>';
    html += '<button type="button" class="flash-live-entry" onclick="showFlashLiveSetup()">Set up live trial →</button></div>';
    html += '<button type="button" class="flash-live-entry flash-tracking-entry" onclick="showFlashTracking()">Live tracking · camera only →</button>';
    html += '<button type="button" class="flash-review-return" onclick="showFlashReview()">← Back to recordings</button>';
    html += flashTrackingForm();
    html += '<details class="flash-manual-setup" id="run-flash-manual-setup"><summary>Open another dataset or run a live trial</summary>';
    html += '<div class="form-grid">';
    html += '<label>Dataset</label>';
    html += '<select id="run-flash-dataset" onchange="_onFlashDatasetChange()"><option value="">Choose an opened dataset...</option></select>';
    html += '</div>';
    html += '<div class="form-grid">';
    html += `<label>Local dataset directory${_REQ}</label>`;
    html += '<input id="run-flash-root" type="text" placeholder="/absolute/path/to/a/LeRobot/dataset" oninput="_clearFlashPreparedSkills()">';
    html += '</div>';
    html += '<div class="form-grid">';
    html += `<label>Demo episode${_REQ}</label>`;
    html += '<input id="run-flash-episode" type="number" min="0" step="1" value="0" oninput="_clearFlashPreparedSkills()">';
    html += '</div>';
    html += '<div class="flash-demo-actions">';
    html += '<button type="button" class="btn-small secondary" id="run-flash-view-btn" onclick="viewFlashDemo()">View raw episode in Data</button>';
    html += '<button type="button" class="btn-small secondary" id="run-flash-inspect-btn" onclick="inspectFlashDataset()">Inspect FLASH skills</button>';
    html += '</div>';
    html += '<div class="form-section-title">Live robot trial</div>';
    html += '<div class="form-grid">';
    html += '<label>Prepared skill</label>';
    html += '<select id="run-flash-skill" onchange="_flashSkillDetail()"><option value="">Inspect skills to choose one</option></select>';
    html += '</div>';
    html += '<div id="run-flash-skill-detail" class="flash-skill-detail"></div>';
    html += '<div class="form-grid">';
    html += '<label>Robot/camera trial config</label>';
    html += '<input id="run-flash-config" type="text" placeholder="/absolute/path/to/flash-trial.json" oninput="document.getElementById(\'run-flash-config-detail\').replaceChildren(); _updateLaunchButton()">';
    html += '</div>';
    html += '<button type="button" class="btn-small secondary" onclick="inspectFlashTrialConfig()">Inspect trial parameters</button>';
    html += '<div id="run-flash-config-detail" class="flash-skill-detail"></div>';
    html += '</details>';
    html += '<div id="run-flash-status" class="form-hint">Choose a demo above to review its extracted poses.</div>';
    html += '</div>';

    // ---- Episode control (active while a record subprocess runs) ----
    // Sends episode-transition commands to the running lerobot-record
    // subprocess via POST /api/run/control — mirrors the keyboard events
    // the CLI listens for (right-arrow / left-arrow / ESC). Applies to any
    // workflow that launches a record subprocess (teleop record, policy
    // record, HVLA), so it lives outside the per-workflow sections.
    // Buttons start disabled; updateRunUI enables them while a subprocess
    // runs (the endpoint 409s otherwise).
    html += `<div class="form-section" id="run-episode-control" style="${selectedWorkflow === 'flash' ? 'display:none' : ''}">`;
    // Live phase readout sits right in the section title so the operator can
    // see what a "Next episode" click will interrupt ("recording episode 3" /
    // "resetting" / ...). Fed by pollRunStatus from /api/run/status.
    html += '<div class="form-section-title">Episode control <span id="run-phase" class="run-phase-badge"></span></div>';
    html += '<div class="episode-control-row">';
    html += '<button id="run-ctrl-next" class="btn-small secondary" onclick="sendRunControl(\'exit_early\')" disabled title="End the current phase early (hotkey: N)">Next episode (N)</button>';
    html += '<button id="run-ctrl-rerecord" class="btn-small secondary" onclick="sendRunControl(\'rerecord_episode\')" disabled title="Discard the current episode and re-record it (hotkey: R)">Re-record (R)</button>';
    html += '</div>';
    html += '<div class="form-hint" style="margin-top:6px;">Active while a record run is in progress. Stopping a run uses the main Stop button above.</div>';
    html += '</div>';

    form.innerHTML = html;
    _toggleHvlaRecordFields();
    // Refresh debug-model state after the form is in the DOM. Without this,
    // both Load and Unload buttons stay in their default disabled state and
    // the user can't tell whether a model is currently loaded across re-renders.
    _refreshDebugModelStatus();
    // Re-validate the Launch button now that the form's IDs exist again,
    // and re-validate on every input/change inside the form so the button
    // tracks selections continuously rather than failing at click-time.
    form.addEventListener('input', _updateLaunchButton);
    form.addEventListener('change', _updateLaunchButton);
    _updateLaunchButton();
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
    if (selectedWorkflow === 'flash') {
        const { ready, reason } = _validateLaunch();
        if (!ready) { showToast('Cannot run FLASH', reason, 'warning'); return; }
        try {
            const bundlePath = _val('run-flash-skill');
            const res = await fetch('/api/run/flash/run', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_root: _val('run-flash-root'),
                    episode: Number(_val('run-flash-episode')),
                    bundle_path: bundlePath,
                    config_path: _val('run-flash-config'),
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                showToast('FLASH launch failed', data.detail || 'Unknown error', 'error');
                return;
            }
            clearTerminal('run-terminal');
            appendTerminalLine(`FLASH trial started from ${bundlePath}`);
            showToast('Started', `FLASH trial started (PID ${data.pid})`, 'success');
            updateRunUI(true, data.command);
            connectOutputSSE();
            startFlashTelemetry();
        } catch (e) { showToast('FLASH launch failed', e.message, 'error'); }
        return;
    }

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
                    const ok = await Dialogs.confirm(
                        `Dataset "${d.repo_id}" uses ${d.fps} FPS ` +
                        `but you selected ${currentFps} FPS.\n\n` +
                        `A dataset cannot have different FPS across episodes. ` +
                        `The recording will use ${d.fps} FPS.`,
                        { title: 'FPS mismatch', confirmLabel: 'Continue' },
                    );
                    if (!ok) return;
                }

                // Warn on robot type mismatch
                if (d.robot_type && robotData.type && d.robot_type !== robotData.type) {
                    const ok = await Dialogs.confirm(
                        `Dataset was recorded with "${d.robot_type}" ` +
                        `but selected robot is "${robotData.type}".\n\n` +
                        `Recording with a different robot may produce incompatible data.`,
                        { title: 'Robot mismatch', confirmLabel: 'Continue anyway', danger: true },
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
            const ok = await Dialogs.confirm(
                `Dataset was recorded with "${d.robot_type}" ` +
                `but selected robot is "${robotData.type}".\n\n` +
                `Replaying on the wrong robot can send incorrect motor commands.`,
                { title: 'Robot mismatch', confirmLabel: 'Continue anyway', danger: true },
            );
            if (!ok) return;
        }
        endpoint = '/api/run/replay';
        body = {
            robot: robotData,
            repo_id: _resolveDatasetRepoId(d),
            root: d.root,
            episode: parseInt(epIdx),
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
            // Validate the RL Token Encoder field whenever RLT mode is selected
            // (NEW or any existing checkpoint). The actor's input dim depends
            // on the encoder's rl_token_dim; missing → state_dict size mismatch
            // crash on actor.pt load.
            {
                const rltSelEl = document.getElementById('run-hvla-rlt-select');
                const rltSelVal = rltSelEl?.value || '';
                if (rltSelVal) {
                    const tokenCkpt = document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim();
                    if (!tokenCkpt) {
                        showToast(
                            'Error',
                            'RL Token Encoder is required when RLT is enabled. ' +
                            'Set it to the trained Phase-1 encoder dir ' +
                            '(e.g. outputs/rlt_token_v4_4layer_d2048/checkpoint-10000).',
                            'error',
                        );
                        document.getElementById('run-hvla-rlt-token-ckpt')?.focus();
                        return;
                    }
                }
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
                s1_checkpoint: _selectedPolicyPath(),
                s2_checkpoint: s2Ckpt,
                task: hvlaTask,
                fps: parseInt(document.getElementById('run-policy-fps')?.value) || 30,
                s1_query_interval: parseInt(document.getElementById('run-hvla-query-interval')?.value) || null,
                denoise_steps: parseInt(document.getElementById('run-hvla-denoise-steps')?.value) || null,
                decode_subtask: document.getElementById('run-hvla-decode-subtask')?.checked || false,
                send_action_shape: document.getElementById('run-hvla-send-action-shape')?.value || 'chunk',
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
                            rlt_shared_noise_per_chunk: document.getElementById('run-hvla-rlt-shared-noise')?.checked === true,
                        };
                    }
                    // Existing checkpoint
                    const runMode = document.getElementById('run-hvla-rlt-run-mode')?.value || 'train';
                    // For ep_N snapshot entries, run_dir points to the parent
                    // run dir so continued training writes back into it
                    // rather than nesting inside the snapshot. For latest/
                    // entries run_dir equals path.
                    const selectedOpt = rltSel.options[rltSel.selectedIndex];
                    const runDir = selectedOpt?.dataset?.runDir || rltVal;
                    return {
                        rlt_mode: true,
                        rlt_checkpoint: rltVal,  // path to existing actor.pt directory
                        rlt_token_checkpoint: document.getElementById('run-hvla-rlt-token-ckpt')?.value?.trim() || null,
                        rlt_deploy: runMode === 'deploy',
                        rl_chunk_length: parseInt(document.getElementById('run-hvla-rlt-chunk-length')?.value) || 10,
                        rlt_output_dir: runDir,
                        rlt_start_engaged: rltStartEngaged,
                        // Only meaningful in train mode (deploy uses deterministic actor)
                        rlt_shared_noise_per_chunk: runMode === 'train'
                            && document.getElementById('run-hvla-rlt-shared-noise')?.checked === true,
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
                policy_path: _selectedPolicyPath(),
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
        updateRunUI(true, data.command);
        connectOutputSSE();
        if (body?.rlt_mode) {
            _startRLTPoll();
            // Remember the output dir of a freshly-launched "+ New training..." run
            // so we can auto-select its new checkpoint in the dropdown when the
            // run ends (see connectOutputSSE done handler).
            _pendingNewRltOutputDir = body.rlt_checkpoint ? null : (body.rlt_output_dir || null);
        }
        // Latency dashboard polls regardless of workflow — the teleop subprocess
        // always emits latency_snapshot.json, and the endpoint returns an empty
        // stub when the file isn't there yet, so this is safe across workflows.
        _startLatencyPoll();
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
        const data = await res.json().catch(() => ({}));
        // A graceful stop means the record subprocess saved the buffered
        // episode before exiting; say so, because the operator's next question
        // after stopping mid-reset is whether their episode survived.
        showToast('Stopped', data.graceful ? 'Episode saved, recorder exited cleanly' : 'Process stopped', 'info');
        stopObsStreamViewer();
        updateRunUI(false);
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

// ============================================================================
// Episode control (episode transitions during an active record run)
// ============================================================================

// Human-readable labels for toasts, keyed by the command sent to
// POST /api/run/control.
const _RUN_CONTROL_LABELS = {
    exit_early: 'Next episode',
    rerecord_episode: 'Re-record episode',
    stop_recording: 'Stop recording',
};

async function sendRunControl(cmd) {
    try {
        const res = await fetch('/api/run/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cmd }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            showToast('Episode control', err.detail || `Command "${cmd}" failed`, 'error');
            return;
        }
        showToast('Episode control', `${_RUN_CONTROL_LABELS[cmd] || cmd} sent`, 'info');
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

// Hotkeys mirror the record CLI's own keyboard controls. Bound once from
// runTabInit; active only while the Run tab is visible and a subprocess is
// running, and never while typing in a form field.
let _runControlHotkeysBound = false;

function _bindRunControlHotkeys() {
    if (_runControlHotkeysBound) return;
    _runControlHotkeysBound = true;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const activeTab = document.querySelector('.tab.active')?.dataset.tab;
        if (activeTab !== 'run') return;
        // Same gate as the buttons: during a teleop run, N would not advance
        // an episode — it would end the whole session.
        if (!episodeControlsAvailable(_isRunning, _runCommand)) return;
        const target = e.target;
        if (target && (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable)) return;
        const cmd = { n: 'exit_early', r: 'rerecord_episode' }[e.key.toLowerCase()];
        if (!cmd) return;
        e.preventDefault();
        sendRunControl(cmd);
    });
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
            _stopLatencyPoll();
            // Sliders only make sense during an active run — disable
            // them. The diag button is sticky (preference persists in
            // localStorage), so just refresh its tooltip to reflect the
            // staged-for-next-launch semantics.
            _setRLTSlidersEnabled(false);
            const diagOn = document.getElementById('rlt-btn-diagnostic')
                ?.classList.contains('active') || false;
            _applyDiagBtnVisual(diagOn, false);
            pollRunStatus();
            if (selectedWorkflow === 'flash') void _loadFlashRecordings();
            // Re-scan dataset sources and refresh opened datasets
            if (typeof window.refreshExpandedSources === 'function') {
                window.refreshExpandedSources();
            }
            if (typeof window.refreshOpenedDatasets === 'function') {
                window.refreshOpenedDatasets();
            }
            // Refresh the RLT checkpoint dropdown unconditionally after any
            // RLT run so the displayed episode count reflects the latest
            // saved checkpoint, not the value from when the dropdown was
            // first populated. After "+ New training..." also auto-select
            // the newly-written run dir so the next Start naturally resumes
            // from it instead of silently re-starting a fresh training.
            const newOutDir = _pendingNewRltOutputDir;
            _pendingNewRltOutputDir = null;
            _refreshRltCheckpoints().then(() => {
                if (!newOutDir) return;
                const sel = document.getElementById('run-hvla-rlt-select');
                if (!sel) return;
                // Paths in the dropdown are absolute; the user's output dir
                // may be relative — match by suffix in either direction.
                const match = [...sel.options].find(o => {
                    if (!o.value || o.value === '__new__') return false;
                    return o.value === newOutDir
                        || o.value.endsWith('/' + newOutDir)
                        || newOutDir.endsWith('/' + o.value);
                });
                if (match) {
                    sel.value = match.value;
                    sel.dispatchEvent(new Event('change'));
                }
            });
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
        // SSE can drop for many reasons: server-side close after a `done`
        // event raced with the connection teardown, network blip, browser
        // backgrounding the tab, etc. If we don't reconcile here the
        // frontend can stay locked at _isRunning=true forever even though
        // the subprocess already exited — Stop AND Launch both end up
        // greyed out and the user has to refresh the page to unstick.
        disconnectOutputSSE();
        pollRunStatus();
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
    _copyTextToClipboard(lines).then(ok => {
        if (ok) showToast('Terminal', 'Copied to clipboard', 'info');
        else showToast('Terminal', 'Copy failed', 'error');
    });
}

// Clipboard helper with fallback for non-secure contexts (HTTP over LAN IP).
// navigator.clipboard is only defined on HTTPS or localhost; otherwise fall
// back to a hidden textarea + document.execCommand('copy').
function _copyTextToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).then(() => true).catch(() => false);
    }
    return new Promise(resolve => {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '0';
        ta.style.left = '0';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        resolve(ok);
    });
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
        _reportRunFailure(status.last_error);
        updateRunUI(status.running, status.command);
        if (status.running && status.command === 'flash_tracking' && selectedWorkflow === 'flash' && flashUiMode !== 'tracking') await showFlashTracking();
        if (status.running && status.command === 'flash' && selectedWorkflow === 'flash') startFlashTelemetry();
        else if (!status.running) stopFlashTelemetry();

        // Live record-phase readout next to the episode-control buttons
        // ("recording episode 3" / "resetting" / ...). Empty when idle or
        // before the first phase transition.
        const phaseEl = document.getElementById('run-phase');
        const phase = status.running ? (status.phase || '') : '';
        if (phaseEl) {
            phaseEl.textContent = phase;
        }

        // "Next episode" means different things per phase: while recording it
        // ENDS the episode early (and keeps it), during reset it SKIPS the
        // remaining reset time and starts the next episode. Say which.
        const nextBtn = document.getElementById('run-ctrl-next');
        if (nextBtn) {
            if (phase.startsWith('recording episode')) {
                nextBtn.textContent = 'End episode → reset (N)';
                nextBtn.title = 'Finish and keep the current episode now, then enter the reset phase (hotkey: N)';
            } else if (phase === 'resetting') {
                nextBtn.textContent = 'Start next episode (N)';
                nextBtn.title = 'Skip the rest of the reset phase and start recording the next episode (hotkey: N)';
            } else {
                nextBtn.textContent = 'Next episode (N)';
                nextBtn.title = 'End the current phase early (hotkey: N)';
            }
        }

        // If running but no SSE, reconnect
        if (status.running && !runEventSource) {
            connectOutputSSE();
            if (status.command !== 'flash_prepare' && status.command !== 'flash') startObsStreamViewer();
        }
    } catch (e) {
        console.error('Status poll failed:', e);
    }
}

// ============================================================================
// UI state management
// ============================================================================

// Episode flow control only exists in lerobot-record. Enabling the buttons
// for any other run kind hands the operator a live "Next episode" during a
// teleop session — where exit_early doesn't advance an episode, it ends the
// whole run.
function episodeControlsAvailable(isRunning, command) {
    return !!isRunning && command === 'record';
}

// The reason the last run stopped, shown once. A run whose robot is missing
// a dependency dies in its own process, and the sentence naming the package
// is the only thing that can tell the operator what to do about it.
let _reportedFailure = null;

function _reportRunFailure(failure) {
    if (!failure) { _reportedFailure = null; return; }
    const key = `${failure.command}:${failure.returncode}:${failure.reason}`;
    if (key === _reportedFailure) return;  // the poll repeats; the toast should not
    _reportedFailure = key;
    showToast(`${failure.command || 'Run'} stopped`, failure.reason, 'error', 15000);
}

function updateRunUI(isRunning, command = null) {
    _isRunning = isRunning;  // mirror for _validateLaunch
    _runCommand = command;  // mirror for the N/R hotkey gate
    _flashUpdateWorkspace();
    const stopBtn = document.getElementById('run-stop-btn');
    const formInputs = document.querySelectorAll('#run-form input, #run-form select');
    const workflowBtns = document.querySelectorAll('.workflow-btn');

    if (stopBtn) stopBtn.disabled = !isRunning;

    const controlsLive = episodeControlsAvailable(isRunning, command);
    for (const id of ['run-ctrl-next', 'run-ctrl-rerecord']) {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = !controlsLive;
    }

    formInputs.forEach(el => el.disabled = isRunning && !(command === 'flash_tracking' && el.closest('#flash-tracking-setup')));
    workflowBtns.forEach(el => el.disabled = isRunning);

    const indicator = document.getElementById('run-status-indicator');
    if (indicator) {
        indicator.textContent = isRunning ? 'Running' : 'Idle';
        indicator.className = isRunning ? 'run-status running' : 'run-status idle';
    }
    // Launch button is now state-driven via _updateLaunchButton (running
    // state + form validity). Don't unconditionally disable it here —
    // _validateLaunch handles the "running" case via _isRunning.
    _updateLaunchButton();
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

// ============================================================================
// Low Bandwidth: the cameras as video, and the readouts with them
// ============================================================================
//
// Design: gui/docs/live_camera_video.md. The tiles, the joint readouts and the
// visualizer all come from one connection, so nothing here is on a clock and
// nothing waits for anything else.

document.addEventListener('DOMContentLoaded', () => {
    if (!window.VideoMode) return;
    window.VideoMode.bindSelect('run-video-mode-select', {
        usable: _liveVideoUsable,
        reason: 'Low Bandwidth needs WebRTC, which this browser does not have',
        // Mid-run, switching is the operator reaching for the other path
        // because this one is wrong: rebuild the viewer on the new one.
        onChange: (mode) => {
            if (mode === 'low-bandwidth') _openLiveVideoEarly();
            else _stopLiveVideo();
            if (obsStreamMeta || _isRunning) startObsStreamViewer();
        },
    });
});

function _liveVideoUsable() {
    return typeof RTCPeerConnection !== 'undefined' && !!window.LiveVideo;
}

function _showLiveVideoState(state, client) {
    const el = document.getElementById('run-live-video-state');
    if (!el) return;
    if (!state || state.name === 'off') { el.textContent = ''; return; }
    if (state.name === 'streaming') {
        const width = client && client.profile ? `${client.profile.width} wide` : '';
        const ages = (client ? client.cameras : []).map((c) => client.ageMs(c)).filter((a) => a !== null);
        // One number for the tab: the worst tile is what an operator would
        // notice, and a tile the page cannot place yet contributes nothing.
        const worst = ages.length ? Math.round(Math.max(...ages)) : null;
        el.textContent = ['Streaming', width, worst === null ? '' : `${worst} ms`]
            .filter(Boolean).join(' · ');
        return;
    }
    if (state.name === 'failed') { el.textContent = `Low Bandwidth failed: ${state.reason}`; return; }
    if (state.name === 'idle') { el.textContent = state.reason || 'Nothing to watch yet'; return; }
    el.textContent = 'Connecting…';
}

function _openLiveVideoEarly() {
    if (!_liveVideoUsable() || !window.VideoMode) return;
    if (window.VideoMode.stored() !== 'low-bandwidth') return;
    if (liveVideo) return;
    liveVideo = _makeLiveVideoClient();
    liveVideo.open();
}

// One client for the tab, not for a grid: the tiles are rebuilt whenever a
// run starts or the operator switches paths, and the connection must not be.
function _makeLiveVideoClient() {
    let cycles = 0;
    // A client the tab has let go of still reports: a frame painted before
    // the switch to Full Quality calls back after it, and wrote "Streaming"
    // over the stopped tab until the next switch or reload.
    const current = () => liveVideo === client;
    const client = window.LiveVideo.createClient({
        onState: (state) => {
            if (!current()) return;
            window.__liveVideoState = state;
            _showLiveVideoState(state, client);
        },
        onTrack: (camera, track) => {
            if (!current()) return;
            const video = (client.tiles || {})[camera];
            if (video) _playTrack(client, camera, video, track);
        },
        onCycle: (message) => {
            if (!current()) return;
            cycles++;
            // What the page has seen of the run, for the tests that watch it
            // from outside; the tab itself draws from the message below.
            window.__liveVideoCycles = cycles;
            window.__liveVideoLastCycle = message.cycle;
            const frame = client.urdfFrame;
            if (frame && frame.contentWindow && message.pose && message.pose.available) {
                frame.contentWindow.postMessage({ type: 'pose', source: 'state', data: message.pose }, '*');
            }
            // The bar's age moves with the pictures, not on a clock of its own.
            if (cycles % 15 === 0) _showLiveVideoState(client.state, client);
        },
    });
    client.tiles = {};
    client.urdfFrame = null;
    return client;
}

// The run's cameras exist and the grid is built: give the tiles the tracks
// the connection is already carrying, and ask for any that are missing.
function _attachLiveVideo(videoElements, urdfFrame) {
    if (!liveVideo) liveVideo = _makeLiveVideoClient();
    liveVideo.tiles = videoElements;
    liveVideo.urdfFrame = urdfFrame;
    for (const [camera, track] of Object.entries(liveVideo.tracks)) {
        const video = videoElements[camera];
        if (video && track) _playTrack(liveVideo, camera, video, track);
    }
    liveVideo.attach(Object.keys(videoElements));
}

function _playTrack(client, camera, video, track) {
    video.srcObject = new MediaStream([track]);
    // Every painted frame carries the timestamp the age is measured from;
    // reading it here is also what says the stream is running.
    if (!video.requestVideoFrameCallback) return;
    const onFrame = (now, meta) => {
        if (meta && meta.rtpTimestamp !== undefined) {
            client.notePainted(camera, meta.rtpTimestamp, Date.now() / 1000);
        }
        video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
}

function _stopLiveVideo() {
    if (liveVideo) { liveVideo.close(); liveVideo = null; }
    window.__liveVideoState = { name: 'off', reason: '' };
    // What the page has seen of a stream describes that stream; leaving it
    // behind lets a count from the last run stand in for the next one's.
    window.__liveVideoCycles = 0;
    window.__liveVideoLastCycle = null;
    _showLiveVideoState(null, null);
}

window.addEventListener('beforeunload', _stopLiveVideo);

async function startObsStreamViewer() {
    stopObsStreamViewer();
    const myGen = ++obsStreamGen;  // claim this viewer; stop() / a newer start() supersedes it

    const container = document.getElementById('rerun-viewer');
    if (!container) return;

    // Wait for the obs-stream to come up. When a run is tracking it there's no good fixed deadline
    // (robot connect + a torch.compile warmup can take seconds or minutes), so we wait for the whole
    // RUN LIFECYCLE (_isRunning, reconciled by pollRunStatus on the SSE `done`/backstop). But an
    // externally-published stream (an out-of-GUI teleop, or the data publisher) has no _isRunning to
    // bound it — so fall back to a 15s poll in that case rather than ignore the stream. Ends on: stream
    // available, the run ending, superseded (stop / a newer start bumps obsStreamGen), or — with no
    // run — the fallback deadline. NOTE: the URDF probe must stay AFTER this wait — /api/run/urdf-viz
    // is only answerable once the run's obs stream has data (same moment this loop exits), so an
    // up-front probe would latch false for every GUI-launched run and hide the visualizer tile.
    let attempts = 0;
    const maxAttempts = 30; // 30 × 500ms = 15s fallback when no run is tracking the stream
    while (obsStreamGen === myGen && (_isRunning || attempts < maxAttempts)) {
        try {
            obsStreamMeta = await (await fetch('/api/run/obs-stream/meta')).json();
            if (obsStreamMeta?.available) break;
        } catch (e) { obsStreamMeta = null; }
        await new Promise(r => setTimeout(r, 500));
        attempts++;
    }
    if (obsStreamGen !== myGen) return;  // superseded while waiting

    // Probe the URDF visualization — it's on by default, appearing as one
    // grid tile whenever the robot has a vendored URDF (see /api/run/urdf-viz).
    let urdfVizActive = false;
    try {
        const res = await fetch('/api/run/urdf-viz');
        urdfVizActive = !!(await res.json()).available;
    } catch (e) { /* probe failed; leave inactive */ }

    if (!obsStreamMeta?.available && !urdfVizActive) {
        return;  // the run ended before the stream or the URDF viz came up — nothing to show
    }

    const placeholder = document.getElementById('rerun-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    // Which path draws the cameras. One setting for both tabs, and this tab
    // can only honour Low Bandwidth where the browser has WebRTC.
    const lowBandwidth = _liveVideoUsable() && window.VideoMode
        && window.VideoMode.stored() === 'low-bandwidth';

    // Remove any old content (iframe or previous grid)
    const oldIframe = container.querySelector('iframe');
    if (oldIframe) oldIframe.remove();
    let grid = container.querySelector('.obs-cam-grid');
    if (grid) grid.remove();

    // Build the tile grid: cameras + (optionally) the URDF viz tile.
    const camKeys = obsStreamMeta?.available ? Object.keys(obsStreamMeta.image_keys) : [];
    if (camKeys.length === 0 && !urdfVizActive) return;

    const tileCount = camKeys.length + (urdfVizActive ? 1 : 0);
    grid = document.createElement('div');
    grid.className = 'obs-cam-grid';
    const cols = tileCount <= 2 ? tileCount : Math.min(tileCount, 3);
    const rows = Math.max(1, Math.ceil(tileCount / cols));
    grid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(${cols}, 1fr);
        grid-template-rows: repeat(${rows}, 1fr);
        gap: 4px;
        width: 100%; height: 100%;
        padding: 4px;
        box-sizing: border-box;
    `;

    const imgElements = {};
    const videoElements = {};  // camera -> <video> on its track, at Low Bandwidth
    const cellByKey = {};      // camera -> tile cell, for the enlarge toggle
    const overlayElements = {};  // key -> live Overlays result <img> over each tile
    // Completion-gated latest-wins overlay loading, the same helper the data tab uses.
    // Without it this tile loop assigned ov.src every tick, and reassigning an <img>
    // src ABORTS the in-flight download: an overlay that takes longer than one tick to
    // arrive never finishes, so the tile shows no overlay at all while the worker fps
    // badge reads healthy. Observed on real teleop as drawn=[] blank=['front','top']
    // in the live/diag log while the worker was detecting every object correctly.
    const overlayLoader = window.OverlayPullGate ? window.OverlayPullGate.createLoader() : null;
    const nextOverlay = (key, ov) => {
        if (!overlayLoader) return;
        const url = overlayLoader.done(key);
        if (url) ov.src = url;
    };
    for (const key of camKeys) {
        const cell = document.createElement('div');
        cell.style.cssText = 'position: relative; overflow: hidden; background: #111; border-radius: 4px;';

        // The picture: a polled JPEG, or a video element on this camera's
        // track. Everything else about the tile is the same either way.
        let img = null;
        if (lowBandwidth) {
            const video = document.createElement('video');
            video.autoplay = true;
            video.muted = true;
            video.playsInline = true;
            video.dataset.camera = key;
            video.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
            cell.appendChild(video);
            videoElements[key] = video;
        } else {
            img = document.createElement('img');
            img.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
            img.alt = key;
            cell.appendChild(img);
        }

        // Live Overlays result (RGBA PNG) composited over the obs frame.
        const ov = document.createElement('img');
        ov.className = 'overlay-layer';
        ov.alt = '';
        // onload/onerror are bound ONCE, here: the tick below must never reassign them,
        // or the loader's in-flight bookkeeping is orphaned and every camera wedges.
        ov.onload = () => { ov.style.display = 'block'; nextOverlay(key, ov); };
        ov.onerror = () => { ov.style.display = 'none'; nextOverlay(key, ov); };
        cell.appendChild(ov);
        // At Low Bandwidth the overlay is drawn on the frame before it is
        // encoded, so this layer would be the same picture a second time.
        if (lowBandwidth) ov.style.display = 'none';
        else overlayElements[key] = ov;

        if (selectedWorkflow === 'flash' && key.includes('overhead') && !key.includes('depth')) {
            const poseCanvas = document.createElement('canvas');
            poseCanvas.className = 'flash-pose-canvas';
            cell.appendChild(poseCanvas);
            if (img) img.addEventListener('load', _flashDrawOverlay);
        }

        // Same chip family as the enlarge button beside it, and as the data tab's
        // tiles -- one class rather than two hand-written inline plates that had
        // already drifted apart in inset, padding and background.
        const label = document.createElement('div');
        label.className = 'camera-chip camera-title';
        label.textContent = key;
        label.title = key;
        cell.appendChild(label);

        // Enlarge this camera to fill the grid (click again or Esc restores). A corner
        // button rather than a click on the tile: the tile surface stays free for
        // features that give clicks meaning (and stopPropagation keeps it that way).
        const zoom = document.createElement('button');
        zoom.className = 'camera-chip obs-cam-zoom';
        zoom.textContent = '⤢';
        zoom.title = 'Enlarge this camera (click again to restore)';
        zoom.addEventListener('click', (e) => { e.stopPropagation(); focusTile(key); });
        cell.appendChild(zoom);
        cell.dataset.camCell = key;  // marks a CAMERA tile, so focus can hide the rest
        cellByKey[key] = cell;

        // Per-camera latency overlay. Populated by _updateCameraOverlays
        // when the latency snapshot includes this camera (cam_* stages
        // merged across all fresh tracks). The short name (last
        // dot-separated segment) is what the tracer uses (cam_<short>_*).
        const shortName = _camShortName(key);
        const overlay = document.createElement('div');
        overlay.className = 'cam-latency-overlay';
        overlay.dataset.camShort = shortName;
        overlay.style.display = 'none';
        cell.appendChild(overlay);

        grid.appendChild(cell);
        if (img) imgElements[key] = img;
    }

    // URDF visualization tile — the in-browser three.js/urdf-loader viewer,
    // served same-origin (no separate process or port).
    let urdfFrame = null;   // the tile to push poses into, at Low Bandwidth
    if (urdfVizActive) {
        const cell = document.createElement('div');
        cell.style.cssText = 'position: relative; overflow: hidden; background: #111; border-radius: 4px;';
        const iframe = document.createElement('iframe');
        // `?urdfGhost=on` on the parent URL propagates as the iframe's
        // initial ghost state — bookmarkable, and what the screenshot
        // script keys off.
        const ghostInit = new URLSearchParams(location.search).get('urdfGhost') === 'on' ? '&ghost=on' : '';
        // `pushed` stops the tile polling for a pose: at Low Bandwidth it is
        // fed from the same stream as the pictures, one message per cycle.
        const pushed = lowBandwidth ? '&pushed=1' : '';
        iframe.src = `/static/urdf_viz.html?v=8${ghostInit}${pushed}`;
        if (lowBandwidth) urdfFrame = iframe;
        iframe.style.cssText = 'width: 100%; height: 100%; border: none; background: #1a1a1a;';
        iframe.title = 'Robot visualizer';
        cell.appendChild(iframe);
        const label = document.createElement('div');
        label.className = 'camera-chip camera-title';
        label.textContent = 'visualizer';
        cell.appendChild(label);
        grid.appendChild(cell);
    }

    container.appendChild(grid);
    _flashDrawOverlay();
    if (lowBandwidth) _attachLiveVideo(videoElements, urdfFrame);
    // Click / drag-a-box on a tile to segment what is under it — one delegated handler for
    // the whole grid (see Overlays.installTileGestures). Run tab only — see clickCapable().
    if (window.Overlays && window.Overlays.installTileGestures) window.Overlays.installTileGestures(grid, 'live');

    // Enlarge one tile: hide the others rather than restyling each, so restore is
    // trivial; the grid template is captured so it comes back exactly as it was.
    const gridCols = grid.style.gridTemplateColumns;
    const gridRows = grid.style.gridTemplateRows;
    let focusedTile = null;
    function focusTile(key) {
        focusedTile = focusedTile === key ? null : key;
        for (const [k, cell] of Object.entries(cellByKey)) {
            cell.style.display = (!focusedTile || k === focusedTile) ? '' : 'none';
        }
        const others = grid.querySelectorAll(':scope > div:not([data-cam-cell])');
        others.forEach((el) => { el.style.display = focusedTile ? 'none' : ''; });
        grid.style.gridTemplateColumns = focusedTile ? '1fr' : gridCols;
        grid.style.gridTemplateRows = focusedTile ? '1fr' : gridRows;
    }
    if (selectedWorkflow === 'flash') {
        const overhead = camKeys.find(key => key.includes('overhead') && !key.includes('depth'));
        if (overhead) focusTile(overhead);
    }
    // Deliberately NO Esc hotkey: six other components already bind Escape (log modal,
    // process modal, bug-report and add-host dialogs, data-tab handlers), so a global
    // listener here would fire alongside whichever modal is topmost — and, added per
    // grid build, it would also accumulate across stream rebuilds. The button is the
    // whole interface.

    // Poll camera frames at ~10fps — the ones drawn as images, which at Low
    // Bandwidth is none of them: those tiles hold a video on their own track,
    // and the pose rides the same stream, so the only thing left on a clock
    // is the subtask text.
    let frameSeq = 0;
    obsStreamTimer = setInterval(() => {
        const seq = ++frameSeq;
        for (const key of Object.keys(imgElements)) {
            const img = imgElements[key];
            if (!img) continue;
            // Append seq to bust browser cache
            img.src = `/api/run/obs-stream/image/${encodeURIComponent(key)}?_=${seq}`;
            const ov = overlayElements[key];
            if (ov) {
                const url = (window.Overlays && window.Overlays.liveFrameUrl) ? window.Overlays.liveFrameUrl(key, seq) : null;
                if (url) {
                    const now = overlayLoader ? overlayLoader.request(key, url) : url;
                    if (now) ov.src = now;  // else: queued as pending, assigned on completion
                } else {
                    ov.style.display = 'none';
                    ov.removeAttribute('src');
                    // Scoped to THIS camera: an unscoped reset here would clear the
                    // in-flight flag of the cameras that ARE on — every tick, for every
                    // camera that is off — disabling the gate entirely.
                    if (overlayLoader) overlayLoader.reset(key);
                }
            }
        }
        _pollSubtaskOverlay();
    }, 50);
}

function stopObsStreamViewer() {
    obsStreamGen++;  // cancel any in-flight "wait for the obs-stream" loop
    if (liveVideo) {
        // The tiles are about to be removed, and a track left attached keeps
        // its per-frame callback running on an element nothing shows.
        for (const video of Object.values(liveVideo.tiles || {})) {
            if (video) video.srcObject = null;
        }
        // These tracks belong to the run that just ended, and the encoders
        // behind them read a tap that is about to be unlinked. Build the
        // connection again, now, while nothing is waiting for a picture:
        // the next run then finds it open, which is the whole point of
        // opening early.
        _stopLiveVideo();
        _openLiveVideoEarly();
    }
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
// Output dir of the most recent "+ New training..." RLT launch. When the
// run ends we refresh the checkpoint dropdown and auto-select the entry
// that matches this path, so the next launch resumes from it instead of
// starting a fresh training.
let _pendingNewRltOutputDir = null;

function _switchBottomTab(tab) {
    document.querySelectorAll('.run-bottom-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tab);
    });
    const outputPanel = document.getElementById('run-bottom-panel-output');
    const rlPanel = document.getElementById('run-bottom-panel-rl');
    const latencyPanel = document.getElementById('run-bottom-panel-latency');
    if (outputPanel) outputPanel.style.display = (tab === 'output') ? '' : 'none';
    if (rlPanel) rlPanel.style.display = (tab === 'rl') ? '' : 'none';
    if (latencyPanel) latencyPanel.style.display = (tab === 'latency') ? '' : 'none';
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

// =============================================================================
// Latency dashboard — polls /api/run/latency-sources at 1 Hz to discover
// which loops are publishing snapshots, then fetches each fresh source and
// renders it as its own track. Multi-thread loops (e.g. HVLA's main +
// inference threads) thus appear as two stacked tracks of the same process;
// single-thread loops (teleop, record) render as one. The track DOM block
// is built from a template per source key — the rendering functions here
// take a ``scope`` element and query data-role children inside it, so the
// same code drives one track or many.
// =============================================================================

let _latencyPollTimer = null;
// Per-track render state keyed by source key (e.g. "teleop", "hvla_main").
// Each entry holds the latest iterations dict (for the percentile picker
// to re-render without a fresh fetch) and the only-widen Gantt X range
// (so per-iteration jitter doesn't move the axis bounds visibly).
const _trackStates = new Map();

function _showLatencyTab(show) {
    const tab = document.getElementById('run-bottom-tab-latency');
    if (tab) tab.style.display = show ? '' : 'none';
}

// Camera key in the obs stream is the full feature path (e.g. "top" or
// "observation.images.top"); the latency tracer uses just the last
// dot-segment to build "cam_<short>_stale_ms" / "cam_<short>_period_ms".
// This must match the cam_key used by the robot's get_observation().
function _camShortName(featureKey) {
    const idx = featureKey.lastIndexOf('.');
    return idx >= 0 ? featureKey.slice(idx + 1) : featureKey;
}

// Color palette for span bars in the Gantt. Picked to be distinct without
// being garish; consistent across iterations so the operator builds muscle
// memory ("the wide green bar is process_action").
const _GANTT_SPAN_COLORS = {
    // Teleop / record
    get_observation: '#4fc3f7',
    process_obs:     '#9b59b6',
    process_action:  '#2ecc71',
    action_send:     '#e67e22',
    infer_total:     '#e74c3c',
    inference:       '#e74c3c',
    infer_forward:   '#e74c3c',  // legacy V2 doc name; kept for back-compat
    dataset_write:   '#95a5a6',
    video_encode:    '#7f8c8d',
    // HVLA main loop
    publish_obs:     '#9b59b6',
    get_chunk:       '#f1c40f',
    // HVLA inference thread (GPU stages, sequential on the CUDA stream)
    batch_prep:      '#3498db',
    enc_obs:         '#5dade2',
    rl_tok:          '#af7ac5',
    s1_denoise:      '#e74c3c',
    actor:           '#f39c12',
};
const _GANTT_DEFAULT_SPAN_COLOR = '#aaaaaa';

// Snap the X-axis range to multiples of this many ms so small per-iteration
// jitter doesn't keep redrawing the timeline at slightly different scales.
const _GANTT_AXIS_STEP_MS = 10;

// Pick a "nice" tick interval (1, 2, 5, 10, 20, 50, ... ms) that produces
// roughly `targetCount` ticks across `range_ms`. Standard chart-axis trick:
// scale to a power of 10, snap to {1, 2, 5}, scale back. Avoids ugly tick
// values like "13.7 ms" between gridlines.
function _niceTickStep(range_ms, targetCount) {
    const raw = range_ms / Math.max(targetCount, 1);
    const exp = Math.floor(Math.log10(raw));
    const base = Math.pow(10, exp);
    const norm = raw / base;
    let nice;
    if (norm <= 1) nice = 1;
    else if (norm <= 2) nice = 2;
    else if (norm <= 5) nice = 5;
    else nice = 10;
    return nice * base;
}

// One-time outliers (torch.compile autotune, GC pause, etc.) should NOT
// stretch the shared Gantt axis. A single 20-second compile spike at
// step=0 would otherwise squish every normal iteration's bars (10-90 ms)
// into a sub-pixel sliver against the left edge. Cap any iteration's
// contribution to the union at this many times the p99 reference. The
// outlier card itself still renders fine on its own when explicitly
// selected (the cap is only on the SHARED axis range).
const _GANTT_OUTLIER_MULTIPLIER = 10;
// Absolute floor for the cap — when p99 is very small (warming up, few
// samples), don't let the multiplier shrink the axis below something
// reasonable for a 30 Hz loop.
const _GANTT_OUTLIER_FLOOR_MS = 200;

function _computeStableRange(iterations) {
    // Pessimistic union over the representative iterations so switching
    // between median/p95/max in the dropdown doesn't rescale the axis —
    // EXCEPT we cap each iteration's contribution against a reference
    // derived from p99/p95 so a compile spike can't pollute the axis.
    const ref = iterations.p99?.loop_dt_ms ?? iterations.p95?.loop_dt_ms ?? 0;
    const cap = Math.max(_GANTT_OUTLIER_FLOOR_MS, ref * _GANTT_OUTLIER_MULTIPLIER);

    let minMs = 0;
    let maxMs = 0;
    for (const rec of Object.values(iterations)) {
        if (!rec) continue;
        // Skip records whose loop_dt is wildly above the typical iteration
        // — they're one-time outliers (compile, eviction, GC) and surfacing
        // them via the shared axis would squish every other bar. Their own
        // card / dropdown selection still works because _drawGantt uses
        // whatever range we pass and a single-record render isn't subject
        // to this filter.
        if ((rec.loop_dt_ms || 0) > cap) continue;
        if ((rec.loop_dt_ms || 0) > maxMs) maxMs = rec.loop_dt_ms;
        for (const [, [s, e]] of Object.entries(rec.spans || {})) {
            if (s < minMs) minMs = s;
            if (e > maxMs) maxMs = e;
        }
        for (const ev of Object.values(rec.cam_events || {})) {
            if (ev.captured_at_ms < minMs) minMs = ev.captured_at_ms;
            if (ev.consumed_at_ms > maxMs) maxMs = ev.consumed_at_ms;
        }
    }
    // Round to a coarse grid so small jitter doesn't move the bounds.
    const step = _GANTT_AXIS_STEP_MS;
    minMs = Math.floor(minMs / step) * step;
    maxMs = Math.ceil(maxMs / step) * step;
    if (maxMs - minMs < step) maxMs = minMs + step;
    return [minMs, maxMs];
}

function _renderGantt(scope, iterations, state) {
    state.latestIterations = iterations;
    const select = scope.querySelector('[data-role="gantt-pick"]');
    const which = (select && select.value) || 'aggregate_median';
    const rec = iterations[which];
    const empty = scope.querySelector('[data-role="gantt-empty"]');
    const canvas = scope.querySelector('[data-role="gantt-canvas"]');
    if (!canvas) return;

    if (!rec || !rec.spans || Object.keys(rec.spans).length === 0) {
        if (empty) empty.style.display = '';
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        const meta = scope.querySelector('[data-role="gantt-meta"]');
        if (meta) meta.textContent = '';
        const legend = scope.querySelector('[data-role="gantt-legend"]');
        if (legend) legend.innerHTML = '';
        return;
    }
    if (empty) empty.style.display = 'none';

    const meta = scope.querySelector('[data-role="gantt-meta"]');
    if (meta) {
        if (rec.synthetic) {
            const p = rec.percentile != null ? `p${Math.round(rec.percentile)}` : 'aggregate';
            const n = rec.n_aggregated || 0;
            meta.textContent = `${p} of ${n} iters · loop ${(rec.loop_dt_ms || 0).toFixed(1)} ms`;
        } else {
            meta.textContent = `step ${rec.step ?? '?'} · loop ${(rec.loop_dt_ms || 0).toFixed(1)} ms`;
        }
    }

    // Stable range: compute from all reps, then take the union with the
    // previous render's range (only widen, never shrink). Keeps the axis
    // visually stable even as new outliers stretch the worst case.
    //
    // Exception: if the new computed range is MUCH smaller than the cached
    // one (>= 10x), bust the cache. The cached value was inflated by an
    // earlier outlier (e.g. a torch.compile spike at step=0 polluting the
    // axis for the whole session). Without this, one bad iteration locks
    // a 20-second axis on a 30 Hz loop, making every normal bar render at
    // < 1 px width.
    const [newMin, newMax] = _computeStableRange(iterations);
    const _STALE_CACHE_RATIO = 10;
    const cachedSpan = state.ganttRange ? state.ganttRange[1] - state.ganttRange[0] : 0;
    const newSpan = newMax - newMin;
    if (state.ganttRange === null || (newSpan > 0 && cachedSpan > newSpan * _STALE_CACHE_RATIO)) {
        state.ganttRange = [newMin, newMax];
    } else {
        state.ganttRange = [Math.min(state.ganttRange[0], newMin), Math.max(state.ganttRange[1], newMax)];
    }

    _drawGantt(canvas, rec, state.ganttRange);
    _renderGanttLegend(scope, rec);
}

function _renderGanttLegend(scope, rec) {
    const legend = scope.querySelector('[data-role="gantt-legend"]');
    if (!legend) return;
    legend.innerHTML = '';
    const spans = rec.spans || {};
    const camEvents = rec.cam_events || {};
    // Span entries with matching swatch colors and per-stage durations so
    // the legend is also a quick numeric breakdown when a bar is too short
    // to label inside the canvas.
    for (const [name, [s, e]] of Object.entries(spans)) {
        const item = document.createElement('div');
        item.className = 'latency-gantt-legend-item';
        const color = _GANTT_SPAN_COLORS[name] || _GANTT_DEFAULT_SPAN_COLOR;
        item.innerHTML = `
            <span class="latency-gantt-legend-swatch" style="background:${color}"></span>
            <span>${name}</span>
            <span class="latency-gantt-legend-dur">${(e - s).toFixed(1)} ms</span>
        `;
        legend.appendChild(item);
    }
    // Cameras as a single legend row (they all share the diamond marker
    // color); per-camera staleness is already in the Cameras card row.
    const camKeys = Object.keys(camEvents);
    if (camKeys.length > 0) {
        const camRow = document.createElement('div');
        camRow.className = 'latency-gantt-legend-cam';
        camRow.innerHTML = `
            <span class="latency-gantt-legend-cam-marker">◆</span>
            <span>cameras (${camKeys.length})</span>
        `;
        legend.appendChild(camRow);
    }
}

function _drawGantt(canvas, rec, [minMs, maxMs]) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const spans = rec.spans || {};
    const camEvents = rec.cam_events || {};
    const loopDt = rec.loop_dt_ms || 0;

    const padL = 6, padR = 12, padT = 16, padB = 18;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xOf = ms => padL + plotW * (ms - minMs) / (maxMs - minMs);

    // Vertical zero line — separates background camera grab (left) from
    // in-iteration work (right). Subtle but important visually.
    const zeroX = xOf(0);
    ctx.strokeStyle = '#444';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(zeroX, padT - 2);
    ctx.lineTo(zeroX, H - padB + 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('iter_start', zeroX, padT - 4);

    // Loop-end line (right edge of the iteration's work).
    const endX = xOf(loopDt);
    ctx.strokeStyle = '#444';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(endX, padT - 2);
    ctx.lineTo(endX, H - padB + 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('action_sent', endX, padT - 4);

    // X-axis: pick a "nice" tick interval based on the visible range, then
    // draw subtle vertical gridlines + ms labels at each tick. Lets the
    // operator read camera staleness (or any duration) directly off the
    // canvas instead of guessing from bar lengths.
    const range = maxMs - minMs;
    const tickStep = _niceTickStep(range, 8);  // aim for ~8 ticks max
    const firstTick = Math.ceil(minMs / tickStep) * tickStep;
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    // Edge-label width (corners always draw "{minMs} ms" / "{maxMs} ms").
    // Reserve a band around each corner so tick labels never collide
    // with the corner labels — that was the "-40 -40ms" overlap.
    const leftEdgeX = padL;
    const rightEdgeX = W - padR;
    const cornerLabelHalfWidth = 32;  // covers "-180 ms" at 10 px monospace
    for (let v = firstTick; v <= maxMs; v += tickStep) {
        const x = xOf(v);
        // Skip ticks too close to the iter_start / action_sent lines so
        // their labels don't visually collide with those distinguished ones.
        const tooCloseToZero = Math.abs(x - zeroX) < 14;
        const tooCloseToEnd = Math.abs(x - endX) < 14;
        // Skip ticks too close to either corner — the corner extent
        // labels always render and would visually merge with the tick.
        const tooCloseToLeft = (x - leftEdgeX) < cornerLabelHalfWidth;
        const tooCloseToRight = (rightEdgeX - x) < cornerLabelHalfWidth;
        if (!tooCloseToZero && !tooCloseToEnd && !tooCloseToLeft && !tooCloseToRight) {
            ctx.strokeStyle = '#1f1f3a';
            ctx.beginPath();
            ctx.moveTo(x, padT - 2);
            ctx.lineTo(x, H - padB);
            ctx.stroke();
            ctx.fillStyle = '#666';
            ctx.fillText(`${v}`, x, H - 4);
        }
    }
    // Always show the extent labels at the corners as a fallback for
    // narrow ranges where no tick happens to land near the edges.
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${minMs.toFixed(0)} ms`, padL, H - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxMs.toFixed(0)} ms`, W - padR, H - 4);

    // Reserve a fixed sliver at the bottom for the camera markers (one
    // sub-row per camera so labels don't stack on top of each other).
    const camKeys = Object.keys(camEvents);
    const camRowCount = Math.max(camKeys.length, 1);
    const camRowH = 12;
    const camAreaH = camKeys.length > 0 ? (camRowCount * camRowH + 4) : 0;

    // Lay out span bars in declaration order in the remaining vertical space.
    const spanNames = Object.keys(spans);
    const spanCount = Math.max(spanNames.length, 1);
    const spanRegionH = plotH - camAreaH;
    // Row height bumped from 20→30 max to give 10 px font + 4 px bar
    // margin comfortable headroom; otherwise labels clip on dense
    // iterations with 5+ spans plus 4 camera rows.
    const rowH = Math.min(30, Math.max(15, spanRegionH / spanCount));

    spanNames.forEach((name, i) => {
        const [s, e] = spans[name];
        const x0 = xOf(s);
        const x1 = xOf(e);
        const y = padT + i * rowH;
        const w = Math.max(2, x1 - x0);
        ctx.fillStyle = _GANTT_SPAN_COLORS[name] || _GANTT_DEFAULT_SPAN_COLOR;
        ctx.fillRect(x0, y, w, rowH - 4);
        // Label — name and duration if there's room. Drawn outside the bar
        // (to the right) when the bar is too narrow to hold any text, so
        // names are always readable.
        const dur = e - s;
        ctx.fillStyle = '#0a0a1a';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        const labelInside = w > 80
            ? `${name}  ${dur.toFixed(1)} ms`
            : (w > 36 ? name : '');
        if (labelInside) {
            ctx.fillText(labelInside, x0 + 3, y + rowH - 7);
        } else {
            // Place the label outside the bar to the right, in muted color.
            ctx.fillStyle = '#aaa';
            ctx.fillText(`${name} ${dur.toFixed(1)}ms`, x1 + 4, y + rowH - 7);
        }
    });

    // Camera markers — one sub-row per camera so the captured-→-consumed
    // line and the diamonds don't overlap with sibling cameras' markers.
    // Labels are intentionally NOT drawn here (per-camera staleness/period
    // is shown in the Cameras card row below the panel; duplicating it on
    // the Gantt was the source of overlapping text).
    const camAreaTop = H - padB - camAreaH + 2;
    camKeys.forEach((key, i) => {
        const ev = camEvents[key];
        const x = xOf(ev.captured_at_ms);
        const cx = xOf(ev.consumed_at_ms);
        const y = camAreaTop + i * camRowH + camRowH / 2;

        // Stale segment.
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(cx, y);
        ctx.stroke();

        // Capture diamond (filled).
        ctx.fillStyle = '#4fc3f7';
        ctx.beginPath();
        ctx.moveTo(x, y - 3); ctx.lineTo(x + 3, y); ctx.lineTo(x, y + 3); ctx.lineTo(x - 3, y); ctx.closePath();
        ctx.fill();
        // Consume diamond (hollow).
        ctx.strokeStyle = '#4fc3f7';
        ctx.beginPath();
        ctx.moveTo(cx, y - 2.5); ctx.lineTo(cx + 2.5, y); ctx.lineTo(cx, y + 2.5); ctx.lineTo(cx - 2.5, y); ctx.closePath();
        ctx.stroke();

        // Tiny camera label at the right edge so the row is identifiable.
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(key, padL + plotW, y + 3);
    });
}

// Re-render the Gantt when the operator switches the percentile picker on
// any track. The track scope is the closest .latency-track ancestor; its
// state lives in _trackStates keyed by data-track-key.
document.addEventListener('change', e => {
    if (e.target && e.target.matches('[data-role="gantt-pick"]')) {
        const track = e.target.closest('.latency-track');
        if (!track) return;
        const state = _trackStates.get(track.dataset.trackKey);
        if (state) _renderGantt(track, state.latestIterations || {}, state);
    }
});

function _updateCameraOverlays(stages) {
    const overlays = document.querySelectorAll('.cam-latency-overlay');
    if (!overlays.length) return;
    for (const overlay of overlays) {
        const short = overlay.dataset.camShort;
        const stale = stages[`cam_${short}_stale_ms`];
        const period = stages[`cam_${short}_period_ms`];
        if (!stale) {
            overlay.style.display = 'none';
            continue;
        }
        const fps = (period && period.p50) ? (1000 / period.p50) : null;
        const stalePalette = stale.p95 < 40 ? 'ok' : stale.p95 < 70 ? 'warn' : 'bad';
        overlay.style.display = '';
        overlay.className = `cam-latency-overlay ${stalePalette}`;
        const fpsTxt = fps != null ? `${fps.toFixed(1)}&nbsp;Hz` : '—';
        overlay.innerHTML = `
            <div class="cam-latency-line">${fpsTxt}</div>
            <div class="cam-latency-line">stale&nbsp;${stale.p50.toFixed(0)}/${stale.p95.toFixed(0)}&nbsp;ms</div>
        `;
    }
}

function _startLatencyPoll() {
    _stopLatencyPoll();
    _showLatencyTab(true);
    // 1 Hz matches the snapshot writer interval; faster polling gains nothing.
    _latencyPollTimer = setInterval(_fetchLatencyMetrics, 1000);
    _fetchLatencyMetrics();
}

function _stopLatencyPoll() {
    if (_latencyPollTimer) { clearInterval(_latencyPollTimer); _latencyPollTimer = null; }
    // Reset per-track state so the next session starts fresh instead of
    // inheriting bounds / iteration caches from the previous run.
    _trackStates.clear();
    const container = document.getElementById('latency-tracks');
    if (container) {
        container.querySelectorAll('.latency-track').forEach(el => el.remove());
    }
}

async function _fetchLatencyMetrics() {
    // Discover available sources first; render only the fresh ones. Stale
    // sources (>5s old) are dropped from the display until they publish
    // again — this matches the polling cadence on the producer side.
    let sourcesData;
    try {
        const res = await fetch('/api/run/latency-sources');
        if (!res.ok) return;
        sourcesData = await res.json();
    } catch (e) { return; }
    const fresh = (sourcesData.sources || []).filter(s => s.fresh);
    _reconcileTracks(fresh);

    // Fetch each fresh source in parallel and render. Cameras live on the
    // robot, not on the inference thread; we collect cam_* stages from
    // every track and feed them to _updateCameraOverlays once so the
    // per-frame badges work regardless of which track owns the cameras.
    const results = await Promise.all(fresh.map(async source => {
        try {
            const res = await fetch(`/api/run/latency-metrics?source=${encodeURIComponent(source.key)}`);
            if (!res.ok) return null;
            return { source, data: await res.json() };
        } catch (e) { return null; }
    }));
    const mergedCamStages = {};
    for (const result of results) {
        if (!result) continue;
        const { source, data } = result;
        const trackEl = document.querySelector(`.latency-track[data-track-key="${CSS.escape(source.key)}"]`);
        if (!trackEl) continue;
        const state = _trackStates.get(source.key);
        if (!state) continue;
        _updateTrack(trackEl, data, state);
        for (const [k, v] of Object.entries(data.stages || {})) {
            if (k.startsWith('cam_')) mergedCamStages[k] = v;
        }
    }
    _updateCameraOverlays(mergedCamStages);
}

function _reconcileTracks(sources) {
    const container = document.getElementById('latency-tracks');
    if (!container) return;
    const noSources = document.getElementById('latency-no-sources');
    if (noSources) noSources.style.display = sources.length === 0 ? '' : 'none';

    // Remove tracks for sources that disappeared.
    const wantKeys = new Set(sources.map(s => s.key));
    container.querySelectorAll('.latency-track').forEach(el => {
        if (!wantKeys.has(el.dataset.trackKey)) {
            _trackStates.delete(el.dataset.trackKey);
            el.remove();
        }
    });

    // Add tracks that are new this poll.
    const have = new Set(
        [...container.querySelectorAll('.latency-track')].map(el => el.dataset.trackKey)
    );
    for (const source of sources) {
        if (have.has(source.key)) continue;
        container.appendChild(_makeTrackBlock(source));
        _trackStates.set(source.key, { latestIterations: {}, ganttRange: null });
    }
}

// Build a self-contained track DOM block. data-role attributes scope the
// Loop Health grid, Gantt selector / canvas / legend, camera grid, and
// meta footer so the rendering functions can find them with querySelector
// inside the track element instead of using global IDs.
function _makeTrackBlock(source) {
    const track = document.createElement('div');
    track.className = 'latency-track';
    track.dataset.trackKey = source.key;
    const display = source.loop_kind || source.key;
    track.innerHTML = `
        <div class="latency-track-header">${display}</div>
        <div class="latency-section">
            <div class="latency-section-title">Loop Health</div>
            <div class="latency-grid" data-role="grid-system"></div>
        </div>
        <div class="latency-section">
            <div class="latency-section-title">
                Iteration timeline
                <span class="latency-gantt-controls">
                    <select data-role="gantt-pick" title="Aggregate views synthesize a typical iteration where each stage independently shows its own percentile. Sample views show one real captured iteration whose per-stage values are whatever happened in that one iteration.">
                        <optgroup label="Aggregate (per-stage percentile)">
                            <option value="aggregate_median" selected>median</option>
                            <option value="aggregate_p95">p95</option>
                        </optgroup>
                        <optgroup label="Sample (real iteration)">
                            <option value="median">median sample</option>
                            <option value="p95">p95 sample</option>
                            <option value="p99">p99 sample</option>
                            <option value="max">worst (max loop)</option>
                            <option value="latest">latest</option>
                        </optgroup>
                    </select>
                    <span data-role="gantt-meta"></span>
                </span>
            </div>
            <div class="latency-gantt-panel" title="Per-iteration timeline. Bars are tracer spans; diamonds are camera frame captures (often before t=0 because cameras grab in a background thread). Gaps in the timeline indicate uninstrumented work; overlaps indicate parallel stages.">
                <div class="latency-gantt-canvas-wrap"><canvas data-role="gantt-canvas"></canvas></div>
                <div class="latency-gantt-legend" data-role="gantt-legend"></div>
                <div class="latency-empty" data-role="gantt-empty">No timeline yet — waiting for first snapshot…</div>
            </div>
        </div>
        <div class="latency-section">
            <div class="latency-section-title">Cameras</div>
            <div class="latency-cam-grid" data-role="cam-grid">
                <div class="latency-empty">No camera data — this track may not own cameras (e.g. inference thread).</div>
            </div>
        </div>
        <div class="latency-meta">
            <span data-role="meta-records">0 records</span>
            <span data-role="meta-dropped"></span>
            <span data-role="meta-updated"></span>
        </div>
    `;
    return track;
}

// Color thresholds for the system cards.
//
// Two of these (loop_dt_ms, get_observation_ms) are derived from the loop's
// target_period_ms (1000/fps), which the snapshot writer publishes — so
// they automatically scale to whatever FPS the teleop session is running.
// The rest are heuristic constants tuned for typical Feetech / 30 Hz
// camera setups; calibration drift detection (V2) will replace them.
//
// `ok`   → green: comfortably within budget.
// `warn` → yellow: approaching or just over the budget.
// `bad`  → red: clearly over budget; investigate.
function _budgetsFromTarget(targetPeriodMs) {
    // Fall back to 60 Hz when no target period is published (e.g. a session
    // that started before this field existed, or non-teleop loops).
    const t = targetPeriodMs || (1000 / 60);
    return {
        loop_dt_ms:         { ok: 0.7 * t, warn: t },         // 70% / 100% of FPS budget
        get_observation_ms: { ok: 0.4 * t, warn: 0.7 * t },   // obs read shouldn't dominate
        action_send_ms:     { ok: 2,       warn: 5 },         // sync_write is fire-and-forget
    };
}

// Per-card explanatory tooltips. Shown on hover via the standard `title`
// attribute. Keep terse — they should answer "what is this and why does
// the color change" in two sentences.
const _LAT_TOOLTIPS = {
    loop_dt_ms: 'Wall-clock time per iteration (work only; sleep excluded). The per-iteration budget is 1000 ms ÷ target FPS — at 60 fps that is 16.67 ms; at 30 fps, 33.33 ms. Color is driven by the p95 tail (the value on the right), not p50: a typical iteration may sit well under budget while the tail still hits overrun. Yellow at p95 ≥ 70% of budget; red when p95 exceeds it.',
    get_observation_ms: 'Cost of robot.get_observation(): the follower bus.sync_read("Present_Position") plus per-camera read_latest() calls. Cameras are cached by their grab thread so the camera reads are essentially free; this number is dominated by the motor sync_read (Feetech/Dynamixel round-trip + any retries).',
    action_send_ms: 'Cost of bus.sync_write("Goal_Position"). Fire-and-forget on Feetech, so this is bus-TX only — not motor motion.',
    overrun: 'Fraction of iterations whose work time exceeded the per-iteration budget (1000 ms ÷ target FPS). Sleep time is excluded — this only fires when the work itself is too slow to sustain the target rate.',
    cam_stale: 'Age of the cached camera frame at the moment the consumer reads it (now − cam.latest_timestamp). Yellow when p95 exceeds ~1.2× the camera frame period.',
    cam_period: 'Wall-clock interval between successive frame captures by the camera grab thread. Inverse is effective FPS.',
};

function _classifyLatency(value, budget) {
    if (budget == null) return '';
    if (value <= budget.ok) return 'ok';
    if (value <= budget.warn) return 'warn';
    return 'bad';
}

function _renderLatencyCard(parent, key, title, p50, p95, budget, sparkData, tooltip) {
    const card = document.createElement('div');
    card.className = 'latency-card';
    if (tooltip) card.title = tooltip;
    // Color reflects the TAIL (p95) — that's what determines whether the
    // loop is at risk of overrun under load, not the typical-case p50.
    // The headline value shows BOTH so the operator can see at a glance
    // which number is driving the color; the colored half is p95.
    const cls = _classifyLatency(p95 != null ? p95 : p50, budget);
    const p95Html = p95 != null
        ? ` <span class="latency-card-p95 ${cls}">p95 ${p95.toFixed(1)}</span>`
        : '';
    card.innerHTML = `
        <div class="latency-card-title">${title}</div>
        <div class="latency-card-value"><span class="latency-card-p50">p50 ${p50.toFixed(1)}</span>${p95Html}</div>
        <canvas data-spark-key="${key}"></canvas>
    `;
    parent.appendChild(card);
    if (sparkData && sparkData.length > 0) {
        const canvas = card.querySelector('canvas');
        canvas.id = `latency-spark-${key}`;
        // Reuse the same sparkline primitive RLT uses; auto-scale (no fixed range).
        drawChart(canvas.id, {series: [{data: sparkData, color: '#4fc3f7'}]});
    }
}

function _updateTrack(scope, data, state) {
    const stages = data.stages || {};
    const series = data.series || {};
    const targetPeriodMs = data.target_period_ms || null;
    const budgets = _budgetsFromTarget(targetPeriodMs);

    // Health badge — driven by snapshot.health.issues (populated by the
    // LoopHealthDetector on the producer side). Hidden when healthy; shows
    // an "error"-colored badge when any issue's severity is error, otherwise
    // a "warn"-colored one. Tooltip lists each issue's message so the
    // operator can drill in without leaving the panel.
    const issues = (data.health && data.health.issues) || [];
    const header = scope.querySelector('.latency-track-header');
    let badge = scope.querySelector('.latency-track-health');
    if (header) {
        if (issues.length === 0) {
            if (badge) badge.remove();
        } else {
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'latency-track-health';
                header.appendChild(badge);
            }
            const hasError = issues.some(i => i.severity === 'error');
            badge.classList.toggle('error', hasError);
            badge.classList.toggle('warn', !hasError);
            badge.textContent = `${issues.length} issue${issues.length > 1 ? 's' : ''}`;
            badge.title = issues.map(i => `[${i.severity}] ${i.message}`).join('\n');
        }
    }

    // System metrics row — sparkline cards for the stages we want to track
    // over time. Stages not in this set still appear in the Gantt below;
    // the Loop Health row is just the headline numbers per loop kind.
    const sysGrid = scope.querySelector('[data-role="grid-system"]');
    if (sysGrid) {
        sysGrid.innerHTML = '';
        // Top-row metrics are the LOAD-BEARING headline. Only show stats
        // that are at a comparable level of granularity:
        //   - loop_dt_ms: the umbrella (includes every stage, sleep
        //     excluded)
        //   - overrun_ratio: a derived rate, conceptually different from
        //     a duration but still a headline KPI
        // Per-stage breakdowns belong in the Gantt below, NOT here —
        // mixing them at top creates misleading impression that loop
        // ≈ obs + send (it doesn't; process_obs, process_action,
        // inference, dataset_write are also inside loop and would
        // distort the picture if absent from the headline).
        const sysOrder = [
            { key: 'loop_dt_ms', title: 'Loop' },
        ];
        for (const item of sysOrder) {
            const stage = stages[item.key];
            if (!stage) continue;
            const sparkPoints = (series[item.key] || []).map(([_t, v]) => v);
            _renderLatencyCard(sysGrid, item.key, item.title,
                stage.p50 ?? 0, stage.p95 ?? null,
                budgets[item.key], sparkPoints, _LAT_TOOLTIPS[item.key]);
        }

        // Overrun ratio.
        const overrun = (data.overrun_ratio || 0) * 100;
        const ovCard = document.createElement('div');
        ovCard.className = 'latency-card';
        ovCard.title = _LAT_TOOLTIPS.overrun + (
            targetPeriodMs ? ` Budget: ${targetPeriodMs.toFixed(2)} ms (${(1000 / targetPeriodMs).toFixed(0)} Hz target).` : ''
        );
        const ovCls = overrun < 1 ? 'ok' : overrun < 5 ? 'warn' : 'bad';
        const budgetLabel = targetPeriodMs
            ? `> ${targetPeriodMs.toFixed(1)} ms (${(1000 / targetPeriodMs).toFixed(0)} Hz)`
            : 'budget unknown';
        ovCard.innerHTML = `
            <div class="latency-card-title">Overrun</div>
            <div class="latency-card-value ${ovCls}">${overrun.toFixed(1)}%</div>
            <div class="latency-card-sub">Iterations with work time ${budgetLabel}.</div>
        `;
        sysGrid.appendChild(ovCard);
    }

    // Gantt timeline — pick one representative iteration and draw it.
    _renderGantt(scope, data.iterations || {}, state);

    // Per-camera cards.
    const camGrid = scope.querySelector('[data-role="cam-grid"]');
    if (camGrid) {
        const camKeys = Object.keys(stages).filter(k => k.startsWith('cam_') && k.endsWith('_stale_ms'));
        if (camKeys.length === 0) {
            camGrid.innerHTML = '<div class="latency-empty">No camera data — this track may not own cameras (e.g. inference thread).</div>';
        } else {
            camGrid.innerHTML = '';
            camKeys.sort();
            for (const staleKey of camKeys) {
                const camName = staleKey.slice('cam_'.length, -'_stale_ms'.length);
                const periodKey = `cam_${camName}_period_ms`;
                const stale = stages[staleKey];
                const period = stages[periodKey];
                const fps = period && period.p50 ? (1000 / period.p50) : null;
                // Stale threshold scales with the camera's actual period (when
                // we know it): yellow at 1.2× period, red at 2× period.
                const stalePeriod = period ? period.p50 : null;
                const okStale = stalePeriod ? stalePeriod * 1.2 : 40;
                const warnStale = stalePeriod ? stalePeriod * 2 : 70;
                const stalePalette = stale.p95 < okStale ? 'ok' : stale.p95 < warnStale ? 'warn' : 'bad';
                const card = document.createElement('div');
                card.className = 'latency-cam-card';
                card.title = `${_LAT_TOOLTIPS.cam_stale}\n\n${_LAT_TOOLTIPS.cam_period}`;
                card.innerHTML = `
                    <div class="latency-cam-card-title">${camName}</div>
                    <div class="latency-cam-stat">stale p50/p95 <b class="${stalePalette}">${stale.p50.toFixed(0)}/${stale.p95.toFixed(0)} ms</b></div>
                    <div class="latency-cam-stat">period p50 <b>${period ? period.p50.toFixed(1) : '—'} ms</b></div>
                    <div class="latency-cam-stat">effective fps <b>${fps != null ? fps.toFixed(1) : '—'} Hz</b></div>
                `;
                camGrid.appendChild(card);
            }
        }
    }

    // Footer.
    const recordsEl = scope.querySelector('[data-role="meta-records"]');
    if (recordsEl) recordsEl.textContent = `${data.n_records || 0} records`;
    const droppedEl = scope.querySelector('[data-role="meta-dropped"]');
    if (droppedEl) droppedEl.textContent = data.dropped_records ? `${data.dropped_records} dropped` : '';
    const updatedEl = scope.querySelector('[data-role="meta-updated"]');
    if (updatedEl) updatedEl.textContent = data.t ? `updated ${new Date(data.t * 1000).toLocaleTimeString()}` : '';
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

    // Three independent groups, each with its own timestamps. The chart
    // for each group reads from its own group sub-dict; the hover-time
    // label per chart uses that group's timestamps. This is the layout
    // change that fixed the actor_deltas piggyback bug — actor delta is
    // a per-inference event, Q values / critic loss are per-grad-update,
    // and they tick at different rates.
    const series = data.series || {};
    const inf = series.inferences || {deltas: [], timestamps: []};
    const grad = series.grad_updates || {
        critic_losses: [], critic_grad_norms: [], actor_losses: [],
        q_values_mean: [], q_values_min: [], q_values_max: [],
        actor_q_terms: [], actor_bc_terms: [], update_rates: [], timestamps: [],
    };

    const nMax = parseInt(document.getElementById('rlt-stat-history')?.value) || 1000;
    const clip = arr => (arr && arr.length > nMax) ? arr.slice(-nMax) : (arr || []);

    // Success rate (episode-axis; standalone — not part of the synced group)
    if (series.autonomous_rate_rolling && series.autonomous_rate_rolling.length > 0) {
        drawChart('rlt-chart-success', {
            series: [{data: series.autonomous_rate_rolling, color: '#4fc3f7'}],
            fixedMin: 0, fixedMax: 1, percentage: true,
        });
    }

    const rltStep = data.total_updates || 0;

    // Per-grad-update synced charts (Q, critic loss, actor components, update rate).
    // All share the grad_updates timestamps.
    const gradTs = clip(grad.timestamps);
    if (grad.q_values_mean && grad.q_values_mean.length > 0) {
        drawChart('rlt-chart-qvalues', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.q_values_min), color: '#2ecc71', label: 'min',
             bandPair: 'min', bandColor: '#2ecc71', hideLine: true},
            {data: clip(grad.q_values_max), color: '#2ecc71', label: 'max',
             bandPair: 'max', hideLine: true},
            {data: clip(grad.q_values_mean), color: '#2ecc71', label: 'mean'},
        ]});
    }
    if (grad.critic_losses && grad.critic_losses.length > 0) {
        drawChart('rlt-chart-critic', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.critic_losses), color: '#f39c12', label: 'critic'},
        ]});
    }
    if ((grad.actor_q_terms && grad.actor_q_terms.length > 0) ||
        (grad.actor_bc_terms && grad.actor_bc_terms.length > 0)) {
        drawChart('rlt-chart-actor-components', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.actor_q_terms), color: '#e06c75', label: 'q'},
            {data: clip(grad.actor_bc_terms), color: '#4fc3f7', label: 'bc'},
        ]});
    }
    if (grad.update_rates && grad.update_rates.length > 0) {
        drawChart('rlt-chart-update-rate', {syncGroup: 'rlt', latestStep: rltStep, timestamps: gradTs, series: [
            {data: clip(grad.update_rates), color: '#bd93f9', label: 'rate'},
        ]});
    }

    // Per-inference chart (actor delta). Its own timestamps — these
    // tick faster than grad_updates because inference fires every
    // query interval, while grad_updates fires only when the buffer
    // has reached the training threshold.
    if (inf.deltas && inf.deltas.length > 0) {
        drawChart('rlt-chart-delta', {syncGroup: 'rlt', latestStep: rltStep, timestamps: clip(inf.timestamps), series: [
            {data: clip(inf.deltas), color: '#e5c07b', label: 'delta'},
        ]});
    }
}

let _rltConfigTimer = null;
// Slider live-tuning controls only make sense during an active run —
// β / σ have no "preference" semantics, just a current value that the
// subprocess polls. Disable them visibly when no run is active.
// (The diag button is sticky — see _applyDiagBtnVisual.)
function _setRLTSlidersEnabled(enabled) {
    const tooltip = enabled ? '' : 'No active RLT training session';
    for (const id of ['rlt-slider-beta', 'rlt-slider-sigma']) {
        const el = document.getElementById(id);
        if (!el) continue;
        el.disabled = !enabled;
        el.title = tooltip;
        el.classList.toggle('disabled', !enabled);
    }
}

// Sticky-preference model for the Diagnostic toggle: the user's intent
// is stored in localStorage and survives across runs / page reloads.
// If toggled while no run is active, the click is staged locally and
// auto-applied at the start of the next run (see _initRLTSliders). If
// toggled mid-run, both the local pref and the backend are updated.
const _RLT_DIAG_PREF_KEY = 'rlt_dump_chunks_pref';

function _readDiagPref() {
    return localStorage.getItem(_RLT_DIAG_PREF_KEY) === '1';
}

function _writeDiagPref(on) {
    localStorage.setItem(_RLT_DIAG_PREF_KEY, on ? '1' : '0');
}

function _applyDiagBtnVisual(on, sessionActive) {
    const diagBtn = document.getElementById('rlt-btn-diagnostic');
    if (!diagBtn) return;
    diagBtn.classList.toggle('active', on);
    diagBtn.textContent = on ? 'Diagnostic: ON' : 'Diagnostic: OFF';
    diagBtn.title = sessionActive
        ? ''
        : `Will apply when training starts (currently ${on ? 'ON' : 'OFF'})`;
}

async function _initRLTSliders() {
    const controls = document.getElementById('rlt-controls');
    if (controls) controls.style.display = '';

    const betaSlider = document.getElementById('rlt-slider-beta');
    const sigmaSlider = document.getElementById('rlt-slider-sigma');
    const diagBtn = document.getElementById('rlt-btn-diagnostic');
    const betaVal = document.getElementById('rlt-slider-beta-val');
    const sigmaVal = document.getElementById('rlt-slider-sigma-val');

    // Seed controls from the backend so sliders show the values actually
    // applied to the training process, not the HTML's hardcoded defaults.
    // Without this, a page reload makes the UI lie about current β / σ.
    let sessionActive = false;
    let serverDiag = false;
    try {
        const resp = await fetch('/api/run/rlt-config');
        if (resp.ok) {
            const cfg = await resp.json();
            sessionActive = !!cfg.active;
            serverDiag = !!cfg.dump_chunks;
            if (betaSlider && typeof cfg.beta === 'number') {
                betaSlider.value = cfg.beta;
                if (betaVal) betaVal.textContent = cfg.beta.toFixed(2);
            }
            if (sigmaSlider && typeof cfg.exploration_sigma === 'number') {
                sigmaSlider.value = cfg.exploration_sigma;
                if (sigmaVal) sigmaVal.textContent = cfg.exploration_sigma.toFixed(3);
            }
        }
    } catch (e) { /* backend unreachable — keep HTML defaults */ }
    _setRLTSlidersEnabled(sessionActive);

    // Resolve diag state. Without a stored pref the button mirrors the
    // server (or HTML default OFF when no session). With a stored pref,
    // that's the user's intent and wins. If a session is active and the
    // pref differs from what the server is currently applying, sync the
    // pref to the backend now — this is how toggles staged before the
    // run actually take effect when the run starts.
    const havePref = localStorage.getItem(_RLT_DIAG_PREF_KEY) !== null;
    const desired = havePref ? _readDiagPref() : serverDiag;
    _applyDiagBtnVisual(desired, sessionActive);
    if (sessionActive && havePref && desired !== serverDiag) {
        try {
            await fetch('/api/run/rlt-config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dump_chunks: desired}),
            });
        } catch (e) { /* will retry on next click */ }
    }

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

    // Drop focus from RLT controls after release so the operator's arrow-key
    // taps go to the robot (← abort, → success, ↓ ignore — captured by the
    // subprocess's OS-level listener) instead of accidentally re-tweaking
    // whichever slider was last touched. Same problem class as the
    // diagnostic-button SPACE-retoggle bug.
    const _dropFocusAfterRelease = (el) => {
        if (!el) return;
        const blur = () => el.blur();
        el.addEventListener('mouseup', blur);
        el.addEventListener('touchend', blur);
        el.addEventListener('keyup', blur);
    };
    _dropFocusAfterRelease(betaSlider);
    _dropFocusAfterRelease(sigmaSlider);

    const historySelect = document.getElementById('rlt-stat-history');
    if (historySelect) {
        historySelect.onchange = () => _fetchRLTMetrics();
    }
    _dropFocusAfterRelease(historySelect);

    if (diagBtn) {
        diagBtn.onclick = async () => {
            // Drop focus so SPACE/ENTER don't keep re-triggering this button.
            // The subprocess captures SPACE for intervention; if this button
            // keeps focus after click, SPACE fires both the intervention and
            // this toggle, flipping the diagnostic flag on every intervention.
            diagBtn.blur();
            const on = !diagBtn.classList.contains('active');
            // Persist intent BEFORE the POST so a refresh / crash mid-flight
            // still preserves the user's choice. The POST is best-effort.
            _writeDiagPref(on);
            // Single POST. The server's 409 is the source of truth for
            // "no active session" — no separate probe GET to race with the
            // POST or to delay the click feedback. 409 → stage; 200 → live.
            let live = false;
            try {
                const resp = await fetch('/api/run/rlt-config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({dump_chunks: on}),
                });
                if (resp.status === 409) {
                    // No active session — preference is staged locally and
                    // will auto-apply at the start of the next run via
                    // _initRLTSliders. Silent (no toast) since this is the
                    // documented sticky-pref path, not an error.
                    live = false;
                } else if (resp.ok) {
                    live = true;
                } else {
                    const err = await resp.json().catch(() => ({}));
                    showToast('Diagnostic toggle failed',
                              err.detail || resp.statusText, 'error');
                }
            } catch (e) {
                showToast('Diagnostic toggle failed', e.message, 'error');
            }
            _applyDiagBtnVisual(on, live);
        };
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
                body: JSON.stringify({beta, exploration_sigma: sigma}),
            });
        } catch (e) { /* ignore */ }
    }, 300);
}
