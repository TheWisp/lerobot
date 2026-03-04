/* LeRobot GUI - Run Tab Logic */

let runTabInitialized = false;
let runEventSource = null;
let selectedWorkflow = 'teleop'; // 'teleop' | 'replay' | 'policy'
let rerunPorts = null; // {available, web_port, grpc_port}

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
    renderRunForm();
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

function renderRunForm() {
    const form = document.getElementById('run-form');
    if (!form) return;

    let html = '<div class="form-grid">';

    // Robot profile (always shown)
    html += `<label>Robot</label>`;
    html += `<select id="run-robot-profile">${_robotProfileOptions()}</select>`;

    if (selectedWorkflow === 'teleop') {
        // Teleop profile
        html += `<label>Teleop</label>`;
        html += `<select id="run-teleop-profile">${_teleopProfileOptions()}</select>`;
        html += `<label>FPS</label>`;
        html += `<input type="number" id="run-fps" value="60" min="1" max="200">`;

        // Output dataset section (optional recording)
        html += '</div>';
        html += '<div class="form-section">';
        html += '<div class="form-section-title">Record dataset (optional)</div>';
        html += '<div class="form-grid">';
        html += `<label>Repo ID</label>`;
        html += `<input type="text" id="run-repo-id" placeholder="user/dataset_name">`;
        html += `<label>Task</label>`;
        html += `<input type="text" id="run-single-task" placeholder="Pick up the cube">`;
        html += `<label>Episodes</label>`;
        html += `<input type="number" id="run-num-episodes" value="50" min="1">`;
        html += `<label>Episode (s)</label>`;
        html += `<input type="number" id="run-episode-time" value="60" min="1">`;
        html += `<label>Reset (s)</label>`;
        html += `<input type="number" id="run-reset-time" value="60" min="0">`;
        html += `<label>FPS</label>`;
        html += `<input type="number" id="run-record-fps" value="30" min="1" max="200">`;
        html += '</div>';
        html += '</div>';

    } else if (selectedWorkflow === 'replay') {
        // Replay: dataset + episode
        html += `<label>Repo ID</label>`;
        html += `<input type="text" id="run-replay-repo-id" placeholder="user/dataset_name">`;
        html += `<label>Episode</label>`;
        html += `<input type="number" id="run-replay-episode" value="0" min="0">`;
        html += `<label>FPS</label>`;
        html += `<input type="number" id="run-replay-fps" value="30" min="1" max="200">`;

    } else if (selectedWorkflow === 'policy') {
        // Policy: placeholder
        html += `<label>Policy</label>`;
        html += `<select disabled><option>Coming with Model tab</option></select>`;
    }

    if (!html.endsWith('</div>')) html += '</div>';
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

async function launchRun() {
    const robotSelect = document.getElementById('run-robot-profile');
    if (!robotSelect || !robotSelect.value) {
        showToast('Error', 'Select a robot profile', 'error');
        return;
    }

    const robotData = await _getProfileData('robot', robotSelect.value);
    if (!robotData) {
        showToast('Error', `Robot profile "${robotSelect.value}" not found`, 'error');
        return;
    }

    let endpoint, body;

    if (selectedWorkflow === 'teleop') {
        const teleopSelect = document.getElementById('run-teleop-profile');
        const repoId = document.getElementById('run-repo-id')?.value?.trim();

        if (repoId) {
            // Record mode
            const teleopData = await _getProfileData('teleop', teleopSelect?.value);
            if (!teleopData) {
                showToast('Error', 'Select a teleop profile', 'error');
                return;
            }
            const singleTask = document.getElementById('run-single-task')?.value?.trim();
            if (!singleTask) {
                showToast('Error', 'Task description is required for recording', 'error');
                return;
            }
            endpoint = '/api/run/record';
            body = {
                robot: robotData,
                teleop: teleopData,
                repo_id: repoId,
                single_task: singleTask,
                fps: parseInt(document.getElementById('run-record-fps')?.value) || 30,
                episode_time_s: parseFloat(document.getElementById('run-episode-time')?.value) || 60,
                reset_time_s: parseFloat(document.getElementById('run-reset-time')?.value) || 60,
                num_episodes: parseInt(document.getElementById('run-num-episodes')?.value) || 50,
            };
        } else {
            // Pure teleoperate
            const teleopData = await _getProfileData('teleop', teleopSelect?.value);
            if (!teleopData) {
                showToast('Error', 'Select a teleop profile', 'error');
                return;
            }
            endpoint = '/api/run/teleoperate';
            body = {
                robot: robotData,
                teleop: teleopData,
                fps: parseInt(document.getElementById('run-fps')?.value) || 60,
            };
        }
    } else if (selectedWorkflow === 'replay') {
        const repoId = document.getElementById('run-replay-repo-id')?.value?.trim();
        if (!repoId) {
            showToast('Error', 'Repo ID is required for replay', 'error');
            return;
        }
        endpoint = '/api/run/replay';
        body = {
            robot: robotData,
            repo_id: repoId,
            episode: parseInt(document.getElementById('run-replay-episode')?.value) || 0,
            fps: parseInt(document.getElementById('run-replay-fps')?.value) || 30,
        };
    } else {
        showToast('Error', 'Policy workflow not yet implemented', 'error');
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
        showRerunViewer();
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

    // Only create iframe if not already there
    if (!container.querySelector('iframe')) {
        const iframe = document.createElement('iframe');
        // The ?url= param tells the Rerun WASM viewer where to connect for data
        const grpcUrl = encodeURIComponent(`rerun+http://localhost:${rerunPorts.grpc_port}/proxy`);
        iframe.src = `http://localhost:${rerunPorts.web_port}?url=${grpcUrl}&hide_welcome_screen`;
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.border = 'none';
        container.appendChild(iframe);
    }
}
