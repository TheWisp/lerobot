/* LeRobot GUI - Robot Tab Logic */

let robotSchemas = null;
let teleopSchemas = null;
let robotProfiles = [];
let teleopProfiles = [];
let currentProfile = null; // {kind: 'robot'|'teleop', name, data}
let savedProfileData = null; // deep copy of data at last save/load — for dirty detection
let detectedCameras = [];
let previewInterval = null;
let scannedPorts = null; // {ports, allAssignments} from last scan
let robotTabInitialized = false;

// ============================================================================
// Initialization
// ============================================================================

async function robotTabInit() {
    if (robotTabInitialized) return;
    robotTabInitialized = true;
    await Promise.all([
        loadRobotSchemas(),
        loadTeleopSchemas(),
        loadRobotProfiles(),
        loadTeleopProfiles(),
    ]);
}

async function loadRobotSchemas() {
    try {
        const res = await fetch('/api/robot/schemas');
        robotSchemas = await res.json();
    } catch (e) {
        console.error('Failed to load robot schemas:', e);
    }
}

async function loadTeleopSchemas() {
    try {
        const res = await fetch('/api/robot/teleop-schemas');
        teleopSchemas = await res.json();
    } catch (e) {
        console.error('Failed to load teleop schemas:', e);
    }
}

// ============================================================================
// Profile list loading and rendering
// ============================================================================

async function loadRobotProfiles() {
    try {
        const res = await fetch('/api/robot/profiles');
        robotProfiles = await res.json();
    } catch (e) {
        robotProfiles = [];
    }
    renderRobotProfileList();
    if (typeof refreshRunProfileSelects === 'function') refreshRunProfileSelects();
}

async function loadTeleopProfiles() {
    try {
        const res = await fetch('/api/robot/teleop-profiles');
        teleopProfiles = await res.json();
    } catch (e) {
        teleopProfiles = [];
    }
    renderTeleopProfileList();
    if (typeof refreshRunProfileSelects === 'function') refreshRunProfileSelects();
}

function renderRobotProfileList() {
    const list = document.getElementById('robot-profile-list');
    if (!list) return;
    if (robotProfiles.length === 0) {
        list.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No profiles</div>';
        return;
    }
    list.innerHTML = robotProfiles.map(p => {
        const isActive = currentProfile?.name === p.name && currentProfile?.kind === 'robot';
        return `<div class="profile-item ${isActive ? 'active' : ''}"
                     onclick="selectProfile('robot', '${esc(p.name)}')">
            <span>${esc(p.name)}<span class="profile-type">${esc(p.type)}</span></span>
            <span class="profile-delete" onclick="event.stopPropagation(); deleteProfile('robot', '${esc(p.name)}')">&times;</span>
        </div>`;
    }).join('');
}

function renderTeleopProfileList() {
    const list = document.getElementById('teleop-profile-list');
    if (!list) return;
    if (teleopProfiles.length === 0) {
        list.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No profiles</div>';
        return;
    }
    list.innerHTML = teleopProfiles.map(p => {
        const isActive = currentProfile?.name === p.name && currentProfile?.kind === 'teleop';
        return `<div class="profile-item ${isActive ? 'active' : ''}"
                     onclick="selectProfile('teleop', '${esc(p.name)}')">
            <span>${esc(p.name)}<span class="profile-type">${esc(p.type)}</span></span>
            <span class="profile-delete" onclick="event.stopPropagation(); deleteProfile('teleop', '${esc(p.name)}')">&times;</span>
        </div>`;
    }).join('');
}

// ============================================================================
// Profile selection and editor
// ============================================================================

async function selectProfile(kind, name) {
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(name)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        currentProfile = { kind, name, data };
        savedProfileData = JSON.parse(JSON.stringify(data));
        renderEditor();
        renderRobotProfileList();
        renderTeleopProfileList();
    } catch (e) {
        showToast('Error', `Failed to load profile: ${e.message}`, 'error');
    }
}

function renderEditor() {
    const emptyEl = document.getElementById('robot-empty');
    const editorEl = document.getElementById('robot-editor');
    if (!currentProfile) {
        emptyEl.style.display = 'flex';
        editorEl.style.display = 'none';
        return;
    }
    emptyEl.style.display = 'none';
    editorEl.style.display = 'block';

    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);

    let html = '';

    // Header: name + actions
    html += '<div class="editor-header">';
    html += '<div class="editor-header-row">';
    html += `<h2>${esc(currentProfile.name)}</h2>`;
    html += `<button class="btn-small secondary" onclick="renameProfile()" title="Rename profile">Rename</button>`;
    html += `<button class="btn-small danger" onclick="deleteProfile('${esc(currentProfile.kind)}', '${esc(currentProfile.name)}')" title="Delete profile">Delete</button>`;
    html += '</div>';
    html += '</div>';

    // Form fields (type selector as first row)
    html += '<div class="editor-form" id="editor-form">';
    html += `<label>Type</label>`;
    html += `<select onchange="changeProfileType(this.value)">`;
    if (schemas) {
        for (const s of schemas) {
            const sel = s.type_name === currentProfile.data.type ? 'selected' : '';
            html += `<option value="${esc(s.type_name)}" ${sel}>${esc(s.type_name)}</option>`;
        }
    }
    html += '</select>';
    if (schema) {
        for (const field of schema.fields) {
            const value = currentProfile.data.fields?.[field.name] ?? field.default ?? '';
            html += renderFormField(field, value);
        }
    }
    html += '</div>';

    // Cameras section (robot profiles only)
    if (currentProfile.kind === 'robot') {
        html += renderCamerasSection();
    }

    // Ports section (both robot and teleop — leaders have ports too)
    if (_getPortFields().length > 0) {
        html += renderPortsSection();
    }

    // Rest position section (robot profiles only)
    if (currentProfile.kind === 'robot') {
        html += renderRestPositionSection();
    }

    // Save/discard bar (hidden until dirty)
    html += '<div class="editor-actions" id="editor-actions" style="display:none">';
    html += `<button class="btn-save" onclick="saveProfile()">Save Changes</button>`;
    html += `<button class="btn-discard" onclick="discardProfileChanges()">Discard</button>`;
    html += '</div>';

    editorEl.innerHTML = html;

    // Listen for input changes to track dirty state
    const form = document.getElementById('editor-form');
    if (form) {
        form.addEventListener('input', _updateDirtyState);
        form.addEventListener('change', _updateDirtyState);
    }
}

function renderFormField(field, value) {
    const id = `field-${field.name}`;
    const label = `<label for="${id}">${esc(field.name)}${field.required ? ' *' : ''}</label>`;
    const typeStr = field.type_str.toLowerCase();

    if (typeStr === 'bool' || typeStr === 'bool | none') {
        const trueSelected = value === true ? 'selected' : '';
        const falseSelected = value === false ? 'selected' : '';
        const noneSelected = (value === null || value === '' || value === undefined) ? 'selected' : '';
        return label + `<select id="${id}">
            ${!field.required ? `<option value="null" ${noneSelected}>--</option>` : ''}
            <option value="true" ${trueSelected}>true</option>
            <option value="false" ${falseSelected}>false</option>
        </select>`;
    }

    if (typeStr.includes('int') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="1">`;
    }

    if (typeStr.includes('float') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="any">`;
    }

    // Default: text input
    const displayValue = (value === null || value === undefined) ? '' : String(value);
    return label + `<input type="text" id="${id}" value="${esc(displayValue)}"
        ${!field.required ? 'placeholder="(optional)"' : ''}>`;
}

function changeProfileType(newType) {
    if (!currentProfile) return;
    currentProfile.data.type = newType;
    currentProfile.data.fields = {};
    currentProfile.data.cameras = {};
    currentProfile.data.rest_position = {};
    _rerender();
    _updateDirtyState();
}

// ============================================================================
// Dirty tracking
// ============================================================================

function _serializeProfile(data) {
    return JSON.stringify({
        type: data.type,
        name: data.name,
        fields: data.fields || {},
        cameras: data.cameras || {},
        rest_position: data.rest_position || {},
    });
}

function _collectFormFields() {
    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);
    const fields = {};
    if (schema) {
        for (const field of schema.fields) {
            const input = document.getElementById(`field-${field.name}`);
            if (!input) continue;
            fields[field.name] = parseFieldValue(field, input.value);
        }
    }
    return fields;
}

function _isDirty() {
    if (!currentProfile || !savedProfileData) return false;
    const current = _serializeProfile({
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    });
    return current !== _serializeProfile(savedProfileData);
}

function _updateDirtyState() {
    const bar = document.getElementById('editor-actions');
    if (bar) bar.style.display = _isDirty() ? 'flex' : 'none';
}

function discardProfileChanges() {
    if (!currentProfile || !savedProfileData) return;
    currentProfile.data = JSON.parse(JSON.stringify(savedProfileData));
    _rerender();
}

// ============================================================================
// Profile CRUD
// ============================================================================

async function newRobotProfile() {
    const name = prompt('Robot profile name:');
    if (!name || !name.trim()) return;
    const defaultType = robotSchemas?.[0]?.type_name || 'so107_follower';
    const profile = { type: defaultType, name: name.trim(), fields: {}, cameras: {} };
    try {
        const res = await fetch('/api/robot/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadRobotProfiles();
        await selectProfile('robot', name.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function newTeleopProfile() {
    const name = prompt('Teleop profile name:');
    if (!name || !name.trim()) return;
    const defaultType = teleopSchemas?.[0]?.type_name || 'keyboard';
    const profile = { type: defaultType, name: name.trim(), fields: {}, cameras: {} };
    try {
        const res = await fetch('/api/robot/teleop-profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadTeleopProfiles();
        await selectProfile('teleop', name.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function saveProfile() {
    if (!currentProfile) return;

    const payload = {
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    };

    const endpoint = currentProfile.kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(currentProfile.name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        currentProfile.data = payload;
        savedProfileData = JSON.parse(JSON.stringify(payload));
        _updateDirtyState();
        showToast('Saved', `Profile "${currentProfile.name}" saved`, 'success');
        // Reload lists in case type changed
        if (currentProfile.kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function renameProfile() {
    if (!currentProfile) return;
    const newName = prompt('New profile name:', currentProfile.name);
    if (!newName || !newName.trim() || newName.trim() === currentProfile.name) return;

    const endpoint = currentProfile.kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(currentProfile.name)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_name: newName.trim() }),
        });
        if (!res.ok) throw new Error(await res.text());
        const oldName = currentProfile.name;
        currentProfile.name = newName.trim();
        currentProfile.data.name = newName.trim();
        if (currentProfile.kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
        _rerender();
        showToast('Renamed', `"${oldName}" renamed to "${newName.trim()}"`, 'success');
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function deleteProfile(kind, name) {
    if (!confirm(`Delete profile "${name}"?`)) return;
    const endpoint = kind === 'robot' ? '/api/robot/profiles' : '/api/robot/teleop-profiles';
    try {
        const res = await fetch(`${endpoint}/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(await res.text());
        if (currentProfile?.name === name && currentProfile?.kind === kind) {
            currentProfile = null;
            renderEditor();
        }
        if (kind === 'robot') await loadRobotProfiles();
        else await loadTeleopProfiles();
        showToast('Deleted', `Profile "${name}" deleted`, 'info');
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

function parseFieldValue(field, rawValue) {
    if (rawValue === '' || rawValue === 'null') return null;
    const typeStr = field.type_str.toLowerCase();
    if (typeStr === 'bool' || typeStr === 'bool | none') {
        if (rawValue === 'true') return true;
        if (rawValue === 'false') return false;
        return null;
    }
    if (typeStr.includes('int') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        const n = parseInt(rawValue, 10);
        return isNaN(n) ? null : n;
    }
    if (typeStr.includes('float') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        const n = parseFloat(rawValue);
        return isNaN(n) ? null : n;
    }
    return rawValue;
}

// ============================================================================
// Camera detection and preview
// ============================================================================

function renderCamerasSection() {
    const cameras = currentProfile?.data?.cameras || {};
    const cameraEntries = Object.entries(cameras);

    let html = '<div class="camera-detect-section"><h3>Cameras</h3>';

    // Show assigned cameras with editable config
    if (cameraEntries.length > 0) {
        for (const [role, cam] of cameraEntries) {
            const camType = cam.type || 'unknown';
            const camId = cam.index_or_path ?? cam.serial_number_or_name ?? '?';
            const isRealsense = camType === 'intelrealsense';
            html += `<div class="camera-assigned">
                <div class="cam-header">
                    <span class="cam-role">${esc(role)}</span>
                    <span class="cam-info">${esc(camType)} - ${esc(String(camId))}</span>
                    <span class="cam-remove" onclick="removeCameraRole('${esc(role)}')" title="Remove">&times;</span>
                </div>
                <div class="cam-config">
                    <label>Width<input type="number" value="${cam.width || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'width', this.value)"></label>
                    <label>Height<input type="number" value="${cam.height || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'height', this.value)"></label>
                    <label>FPS<input type="number" value="${cam.fps || ''}" placeholder="auto" onchange="updateCameraConfig('${esc(role)}', 'fps', this.value)"></label>
                    ${!isRealsense ? `<label>Format<select onchange="updateCameraConfig('${esc(role)}', 'fourcc', this.value)">
                        <option value="" ${!cam.fourcc ? 'selected' : ''}>auto</option>
                        <option value="MJPG" ${cam.fourcc === 'MJPG' ? 'selected' : ''}>MJPG</option>
                        <option value="YUYV" ${cam.fourcc === 'YUYV' ? 'selected' : ''}>YUYV</option>
                        <option value="H264" ${cam.fourcc === 'H264' ? 'selected' : ''}>H264</option>
                    </select></label>` : ''}
                    ${isRealsense ? `<label>Depth<select onchange="updateCameraConfig('${esc(role)}', 'use_depth', this.value)">
                        <option value="false" ${!cam.use_depth ? 'selected' : ''}>off</option>
                        <option value="true" ${cam.use_depth ? 'selected' : ''}>on</option>
                    </select></label>` : ''}
                </div>
            </div>`;
        }
    }

    html += '<div class="camera-detect-actions">';
    html += '<button class="btn-small" id="detect-cameras-btn" onclick="detectCameras()">Detect Cameras</button>';
    html += `<button class="btn-small secondary" id="stop-cameras-btn" onclick="stopAllCameras()" style="display:${previewInterval ? 'inline-block' : 'none'}">Stop Preview</button>`;
    html += '</div>';
    html += '<div class="camera-preview-grid" id="camera-preview-grid"></div>';
    html += '</div>';
    return html;
}

async function detectCameras() {
    const btn = document.getElementById('detect-cameras-btn');
    if (btn) { btn.textContent = 'Detecting...'; btn.disabled = true; }

    try {
        const res = await fetch('/api/robot/detect-cameras', { method: 'POST' });
        detectedCameras = await res.json();
        renderCameraPreview();
        startCameraPreview();
    } catch (e) {
        showToast('Error', `Camera detection failed: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.textContent = 'Detect Cameras'; btn.disabled = false; }
    }
}

function renderCameraPreview() {
    const grid = document.getElementById('camera-preview-grid');
    if (!grid) return;
    if (detectedCameras.length === 0) {
        grid.innerHTML = '<div style="color: #666; font-size: 13px; padding: 8px;">No cameras detected</div>';
        return;
    }

    // Default camera role names
    const roles = ['front', 'left_wrist', 'right_wrist', 'top'];

    // Build a reverse lookup: camera id -> assigned role
    const assignedRoles = {};
    const cameras = currentProfile?.data?.cameras || {};
    for (const [role, cam] of Object.entries(cameras)) {
        const camId = String(cam.index_or_path ?? cam.serial_number_or_name ?? '');
        if (camId) assignedRoles[camId] = role;
    }

    grid.innerHTML = detectedCameras.map((cam, i) => {
        const profile = cam.default_stream_profile || {};
        // Build identifier: serial number for RealSense, path for OpenCV
        const camId = String(cam.id || '');
        const identifier = cam.type === 'RealSense' ? `S/N: ${camId}` : camId;
        // Check if this camera is already assigned to a role
        const currentRole = assignedRoles[camId] || '';

        return `<div class="camera-preview-card">
            <img id="preview-cam-${i}" src="" alt="${esc(cam.name || 'Camera')}">
            <div class="camera-preview-info">
                <div>${esc(cam.name || `Camera ${i}`)}</div>
                <div class="cam-meta">${esc(cam.type)} | ${esc(identifier)} | ${profile.width || '?'}x${profile.height || '?'} @ ${Math.round(profile.fps) || '?'}fps</div>
                <select id="cam-role-${i}" onchange="assignCameraRole(${i}, this.value)">
                    <option value="">-- assign role --</option>
                    ${roles.map(r => `<option value="${r}" ${currentRole === r ? 'selected' : ''}>${r}</option>`).join('')}
                    <option value="__custom" ${currentRole && !roles.includes(currentRole) ? 'selected' : ''}>custom...</option>
                </select>
            </div>
        </div>`;
    }).join('');
}

function startCameraPreview() {
    stopCameraPreview();
    previewInterval = setInterval(() => {
        detectedCameras.forEach((_, i) => {
            const img = document.getElementById(`preview-cam-${i}`);
            if (img) {
                img.src = `/api/robot/camera-frame/${i}?t=${Date.now()}`;
            }
        });
    }, 100); // ~10fps
    const stopBtn = document.getElementById('stop-cameras-btn');
    if (stopBtn) stopBtn.style.display = 'inline-block';
}

function stopCameraPreview() {
    if (previewInterval) {
        clearInterval(previewInterval);
        previewInterval = null;
    }
    const stopBtn = document.getElementById('stop-cameras-btn');
    if (stopBtn) stopBtn.style.display = 'none';
}

async function stopAllCameras() {
    stopCameraPreview();
    try {
        await fetch('/api/robot/stop-cameras', { method: 'POST' });
    } catch (e) { /* ignore */ }
    detectedCameras = [];
    const grid = document.getElementById('camera-preview-grid');
    if (grid) grid.innerHTML = '';
}

function assignCameraRole(cameraIndex, role) {
    if (!currentProfile || !detectedCameras[cameraIndex]) return;

    if (role === '__custom') {
        role = prompt('Camera role name:');
        if (!role || !role.trim()) {
            // Reset dropdown
            const sel = document.getElementById(`cam-role-${cameraIndex}`);
            if (sel) sel.value = '';
            return;
        }
        role = role.trim();
    }

    if (!role) return;

    const cam = detectedCameras[cameraIndex];
    const camConfig = {};
    if (cam.type === 'OpenCV') {
        camConfig.type = 'opencv';
        camConfig.index_or_path = cam.id;
    } else if (cam.type === 'RealSense') {
        camConfig.type = 'intelrealsense';
        camConfig.serial_number_or_name = String(cam.id);
    }
    const profile = cam.default_stream_profile || {};
    if (profile.width) camConfig.width = profile.width;
    if (profile.height) camConfig.height = profile.height;
    if (profile.fps) camConfig.fps = profile.fps;
    if (profile.fourcc) camConfig.fourcc = profile.fourcc;

    if (!currentProfile.data.cameras) currentProfile.data.cameras = {};
    currentProfile.data.cameras[role] = camConfig;
    _rerender();
    _updateDirtyState();
    showToast('Camera assigned', `${cam.name || 'Camera'} assigned as "${role}"`, 'success');
}

// Re-render the entire editor from currentProfile state.
// Re-populates camera preview and port list from cached scan data.
function _rerender() {
    renderEditor();
    if (detectedCameras.length > 0) renderCameraPreview();
    if (scannedPorts) renderPortList(scannedPorts.ports, scannedPorts.allAssignments);
}

function removeCameraRole(role) {
    if (!currentProfile?.data?.cameras) return;
    delete currentProfile.data.cameras[role];
    _rerender();
    _updateDirtyState();
}

function updateCameraConfig(role, key, value) {
    if (!currentProfile?.data?.cameras?.[role]) return;
    if (key === 'use_depth') {
        currentProfile.data.cameras[role][key] = value === 'true';
    } else if (key === 'fourcc') {
        if (value) currentProfile.data.cameras[role][key] = value;
        else delete currentProfile.data.cameras[role][key];
    } else {
        // width, height, fps — numeric
        const n = parseInt(value, 10);
        if (isNaN(n) || n <= 0) delete currentProfile.data.cameras[role][key];
        else currentProfile.data.cameras[role][key] = n;
    }
    _updateDirtyState();
}

// ============================================================================
// Port scanning and arm identification
// ============================================================================

function renderPortsSection() {
    let html = '<div class="port-section"><h3>Arm Ports</h3>';
    html += '<div style="display: flex; gap: 8px; margin-bottom: 8px;">';
    html += '<button class="btn-small" id="scan-ports-btn" onclick="scanPorts()">Scan Ports</button>';
    html += '</div>';
    html += '<div class="port-list" id="port-list"></div>';
    html += '</div>';
    return html;
}

async function scanPorts() {
    const btn = document.getElementById('scan-ports-btn');
    if (btn) { btn.textContent = 'Scanning...'; btn.disabled = true; }

    try {
        const [portsRes, assignRes] = await Promise.all([
            fetch('/api/robot/ports'),
            fetch('/api/robot/all-port-assignments'),
        ]);
        const ports = await portsRes.json();
        const allAssignments = await assignRes.json();
        scannedPorts = { ports, allAssignments };
        renderPortList(ports, allAssignments);
    } catch (e) {
        showToast('Error', `Port scan failed: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.textContent = 'Scan Ports'; btn.disabled = false; }
    }
}

function _getPortFields() {
    if (!currentProfile) return [];
    const schemas = currentProfile.kind === 'robot' ? robotSchemas : teleopSchemas;
    const schema = schemas?.find(s => s.type_name === currentProfile.data.type);
    if (!schema) return [];
    return schema.fields.filter(f =>
        f.name.includes('port') && f.type_str.toLowerCase().includes('str')
    );
}

function renderPortList(ports, allAssignments) {
    const list = document.getElementById('port-list');
    if (!list) return;
    if (ports.length === 0) {
        list.innerHTML = '<div style="color: #666; font-size: 13px; padding: 8px;">No serial ports found</div>';
        return;
    }
    const portFields = _getPortFields();

    // Build map of ports claimed by OTHER profiles: { "/dev/ttyACM0": { profile_name, profile_kind, field_name } }
    const usedByOthers = {};
    if (allAssignments) {
        for (const a of allAssignments) {
            // Skip assignments belonging to the currently edited profile
            if (currentProfile && a.profile_name === currentProfile.name && a.profile_kind === currentProfile.kind) {
                continue;
            }
            usedByOthers[a.port] = a;
        }
    }

    list.innerHTML = ports.map(p => {
        const meta = [p.name || ''];
        if (p.manufacturer) meta.push(p.manufacturer);
        if (p.vid_pid) meta.push(p.vid_pid);

        const claimedBy = usedByOthers[p.path];

        // Build assign dropdown + "in use by" trailing text (always 5 children for grid alignment)
        let assignHtml = '<span></span>';
        let claimedHtml = '<span></span>';
        if (portFields.length > 0) {
            const currentField = portFields.find(f => {
                const input = document.getElementById(`field-${f.name}`);
                return input && input.value === p.path;
            })?.name || '';
            if (claimedBy) {
                // Port is used by another profile — still allow reassign, but with confirmation
                assignHtml = `<select class="port-assign-select port-assign-claimed" onchange="assignPort('${esc(p.path)}', this.value, '${esc(claimedBy.profile_name)}', '${esc(claimedBy.profile_kind)}')">
                    <option value="">${esc(claimedBy.field_name)}</option>
                    ${portFields.map(f => `<option value="${esc(f.name)}">${esc(f.name)}</option>`).join('')}
                </select>`;
                claimedHtml = `<span class="port-claimed">in use by <a onclick="selectProfile('${esc(claimedBy.profile_kind)}', '${esc(claimedBy.profile_name)}')">${esc(claimedBy.profile_name)}</a></span>`;
            } else {
                const unassignOpt = currentField ? `<option value="__unassign__">-- clear --</option>` : '';
                assignHtml = `<select class="port-assign-select" onchange="if(this.value==='__unassign__'){unassignPort('${esc(p.path)}')}else{assignPort('${esc(p.path)}', this.value)}">
                    <option value="">-- assign --</option>
                    ${unassignOpt}
                    ${portFields.map(f => `<option value="${esc(f.name)}" ${currentField === f.name ? 'selected' : ''}>${esc(f.name)}</option>`).join('')}
                </select>`;
            }
        }

        return `<div class="port-item">
            <span class="port-path">${esc(p.path)}</span>
            <span class="port-name">${esc(meta.join(' | '))}</span>
            <button class="btn-small" onclick="identifyArm('${esc(p.path)}', this)">Wiggle</button>
            ${assignHtml}
            ${claimedHtml}
        </div>`;
    }).join('');
}

function assignPort(port, fieldName, claimedByProfile, claimedByKind) {
    // If port is claimed by another profile, confirm before reassigning
    if (claimedByProfile) {
        if (!confirm(`This port is currently in use by "${claimedByProfile}" (${claimedByKind}). Reassign it anyway?`)) {
            // Reset the dropdown to show the claimed field
            if (scannedPorts) renderPortList(scannedPorts.ports, scannedPorts.allAssignments);
            return;
        }
    }
    if (!fieldName) return;
    if (!currentProfile.data.fields) currentProfile.data.fields = {};
    currentProfile.data.fields[fieldName] = port;
    _rerender();
    _updateDirtyState();
    showToast('Port set', `${fieldName} = ${port}`, 'info');
}

function unassignPort(port) {
    if (!currentProfile || !currentProfile.data.fields) return;
    const portFields = _getPortFields();
    for (const f of portFields) {
        if (currentProfile.data.fields[f.name] === port) {
            delete currentProfile.data.fields[f.name];
        }
    }
    _rerender();
    _updateDirtyState();
    showToast('Port cleared', `${port} unassigned`, 'info');
}

async function identifyArm(port, btn) {
    if (btn) {
        btn.textContent = 'Wiggling...';
        btn.classList.add('wiggling');
        btn.disabled = true;
    }
    try {
        const res = await fetch('/api/robot/identify-arm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ port }),
        });
        const result = await res.json();
        if (result.status === 'ok') {
            showToast('Arm moved', `Motor wiggled on ${port}`, 'success');
        } else {
            showToast('Error', result.message || 'Identification failed', 'error');
        }
    } catch (e) {
        showToast('Error', `Arm identification failed: ${e.message}`, 'error');
    } finally {
        if (btn) {
            btn.textContent = 'Wiggle';
            btn.classList.remove('wiggling');
            btn.disabled = false;
        }
    }
}

// ============================================================================
// Rest position
// ============================================================================

let _restRecordingActive = false;

function renderRestPositionSection() {
    const restPos = currentProfile?.data?.rest_position || {};
    const hasRest = Object.keys(restPos).length > 0;

    let html = '<div class="rest-position-section"><h3>Rest Position</h3>';

    if (hasRest) {
        html += '<div class="rest-position-values">';
        for (const [key, val] of Object.entries(restPos)) {
            const label = key.replace('.pos', '');
            html += `<span class="rest-pos-entry"><span class="rest-pos-label">${esc(label)}</span><span class="rest-pos-value">${Number(val).toFixed(1)}</span></span>`;
        }
        html += '</div>';
    } else {
        html += '<div class="rest-position-empty">No rest position recorded. Click Record to start.</div>';
    }

    html += '<div class="rest-position-actions">';
    if (_restRecordingActive) {
        html += '<button class="btn-small" id="finish-rest-btn" onclick="finishRestRecording()">Done</button>';
        html += '<button class="btn-small secondary" id="cancel-rest-btn" onclick="cancelRestRecording()">Cancel</button>';
    } else {
        html += '<button class="btn-small" id="record-rest-btn" onclick="startRestRecording()">Record Rest Position</button>';
        html += `<button class="btn-small secondary" id="move-rest-btn" onclick="moveToRestPosition()" ${hasRest ? '' : 'disabled'}>Move to Rest</button>`;
        html += '<button class="btn-small danger" id="clear-rest-btn" onclick="clearRestPosition()" ' + (hasRest ? '' : 'disabled') + '>Clear</button>';
    }
    html += '</div>';
    html += '<div id="rest-position-status" class="rest-position-status"></div>';
    html += '</div>';
    return html;
}

async function startRestRecording() {
    if (!currentProfile) return;

    const btn = document.getElementById('record-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Connecting...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting and disabling torque...';

    try {
        const res = await fetch('/api/robot/start-rest-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ robot: currentProfile.data }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to start recording');
        }

        _restRecordingActive = true;
        _rerender();
        const newStatus = document.getElementById('rest-position-status');
        if (newStatus) newStatus.textContent = 'Torque disabled — move the robot to the desired rest pose, then click Done.';
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Record Rest Position'; btn.disabled = false; }
    }
}

async function finishRestRecording() {
    const btn = document.getElementById('finish-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Reading...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Reading positions...';

    try {
        const res = await fetch('/api/robot/finish-rest-recording', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to finish recording');
        }
        const result = await res.json();

        currentProfile.data.rest_position = result.rest_position;
        await _saveRestPositionToProfile();

        _restRecordingActive = false;
        _rerender();
        showToast('Recorded', 'Rest position saved to profile', 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = 'Error reading positions. You can retry or cancel.';
        showToast('Error', e.message, 'error');
        if (btn) { btn.textContent = 'Done'; btn.disabled = false; }
    }
}

async function cancelRestRecording() {
    try {
        await fetch('/api/robot/cancel-rest-recording', { method: 'POST' });
    } catch (e) { /* ignore */ }
    _restRecordingActive = false;
    _rerender();
}

async function moveToRestPosition() {
    if (!currentProfile) return;
    const restPos = currentProfile.data.rest_position;
    if (!restPos || Object.keys(restPos).length === 0) return;

    const btn = document.getElementById('move-rest-btn');
    const statusEl = document.getElementById('rest-position-status');
    if (btn) { btn.textContent = 'Moving...'; btn.disabled = true; }
    if (statusEl) statusEl.textContent = 'Connecting and moving to rest position...';

    try {
        const res = await fetch('/api/robot/move-to-rest-position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                robot: currentProfile.data,
                rest_position: restPos,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Failed to move to rest position');
        }
        if (statusEl) statusEl.textContent = '';
        showToast('Done', 'Robot is at rest position', 'success');
    } catch (e) {
        if (statusEl) statusEl.textContent = '';
        showToast('Error', e.message, 'error');
    } finally {
        if (btn) { btn.textContent = 'Move to Rest'; btn.disabled = false; }
    }
}

function clearRestPosition() {
    if (!currentProfile) return;
    if (!confirm('Clear the saved rest position?')) return;
    currentProfile.data.rest_position = {};
    _rerender();
    _updateDirtyState();
}

async function _saveRestPositionToProfile() {
    /* Persist the current rest_position to the profile JSON on disk. */
    if (!currentProfile) return;
    const payload = {
        ...currentProfile.data,
        name: currentProfile.name,
        fields: _collectFormFields(),
    };
    try {
        const res = await fetch(`/api/robot/profiles/${encodeURIComponent(currentProfile.name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        currentProfile.data = payload;
        savedProfileData = JSON.parse(JSON.stringify(payload));
        _updateDirtyState();
    } catch (e) {
        console.error('Failed to auto-save rest position:', e);
    }
}

// ============================================================================
// Utilities
// ============================================================================

function esc(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}
