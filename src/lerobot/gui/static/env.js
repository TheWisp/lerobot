/* LeRobot GUI - Environment (sim) Tab Logic
 *
 * Mirrors robot.js's profile CRUD pattern, but for sim env profiles
 * (gym-hil / aloha / libero / metaworld / isaaclab_arena / EnvHub).
 * Stored at ~/.config/lerobot/envs/<name>.json.
 *
 * Kept entirely separate from robot.js (own globals, own DOM ids) so the
 * robot/env codepaths don't share mutable state. The price is some
 * duplicated render code — acceptable for the prototype, can be DRY'd
 * later if the patterns prove identical.
 */

let envSchemas = null;
let envProfiles = [];
let currentEnvProfile = null; // {name, data}
let savedEnvProfileData = null;
let envTabInitialized = false;
// Cache of registered tasks per namespace ("gym_hil" -> {source, tasks, warning?}).
// Filled lazily on first render of a profile that wants the task dropdown so
// the editor can render synchronously after the cache hit.
let envTaskCache = {};

// ============================================================================
// Initialization
// ============================================================================

async function envTabInit() {
    if (envTabInitialized) return;
    envTabInitialized = true;
    await Promise.all([loadEnvSchemas(), loadEnvProfiles()]);
}

async function loadEnvSchemas() {
    try {
        const res = await fetch('/api/env/schemas');
        envSchemas = await res.json();
    } catch (e) {
        console.error('Failed to load env schemas:', e);
        envSchemas = [];
    }
}

async function loadEnvProfiles() {
    try {
        const res = await fetch('/api/env/profiles');
        envProfiles = await res.json();
    } catch (e) {
        envProfiles = [];
    }
    renderEnvProfileList();
    if (typeof refreshRunProfileSelects === 'function') refreshRunProfileSelects();
}

/** Fetch and cache the registered-tasks list for a namespace ("gym_hil").
 *  Re-renders the editor on success so the task dropdown picks up live data. */
async function _ensureEnvTasksLoaded(name) {
    if (!name) return;
    if (envTaskCache[name]) return;
    try {
        const res = await fetch(`/api/env/registered-tasks?name=${encodeURIComponent(name)}`);
        if (!res.ok) return;
        envTaskCache[name] = await res.json();
        // Refresh editor so the (now-populated) dropdown shows up if this
        // profile is the one being viewed.
        if (currentEnvProfile && currentEnvProfile.data?.fields?.name === name) {
            renderEnvEditor();
        }
    } catch (e) {
        console.warn('failed to load registered-tasks for', name, e);
    }
}

// ============================================================================
// Rendering
// ============================================================================

function _envEsc(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function renderEnvProfileList() {
    const list = document.getElementById('env-profile-list');
    if (!list) return;
    if (envProfiles.length === 0) {
        list.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No profiles</div>';
        return;
    }
    list.innerHTML = envProfiles.map(p => {
        const isActive = currentEnvProfile?.name === p.name;
        return `<div class="profile-item ${isActive ? 'active' : ''}"
                     onclick="selectEnvProfile('${_envEsc(p.name)}')">
            <span>${_envEsc(p.name)}<span class="profile-type">${_envEsc(p.type)}</span></span>
            <span class="profile-delete" onclick="event.stopPropagation(); deleteEnvProfile('${_envEsc(p.name)}')">&times;</span>
        </div>`;
    }).join('');
}

async function selectEnvProfile(name) {
    try {
        const res = await fetch(`/api/env/profiles/${encodeURIComponent(name)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        currentEnvProfile = { name, data };
        savedEnvProfileData = JSON.parse(JSON.stringify(data));
        renderEnvEditor();
        renderEnvProfileList();
    } catch (e) {
        showToast('Error', `Failed to load env profile: ${e.message}`, 'error');
    }
}

function renderEnvEditor() {
    const emptyEl = document.getElementById('env-empty');
    const editorEl = document.getElementById('env-editor');
    if (!currentEnvProfile) {
        emptyEl.style.display = 'flex';
        editorEl.style.display = 'none';
        return;
    }
    emptyEl.style.display = 'none';
    editorEl.style.display = 'block';

    const schema = envSchemas?.find(s => s.type_name === currentEnvProfile.data.type);

    let html = '';

    // Header
    html += '<div class="editor-header"><div class="editor-header-row">';
    html += `<h2>${_envEsc(currentEnvProfile.name)}</h2>`;
    html += `<button class="btn-small secondary" onclick="renameEnvProfile()" title="Rename">Rename</button>`;
    html += `<button class="btn-small danger" onclick="deleteEnvProfile('${_envEsc(currentEnvProfile.name)}')" title="Delete">Delete</button>`;
    html += '</div></div>';

    // Form
    html += '<div class="editor-form" id="env-editor-form">';
    html += `<label>Type</label>`;
    html += `<select onchange="changeEnvProfileType(this.value)">`;
    if (envSchemas) {
        for (const s of envSchemas) {
            const sel = s.type_name === currentEnvProfile.data.type ? 'selected' : '';
            html += `<option value="${_envEsc(s.type_name)}" ${sel}>${_envEsc(s.type_name)}</option>`;
        }
    }
    html += '</select>';

    // For gym_manipulator profiles we render the `task` field as a
    // dropdown sourced from /api/env/registered-tasks for the current
    // namespace. Other env types and other fields render as before.
    const isGymManipulator = currentEnvProfile.data.type === 'gym_manipulator';
    const namespace = isGymManipulator ? (currentEnvProfile.data.fields?.name || 'gym_hil') : null;
    const taskInfo = namespace ? envTaskCache[namespace] : null;
    if (namespace && !taskInfo) {
        // Kick off load; renderEnvEditor will be re-called on completion
        // (handled inside _ensureEnvTasksLoaded).
        _ensureEnvTasksLoaded(namespace);
    }

    if (schema) {
        for (const field of schema.fields) {
            const value = currentEnvProfile.data.fields?.[field.name] ?? field.default ?? '';
            if (field.name === 'task' && isGymManipulator && taskInfo?.tasks?.length) {
                html += `<label for="env-field-task">task${field.required ? ' *' : ''}</label>`;
                html += `<select id="env-field-task">`;
                let matched = false;
                for (const t of taskInfo.tasks) {
                    const sel = (t === value) ? 'selected' : '';
                    if (sel) matched = true;
                    html += `<option value="${_envEsc(t)}" ${sel}>${_envEsc(t)}</option>`;
                }
                // Stored value not in the live list — keep it visible so the
                // user can tell it's stale rather than silently overwriting.
                if (value && !matched) {
                    html += `<option value="${_envEsc(value)}" selected>${_envEsc(value)} (not registered)</option>`;
                }
                html += `</select>`;
            } else {
                html += renderEnvFormField(field, value);
            }
        }
    }

    // gym_manipulator profiles need a `device` field that lives at the
    // top of GymManipulatorConfig (sibling of env), not inside env. Surface
    // it here for convenience.
    if (isGymManipulator) {
        const dev = currentEnvProfile.data.fields?.device ?? 'cuda';
        html += `<label for="env-field-device">device</label>`;
        html += `<input type="text" id="env-field-device" value="${_envEsc(dev)}" placeholder="cuda or cpu">`;
    }

    html += '</div>';

    // Loud warning: gym-hil's Gamepad variants exit cleanly with rc=0
    // within ~6s when no controller is plugged in — looks like an instant
    // crash from the GUI side. Surface this *before* launch.
    if (isGymManipulator) {
        const task = String(currentEnvProfile.data.fields?.task || '');
        if (task.includes('Gamepad')) {
            html += '<div class="form-section" style="border:1px solid #b58900; background:rgba(181,137,0,0.08); padding:10px;">';
            html += '<div class="form-section-title" style="color:#b58900">⚠ Gamepad task selected</div>';
            html += '<div class="form-hint" style="line-height:1.6">';
            html += 'This task <b>requires a USB gamepad</b>. Without one, gym-hil prints "No gamepad detected" and the sim exits silently in ~6 seconds (rc=0). The Launch button will refuse with a 400 if no controller is plugged in.<br><br>';
            html += 'Pick a <code>*Keyboard-v0</code> or <code>*Base-v0</code> variant from the <code>task</code> dropdown for a no-hardware setup.';
            html += '</div></div>';
        }
        // Show source provenance (registry vs fallback) so users can see
        // when the dropdown is stale because gym_hil isn't installed.
        if (taskInfo) {
            html += '<div class="form-section">';
            html += '<div class="form-section-title">Tasks registered for ' + _envEsc(namespace) + '</div>';
            html += '<div class="form-hint" style="line-height:1.6">';
            if (taskInfo.source === 'registry') {
                html += `<code>${taskInfo.tasks.length}</code> task${taskInfo.tasks.length === 1 ? '' : 's'} discovered from <code>gymnasium.envs.registry</code>. The dropdown above will reflect any tasks the package adds.`;
            } else {
                html += `<span style="color:#b58900">⚠ Using built-in fallback list</span> — ${_envEsc(taskInfo.warning || 'package import failed')}.`;
            }
            html += '</div></div>';
        }
    }

    // Save bar
    html += '<div class="editor-actions" id="env-editor-actions" style="display:none">';
    html += `<button class="btn-save" onclick="saveEnvProfile()">Save Changes</button>`;
    html += `<button class="btn-discard" onclick="discardEnvChanges()">Discard</button>`;
    html += '</div>';

    editorEl.innerHTML = html;

    const form = document.getElementById('env-editor-form');
    if (form) {
        form.addEventListener('input', _updateEnvDirtyState);
        form.addEventListener('change', _updateEnvDirtyState);
    }
}

function renderEnvFormField(field, value) {
    const id = `env-field-${field.name}`;
    const label = `<label for="${id}">${_envEsc(field.name)}${field.required ? ' *' : ''}</label>`;
    const typeStr = (field.type_str || '').toLowerCase();

    if (typeStr === 'bool' || typeStr === 'bool | none') {
        const t = value === true ? 'selected' : '';
        const f = value === false ? 'selected' : '';
        const n = (value === null || value === '' || value === undefined) ? 'selected' : '';
        return label + `<select id="${id}">
            ${!field.required ? `<option value="null" ${n}>--</option>` : ''}
            <option value="true" ${t}>true</option>
            <option value="false" ${f}>false</option>
        </select>`;
    }
    if (typeStr.includes('int') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="1">`;
    }
    if (typeStr.includes('float') && !typeStr.includes('dict') && !typeStr.includes('|')) {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="any">`;
    }
    const display = (value === null || value === undefined) ? '' : String(value);
    return label + `<input type="text" id="${id}" value="${_envEsc(display)}"
        ${!field.required ? 'placeholder="(optional)"' : ''}>`;
}

function changeEnvProfileType(newType) {
    if (!currentEnvProfile) return;
    currentEnvProfile.data.type = newType;
    currentEnvProfile.data.fields = {};
    renderEnvEditor();
    _updateEnvDirtyState();
}

// ============================================================================
// Dirty tracking
// ============================================================================

function _envParseValue(field, raw) {
    if (raw === '' || raw === null) return null;
    const t = (field.type_str || '').toLowerCase();
    if (t === 'bool' || t === 'bool | none') {
        if (raw === 'true') return true;
        if (raw === 'false') return false;
        return null;
    }
    if (t.includes('int') && !t.includes('|')) {
        const n = parseInt(raw);
        return isNaN(n) ? null : n;
    }
    if (t.includes('float') && !t.includes('|')) {
        const n = parseFloat(raw);
        return isNaN(n) ? null : n;
    }
    return raw;
}

function _collectEnvFormFields() {
    const schema = envSchemas?.find(s => s.type_name === currentEnvProfile.data.type);
    const fields = {};
    if (schema) {
        for (const field of schema.fields) {
            const input = document.getElementById(`env-field-${field.name}`);
            if (!input) continue;
            const v = _envParseValue(field, input.value);
            if (v !== null) fields[field.name] = v;
        }
    }
    // Special: device for gym_manipulator
    if (currentEnvProfile.data.type === 'gym_manipulator') {
        const devEl = document.getElementById('env-field-device');
        if (devEl && devEl.value) fields.device = devEl.value.trim();
    }
    return fields;
}

function _serializeEnvProfile(data) {
    return JSON.stringify({
        type: data.type,
        name: data.name,
        fields: data.fields || {},
    });
}

function _isEnvDirty() {
    if (!currentEnvProfile || !savedEnvProfileData) return false;
    const current = _serializeEnvProfile({
        ...currentEnvProfile.data,
        name: currentEnvProfile.name,
        fields: _collectEnvFormFields(),
    });
    return current !== _serializeEnvProfile(savedEnvProfileData);
}

function _updateEnvDirtyState() {
    const bar = document.getElementById('env-editor-actions');
    if (bar) bar.style.display = _isEnvDirty() ? 'flex' : 'none';
}

function discardEnvChanges() {
    if (!currentEnvProfile || !savedEnvProfileData) return;
    currentEnvProfile.data = JSON.parse(JSON.stringify(savedEnvProfileData));
    renderEnvEditor();
    _updateEnvDirtyState();
}

// ============================================================================
// CRUD
// ============================================================================

async function newEnvProfile() {
    const name = prompt('Env profile name:');
    if (!name || !name.trim()) return;
    const defaultType = 'gym_manipulator';
    const profile = {
        type: defaultType,
        name: name.trim(),
        fields: { name: 'gym_hil', task: 'PandaPickCubeGamepad-v0', fps: 10, device: 'cuda' },
    };
    try {
        const res = await fetch('/api/env/profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadEnvProfiles();
        await selectEnvProfile(name.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function saveEnvProfile() {
    if (!currentEnvProfile) return;
    const payload = {
        ...currentEnvProfile.data,
        name: currentEnvProfile.name,
        fields: _collectEnvFormFields(),
    };
    try {
        const res = await fetch(`/api/env/profiles/${encodeURIComponent(currentEnvProfile.name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        currentEnvProfile.data = payload;
        savedEnvProfileData = JSON.parse(JSON.stringify(payload));
        _updateEnvDirtyState();
        showToast('Saved', `Env profile "${currentEnvProfile.name}" saved`, 'success');
        await loadEnvProfiles();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function deleteEnvProfile(name) {
    if (!confirm(`Delete env profile "${name}"?`)) return;
    try {
        const res = await fetch(`/api/env/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(await res.text());
        if (currentEnvProfile?.name === name) {
            currentEnvProfile = null;
            savedEnvProfileData = null;
            renderEnvEditor();
        }
        await loadEnvProfiles();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function renameEnvProfile() {
    if (!currentEnvProfile) return;
    const newName = prompt('New name:', currentEnvProfile.name);
    if (!newName || newName === currentEnvProfile.name) return;
    try {
        const res = await fetch(`/api/env/profiles/${encodeURIComponent(currentEnvProfile.name)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_name: newName.trim() }),
        });
        if (!res.ok) throw new Error(await res.text());
        await loadEnvProfiles();
        await selectEnvProfile(newName.trim());
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}
