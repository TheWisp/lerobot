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
    // Guard against silent loss when switching with unsaved edits in flight.
    // Don't prompt when re-selecting the same profile (idempotent click).
    if (currentEnvProfile && currentEnvProfile.name !== name && _isEnvDirty()) {
        const ok = confirm(
            `"${currentEnvProfile.name}" has unsaved changes. ` +
            `Discard them and switch to "${name}"?`
        );
        if (!ok) return;
    }
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
    // it here as the same enum-alike select used elsewhere; the choices
    // mirror _GLOBAL_FIELD_CHOICES["device"] in env.py — duplicated here
    // because this is the only field rendered outside the schema loop.
    if (isGymManipulator) {
        const dev = currentEnvProfile.data.fields?.device ?? 'cuda';
        html += `<label for="env-field-device">device</label>`;
        html += _renderEnvChoiceSelect('env-field-device', dev, ['cuda', 'cpu', 'mps'], false);
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
        // gym-hil controls — only for variants that take human input.
        // Keyboard and Gamepad have *fundamentally different* intervention
        // models (toggle vs hold) and entirely different keymaps, so split
        // the rendering rather than papering over with shared text.
        const isKeyboardTask = task.includes("Keyboard");
        const isGamepadTask = task.includes("Gamepad");
        if (isKeyboardTask) {
            html += '<div class="form-section">';
            html += '<div class="form-section-title">Keyboard controls (read this before Launch)</div>';
            html += '<div class="form-hint" style="line-height:1.6">';
            html += '<b>Press <code>Space</code> first</b> to enable intervention — Space <i>toggles</i> (every press flips on/off, no LED to tell you which state you\'re in). The arm stays still until you toggle on.<br><br>';
            html += '<b>Movement</b> (after Space): Arrows = X/Y, <code>Shift</code> / <code>RShift</code> = Z down/up, <code>LCtrl</code> / <code>RCtrl</code> = gripper close/open.<br>';
            html += '<b>End episode</b>: <code>Enter</code> = success, <code>Esc</code> = failure (despite the printed help text saying "ESC: Exit", it actually ends the episode). <code>r</code> = rerecord. The env <i>also</i> auto-ends on natural success (cube lifted &gt;10cm) or if the cube goes off-table.<br>';
            html += '<b>Stop the run</b>: hit the GUI <code>Stop</code> button (the printed "Press Ctrl+C" hint is for direct CLI use, not the GUI).<br><br>';
            html += '<span style="color:#b58900">⚠ System-wide keyboard capture.</span> gym-hil uses <code>pynput</code>, which grabs keys from the focused window — any window. Typing <code>Enter</code> in the terminal, browser, IDE, etc. will end the current episode. Be deliberate about what you type while the env is running.';
            html += '</div></div>';
        } else if (isGamepadTask) {
            // Defaults below are Logitech F310 button labels; gym-hil maps
            // by name via controller_config.json so PS/Xbox controllers
            // get their own labels. We can't know the user's controller
            // here, so we name the *function* and the F310 label in
            // parentheses.
            html += '<div class="form-section">';
            html += '<div class="form-section-title">Gamepad controls (read this before Launch)</div>';
            html += '<div class="form-hint" style="line-height:1.6">';
            html += '<b>Hold <code>RB</code></b> (right shoulder button) to engage intervention. This is hold-to-engage, not a toggle — release RB and the arm goes back to autonomous. Different from the keyboard variant.<br><br>';
            html += '<b>Movement</b> (while holding RB): Left analog stick = X/Y, right stick (vertical) = Z, <code>LT</code> = close gripper, <code>RT</code> = open gripper.<br>';
            html += '<b>End episode</b>: <code>Y</code> / Triangle = success, <code>A</code> / Cross = failure, <code>X</code> / Square = rerecord, <code>B</code> / Circle = exit. The env <i>also</i> auto-ends on natural success (cube lifted &gt;10cm) or if the cube goes off-table.<br>';
            html += '<b>Stop the run</b>: hit the GUI <code>Stop</code> button.<br><br>';
            html += '<span style="color:#999">Button labels above are Logitech F310 defaults. gym-hil reads button mappings from <code>controller_config.json</code> in the gym-hil package — PS / Xbox controllers map to their own labels but the same functions.</span>';
            html += '</div></div>';
        }

        // Behaviour reference — termination rules, reward, action space
        // apply to ALL gym-hil variants (autonomous and human-input
        // alike). A user setting up an autonomous policy eval against
        // PandaPickCubeBase-v0 needs to know about the bounds-termination
        // and the +10cm success threshold just as much as a teleop user.
        // Render this independently of human-input gating.
        const srcUrl = _gymHilSourceUrl(task);
        html += '<div class="form-section">';
        html += '<div class="form-section-title">Behaviour reference</div>';
        html += '<div class="form-hint" style="line-height:1.6">';
        html += 'gym-hil envs have implicit termination / reward rules not surfaced in the form. ';
        html += 'For PickCube: success when the cube is lifted &gt;10cm; episode also ends if the cube is bumped off-table. ';
        html += 'For ArrangeBoxes: similar — see source. To read the exact rules:<br>';
        html += `&nbsp;&bull; <a href="${_envEsc(srcUrl)}" target="_blank" rel="noopener">📖 Source for this task family</a> on GitHub<br>`;
        html += `&nbsp;&bull; <a href="https://github.com/huggingface/gym-hil" target="_blank" rel="noopener">gym-hil package</a> README<br>`;
        html += `&nbsp;&bull; <a href="https://huggingface.co/docs/lerobot/hilserl_sim" target="_blank" rel="noopener">LeRobot HIL-SERL sim guide</a>`;
        html += '</div></div>';
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

/** Render a <select> with one option per choice; preserve out-of-list
 *  values as a labelled option so stale profiles don't get silently
 *  rewritten on first save. */
function _renderEnvChoiceSelect(id, value, choices, optional) {
    let html = `<select id="${id}">`;
    if (optional) {
        const n = (value === null || value === '' || value === undefined) ? 'selected' : '';
        html += `<option value="" ${n}>(unset)</option>`;
    }
    let matched = false;
    for (const c of choices) {
        const sel = (c === value) ? 'selected' : '';
        if (sel) matched = true;
        html += `<option value="${_envEsc(c)}" ${sel}>${_envEsc(c)}</option>`;
    }
    if (value && !matched) {
        // Pre-existing profile carries a value not in the current registry —
        // keep it visible (and labeled) rather than silently overwriting.
        html += `<option value="${_envEsc(value)}" selected>${_envEsc(value)} (custom)</option>`;
    }
    html += '</select>';
    return html;
}

function renderEnvFormField(field, value) {
    const id = `env-field-${field.name}`;
    const label = `<label for="${id}">${_envEsc(field.name)}${field.required ? ' *' : ''}</label>`;
    const typeStr = (field.type_str || '').toLowerCase();

    // Enum-alike fields the backend marked with a choices list. Hits before
    // the type-string fallthroughs because choices imply discrete values.
    if (Array.isArray(field.choices) && field.choices.length > 0) {
        return label + _renderEnvChoiceSelect(id, value, field.choices, !field.required);
    }

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
    // Render exact int/float types AND their `T | None` Optional variants
    // as <input type="number">. The previous heuristic excluded the | None
    // case and rendered as text, which silently saved the value as a
    // string (the parser bug above is the symmetric counterpart).
    if (typeStr === 'int' || typeStr === 'int | none') {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="1">`;
    }
    if (typeStr === 'float' || typeStr === 'float | none') {
        return label + `<input type="number" id="${id}" value="${value ?? ''}" step="any">`;
    }
    const display = (value === null || value === undefined) ? '' : String(value);
    return label + `<input type="text" id="${id}" value="${_envEsc(display)}"
        ${!field.required ? 'placeholder="(optional)"' : ''}>`;
}

/** Map a gym-hil task id (e.g. "PandaPickCubeKeyboard-v0") to the GitHub
 *  source URL of the env class that implements it. Lets us surface a
 *  "📖 Source" link the user can click to read termination / reward /
 *  bounds rules — these aren't documented anywhere user-facing today,
 *  they live as Python in the gym-hil package. */
function _gymHilSourceUrl(task) {
    const t = String(task || "");
    const repo = "https://github.com/huggingface/gym-hil/blob/main";
    if (t.includes("PandaPickCube")) return `${repo}/gym_hil/envs/panda_pick_gym_env.py`;
    if (t.includes("PandaArrangeBoxes")) return `${repo}/gym_hil/envs/panda_arrange_boxes_gym_env.py`;
    return `${repo}/gym_hil/envs/`;
}

/** Sensible starter fields for a fresh profile of this type. Without
 *  this, fields={} causes the form to render schema defaults — which
 *  for HILSerlRobotEnvConfig means name="real_robot", and saving that
 *  routes gym_manipulator down the hardware path that asserts
 *  cfg.robot != None and dies. Mirrors the seed in newEnvProfile so
 *  type-change and new-profile paths agree. */
function _defaultEnvFields(type) {
    if (type === 'gym_manipulator') {
        return {
            name: 'gym_hil',
            task: 'PandaPickCubeKeyboard-v0',
            fps: 10,
            device: 'cuda',
            // Override gym-hil's registered max_episode_steps=100 (which is
            // ~3.3s at fps=30 — too short for human teleop). 1500 ≈ 50s at
            // 30fps; the user can lengthen / shorten in the editor.
            max_episode_steps: 1500,
        };
    }
    return {};
}

function changeEnvProfileType(newType) {
    if (!currentEnvProfile) return;
    currentEnvProfile.data.type = newType;
    currentEnvProfile.data.fields = _defaultEnvFields(newType);
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
    // Match exact int / float types AND their `T | None` Optional variants.
    // The previous heuristic (`includes('int') && !includes('|')`) wrongly
    // excluded `int | None`, causing optional ints (e.g. max_episode_steps)
    // to fall through to the string catch-all and persist as "1500"
    // instead of 1500. draccus coerced the string back to int silently
    // on read, but the JSON on disk was wrong.
    if (t === 'int' || t === 'int | none') {
        const n = parseInt(raw);
        return isNaN(n) ? null : n;
    }
    if (t === 'float' || t === 'float | none') {
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

/** Drop fields whose value equals the schema default. Without this, the
 *  dirty checker fires the moment a profile is rendered: the form
 *  pre-populates every schema field with its default, _collectEnvFormFields
 *  reads those defaults back, and they don't appear in the saved profile
 *  (which only stores fields the user explicitly set). The fix is to
 *  normalize both sides identically before comparison — implicit defaults
 *  on either side are equivalent to absence. */
function _normalizeEnvFields(fields, schema) {
    const out = {};
    const defaults = {};
    if (schema) for (const f of schema.fields) defaults[f.name] = f.default;
    for (const [k, v] of Object.entries(fields || {})) {
        if (k in defaults && defaults[k] !== undefined && defaults[k] !== null && v === defaults[k]) continue;
        out[k] = v;
    }
    // device is a gym_manipulator-only field rendered outside the schema
    // loop (lives at GymManipulatorConfig top level, not inside EnvConfig).
    // "cuda" is the implicit default we ship in newEnvProfile + the form;
    // dropping it keeps profiles that don't override it from looking dirty.
    if (out.device === 'cuda') delete out.device;
    return out;
}

/** Stable, key-sorted JSON for cross-state equality — JSON.stringify alone
 *  reflects insertion order, which differs between freshly-loaded saved
 *  data and freshly-collected current form data even when the contents
 *  match. */
function _stableJSON(obj) {
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) return JSON.stringify(obj);
    const keys = Object.keys(obj).sort();
    return '{' + keys.map(k => JSON.stringify(k) + ':' + _stableJSON(obj[k])).join(',') + '}';
}

function _isEnvDirty() {
    if (!currentEnvProfile || !savedEnvProfileData) return false;
    if (currentEnvProfile.name !== savedEnvProfileData.name) return true;
    if (currentEnvProfile.data.type !== savedEnvProfileData.type) return true;
    const schema = envSchemas?.find(s => s.type_name === currentEnvProfile.data.type);
    const cur = _normalizeEnvFields(_collectEnvFormFields(), schema);
    const persisted = _normalizeEnvFields(savedEnvProfileData.fields, schema);
    return _stableJSON(cur) !== _stableJSON(persisted);
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
        fields: _defaultEnvFields(defaultType),
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
