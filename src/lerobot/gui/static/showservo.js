// Show-and-Servo tab: capture RGB+depth scenes and run the first-contact bench.
//
// Two ways into a session: start a live RealSense (capture new scenes) or reopen an
// existing capture directory (re-analysis, no camera). The bind runs server-side as a
// subprocess of benchmarks/showservo_real.py — the GUI shows its log verbatim and the
// per-scene overlays it writes, so the button and the command line cannot disagree.

let ssPreviewTimer = null;
let ssLogTimer = null;
let ssScenes = [];
let ssTeach = new Set([0]);
let ssLive = false;

async function ssRefreshCameras() {
    const sel = document.getElementById('ss-camera');
    sel.innerHTML = '<option value="">scanning…</option>';
    try {
        const cams = await (await fetch('/api/showservo/cameras')).json();
        sel.innerHTML = cams.length
            ? cams.map(c => `<option value="${c.serial}">${c.serial} ${c.name || ''}</option>`).join('')
            : '<option value="">no RealSense found</option>';
    } catch (e) {
        sel.innerHTML = '<option value="">scan failed</option>';
    }
}

async function ssRefreshSessions() {
    const sel = document.getElementById('ss-existing');
    const sessions = await (await fetch('/api/showservo/sessions')).json();
    sel.innerHTML = sessions.length
        ? sessions.map(s => `<option value="${s.name}">${s.name} (${s.scenes} scenes)</option>`).join('')
        : '<option value="">none yet</option>';
}

function ssSetStatus(text, isError = false) {
    const el = document.getElementById('ss-status');
    el.textContent = text;
    el.style.color = isError ? '#e06c75' : '#888';
}

async function ssStart() {
    const serial = document.getElementById('ss-camera').value;
    if (!serial) { ssSetStatus('pick a camera first', true); return; }
    ssSetStatus('connecting…');
    const r = await fetch('/api/showservo/session/start', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({serial, name: document.getElementById('ss-name').value}),
    });
    if (!r.ok) { ssSetStatus((await r.json()).detail || 'connect failed', true); return; }
    const info = await r.json();
    ssEnterSession(info);
}

async function ssOpen() {
    const name = document.getElementById('ss-existing').value;
    if (!name) return;
    const r = await fetch('/api/showservo/session/open', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name}),
    });
    if (!r.ok) { ssSetStatus((await r.json()).detail || 'open failed', true); return; }
    ssEnterSession(await r.json());
}

function ssEnterSession(info) {
    ssLive = info.live;
    ssScenes = info.scenes || [];
    ssTeach = new Set(ssScenes.map((_, i) => i));  // all teachable by default
    document.getElementById('ss-setup').style.display = 'none';
    document.getElementById('ss-session').style.display = '';
    document.getElementById('ss-session-name').textContent = info.name + (info.live ? ' (live)' : ' (reopened)');
    document.getElementById('ss-capture-btn').style.display = info.live ? '' : 'none';
    document.getElementById('ss-preview-wrap').style.display = info.live ? '' : 'none';
    ssSetStatus(info.live ? 'camera live' : 'session reopened — bind only');
    ssRenderScenes();
    if (info.live) {
        ssPreviewTimer = setInterval(ssPreviewTick, 250);
    }
}

async function ssStop() {
    if (ssPreviewTimer) { clearInterval(ssPreviewTimer); ssPreviewTimer = null; }
    if (ssLogTimer) { clearInterval(ssLogTimer); ssLogTimer = null; }
    await fetch('/api/showservo/session/stop', {method: 'POST'});
    document.getElementById('ss-setup').style.display = '';
    document.getElementById('ss-session').style.display = 'none';
    ssSetStatus('');
    ssRefreshSessions();
}

async function ssCapture() {
    const btn = document.getElementById('ss-capture-btn');
    btn.disabled = true;
    try {
        const r = await fetch('/api/showservo/capture', {method: 'POST'});
        if (!r.ok) { ssSetStatus((await r.json()).detail || 'capture failed', true); return; }
        const info = await r.json();
        ssScenes.push(info);
        ssTeach.add(ssScenes.length - 1);  // new captures teach by default
        ssRenderScenes();
        ssSetStatus(`captured ${info.name} — depth valid ${(info.depth_valid * 100).toFixed(0)}%`);
    } finally {
        btn.disabled = false;
    }
}

function ssToggleTeach(i) {
    if (ssTeach.has(i)) ssTeach.delete(i); else ssTeach.add(i);
    ssRenderScenes();
}

function ssRenderScenes() {
    const strip = document.getElementById('ss-scenes');
    strip.innerHTML = ssScenes.map((s, i) => `
        <div class="ss-scene ${ssTeach.has(i) ? 'ss-teach' : ''}">
            <img src="/api/showservo/scene/${s.name}/${s.has_overlay ? 'overlay.jpg' : (s.has_preview !== false ? 'preview.jpg' : 'rgb.png')}?t=${Date.now()}"
                 title="${s.name}" onclick="window.open(this.src, '_blank')">
            <div class="ss-scene-row">
                <span>${s.name.replace('scene_', '#')}${s.depth_valid !== undefined ? ` · d${(s.depth_valid * 100).toFixed(0)}%` : ''}</span>
                <label title="use this scene as a taught demo">
                    <input type="checkbox" ${ssTeach.has(i) ? 'checked' : ''} onchange="ssToggleTeach(${i})"> teach
                </label>
            </div>
        </div>`).join('') || '<div style="color:#666;padding:12px;">no scenes yet</div>';
}

let ssLiveFit = false;

async function ssLiveToggle() {
    const btn = document.getElementById('ss-live-btn');
    if (ssLiveFit) {
        await fetch('/api/showservo/live/stop', {method: 'POST'});
        ssLiveFit = false;
        btn.textContent = 'Live fit';
        ssSetStatus('live fit stopped');
        return;
    }
    if (!ssTeach.size) { ssSetStatus('mark at least one captured scene as teach first', true); return; }
    const concept = document.getElementById('ss-concept').value;
    if (!concept.trim()) { ssSetStatus('type the concept first (e.g. "blue box")', true); return; }
    const r = await fetch('/api/showservo/live/start', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({concept, teach: [...ssTeach].sort((a, b) => a - b)}),
    });
    if (!r.ok) { ssSetStatus((await r.json()).detail || 'live fit failed to start', true); return; }
    ssLiveFit = true;
    btn.textContent = 'Stop live fit';
    ssSetStatus('live fit: teaching (models load once, ~20 s), then the ghost tracks the object');
}

// One preview poller for every mode: raw camera view normally, the worker's
// annotated frame while a live worker (fit or M1) runs (404s fall back to raw
// until the first result arrives).
function ssPreviewTick() {
    const img = document.getElementById('ss-preview');
    if (!ssLiveFit) { img.src = '/api/showservo/preview.jpg?t=' + Date.now(); return; }
    fetch('/api/showservo/live/status').then(r => r.json()).then(st => {
        if (!st.running && ssLiveFit) {
            ssLiveFit = false;
            document.getElementById('ss-live-btn').textContent = 'Live fit';
            document.getElementById('ss-m1-btn').textContent = 'Start M1';
            ssSetStatus('live worker exited — last output: ' + (st.log || ''), true);
            return;
        }
        // Overwrite any stale exit message while a worker IS running: the red line
        // from a failed attempt outliving the next (running) attempt caused a
        // "still the same error" misread in the field.
        if (st.running) {
            ssSetStatus(st.has_overlay
                ? (st.kind === 'm1' ? 'M1 running — state is in the overlay header' : 'live fit running')
                : 'worker teaching (~20 s)…');
        }
        img.src = st.has_overlay
            ? '/api/showservo/live/overlay.jpg?t=' + Date.now()
            : '/api/showservo/preview.jpg?t=' + Date.now();
    }).catch(() => {});
}

// --- M1: the arm ------------------------------------------------------------------

let ssArmConnected = false;

function ssM1Status(text, isError = false) {
    const el = document.getElementById('ss-m1-status');
    el.textContent = text;
    el.style.color = isError ? '#e06c75' : '#888';
}

async function ssRefreshProfiles() {
    const sel = document.getElementById('ss-m1-profile');
    try {
        const profiles = await (await fetch('/api/robot/profiles')).json();
        const usable = profiles.filter(p => p.type === 'bi_so107_follower');
        sel.innerHTML = usable.length
            ? usable.map(p => `<option value="${p.name}">${p.name}</option>`).join('')
            : '<option value="">no bi_so107 profile</option>';
    } catch (e) {
        sel.innerHTML = '<option value="">profiles unavailable</option>';
    }
}

async function ssRefreshArm() {
    try {
        const st = await (await fetch('/api/showservo/arm/state')).json();
        ssArmConnected = !!st.connected;
        document.getElementById('ss-arm-connect-btn').textContent =
            ssArmConnected ? `Disconnect ${st.arm} arm` : 'Connect arm';
        document.getElementById('ss-m1-btn').disabled = !ssArmConnected;
        if (ssArmConnected && st.stopped) ssM1Status('arm is STOPPED — reconnect to re-arm', true);
    } catch (e) { /* server restart etc. — leave the UI as is */ }
}

async function ssArmToggle() {
    const btn = document.getElementById('ss-arm-connect-btn');
    btn.disabled = true;
    try {
        if (ssArmConnected) {
            await fetch('/api/showservo/arm/disconnect', {method: 'POST'});
            ssM1Status('arm disconnected');
        } else {
            const body = {
                profile: document.getElementById('ss-m1-profile').value,
                arm: document.getElementById('ss-m1-arm').value,
            };
            const r = await fetch('/api/showservo/arm/connect', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            if (!r.ok) { ssM1Status((await r.json()).detail || 'arm connect failed', true); return; }
            ssM1Status(`arm connected (${body.arm}) — steps clamped to 3 units, travel to ±30`);
        }
    } finally {
        btn.disabled = false;
        ssRefreshArm();
    }
}

async function ssM1Toggle() {
    if (ssLiveFit) {  // the M1 worker occupies the live slot; toggling off = stop it
        await ssM1Stop();
        return;
    }
    if (!ssTeach.size) { ssSetStatus('tick teach on the demo scenes first', true); return; }
    const body = {
        concept: document.getElementById('ss-m1-target').value.trim(),
        held_concept: document.getElementById('ss-m1-held').value.trim(),
        teach: [...ssTeach].sort((a, b) => a - b),
        arm: document.getElementById('ss-m1-arm').value,
    };
    if (!body.concept) { ssM1Status('type the target concept (the object being reached)', true); return; }
    if (!body.held_concept) { ssM1Status('type the held-end concept (the gripper)', true); return; }
    const r = await fetch('/api/showservo/m1/start', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    if (!r.ok) { ssM1Status((await r.json()).detail || 'M1 failed to start', true); return; }
    ssLiveFit = true;
    document.getElementById('ss-m1-btn').textContent = 'Stop M1';
    ssM1Status('M1: teaching (~20 s), then WAIT → PROBE (3 tiny moves) → SERVO. STOP freezes the arm.');
}

async function ssM1Stop() {
    // Freeze the arm FIRST, then kill the worker: the arm must never outlive the stop.
    await fetch('/api/showservo/arm/stop', {method: 'POST'}).catch(() => {});
    await fetch('/api/showservo/live/stop', {method: 'POST'}).catch(() => {});
    ssLiveFit = false;
    document.getElementById('ss-m1-btn').textContent = 'Start M1';
    document.getElementById('ss-live-btn').textContent = 'Live fit';
    ssM1Status('stopped — the arm holds position; reconnect it to re-arm');
    ssRefreshArm();
}

async function ssBind() {
    if (!ssTeach.size) { ssSetStatus('mark at least one scene as teach', true); return; }
    const body = {
        concept: document.getElementById('ss-concept').value,
        mask: document.getElementById('ss-mask').value,
        teach: [...ssTeach].sort((a, b) => a - b),
    };
    const r = await fetch('/api/showservo/bind', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    if (!r.ok) { ssSetStatus((await r.json()).detail || 'bind failed to start', true); return; }
    document.getElementById('ss-bind-btn').disabled = true;
    document.getElementById('ss-log').textContent = 'starting…';
    ssLogTimer = setInterval(ssPollLog, 700);
}

async function ssPollLog() {
    const st = await (await fetch('/api/showservo/bind/log')).json();
    document.getElementById('ss-log').textContent = st.log || '…';
    const pre = document.getElementById('ss-log');
    pre.scrollTop = pre.scrollHeight;
    if (st.done) {
        clearInterval(ssLogTimer); ssLogTimer = null;
        document.getElementById('ss-bind-btn').disabled = false;
        ssSetStatus(st.ok ? 'bind finished' : 'bind FAILED — see log', !st.ok);
        const state = await (await fetch('/api/showservo/state')).json();
        if (state.session) { ssScenes = state.session.scenes; ssRenderScenes(); }
    }
}

function ssInitTab() {
    ssRefreshCameras();
    ssRefreshSessions();
    ssRefreshProfiles();
    ssRefreshArm();
    // Session state (camera handle included) lives server-side; a page reload must
    // re-attach rather than orphan it behind the setup screen.
    fetch('/api/showservo/state').then(r => r.json()).then(st => {
        if (st.session && !ssPreviewTimer) ssEnterSession(st.session);
    }).catch(() => {});
}


// ── Jog panel: one arm owned by the server, driven from the URDF tile's gizmo ─
let jogConnected = false, jogTimer = null;

async function jogRefreshProfiles() {
    const sel = document.getElementById('jog-profile');
    try {
        const profiles = await (await fetch('/api/robot/profiles')).json();
        const usable = profiles.filter(p => p.type === 'bi_so107_follower' || p.type === 'so107_follower');
        sel.innerHTML = usable.length
            ? usable.map(p => `<option value="${p.name}">${p.name}</option>`).join('')
            : '<option value="">no SO-107 profile</option>';
    } catch (e) { sel.innerHTML = '<option value="">profiles unavailable</option>'; }
}

function jogStatus(text, isError = false) {
    const el = document.getElementById('jog-status');
    el.textContent = text;
    el.style.color = isError ? '#e06c75' : '#888';
}

async function jogToggle() {
    const btn = document.getElementById('jog-connect-btn');
    btn.disabled = true;
    try {
        if (jogConnected) {
            await fetch('/api/jog/disconnect', {method: 'POST'});
            jogConnected = false;
            clearInterval(jogTimer);
            btn.textContent = 'Connect';
            document.getElementById('jog-stop-btn').disabled = true;
            jogStatus('disconnected — the arm holds, torque on');
            return;
        }
        const num = (id) => { const v = document.getElementById(id).value; return v === '' ? null : Number(v); };
        const body = {
            profile: document.getElementById('jog-profile').value,
            arm: document.getElementById('jog-arm').value,
            p_coefficient: num('jog-p'), i_coefficient: num('jog-i'), gravity_ff_alpha: num('jog-alpha'),
        };
        // The tile polls /api/jog/meta until the arm is up, so it can start now.
        const tile = document.getElementById('jog-tile');
        if (!tile.src) {
            tile.src = '/static/urdf_viz.html?mode=jog&v=6';
            tile.addEventListener('load', jogGhostFloor, {once: true});
        }
        jogLimits();
        const r = await fetch('/api/jog/connect', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        });
        if (!r.ok) { jogStatus((await r.json()).detail || 'connect failed', true); return; }
        jogConnected = true;
        btn.textContent = 'Disconnect';
        document.getElementById('jog-stop-btn').disabled = false;
        jogStatus('connected — drag the gizmo in the view');
        jogTimer = setInterval(jogPoll, 500);
    } finally {
        btn.disabled = false;
    }
}

function jogMode(mode) {
    const tile = document.getElementById('jog-tile');
    if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-mode', mode}, '*');
}

function jogGhostFloor() {
    const mm = Number(document.getElementById('jog-ghost-floor').value);
    document.getElementById('jog-ghost-floor-val').textContent = mm;
    const tile = document.getElementById('jog-tile');
    if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-ghost-floor', mm}, '*');
}

let jogLimitsTimer = null;
function jogLimits() {
    const lin = Number(document.getElementById('jog-speed').value);
    const ang = Number(document.getElementById('jog-turn').value);
    const cap = Number(document.getElementById('jog-cap').value);
    document.getElementById('jog-speed-val').textContent = lin;
    document.getElementById('jog-turn-val').textContent = ang;
    document.getElementById('jog-cap-val').textContent = cap;
    // Coalesce a drag into one request in flight at a time.
    clearTimeout(jogLimitsTimer);
    jogLimitsTimer = setTimeout(() => {
        fetch('/api/jog/limits', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({linear_mm_s: lin, angular_deg_s: ang, rotation_cap_deg: cap}),
        }).catch(() => {});
    }, 80);
}

async function jogReattach() {
    // The arm lives server-side; a page reload must pick the connection up, not fight it.
    try {
        const st = await (await fetch('/api/jog/state')).json();
        if (!st.connected || jogConnected) return;
        jogConnected = true;
        document.getElementById('jog-connect-btn').textContent = 'Disconnect';
        document.getElementById('jog-stop-btn').disabled = false;
        const tile = document.getElementById('jog-tile');
        if (!tile.src) {
            tile.src = '/static/urdf_viz.html?mode=jog&v=6';
            tile.addEventListener('load', jogGhostFloor, {once: true});
        }
        if (!jogTimer) jogTimer = setInterval(jogPoll, 500);
        jogStatus('re-attached to the connected arm');
    } catch (e) { /* no server */ }
}

async function jogRecover() {
    const btn = document.getElementById('jog-recover-btn');
    btn.disabled = true;
    try {
        const r = await fetch('/api/jog/recover', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { jogStatus(d.detail || 'recover failed', true); return; }
        jogStatus(d.cleared.length ? `cleared overload on ${d.cleared.join(', ')} — resumed from the present pose` : 'no latched motor found — resumed from the present pose');
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { jogStatus(String(e), true); }
    finally { btn.disabled = false; }
}

async function jogStop() {
    await fetch('/api/jog/stop', {method: 'POST'});
    jogStatus('frozen at the current command — disconnect and reconnect to resume', true);
}

async function jogPoll() {
    try {
        const st = await (await fetch('/api/jog/state')).json();
        if (!st.connected) return;
        const t = st.temps || {};
        const hottest = Object.keys(t).length ? Math.max(...Object.values(t)) : null;
        if (st.gripper && !jogGripDragging) {
            document.getElementById('jog-grip').value = Math.round(st.gripper.obs);
            document.getElementById('jog-grip-val').textContent = Math.round(st.gripper.obs);
        }
        jogStatus(`gap ${st.err_mm.toFixed(1)} mm / ${st.err_deg.toFixed(1)}°` +
                  (hottest !== null ? ` · hottest motor ${hottest} °C` : '') +
                  (st.halted ? ` · FROZEN: ${st.reason}` : ''), !!st.halted);
    } catch (e) { /* transient */ }
}


// ── Touch calibration: a guided flow — fingertip (tool point), then camera to base ─
// The server holds the touches and solves after every one; this side only decides
// which step is showing and what the operator should do next.
let calibTimer = null, calibState = null;
const calibUI = { step: null, force: null, skipTool: false, camSaved: false, target: null, lastImg: '', instrKey: '', whyOpen: false };
const CALIB_STEPS = [['setup', 'Arm & camera'], ['tool', 'Fingertip'], ['detect', 'Markers'], ['corners', 'Corners'], ['done', 'Done']];
const CALIB_MIN_ROT_DEG = 20;

function calibSet(id, text, isError = false) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.style.color = isError ? '#e06c75' : '#888';
}

async function calibPost(path, body) {
    const r = await fetch(path, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `${path} failed`);
    return data;
}

function calibStart() {
    if (!calibTimer) calibTimer = setInterval(calibRefresh, 700);
    calibRefresh();
}

function calibCanLeave(step, st) {
    // The reason the operator cannot go on yet, or '' when they can.
    if (step === 'setup') return (st.arm_connected && st.camera_live) ? '' : 'connect the arm and the camera first';
    if (step === 'tool') {
        if (calibUI.skipTool) return '';
        if (!st.saved.tool_point) return 'save the fingertip (or Skip) first';
        if (!(st.live && st.live.tip_calibrated)) return 'saved — disconnect and reconnect the jog to load the measured tip';
        return '';
    }
    if (step === 'detect') return st.markers ? '' : 'detect the markers first';
    if (step === 'corners') return (calibUI.camSaved || st.saved.camera) ? '' : 'save the camera fit first';
    return '';
}

function calibRenderNav(step, st) {
    const nav = document.getElementById('calib-nav');
    const idx = CALIB_STEPS.findIndex(s => s[0] === step);
    if (!nav.children.length) {
        nav.append(calibButton('◂ Back', () => { calibUI.force = CALIB_STEPS[Math.max(0, CALIB_STEPS.findIndex(s => s[0] === calibUI.step) - 1)][0]; calibRefresh(); }));
        nav.append(calibButton('Next ▸', () => { calibUI.force = CALIB_STEPS[Math.min(CALIB_STEPS.length - 1, CALIB_STEPS.findIndex(s => s[0] === calibUI.step) + 1)][0]; calibRefresh(); }));
        const hint = document.createElement('span'); hint.id = 'calib-nav-hint'; hint.style.cssText = 'color:#888; font-size:12px;';
        nav.append(hint);
    }
    const reason = calibCanLeave(step, st);
    nav.children[0].disabled = idx === 0;
    nav.children[1].disabled = idx === CALIB_STEPS.length - 1 || !!reason;
    document.getElementById('calib-nav-hint').textContent = reason;
}

function calibDeriveStep(st) {
    if (!st.arm_connected || !st.camera_live) return 'setup';
    if (calibUI.force) return calibUI.force;
    const toolDone = calibUI.skipTool || (st.saved.tool_point && st.live && st.live.tip_calibrated);
    if (!toolDone) return 'tool';
    if (!st.markers) return 'detect';
    if (!calibUI.camSaved && !st.saved.camera) return 'corners';
    return 'done';
}

function calibInstr(title, now, why) {
    // Rewriting the HTML every poll would snap the 'why' shut; only touch what changed.
    const el = document.getElementById('calib-instruction');
    const key = `${title}|${why}`;
    if (calibUI.instrKey !== key) {
        calibUI.instrKey = key;
        el.innerHTML = `<b>${title}</b> — <span id="calib-now"></span>` +
            (why ? `<details id="calib-why" style="margin-top:4px; color:#999;"${calibUI.whyOpen ? ' open' : ''}><summary style="cursor:pointer;">why</summary>${why}</details>` : '');
        const d = document.getElementById('calib-why');
        if (d) d.addEventListener('toggle', () => { calibUI.whyOpen = d.open; });
    }
    const nowEl = document.getElementById('calib-now');
    if (nowEl && nowEl.textContent !== now) nowEl.textContent = now;
}

function calibButton(label, onclick, opts = {}) {
    const b = document.createElement('button');
    b.className = 'btn-small'; b.textContent = label; b.onclick = onclick;
    if (opts.title) b.title = opts.title;
    if (opts.disabled) b.disabled = true;
    return b;
}

function calibRenderSteps(step) {
    const idx = CALIB_STEPS.findIndex(s => s[0] === step);
    document.getElementById('calib-steps').innerHTML = CALIB_STEPS.map(([key, label], i) => {
        const state = i < idx ? 'done' : (i === idx ? 'now' : 'todo');
        const col = state === 'now' ? '#4dd0ff' : (state === 'done' ? '#7c7' : '#666');
        const bg = state === 'now' ? 'rgba(77,208,255,0.12)' : 'transparent';
        return `<span style="padding:2px 10px; border:1px solid ${col}; border-radius:12px; color:${col}; background:${bg};">${i + 1} ${label}</span>`;
    }).join('');
}

async function calibRefresh() {
    let st;
    try { st = await (await fetch('/api/calib/state')).json(); } catch (e) { return; }
    calibState = st;
    const step = calibDeriveStep(st);
    const changed = step !== calibUI.step;
    calibUI.step = step;
    calibRenderSteps(step);
    calibRenderNav(step, st);
    const mm = (v) => (v * 1000).toFixed(1);
    const live = document.getElementById('calib-live');
    const controls = document.getElementById('calib-controls');
    const img = document.getElementById('calib-markers');
    const list = document.getElementById('calib-list');
    if (changed) { controls.innerHTML = ''; calibSet('calib-msg', ''); }
    live.textContent = st.live ? `fingertip now (${st.live.tip_mm.map(v => v.toFixed(1)).join(', ')}) mm` +
        (st.live.tip_calibrated ? ' · measured tip in use' : ' · URDF tip in use') : '';
    img.style.display = (step === 'corners' || step === 'done') && st.markers ? '' : 'none';
    list.style.display = (step === 'tool' || step === 'corners' || step === 'done') ? '' : 'none';

    if (step === 'setup') {
        const missing = [];
        if (!st.arm_connected) missing.push('connect the arm in the Jog panel above');
        if (!st.camera_live) missing.push('start the live camera session at the top of this tab');
        calibInstr('Step 1 · Arm and camera', `${missing.join(', and ')}.`, '');
        if (changed) {
            controls.append('Markers to print: ');
            controls.append(calibSelect('calib-dict', [['DICT_4X4_50', '4x4_50'], ['DICT_5X5_50', '5x5_50'], ['DICT_6X6_50', '6x6_50'], ['DICT_APRILTAG_36h11', 'AprilTag 36h11']]));
            controls.append(calibInput('calib-side', 'side mm', '25'));
            controls.append(calibButton('Print markers', calibSheet, {title: 'PDF at 100 % scale; check the bar with a ruler'}));
        }
        return;
    }

    if (step === 'tool') {
        const n = st.tool.touches.length, res = st.tool.result;
        const rots = (st.live && st.live.rotation_from_touches_deg) || [];
        const minRot = rots.length ? Math.min(...rots) : null;
        const tooClose = minRot !== null && minRot < CALIB_MIN_ROT_DEG;
        const why = 'Measures where the physical jaw tip is relative to the wrist link, the one part of the arm the URDF has wrong. Each touch records the wrist pose from the encoders; one tip offset must explain every touch of the same point from different wrist orientations. The point itself is never needed, it only has to be the same each time. No camera. Spread the orientations wide (45° or more, roll as well as pitch): the solve amplifies placement error by about 1/sin of the spread.';
        let now;
        if (n === 0) now = 'put the jaw tip on one fixed point (a marker corner will do), then press Touch.';
        else {
            const rot = minRot === null ? '' : `${minRot.toFixed(0)}° so far${tooClose ? `, need ${CALIB_MIN_ROT_DEG}°` : ', enough'}`;
            now = `touch ${n + 1}: (a) in the 3D view press Rotate and drag a ring to turn the wrist — ${rot}; (b) press Move and put the tip back on the same point; (c) press Touch.` +
                (res && !res.error ? ` Fit so far: rms ${mm(res.rms_m)} mm over ${n} touches; Save fingertip when happy.` : (res && res.error ? ` ${res.error}` : ' Three touches minimum.'));
        }
        calibInstr('Step 2 · Fingertip offset', now, why);
        if (minRot !== null) { live.textContent += ` · wrist turned ${minRot.toFixed(0)}° from the closest touch`; live.style.color = tooClose ? '#e5b93c' : '#7c7'; } else live.style.color = '#9aa';
        if (changed) {
            controls.append(calibButton('Touch', () => calibTool('touch')));
            controls.append(calibButton('Undo', () => calibTool('undo')));
            controls.append(calibButton('Clear', () => calibTool('clear')));
            controls.append(calibButton('Save fingertip', () => calibTool('save')));
            controls.append(calibButton('Skip (keep URDF tip)', () => { calibUI.skipTool = true; calibRefresh(); }));
        }
        const saveBtn = [...controls.children].find(b => b.textContent === 'Save fingertip');
        if (saveBtn) saveBtn.disabled = !(res && !res.error);
        const lines = st.tool.touches.map((t, i) => `#${i + 1} ${t.at}  tip (${t.tip_m.map(mm).join(', ')}) mm` +
            (res && !res.error ? `  residual ${mm(res.residuals_m[i])} mm` : ''));
        if (res && !res.error) lines.push(`fingertip offset from the wrist link (${res.offset_m.map(mm).join(', ')}) mm · rms ${mm(res.rms_m)} · max ${mm(res.max_m)} mm`);
        if (st.saved.tool_point) lines.push(`saved: (${st.saved.tool_point.offset_m.map(mm).join(', ')}) mm, rms ${mm(st.saved.tool_point.rms_m)} mm` +
            (st.live && st.live.tip_calibrated ? '' : ' — disconnect and reconnect the jog to use it'));
        list.textContent = lines.join('\n') || 'no touches yet';
        return;
    }

    if (step === 'detect') {
        calibInstr('Step 3 · Find the markers', 'move the arm out of the camera\'s view of the markers and press Detect.', 'One frame is taken and each marker corner\'s position in it is recorded. The gripper may cover the markers afterwards.');
        if (changed) {
            controls.append(calibSelect('calib-dict', [['DICT_4X4_50', '4x4_50'], ['DICT_5X5_50', '5x5_50'], ['DICT_6X6_50', '6x6_50'], ['DICT_APRILTAG_36h11', 'AprilTag 36h11']]));
            controls.append(calibInput('calib-side', 'side mm', '25'));
            controls.append(calibButton('Detect markers', calibDetect));
            controls.append(calibButton('Redo fingertip', () => { calibUI.skipTool = false; calibUI.force = 'tool'; calibRefresh(); }));
        }
        return;
    }

    if (step === 'corners') {
        const ids = st.markers.markers.map(m => m.id);
        const touched = new Set(st.camera.touches.map(t => t.marker_id));
        const next = ids.find(id => !touched.has(id));
        if (calibUI.target === null || (!ids.includes(calibUI.target)) || (touched.has(calibUI.target) && next !== undefined)) calibUI.target = next === undefined ? null : next;
        const auto = st.camera.auto.depth && !st.camera.auto.depth.error ? st.camera.auto.depth : null;
        const k = st.camera.touches.length;
        const whyCam = 'Pairs each marker corner\'s 3D position in the camera (from the Detect frame) with the fingertip position from the encoders when you touch it. A rigid fit of three or more pairs is the camera-to-base transform. With the measured fingertip, orientation is free; if you skipped that step, keep one orientation for every corner.';
        if (calibUI.target !== null) calibInstr('Step 4 · Camera to base', `touch the circled corner of marker ${calibUI.target} (red in the image) and press Touch corner. ${k} of ${ids.length} done.`, whyCam);
        else calibInstr('Step 4 · Camera to base', `all ${ids.length} markers touched. Check the residuals, then Save camera.`, whyCam);
        if (changed) {
            controls.append('marker ');
            const sel = calibSelect('calib-marker', ids.map(id => [String(id), String(id)]));
            sel.onchange = () => { calibUI.target = Number(sel.value); calibRefreshImage(true); };
            controls.append(sel);
            controls.append(calibButton('Touch corner', () => calibCamera('touch')));
            controls.append(calibButton('Undo', () => calibCamera('undo')));
            controls.append(calibButton('Clear', () => calibCamera('clear')));
            controls.append(calibButton('Re-detect', () => { calibUI.force = 'detect'; calibRefresh(); }));
            controls.append(calibSelect('calib-source', [['depth', 'fit from depth'], ['pnp', 'fit from marker size']]));
            controls.append(calibButton('Save camera', () => calibCamera('save')));
            controls.append(calibButton('Refine joint zeros', calibRefine, {title: 'fit joint-zero corrections, fingertip and camera pose to every touch at once'}));
            controls.append(calibButton('Save refined', calibRefineSave, {title: 'write zeros + fingertip + camera together'}));
        }
        const sel = document.getElementById('calib-marker');
        if (sel && calibUI.target !== null && sel.value !== String(calibUI.target)) sel.value = String(calibUI.target);
        const saveBtn = [...controls.children].find(b => b.textContent === 'Save camera');
        const pnpAuto = st.camera.auto.pnp && !st.camera.auto.pnp.error ? st.camera.auto.pnp : null;
        if (saveBtn) saveBtn.disabled = !(auto || pnpAuto);
        calibRefreshImage(false);
        const lines = st.camera.touches.map((t) => {
            const key = `${t.marker_id}.${t.corner}`;
            const rd = auto && auto.touch_ids.includes(key) ? ` · depth-fit residual ${mm(auto.residuals_m[auto.touch_ids.indexOf(key)])} mm` : '';
            const rp = pnpAuto && pnpAuto.touch_ids.includes(key) ? ` · size-fit residual ${mm(pnpAuto.residuals_m[pnpAuto.touch_ids.indexOf(key)])} mm` : '';
            return `marker ${t.marker_id} ${t.at}  base (${t.base_m.map(mm).join(', ')}) mm${rd}${rp}`;
        });
        if (auto) lines.push(`depth fit: rms ${mm(auto.rms_m)} · max ${mm(auto.max_m)} mm · similarity scale ${auto.scale.toFixed(4)} (${auto.n} touches)`);
        if (pnpAuto) lines.push(`size fit: rms ${mm(pnpAuto.rms_m)} · max ${mm(pnpAuto.max_m)} mm · similarity scale ${pnpAuto.scale.toFixed(4)} (${pnpAuto.n} touches)`);
        lines.push(...calibRefineLines(st));
        const refineSave = [...controls.children].find(b => b.textContent === 'Save refined');
        if (refineSave) refineSave.disabled = !st.refine;
        list.textContent = lines.join('\n') || 'no corner touches yet';
        return;
    }

    if (step === 'done') {
        const c = st.saved.camera, t = st.saved.tool_point;
        calibInstr('Step 5 · Done', 'saved. The jog uses the measured fingertip on its next connect; the camera transform is stored with this arm.', '');
        if (changed) {
            controls.append('test: go to marker ');
            const ids = st.markers ? st.markers.markers.map(m => [String(m.id), String(m.id)]) : [];
            controls.append(calibSelect('calib-goto-marker', ids));
            controls.append(calibInput('calib-goto-hover', 'hover mm', '10'));
            controls.append(calibButton('Go', calibGoto, {title: 'walk the fingertip above the circled corner, from the camera\'s coordinates through the saved transform'}));
            controls.append(calibButton('Redo corners', () => { calibUI.camSaved = false; calibUI.force = null; calibRefresh(); }));
            controls.append(calibButton('Redo fingertip', () => { calibUI.skipTool = false; calibUI.camSaved = false; calibUI.force = 'tool'; calibRefresh(); }));
        }
        const lines = [];
        if (t) lines.push(`fingertip (${t.offset_m.map(mm).join(', ')}) mm from the wrist link · rms ${mm(t.rms_m)} mm over ${t.n} touches${t.refined ? ' (refined)' : ''}`);
        if (c) lines.push(`camera→base from ${c.source}: rms ${mm(c.rms_m)} · max ${mm(c.max_m)} mm over ${c.n} touches${c.refined ? ' (refined)' : (c.scale ? ` · scale ${c.scale.toFixed(4)}` : '')}`);
        if (st.saved.joint_zero_deg) lines.push('joint zero corrections: ' + Object.entries(st.saved.joint_zero_deg).map(([k, v]) => `${k} ${v.toFixed(2)}°`).join(', ') + (st.live && JSON.stringify(st.live.joint_zero_deg || {}) !== JSON.stringify(st.saved.joint_zero_deg) ? ' — reconnect the jog to apply' : ''));
        lines.push(`file: ${st.saved.path}`);
        list.textContent = lines.join('\n');
        calibRefreshImage(false);
    }
}

function calibSelect(id, options) {
    const sel = document.createElement('select'); sel.id = id;
    sel.innerHTML = options.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    return sel;
}
function calibInput(id, placeholder, value) {
    const inp = document.createElement('input'); inp.id = id; inp.type = 'number'; inp.step = '1';
    inp.placeholder = placeholder; inp.value = value; inp.style.width = '64px';
    return inp;
}

function calibRefreshImage(force) {
    const img = document.getElementById('calib-markers');
    const key = `${calibUI.target}|${(calibState && calibState.camera.touches.length) || 0}|${calibState && calibState.markers && calibState.markers.at}`;
    if (!force && key === calibUI.lastImg) return;
    calibUI.lastImg = key;
    img.src = `/api/calib/markers.jpg?${calibUI.target !== null ? `target=${calibUI.target}&` : ''}t=${Date.now()}`;
}

async function calibTool(action) {
    try {
        const r = await calibPost(`/api/calib/tool/${action}`);
        if (action === 'save') calibSet('calib-msg', `saved — disconnect and reconnect the jog to use the measured tip`);
        else calibSet('calib-msg', `${r.n} touch${r.n === 1 ? '' : 'es'}`);
    } catch (e) { calibSet('calib-msg', e.message, true); }
    calibRefresh();
}

async function calibDetect() {
    const side = document.getElementById('calib-side').value;
    try {
        const r = await calibPost('/api/calib/markers', {
            dictionary: document.getElementById('calib-dict').value, side_mm: side === '' ? null : Number(side),
        });
        calibSet('calib-msg', `${r.n} marker${r.n === 1 ? '' : 's'} found: ${r.ids.join(', ') || 'none'}`);
        calibUI.force = null; calibUI.target = null; calibUI.camSaved = false;
    } catch (e) { calibSet('calib-msg', e.message, true); }
    calibRefresh();
}

async function calibCamera(action) {
    const body = {};
    if (action === 'touch') {
        if (calibUI.target === null) { calibSet('calib-msg', 'pick a marker first', true); return; }
        body.marker_id = calibUI.target; body.corner = 0;
    }
    if (action === 'save') body.source = document.getElementById('calib-source').value;
    try {
        const r = await calibPost(`/api/calib/camera/${action}`, body);
        if (action === 'save') { calibSet('calib-msg', `saved (rms ${r.rms_mm.toFixed(1)} mm)`); calibUI.camSaved = true; calibUI.force = null; }
        else calibSet('calib-msg', `${r.n} corner touch${r.n === 1 ? '' : 'es'}`);
        if (action === 'clear') calibUI.target = null;
    } catch (e) { calibSet('calib-msg', e.message, true); }
    calibRefresh();
}

function calibSheet() {
    const side = document.getElementById('calib-side').value || '25';
    const dict = document.getElementById('calib-dict').value;
    window.open(`/api/calib/markers/sheet.pdf?dictionary=${encodeURIComponent(dict)}&side_mm=${encodeURIComponent(side)}`, '_blank');
}

function calibRefineLines(st) {
    const mm = (v) => (v * 1000).toFixed(1);
    const r = st.refine;
    if (!r) return [];
    return [
        `refined (zeros + fingertip + camera): corners rms ${mm(r.camera_rms_m)} mm (was ${mm(r.before.camera_rms_m)}) · fingertip touches rms ${mm(r.tool_rms_m)} mm (was ${mm(r.before.tool_rms_m)})`,
        '   zero corrections: ' + Object.entries(r.joint_zero_deg).map(([k, v]) => `${k} ${v.toFixed(2)}°`).join(', '),
        `   fingertip (${r.offset_m.map(mm).join(', ')}) mm · corner residuals ${r.camera_residuals_m.map(mm).join(', ')} mm`,
    ];
}

async function calibRefine() {
    calibSet('calib-msg', 'refining…');
    try {
        const r = await calibPost('/api/calib/refine', {source: document.getElementById('calib-source').value});
        calibSet('calib-msg', `refined: corners rms ${(r.camera_rms_m * 1000).toFixed(1)} mm, fingertip rms ${(r.tool_rms_m * 1000).toFixed(1)} mm`);
    } catch (e) { calibSet('calib-msg', e.message, true); }
    calibRefresh();
}

async function calibRefineSave() {
    try {
        const r = await calibPost('/api/calib/refine/save');
        calibSet('calib-msg', `saved zeros + fingertip + camera (corners rms ${r.camera_rms_mm.toFixed(1)} mm) — disconnect and reconnect the jog to apply`);
        calibUI.camSaved = true; calibUI.force = null;
    } catch (e) { calibSet('calib-msg', e.message, true); }
    calibRefresh();
}

async function calibGoto() {
    const id = Number(document.getElementById('calib-goto-marker').value);
    const hover = Number(document.getElementById('calib-goto-hover').value || 10);
    try {
        const r = await calibPost('/api/calib/goto', {marker_id: id, corner: 0, hover_mm: hover});
        calibSet('calib-msg', `walking to ${hover} mm above marker ${id}'s corner at base (${r.corner_base_mm.map(v => v.toFixed(0)).join(', ')}) mm — watch where the fingertip lands`);
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { calibSet('calib-msg', e.message, true); }
}


let jogGripDragging = false;
function jogGripSlider(v) { jogGripDragging = true; document.getElementById('jog-grip-val').textContent = v; }
async function jogGrip(pos) {
    jogGripDragging = false;
    document.getElementById('jog-grip').value = pos;
    document.getElementById('jog-grip-val').textContent = pos;
    try {
        const r = await fetch('/api/jog/gripper', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({pos})});
        if (!r.ok) jogStatus((await r.json()).detail || 'gripper failed', true);
    } catch (e) { jogStatus(String(e), true); }
}

async function jogReady() {
    jogStatus('moving to the ready pose…');
    try {
        const r = await fetch('/api/jog/ready', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { jogStatus(d.detail || 'ready failed', true); return; }
        jogStatus('at the ready pose');
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { jogStatus(String(e), true); }
}

async function jogPark() {
    jogStatus('parking…');
    try {
        const r = await fetch('/api/jog/park', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { jogStatus(d.detail || 'park failed', true); return; }
        jogConnected = false;
        clearInterval(jogTimer); jogTimer = null;
        document.getElementById('jog-connect-btn').textContent = 'Connect';
        document.getElementById('jog-stop-btn').disabled = true;
        jogStatus('parked at rest, torque off, disconnected — Connect brings it back to the ready pose');
    } catch (e) { jogStatus(String(e), true); }
}

// ── Pre-grasp: teach one pose, move the object, go there ────────────────────
const pgUI = { box: null, drag: null, awaiting: false };

function pgSet(text, isError = false) {
    const el = document.getElementById('pg-status');
    el.textContent = text; el.style.color = isError ? '#e06c75' : '#888';
}

function pgRefresh() {
    const img = document.getElementById('pg-frame');
    img.src = `/api/pregrasp/frame.jpg?t=${Date.now()}`;
    pgUI.box = null; document.getElementById('pg-box').style.display = 'none';
    pgState();
}

function pgImgCoords(e) {
    const img = document.getElementById('pg-frame');
    const r = img.getBoundingClientRect();
    const sx = img.naturalWidth / r.width, sy = img.naturalHeight / r.height;
    return { x: Math.round((e.clientX - r.left) * sx), y: Math.round((e.clientY - r.top) * sy), r, sx, sy };
}

function pgDrawBox() {
    const img = document.getElementById('pg-frame'), div = document.getElementById('pg-box');
    if (!pgUI.box) { div.style.display = 'none'; return; }
    const r = img.getBoundingClientRect();
    const sx = r.width / img.naturalWidth, sy = r.height / img.naturalHeight;
    const [x0, y0, x1, y1] = pgUI.box;
    div.style.left = `${Math.min(x0, x1) * sx}px`; div.style.top = `${Math.min(y0, y1) * sy}px`;
    div.style.width = `${Math.abs(x1 - x0) * sx}px`; div.style.height = `${Math.abs(y1 - y0) * sy}px`;
    div.style.display = '';
}

(function pgWireBox() {
    const img = document.getElementById('pg-frame');
    if (!img) return;
    img.addEventListener('mousedown', (e) => { const c = pgImgCoords(e); pgUI.drag = [c.x, c.y]; pgUI.box = [c.x, c.y, c.x, c.y]; pgDrawBox(); e.preventDefault(); });
    img.addEventListener('mousemove', (e) => { if (!pgUI.drag) return; const c = pgImgCoords(e); pgUI.box = [pgUI.drag[0], pgUI.drag[1], c.x, c.y]; pgDrawBox(); });
    window.addEventListener('mouseup', () => { pgUI.drag = null; });
})();

async function pgPost(path, body) {
    const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `${path} failed`);
    return d;
}

let pgPollTimer = null;
function pgModeChanged() {
    const box = document.getElementById('pg-mode').value === 'box';
    document.getElementById('pg-concept').style.display = box ? 'none' : '';
    document.getElementById('pg-worker-btn').style.display = box ? 'none' : '';
    document.getElementById('pg-worker-status').style.display = box ? 'none' : '';
    document.getElementById('pg-teach-btn').textContent = box ? 'Teach (box)' : 'Teach (concept)';
}

async function pgWorkerToggle() {
    const btn = document.getElementById('pg-worker-btn');
    btn.disabled = true;
    try {
        const st = await (await fetch('/api/pregrasp/state')).json();
        if (st.worker.running) await pgPost('/api/pregrasp/worker/stop');
        else await pgPost('/api/pregrasp/worker/start');
    } catch (e) { pgSet(e.message, true); }
    finally { btn.disabled = false; pgState(); }
}

async function pgState() {
    try {
        const st = await (await fetch('/api/pregrasp/state')).json();
        const lines = [];
        const w = st.worker || {};
        document.getElementById('pg-worker-btn').textContent = w.running ? 'Stop worker' : 'Start worker';
        document.getElementById('pg-worker-status').textContent = w.running ? (w.ready ? 'worker ready' : 'worker starting…') : 'worker not running';
        if (st.teach_pending || st.find_pending) {
            lines.push(st.teach_pending ? 'teaching… (SAM3 designation + DINO features in the worker; the first run loads the models)' : 'finding… (SAM3 + DINO in the worker)');
            if (!pgPollTimer) pgPollTimer = setTimeout(() => { pgPollTimer = null; pgState(); if (!st.teach_pending && !st.find_pending) return; }, 1000);
        } else if (pgUI.awaiting) {
            pgUI.awaiting = false;
            document.getElementById('pg-frame').src = `/api/pregrasp/${st.test ? 'test' : 'teach'}.jpg?t=${Date.now()}`;
            if (st.test && !st.test.ok) pgSet(`not found: ${st.test.reason}`, true);
            else if (st.test) pgSet(`object found — ${st.test.n_inliers} of ${st.test.n_matches} matches agree, rms ${(st.test.rms_m * 1000).toFixed(1)} mm; the cross is where the fingertip will go`);
            else if (st.teach) pgSet(`taught by SAM3 + DINO: ${st.teach.n_points} points on "${st.teach.concept}", shape ${st.teach.shape_class}${st.teach.yaw_observable ? '' : ' (turn about the normal not observable from shape alone)'} — now jog the fingertip to the pre-grasp and press Mark`);
            else pgSet('teach failed — see the worker log', true);
        }
        if (w.log && (st.teach_pending || st.find_pending || !w.ready)) lines.push('worker: ' + w.log.split('\n').slice(-3).join(' | '));
        const flat = document.getElementById('pg-flat'); if (flat && document.activeElement !== flat) flat.checked = !!st.flat;
        if (st.teach) lines.push(`taught ${st.teach.at} (${st.teach.mode}): ` + (st.teach.mode === 'features' ? `${st.teach.n_points} DINO points on "${st.teach.concept}", radius ${st.teach.radius_mm.toFixed(0)} mm, shape ${st.teach.shape_class}` : st.teach.mode === 'texture' ? `${st.teach.n_with_depth} of ${st.teach.n_keypoints} keypoints have depth` : `${st.teach.n_points} depth points above the table, ${st.teach.height_mm.toFixed(0)} mm tall${st.teach.colour_cue ? ', colour is a usable cue' : ', colour not distinctive'}`) + (st.teach.tip_mm ? ` · pre-grasp at (${st.teach.tip_mm.map(v => v.toFixed(0)).join(', ')}) mm, gripper ${st.teach.gripper == null ? '?' : st.teach.gripper.toFixed(0)}` : ' · pre-grasp not marked yet'));
        if (st.test) {
            if (st.test.ok && st.test.mode === 'features') lines.push(`found ${st.test.at} (SAM3 + DINO): ${st.test.n_inliers} of ${st.test.n_matches} matches agree · rms ${(st.test.rms_m * 1000).toFixed(1)} mm · scale ${st.test.scale.toFixed(3)} · ` + (st.test.snapped ? `turned ${st.test.yaw_deg.toFixed(0)}° on the table (fit carried ${st.test.tilt_discarded_deg.toFixed(0)}° of axis tilt, discarded)` : `turned ${st.test.motion.rotation_deg.toFixed(0)}° in 6-DoF`) + `${st.test.yaw_observable ? '' : ' · flat object: the turn rests on the features'} · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm`);
            else if (st.test.ok && st.test.mode === 'shape') lines.push(`found ${st.test.at} (shape${st.test.fallback_from ? ', after ' + st.test.fallback_from : ''}${st.test.colour_used ? ', colour-gated' : ''}): ${st.test.n_points} points vs ${st.test.n_points_teach} taught, ${st.test.height_mm.toFixed(0)} mm tall, match score ${st.test.score.toFixed(2)}, footprint overlap ${(st.test.footprint_iou * 100).toFixed(0)}% · object moved ${st.test.motion.translation_mm.toFixed(0)} mm, turned ${st.test.yaw_deg.toFixed(0)}°${st.test.symmetric ? ' (footprint round: no turn measurable)' : ''} · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm`);
            else if (st.test.ok) lines.push(`found ${st.test.at}: ${st.test.n_matches} matches, ${st.test.n_inliers_2d} agree in 2D, ${st.test.n_inliers_3d} in 3D · rms ${(st.test.rms_m * 1000).toFixed(1)} mm · scale ${st.test.scale.toFixed(3)} · object moved ${st.test.motion.translation_mm.toFixed(0)} mm, turned ${st.test.motion.rotation_deg.toFixed(0)}° · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm`);
            else lines.push(`not found ${st.test.at}: ${st.test.reason}`);
        }
        if (!st.camera_live) lines.push('camera session not live');
        if (!st.arm_connected) lines.push('jog arm not connected');
        document.getElementById('pg-info').textContent = lines.join('\n') || 'nothing taught yet';
    } catch (e) { /* no server */ }
}

async function pgTeach() {
    const mode = document.getElementById('pg-mode').value;
    if (mode === 'features') {
        try {
            await pgPost('/api/pregrasp/teach/capture', {mode: 'features', concept: document.getElementById('pg-concept').value});
            pgUI.awaiting = true;
            pgSet('teaching…');
        } catch (e) { pgSet(e.message, true); }
        pgState();
        return;
    }
    if (!pgUI.box || Math.abs(pgUI.box[2] - pgUI.box[0]) < 8) { pgSet('drag a box around the object first', true); return; }
    try {
        const r = await pgPost('/api/pregrasp/teach/capture', {box: pgUI.box});
        pgSet(r.mode === 'texture' ? `taught by texture: ${r.n_with_depth} keypoints with depth — now jog the fingertip to the pre-grasp and press Mark` : `taught by shape: ${r.n_points} depth points, ${r.height_mm.toFixed(0)} mm tall${r.colour_cue ? ', colour will gate the search' : ''} — now jog the fingertip to the pre-grasp and press Mark`);
        document.getElementById('pg-frame').src = `/api/pregrasp/teach.jpg?t=${Date.now()}`;
    } catch (e) { pgSet(e.message, true); }
    pgState();
}

async function pgMark() {
    try {
        const r = await pgPost('/api/pregrasp/teach/mark');
        pgSet(`pre-grasp marked at (${r.tip_mm.map(v => v.toFixed(0)).join(', ')}) mm with gripper at ${r.gripper == null ? '?' : r.gripper.toFixed(0)} — move the object and the arm, then Find object`);
        document.getElementById('pg-frame').src = `/api/pregrasp/teach.jpg?t=${Date.now()}`;
    } catch (e) { pgSet(e.message, true); }
    pgState();
}

async function pgFind() {
    try {
        const r = await pgPost('/api/pregrasp/test/capture');
        if (r.pending) { pgUI.awaiting = true; pgSet('finding…'); pgState(); return; }
        pgSet(r.ok ? (r.mode === 'shape' ? `object found by shape${r.fallback_from ? ' after ' + r.fallback_from : ''} (score ${r.score.toFixed(2)}); the cross is where the fingertip will go` : `object found — ${r.n_inliers_3d} points agree, rms ${(r.rms_m * 1000).toFixed(1)} mm; the cross is where the fingertip will go`) : `not found: ${r.reason}`, !r.ok);
        document.getElementById('pg-frame').src = `/api/pregrasp/test.jpg?t=${Date.now()}`;
    } catch (e) { pgSet(e.message, true); }
    pgState();
}

async function pgGo() {
    try {
        const r = await pgPost('/api/pregrasp/go', {hover_mm: Number(document.getElementById('pg-hover').value || 0)});
        pgSet(`walking to (${r.target_mm.map(v => v.toFixed(0)).join(', ')}) mm`);
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { pgSet(e.message, true); }
}


// ── Approach tab: camera session + jog + calibration + pre-grasp, one place ─
let apKeysWired = false;

async function apInitTab() {
    apCameraRefresh();
    apCameraState();
    jogRefreshProfiles();
    jogReattach();
    calibStart();
    pgState();
    if (!apKeysWired) {
        apKeysWired = true;
        // T move, R rotate, W close a step, E open a step — anywhere on the tab outside a text field.
        document.addEventListener('keydown', (e) => {
            const tab = document.getElementById('tab-approach');
            if (!tab || !tab.classList.contains('active')) return;
            const tag = (e.target && e.target.tagName) || '';
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            const k = e.key.toLowerCase();
            if (k === 't') jogMode('translate');
            else if (k === 'r') jogMode('rotate');
            else if (k === 'w') jogGripStep(+10);
            else if (k === 'e') jogGripStep(-10);
        });
    }
}

function jogGripStep(delta) {
    const cur = Number(document.getElementById('jog-grip').value);
    jogGrip(Math.max(0, Math.min(100, cur + delta)));
}

async function apCameraRefresh() {
    const sel = document.getElementById('ap-camera');
    try {
        const cams = await (await fetch('/api/showservo/cameras')).json();
        sel.innerHTML = cams.length ? cams.map(c => `<option value="${c.serial}">${c.name} ${c.serial}</option>`).join('') : '<option value="">no RealSense found</option>';
    } catch (e) { sel.innerHTML = '<option value="">cameras unavailable</option>'; }
}

async function apCameraState() {
    try {
        const st = await (await fetch('/api/showservo/state')).json();
        const live = !!(st.session && st.session.live);
        document.getElementById('ap-camera-btn').textContent = live ? 'Stop' : 'Start';
        document.getElementById('ap-camera-status').textContent = live ? `live: ${st.session.name}` : 'no live camera';
        return live;
    } catch (e) { return false; }
}

async function apCameraToggle() {
    const live = await apCameraState();
    const btn = document.getElementById('ap-camera-btn');
    btn.disabled = true;
    try {
        if (live) {
            await fetch('/api/showservo/session/stop', {method: 'POST'});
        } else {
            const serial = document.getElementById('ap-camera').value;
            if (!serial) { document.getElementById('ap-camera-status').textContent = 'pick a camera'; return; }
            const r = await fetch('/api/showservo/session/start', {method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({serial, name: 'approach_' + new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '')})});
            if (!r.ok) document.getElementById('ap-camera-status').textContent = (await r.json()).detail || 'start failed';
        }
    } finally {
        btn.disabled = false;
        apCameraState();
    }
}

async function pgFlat() {
    try { await pgPost('/api/pregrasp/options', {flat: document.getElementById('pg-flat').checked}); }
    catch (e) { pgSet(e.message, true); }
}
