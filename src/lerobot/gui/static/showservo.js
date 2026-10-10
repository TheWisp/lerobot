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
        // The profile connected last time is the one wanted next time.
        let last = null;
        try { last = localStorage.getItem('jog-profile'); } catch (e) { /* storage may be unavailable */ }
        if (last && usable.some(p => p.name === last)) sel.value = last;
    } catch (e) { sel.innerHTML = '<option value="">profiles unavailable</option>'; }
    // The leader and the arm's side are the ones used last time, from the saved leader profiles.
    const lead = document.getElementById('jog-leader');
    try {
        const teleops = await (await fetch('/api/robot/teleop-profiles')).json();
        const leaders = teleops.filter(p => String(p.type || '').includes('so107_leader'));
        lead.innerHTML = leaders.length
            ? leaders.map(p => `<option value="${p.name}">${p.name}</option>`).join('')
            : '<option value="">no SO-107 leader profile</option>';
        let lastLeader = null;
        try { lastLeader = localStorage.getItem('jog-leader'); } catch (e) { /* storage may be unavailable */ }
        if (lastLeader && leaders.some(p => p.name === lastLeader)) lead.value = lastLeader;
    } catch (e) { lead.innerHTML = '<option value="">profiles unavailable</option>'; }
    try {
        const side = localStorage.getItem('jog-arm');
        if (side === 'left' || side === 'right') document.getElementById('jog-arm').value = side;
    } catch (e) { /* storage may be unavailable */ }
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
        try {
            localStorage.setItem('jog-profile', document.getElementById('jog-profile').value);
            localStorage.setItem('jog-arm', document.getElementById('jog-arm').value);
        } catch (e) { /* storage may be unavailable */ }
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
        jogUI.mode = st.mode; jogUI.recording = !!st.recording;
        const lb = document.getElementById('jog-leader-btn'); if (lb) lb.textContent = st.mode === 'leader' ? 'Leader stops' : 'Leader drives';
        jogStatus(`gap ${st.err_mm.toFixed(1)} mm / ${st.err_deg.toFixed(1)}°` +
                  (hottest !== null ? ` · hottest motor ${hottest} °C` : '') +
                  (st.mode === 'leader' ? ' · leader drives' : '') +
                  (st.halted ? ` · FROZEN: ${st.reason}` : ''), !!st.halted);
    } catch (e) { /* transient */ }
}

const jogUI = {mode: 'cartesian', recording: false};

async function jogLeaderToggle() {
    const stopping = jogUI.mode === 'leader';
    jogStatus(stopping ? 'handing the arm back to the jog…' : 'meeting the leader arm…');
    try {
        const leader = document.getElementById('jog-leader').value;
        if (!stopping && !leader) throw new Error('choose the leader arm\'s profile under details, Teach');
        try { if (!stopping) localStorage.setItem('jog-leader', leader); } catch (e) { /* storage may be unavailable */ }
        const body = stopping ? undefined : JSON.stringify({profile: leader, arm: document.getElementById('jog-arm').value});
        const r = await fetch(stopping ? '/api/jog/leader/stop' : '/api/jog/leader/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body});
        const d = await r.json().catch(() => ({}));
        if (!r.ok) { jogStatus(d.detail || 'leader failed', true); return; }
        jogStatus(stopping ? 'the jog has the arm again' : `the leader ${d.leader} drives the arm — move it, press Record demo, do the grasp, stop recording`);
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { jogStatus(String(e), true); }
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
    pgCam.holdUntil = 0;  // asked for the camera: a held result gives way
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
    // A click (no drag) in the SAM3 mode teaches whatever is under it; a drag is still the box.
    img.addEventListener('mouseup', (e) => {
        if (!pgUI.drag) return;
        const c = pgImgCoords(e);
        const moved = Math.hypot(c.x - pgUI.drag[0], c.y - pgUI.drag[1]);
        pgUI.drag = null;
        if (moved < 4 && document.getElementById('pg-mode').value === 'features') { pgUI.box = null; pgDrawBox(); pgTeachAt(c.x, c.y); }
    });
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
            if (!pgLive.on) pgShowResult(st.test ? 'test' : 'teach');
            if (st.test && !st.test.ok) pgSet(`not found: ${st.test.reason}`, true);
            else if (st.test) pgSet(`object found — ${st.test.n_inliers} of ${st.test.n_matches} matches agree, rms ${(st.test.rms_m * 1000).toFixed(1)} mm; the camera view shows the path the arm would follow`);
            else if (st.teach && st.teach.ref && st.teach.ref.ok) pgSet(`found "${st.teach.ref.object}": SAM3 cut it out where you clicked and its DINO features matched the demo's view of it (${st.teach.ref.inliers} points, turned ${st.teach.ref.turn_deg.toFixed(0)}°); Point2Pose tracks it from this frame`);
            else if (st.teach && st.teach.ref) pgSet(`what you clicked does not match the demo's view of "${st.teach.ref.object}": ${st.teach.ref.reason}`, true);
            else if (st.teach) pgSet(`taught "${st.teach.concept}": SAM3 cut it out where you clicked and DINO described it (${st.teach.n_points} points); Point2Pose tracks it from this frame${st.demo ? '' : ', so record a demo next'}`);
            else pgSet('teach failed — see the worker log', true);
        }
        if (w.log && (st.teach_pending || st.find_pending || !w.ready)) lines.push('worker: ' + w.log.split('\n').slice(-3).join(' | '));
        const flat = document.getElementById('pg-flat'); if (flat && document.activeElement !== flat) flat.checked = !!st.flat;
        const trust = document.getElementById('pg-trust');
        if (trust && document.activeElement !== trust && st.trust_share != null) { trust.value = st.trust_share; pgTrustLabel(); }
        const depth = document.getElementById('pg-depth'); if (depth && document.activeElement !== depth && st.depth_check != null) depth.checked = !!st.depth_check;
        const depthMm = document.getElementById('pg-depth-mm'); if (depthMm && document.activeElement !== depthMm && st.depth_tol_mm != null) depthMm.value = Math.round(st.depth_tol_mm);
        if (st.teach) lines.push(`taught ${st.teach.at} (${st.teach.mode}): ` + (st.teach.mode === 'features' ? `${st.teach.n_points} DINO points on "${st.teach.concept}", radius ${st.teach.radius_mm.toFixed(0)} mm, visible cloud ${st.teach.shape_class === 'disc' ? 'thin from this view' : st.teach.shape_class}${st.teach.face_planarity != null ? `, ${(st.teach.face_planarity * 100).toFixed(0)}% of the cloud on its face${st.teach.face_usable ? '' : ' (not usable as an axis)'}` : ''}` : st.teach.mode === 'texture' ? `${st.teach.n_with_depth} of ${st.teach.n_keypoints} keypoints have depth` : `${st.teach.n_points} depth points above the table, ${st.teach.height_mm.toFixed(0)} mm tall${st.teach.colour_cue ? ', colour is a usable cue' : ', colour not distinctive'}`) + (st.teach.tip_mm ? ` · demo starts at (${st.teach.tip_mm.map(v => v.toFixed(0)).join(', ')}) mm, gripper ${st.teach.gripper == null ? '?' : st.teach.gripper.toFixed(0)}` : ' · no demo loaded'));
        if (st.test) {
            const armTxt = st.test.arm_turn_deg != null ? ` · the gripper will turn ${st.test.arm_turn_deg.toFixed(0)}° about vertical and lean ${st.test.arm_lean_deg.toFixed(0)}°` : '';
            if (st.test.ok && st.test.mode === 'features') {
                const ax = st.test.axis_source;
                const turn = ax === 'face' ? `turned ${st.test.yaw_deg.toFixed(0)}° about its face, and the face tipped ${st.test.face_tilt_deg.toFixed(0)}°; the raw fit's axis was ${st.test.fit_axis_tilt_deg.toFixed(0)}° off`
                    : ax === 'surface' ? `turned ${(st.test.yaw_deg || 0).toFixed(0)}° about the surface it rests on${st.test.surface_tilt_deg > 1 ? `, which tilted ${st.test.surface_tilt_deg.toFixed(0)}°` : ''}${st.test.face_tilt_deg != null ? ` (face tilt ${st.test.face_tilt_deg.toFixed(0)}° reported only)` : ''}; raw fit's axis ${(st.test.fit_axis_tilt_deg || 0).toFixed(0)}° off${st.test.turn_source === 'footprint' ? ` · turn from the footprint${st.test.footprint_symmetric ? ' (symmetric: no turn measurable)' : ''}, features said ${(st.test.yaw_deg + (st.test.turn_disagreement_deg || 0)).toFixed(0)}°` : ''}`
                    : ax === 'table' ? `turned ${(st.test.yaw_deg || 0).toFixed(0)}° about the table normal${st.test.face_tilt_deg != null ? ` (face tilt ${st.test.face_tilt_deg.toFixed(0)}° ignored: objects stay on the table)` : ''}${st.test.fit_axis_tilt_deg != null ? `; raw fit's axis ${st.test.fit_axis_tilt_deg.toFixed(0)}° off` : ''}`
                    : `turned ${st.test.motion.rotation_deg.toFixed(0)}° in 6-DoF (raw fit)`;
                const by = {p2p: 'Point2Pose', p2p_dense: 'Point2Pose (dense)', dino: 'DINO', klt: 'KLT', refind: 'SAM3 + DINO', depth: 'depth'}[st.test.algo] || 'SAM3 + DINO';
                lines.push(`${st.test.algo ? 'tracked' : 'found'} ${st.test.at} by ${by}: ${st.test.n_inliers} of ${st.test.n_matches} matches agree · rms ${(st.test.rms_m * 1000).toFixed(1)} mm · scale ${st.test.scale.toFixed(3)} · ${turn}${st.test.transported_tip_mm ? ` · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm` : ''}${armTxt}`);
            }
            else if (st.test.ok && st.test.mode === 'shape') lines.push(`found ${st.test.at} (shape${st.test.fallback_from ? ', after ' + st.test.fallback_from : ''}${st.test.colour_used ? ', colour-gated' : ''}): ${st.test.n_points} points vs ${st.test.n_points_teach} taught, ${st.test.height_mm.toFixed(0)} mm tall, match score ${st.test.score.toFixed(2)}, footprint overlap ${(st.test.footprint_iou * 100).toFixed(0)}% · object moved ${st.test.motion.translation_mm.toFixed(0)} mm, turned ${st.test.yaw_deg.toFixed(0)}°${st.test.symmetric ? ' (footprint round: no turn measurable)' : ''}${st.test.transported_tip_mm ? ` · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm` : ''}${armTxt}`);
            else if (st.test.ok) lines.push(`found ${st.test.at}: ${st.test.n_matches} matches, ${st.test.n_inliers_2d} agree in 2D, ${st.test.n_inliers_3d} in 3D · rms ${(st.test.rms_m * 1000).toFixed(1)} mm · scale ${st.test.scale.toFixed(3)} · object moved ${st.test.motion.translation_mm.toFixed(0)} mm, turned ${st.test.motion.rotation_deg.toFixed(0)}°${st.test.transported_tip_mm ? ` · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm` : ''}${armTxt}`);
            else lines.push(`not found ${st.test.at}: ${st.test.reason}`);
            const cc = st.test.camera_check;
            if (cc) lines.push(cc.moved ? `CAMERA OR TRAY MOVED since the calibration: markers shifted ${cc.max_px.toFixed(1)} px (limit ${cc.tol_px}) — Go is refused; redo the camera calibration`
                : cc.checked ? `camera check: ${cc.n_corners} calibration marker corners within ${cc.max_px.toFixed(1)} px of where they were`
                : `camera not checked: only ${cc.n_corners} calibration marker corners visible (uncover the stickers to check)`);
        }
        const trackLine = pgTrackLine(st);
        if (trackLine) lines.push(trackLine);
        pgTrialsRefresh();
        apRenderDemoAct(st);
        if (!st.camera_live) lines.push('camera session not live');
        if (!st.arm_connected) lines.push('jog arm not connected');
        document.getElementById('pg-info').textContent = lines.join('\n') || 'nothing taught yet';
    } catch (e) { /* no server */ }
}

// The page version this tab loaded, as index.html names it in this script's URL; the server says which it serves.
const PG_PAGE_VERSION = ((document.querySelector('script[src*="showservo.js"]') || {}).src || '').match(/[?&]v=(\d+)/)?.[1] || null;

// The designated object the current demo's pre-grasps and grasp are for, if any: a click then finds it from the demo's
// view of it. The object a place goes onto is found the same way, without a track, when the guided row asks for it.
let apMarksObject = '', apPlaceObject = '', apClickFor = 'pick';

async function pgTeachAt(x, y) {
    if (apClickFor === 'place' && apPlaceObject) {
        pgSet(`finding ${apPlaceObject} at (${x}, ${y})…`);
        try {
            const r = await pgPost('/api/pregrasp/locate', {click: [x, y], object: apPlaceObject});
            pgSet(r.ok ? `found ${apPlaceObject}: ${r.inliers} of the demo view's ${r.card_points} points` : `${apPlaceObject} was not found: ${r.reason}`, !r.ok);
        } catch (e) { pgSet(e.message, true); }
        if (typeof apGuideTick === 'function') apGuideTick();
        return;
    }
    try {
        await pgPost('/api/pregrasp/teach/capture', {mode: 'features', concept: document.getElementById('pg-concept').value, click: [x, y], ref_object: apMarksObject});
        pgUI.awaiting = true;
        pgSet(`teaching what is at (${x}, ${y})…`);
    } catch (e) { pgSet(e.message, true); }
    pgState();
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
        pgShowResult('teach');
    } catch (e) { pgSet(e.message, true); }
    pgState();
}

// A teach's or a find's picture in the camera view, held there for a while before the camera's frames return.
function pgShowResult(kind) {
    pgCam.holdUntil = Date.now() + PG_RESULT_HOLD_MS;
    document.getElementById('pg-frame').src = `/api/pregrasp/${kind}.jpg?t=${Date.now()}`;
}




let pgTrialsShown = -1;
async function pgTrialsRefresh(force = false) {
    try {
        const r = await fetch('/api/pregrasp/trials');
        if (!r.ok) return;
        const rows = (await r.json()).rows || [];
        if (!force && rows.length === pgTrialsShown) return;
        pgTrialsShown = rows.length;
        const box = document.getElementById('pg-trials');
        if (!rows.length) { box.innerHTML = ''; return; }
        // What each act did, from its own record: nothing here asks the operator for a verdict.
        const f = (v, d = 1) => (v == null ? '–' : Number(v).toFixed(d));
        // The newest act first, dated: a row from yesterday must not read as one from today.
        const last = rows.map((x, index) => ({...x, index})).slice(-12).reverse();
        const td = 'padding:3px 10px 3px 0; vertical-align:top;';
        const nowrap = td + ' white-space:nowrap;';
        box.innerHTML = `<table style="border-collapse:collapse; width:100%;"><thead><tr style="color:#aaa; text-align:left;">
            <th style="${nowrap}">#</th><th style="${nowrap}">when</th><th style="${nowrap}">demo</th><th style="${td}">result</th>` +
            `<th style="${td}">hold used for the place</th><th></th></tr></thead><tbody>` +
            last.map(x => {
                const p = x.place || {};
                const hold = (p.hold_used ? `${p.hold_used}, corrected ${f(p.shift_mm)} mm ${f(p.shift_deg)}°` : '') +
                    (x.inject || x.correct_hold === false ? ` (${apInjectSummary({at: 'aim', ...Object.fromEntries(AP_INJECT_AXES.map(([k]) => [k, 0])), ...(x.inject || {}), correct_hold: x.correct_hold !== false})})` : '');
                return `<tr style="border-top:1px solid #333;"><td style="${nowrap}">${x.index}</td><td style="${nowrap}">${x.at.slice(5)}</td>` +
                    `<td style="${nowrap}">${x.demo || ''}</td>` +
                    `<td style="${td} color:${x.result === 'done' ? '#7c7' : '#e55'}">${x.result}${x.reason ? ': ' + x.reason : ''}</td><td style="${td}">${hold}</td>` +
                    `<td style="${nowrap}">${x.run ? `<button class="btn-small" onclick="pgReplay(${x.index})">replay</button>` : ''}</td></tr>`;
            }).join('') + '</tbody></table>';
    } catch (e) { /* no server */ }
}

// The replay: an act's recorded frames, each drawn by the server from the act's own record.
let pgReplayAt = null; // {trial, n, frames: [{i, t, step}], held_recorded}
async function pgReplay(trial) {
    const r = await fetch(`/api/pregrasp/replay?trial=${trial}`);
    if (!r.ok) { pgSet(`replay: ${(await r.json()).detail}`, true); return; }
    pgReplayAt = await r.json();
    document.getElementById('pg-replay').hidden = false;
    const slider = document.getElementById('pg-replay-i');
    slider.max = Math.max(0, pgReplayAt.n - 1);
    document.getElementById('pg-replay-title').textContent = `trial ${trial}: ${pgReplayAt.n} frames` +
        (pgReplayAt.held_recorded ? '' : ' · this act did not record where it held the place object');
    pgReplayShow(0);
}
function pgReplayShow(i) {
    if (!pgReplayAt || !pgReplayAt.n) return;
    i = Math.max(0, Math.min(pgReplayAt.n - 1, i));
    document.getElementById('pg-replay-i').value = i;
    const f = pgReplayAt.frames[i];
    document.getElementById('pg-replay-at').textContent = `frame ${i} · +${f.t.toFixed(1)} s · ${f.step}`;
    document.getElementById('pg-replay-img').src = `/api/pregrasp/replay/frame.jpg?trial=${pgReplayAt.trial}&i=${i}`;
}
function pgReplayStep(d) { pgReplayShow(Number(document.getElementById('pg-replay-i').value) + d); }
function pgReplayClose() { document.getElementById('pg-replay').hidden = true; pgReplayAt = null; }

async function pgVerdict(index, verdict) {
    try { await pgPost('/api/pregrasp/trials/verdict', {index, verdict}); } catch (e) { pgSet(e.message, true); }
    pgTrialsRefresh(true);
}


async function pgFind() {
    try {
        const r = await pgPost('/api/pregrasp/test/capture');
        if (r.pending) { pgUI.awaiting = true; pgSet('finding…'); pgState(); return; }
        pgSet(r.ok ? (r.mode === 'shape' ? `object found by shape${r.fallback_from ? ' after ' + r.fallback_from : ''} (score ${r.score.toFixed(2)}); the cross is where the fingertip will go` : `object found — ${r.n_inliers_3d} points agree, rms ${(r.rms_m * 1000).toFixed(1)} mm; the cross is where the fingertip will go`) : `not found: ${r.reason}`, !r.ok);
        pgShowResult('test');
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


// ── the demo (Teach) and the act: record the arm and the object, save a dataset, replay it on the object ─
const demoUI = {recording: false, lastList: ''};

async function demoRecordToggle() {
    try {
        if (demoUI.recording) {
            const name = document.getElementById('demo-name').value.trim();
            const d = await pgPost('/api/pregrasp/demo/record/stop', {name: name || null});
            demoUI.recording = false;
            demoStatus(`recorded ${d.n} samples over ${d.seconds.toFixed(1)} s, object seen ${(d.seen_fraction * 100).toFixed(0)}% of the time — Save demo keeps it`);
        } else {
            const d = await pgPost('/api/pregrasp/demo/record/start', {});
            demoUI.recording = true;
            demoStatus(d.tracking ? 'recording the arm, the object and the camera…' : 'recording the arm (start tracking in Setup to record the object too)…');
        }
    } catch (e) { demoStatus(e.message, true); }
    document.getElementById('demo-record-btn').textContent = demoUI.recording ? 'Stop recording' : 'Record';
    pgState();
}

function demoStatus(text, isError = false) {
    const el = document.getElementById('demo-status');
    if (!el) return;
    el.textContent = text; el.style.color = isError ? '#e55' : '#888';
}

async function demoSave() {
    try {
        const name = document.getElementById('demo-name').value.trim();
        demoStatus('writing the dataset…');
        const d = await pgPost('/api/pregrasp/demo/save', {name: name || null});
        demoStatus(`saved ${d.repo_id} (${d.n} samples${d.frames ? ', ' + d.frames + ' camera frames' : ''}) — play it in the Data tab`);
        demosRefresh(true);
        deLoad(true);
    } catch (e) { demoStatus(e.message, true); }
    pgState();
}

async function demosRefresh(force = false) {
    try {
        const r = await fetch('/api/pregrasp/demos');
        if (!r.ok) return;
        const demos = (await r.json()).demos || [];
        const key = JSON.stringify(demos.map(d => d.name));
        if (!force && key === demoUI.lastList) return;
        demoUI.lastList = key;
        const box = document.getElementById('demo-list');
        if (!box) return;
        if (!demos.length) { box.innerHTML = '<span style="color:#666;">no saved demos yet</span>'; return; }
        box.innerHTML = '<table style="border-collapse:collapse;"><thead><tr style="color:#aaa; text-align:left;"><th>demo</th><th>object</th><th>length</th><th>recorded</th><th></th></tr></thead><tbody>' +
            demos.slice().reverse().map(d => `<tr style="border-top:1px solid #333;"><td style="padding:3px 10px 3px 0;">${d.name}</td><td style="padding-right:10px;">${d.concept}</td><td style="padding-right:10px;">${d.seconds.toFixed(1)} s · ${d.n} samples</td><td style="padding-right:10px;">${d.created}</td><td><button class="btn-small" onclick="demoLoad('${d.name}')">Load</button> <button class="btn-small" onclick="demoOpenData('${d.repo_id}')">Play in Data</button></td></tr>`).join('') + '</tbody></table>';
    } catch (e) { /* no server */ }
}

async function demoLoad(name) {
    try {
        const d = await pgPost('/api/pregrasp/demo/load', {name});
        demoStatus(`loaded ${d.name} (${d.n} samples, object "${d.concept}"); the object is being re-taught from the demo's frame`);
        pgUI.awaiting = true;
    } catch (e) { demoStatus(e.message, true); }
    pgState();
}

function demoOpenData(repoId) {
    try {
        if (typeof switchTab === 'function') switchTab('data');
        if (typeof openDataset === 'function') openDataset(repoId);
    } catch (e) { demoStatus(String(e), true); }
}

async function actGo() {
    try {
        await pgPost('/api/pregrasp/act', {speed: Number(document.getElementById('act-speed').value || 1), ...apInjectBody()});
        const tile = document.getElementById('jog-tile');
        if (tile.contentWindow) tile.contentWindow.postMessage({type: 'jog-reanchor'}, '*');
    } catch (e) { actStatus(e.message, true); }
    pgState();
}

async function actStop() {
    try { await pgPost('/api/pregrasp/act/stop', {}); actStatus('stopped; the arm holds where it is'); } catch (e) { actStatus(e.message, true); }
    pgState();
}

function actStatus(text, isError = false) {
    const el = document.getElementById('act-status');
    if (!el) return;
    el.textContent = text; el.style.color = isError ? '#e55' : '#888';
}

function apRenderDemoAct(st) {
    const demoEl = document.getElementById('act-demo');
    if (demoEl) demoEl.textContent = st.demo ? `demo "${st.demo.name}" on "${st.demo.concept}", ${st.demo.seconds.toFixed(1)} s${st.demo.root ? '' : ' (not saved)'}` : 'no demo: record one in Teach, or load a saved one';
    const rec = document.getElementById('demo-record-btn');
    if (rec) { demoUI.recording = !!st.recording; rec.textContent = st.recording ? `Stop recording (${st.recording.samples} samples)` : 'Record'; }
    if (st.act && (st.act.on || st.act.step)) {
        const txt = `${st.act.step}${st.act.on ? ` ${(st.act.progress * 100).toFixed(0)}%` : ''}${st.act.ok === false ? ' — ' + st.act.reason : st.act.ok ? ' — done' : ''}`;
        actStatus(txt, st.act.ok === false);
    }
}


// ── live tracking: the worker follows the object; the view refreshes as fast as it answers ─
const pgLive = {on: false, timer: null};

function pgTrackBody() {
    return {
        algo: document.getElementById('pg-algo').value,
        follow: document.getElementById('pg-follow').checked,
        hover_mm: Number(document.getElementById('pg-hover').value || 0),
    };
}

async function pgTrackToggle() {
    try {
        if (pgLive.on) {
            await pgPost('/api/pregrasp/track/stop', {});
            pgLiveStop();
        } else {
            await pgPost('/api/pregrasp/track/start', pgTrackBody());
            pgLiveStart();
        }
    } catch (e) { pgSet(e.message, true); }
    pgState();
}

async function pgTrackOptions() {
    if (!pgLive.on) return;
    try { await pgPost('/api/pregrasp/track/options', pgTrackBody()); } catch (e) { pgSet(e.message, true); }
}

function pgLiveStart() {
    pgLive.on = true;
    pgCamStop();  // the tracker's frames take the view over
    document.getElementById('pg-track-btn').textContent = 'Stop tracking';
    pgLiveNext();
}

function pgLiveStop() {
    pgLive.on = false;
    if (pgLive.timer) { clearTimeout(pgLive.timer); pgLive.timer = null; }
    const img = document.getElementById('pg-frame');
    img.onload = null; img.onerror = null;
    document.getElementById('pg-track-btn').textContent = 'Start tracking';
}

function pgLiveNext() {
    if (!pgLive.on) return;
    const img = document.getElementById('pg-frame');
    const next = () => { if (pgLive.on) pgLive.timer = setTimeout(pgLiveNext, 40); };
    img.onload = next; img.onerror = next;
    img.src = '/api/pregrasp/track/live.jpg?t=' + Date.now();
}

// ── the camera view while nothing is tracked: the camera's own frames, for as long as the camera is live ─
// Only tracking used to refresh the view, so with the camera live and the tracker off it kept the last teach's or
// find's picture, which reads as a frozen camera (2026-10-09). A teach's or a find's picture stays up for
// PG_RESULT_HOLD_MS before the camera's frames take over again; the guided row's tick starts and stops this.
const PG_RESULT_HOLD_MS = 4000;
const pgCam = {on: false, active: false, timer: null, holdUntil: 0};

function pgCamSync(on) {
    pgCam.on = on;
    if (!on) { pgCamStop(); return; }
    if (!pgCam.active && !pgLive.on) { pgCam.active = true; pgCamNext(); }
}

function pgCamStop() {
    if (pgCam.timer) { clearTimeout(pgCam.timer); pgCam.timer = null; }
    pgCam.active = false;  // a frame still loading finds the loop ended and does not continue it
}

function pgCamNext() {
    pgCam.timer = null;
    if (!pgCam.on || !pgCam.active || pgLive.on) { pgCam.active = false; return; }
    const wait = pgCam.holdUntil - Date.now();
    if (wait > 0) { pgCam.timer = setTimeout(pgCamNext, wait); return; }
    const img = document.getElementById('pg-frame');
    const then = ms => () => {
        if (pgCam.active && pgCam.on && !pgLive.on) pgCam.timer = setTimeout(pgCamNext, ms);
        else pgCam.active = false;
    };
    img.onload = then(100); img.onerror = then(1000);
    img.src = `/api/pregrasp/frame.jpg?t=${Date.now()}`;
}

function pgTrackLine(st) {
    const t = st.track;
    if (!t) return null;
    if (t.on !== pgLive.on) { if (t.on) pgLiveStart(); else pgLiveStop(); }
    const sel = document.getElementById('pg-algo'); if (sel && document.activeElement !== sel) sel.value = t.algo;
    const fol = document.getElementById('pg-follow'); if (fol && document.activeElement !== fol) fol.checked = !!t.follow;
    const l = t.last || {};
    const status = document.getElementById('pg-track-status');
    if (!t.on) { status.textContent = l.reason ? `stopped: ${l.reason}` : ''; return null; }
    const bits = [`tracking [${t.algo}] ${l.state || ''}`, `${(t.fps || 0).toFixed(0)} fps`, `${(l.ms || 0).toFixed(0)} ms in the worker`];
    if (l.n_inliers != null) bits.push(`${l.n_inliers} of ${l.n_matches} agree`);
    if (l.arm_turn_deg != null) bits.push(`gripper will turn ${l.arm_turn_deg.toFixed(0)}° and lean ${l.arm_lean_deg.toFixed(0)}°`);
    else if (l.yaw_deg != null) bits.push(`turned ${l.yaw_deg.toFixed(0)}° (${l.axis_source})`);
    else if (l.motion) bits.push(`rotated ${l.motion.rotation_deg.toFixed(0)}° (raw fit)`);
    if (l.reason) bits.push(l.reason);
    if (l.follow_error) bits.push(l.follow_error);
    status.textContent = `${l.state || ''} · ${(t.fps || 0).toFixed(0)} fps`;
    return bits.join(' · ');
}


// ── Approach tab: camera session + jog + calibration + pre-grasp, one place ─
let apKeysWired = false;

// The stages are sub-tabs under the shared camera and 3D views; the last one chosen is remembered.
function apSub(name) {
    // With the details folded away under the guide, the sub-panels stay hidden whatever is selected.
    const folded = document.getElementById('ap-subtabs') && document.getElementById('ap-subtabs').style.display === 'none';
    document.querySelectorAll('#tab-approach .ap-sub').forEach(el => { el.style.display = (!folded && el.id === `ap-sub-${name}`) ? '' : 'none'; });
    document.querySelectorAll('#tab-approach .ap-subtab').forEach(b => b.classList.toggle('active', b.dataset.sub === name));
    try { localStorage.setItem('ap-sub', name); } catch (e) { /* storage may be unavailable */ }
    if (name === 'calib') calibRefresh();
    if (name === 'demo' && !folded) { deLoad(); deReachStart(); } else deReachStop();
    grpShow(name === 'groups' && !folded);
}

// ── Point groups: the live view relayed from benchmarks/group_live.py, and its RGB-D recording ─
let grpTimer = null;

// The stream is held open only while the panel shows: an MJPEG connection costs the relay a frame copy per frame.
function grpShow(on) {
    const img = document.getElementById('groups-frame');
    if (!img) return;
    if (on) {
        if (!img.src) img.src = `/api/pregrasp/groups/stream?${Date.now()}`;
        if (!grpTimer) { grpStatus(); grpTimer = setInterval(grpStatus, 2000); }
    } else {
        img.removeAttribute('src');
        if (grpTimer) { clearInterval(grpTimer); grpTimer = null; }
    }
}

let grpErrorUntil = 0;  // a failed Start or Finish stays on the line for a while before the poll overwrites it

function grpRender(st, error) {
    const running = !!(st && st.running);
    document.getElementById('groups-start-btn').disabled = running;
    document.getElementById('groups-finish-btn').disabled = !running;
    const withActs = document.getElementById('groups-with-acts');
    if (st && withActs && document.activeElement !== withActs) withActs.checked = !!st.with_acts;
    const el = document.getElementById('groups-status');
    let text, color = '#888';
    if (error) { text = error; color = '#ff6b6b'; grpErrorUntil = Date.now() + 8000; }
    else if (running && !st.ready) text = 'starting: the tracker is loading…';
    else if (running && st.recording) { text = `recording: ${st.frames} frames so far, to ${st.recording}`; color = '#ff6b6b'; }
    else if (running) text = 'running for the act\'s objects, not recording';
    else if (st && st.last) text = `finished: ${st.frames} frames in ${st.last}`;
    else if (st && st.log && st.log.length) text = `stopped: ${st.log[st.log.length - 1]}`;
    else text = 'not running';
    el.style.color = color;
    el.textContent = text;
}

async function grpStatus() {
    if (Date.now() < grpErrorUntil) return;
    try {
        const r = await fetch('/api/pregrasp/groups/status');
        if (!r.ok) return grpRender(null, (await r.json()).detail || `status ${r.status}`);
        grpRender(await r.json());
    } catch (e) { grpRender(null, `no reply: ${e}`); }
}

// Start runs the view (camera frames, the tracker on the GPU, the recording); Finish stops it all and says
// where the recording went. The buttons follow the server's state, so a reload lands on the right one.
async function grpControl(action) {
    const btn = document.getElementById(action === 'start' ? 'groups-start-btn' : 'groups-finish-btn');
    btn.disabled = true;
    try {
        const r = await fetch(`/api/pregrasp/groups/${action}`, {method: 'POST'});
        if (!r.ok) return grpRender(null, (await r.json()).detail || `status ${r.status}`);
        grpErrorUntil = 0;
        await grpStatus();
    } catch (e) { grpRender(null, `no reply: ${e}`); }
}

// The switch: on, acts run with the view and an object no view places moves with what it rests on; off, an act
// finishes the view before the arm moves and holds such an object where it was last placed.
async function grpWithActs(on) {
    try {
        const r = await fetch('/api/pregrasp/options', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({groups_with_acts: on})});
        if (!r.ok) return grpRender(null, (await r.json()).detail || `status ${r.status}`);
        await grpStatus();
    } catch (e) { grpRender(null, `no reply: ${e}`); }
}

async function apInitTab() {
    let sub = 'setup';
    try { sub = localStorage.getItem('ap-sub') || sub; } catch (e) { /* storage may be unavailable */ }
    if (!document.getElementById(`ap-sub-${sub}`)) sub = 'setup';
    apSub(sub);
    demosRefresh();
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

function pgTrustLabel() {
    const v = Number(document.getElementById('pg-trust').value);
    document.getElementById('pg-trust-val').textContent = `${Math.round(v * 100)}%`;
}

async function pgTrust() {
    pgTrustLabel();
    try { await pgPost('/api/pregrasp/options', {trust_share: Number(document.getElementById('pg-trust').value)}); }
    catch (e) { pgSet(e.message, true); }
}

async function pgDepth() {
    const body = {depth_check: document.getElementById('pg-depth').checked};
    const mm = Number(document.getElementById('pg-depth-mm').value);
    if (mm >= 1 && mm <= 100) body.depth_tol_mm = mm;
    try { await pgPost('/api/pregrasp/options', body); }
    catch (e) { pgSet(e.message, true); }
}

// ---- the guided flow: one step at a time -------------------------------------------------------
// Every 700 ms the server's state is read and reduced to ONE step with ONE primary action. The
// existing handlers do the work; the guide only decides which of them is next.
const apGuide = {timer: null, action: null, busy: false};

// ── error injection: test how the act absorbs a grasp aimed off or an object found wrong. Off unless set, kept in this
// browser; whenever one is set the fold's summary names it, so it is not left on unnoticed. ──
const AP_INJECT_AXES = [['dx_mm', 'x', ' mm'], ['dy_mm', 'y', ' mm'], ['dz_mm', 'z', ' mm'], ['rx_deg', 'about x', '°'], ['ry_deg', 'about y', '°'], ['rz_deg', 'about z', '°']];

function apInjectRead() {
    let v = {};
    try { v = JSON.parse(localStorage.getItem('ap-inject') || '{}') || {}; } catch (e) { /* storage may be unavailable */ }
    const out = {at: v.at === 'find' ? 'find' : 'aim', correct_hold: v.correct_hold !== false};
    for (const [k] of AP_INJECT_AXES) out[k] = Number(v[k]) || 0;
    return out;
}

function apInjectSummary(v) {
    const parts = AP_INJECT_AXES.filter(([k]) => v[k]).map(([k, name, unit]) => `${name} ${v[k]}${unit}`);
    return [parts.length ? `injecting ${v.at === 'find' ? 'a wrong find' : 'a missed aim'}: ${parts.join(', ')}` : '',
        v.correct_hold ? '' : 'the hold not corrected'].filter(Boolean).join('; ');
}

function apInjectFill() {
    const v = apInjectRead();
    const at = document.getElementById('ap-inject-at');
    if (!at) return;
    at.value = v.at;
    for (const [k] of AP_INJECT_AXES) document.getElementById(`ap-inject-${k}`).value = v[k];
    document.getElementById('ap-inject-hold').checked = v.correct_hold;
    const on = apInjectSummary(v), sum = document.getElementById('ap-inject-summary');
    sum.textContent = on || 'inject an error';
    sum.style.color = on ? '#e9a23b' : '#888';
}

function apInjectSave() {
    const v = {at: document.getElementById('ap-inject-at').value, correct_hold: document.getElementById('ap-inject-hold').checked};
    for (const [k] of AP_INJECT_AXES) v[k] = Number(document.getElementById(`ap-inject-${k}`).value) || 0;
    try { localStorage.setItem('ap-inject', JSON.stringify(v)); } catch (e) { /* storage may be unavailable */ }
    apInjectFill();
}

function apInjectClear() {
    try { localStorage.removeItem('ap-inject'); } catch (e) { /* storage may be unavailable */ }
    apInjectFill();
}

function apInjectToggle(open) {
    try { localStorage.setItem('ap-inject-open', open ? '1' : '0'); } catch (e) { /* storage may be unavailable */ }
}

function apInjectBody() {
    const v = apInjectRead();
    const inject = AP_INJECT_AXES.some(([k]) => v[k]) ? Object.fromEntries([['at', v.at], ...AP_INJECT_AXES.map(([k]) => [k, v[k]])]) : null;
    return {inject, correct_hold: v.correct_hold};
}

function apGuideShow(step, text, label, action, extraHtml = '') {
    const fold = document.getElementById('ap-inject');
    if (fold) fold.style.display = step === 'Act' ? '' : 'none';
    document.getElementById('ap-guide-step').textContent = step;
    document.getElementById('ap-guide-text').textContent = text;
    const btn = document.getElementById('ap-guide-btn');
    btn.style.display = label ? '' : 'none';
    btn.textContent = label || '';
    btn.disabled = apGuide.busy;
    apGuide.action = action;
    const extra = document.getElementById('ap-guide-extra');
    if (extra.dataset.html !== extraHtml) { extra.innerHTML = extraHtml; extra.dataset.html = extraHtml; }
}

async function apGuideAction() {
    if (!apGuide.action || apGuide.busy) return;
    apGuide.busy = true;
    document.getElementById('ap-guide-btn').disabled = true;
    try { await apGuide.action(); } catch (e) { document.getElementById('ap-guide-text').textContent = e.message; }
    finally { apGuide.busy = false; apGuideTick(); }
}

async function apGuideTick() {
    const el = document.getElementById('ap-guide');
    if (!el || !document.getElementById('tab-approach').classList.contains('active')) { pgCamSync(false); return; }
    let st, jg;
    try {
        st = await (await fetch('/api/pregrasp/state')).json();
        jg = st.arm_connected ? await (await fetch('/api/jog/state')).json() : {};
    } catch (e) { apGuideShow('offline', 'the server is not answering', null, null); return; }
    const w = st.worker || {}, tr = st.track || {}, act = st.act || {}, demo = st.demo;
    const last = tr.last || {};
    // The camera view follows the server whoever started or stopped the tracker (a script, an act): the tracker's
    // frames while it runs, the camera's own while the camera is live and it does not.
    if (!!tr.on !== pgLive.on) { if (tr.on) pgLiveStart(); else pgLiveStop(); }
    pgCamSync(!!st.camera_live && !tr.on);
    if (typeof deSync === 'function') deSync(st);
    // Before anything else, unless an act runs and needs its Stop: a tab older than the server runs old code.
    if (!act.on && st.page_version && PG_PAGE_VERSION && st.page_version !== PG_PAGE_VERSION) {
        return apGuideShow('Reload', 'this page is older than the server: reload it to run the current version', 'Reload', async () => { location.reload(); });
    }
    const trackText = tr.on ? `${last.state || 'starting'} at ${(tr.fps || 0).toFixed(0)} fps` : 'not tracking';
    if (!st.camera_live) {
        return apGuideShow('Camera', 'the camera is off', 'Start camera', async () => {
            const sel = document.getElementById('ap-camera');
            if (sel && !sel.value && sel.options.length) sel.selectedIndex = sel.options.length - 1;
            await apCameraToggle();
        });
    }
    if (!w.running) return apGuideShow('Worker', 'the tracking worker is off', 'Start worker', async () => { await pgPost('/api/pregrasp/worker/start'); });
    if (!w.ready) return apGuideShow('Worker', 'the worker is loading its models…', null, null);
    apMarksObject = ((demo && demo.keypoints) || []).filter(k => k.kind === 'pregrasp' || k.kind === 'grasp_end').map(k => k.object).find(Boolean) || '';
    apPlaceObject = (demo && demo.place_object) || '';
    apClickFor = 'pick';
    const ref = st.teach && st.teach.ref;
    if (st.teach_pending) return apGuideShow(apMarksObject ? 'Find' : 'Teach', apMarksObject ? `finding ${apMarksObject}…` : 'teaching the object…', null, null);
    if (act.on) return apGuideShow('Act', `acting: ${act.step} ${(100 * (act.progress || 0)).toFixed(0)}%`, 'Stop', async () => { await actStop(); });
    // No verdict is asked for after an act: what happened is in its recording (frames, the arm, what it measured).
    const connectArm = async () => { await jogToggle(); };
    // 1. The demo, recorded first: nothing is taught before it.
    const leader = jg.mode === 'leader';
    if (st.recording) return apGuideShow('Record', `recording the demo (${jg.record_n || 0} samples); do the whole task with the leader, then`, 'Stop recording', async () => { await demoRecordToggle(); });
    if (demo && !demo.root) return apGuideShow('Save', `recorded ${demo.n} samples; name it under details if you like, then`, 'Save demo', async () => { await demoSave(); });
    if (leader && demo) return apGuideShow('Hand back', 'the demo is saved; raise the arm clear with the leader, then', 'Hand the arm back', async () => { await jogLeaderToggle(); });
    if (leader) return apGuideShow('Record', 'the leader drives the arm; bring it to where the task starts, then', 'Record demo', async () => { await demoRecordToggle(); });
    if (!demo) {
        if (!st.arm_connected) return apGuideShow('Arm', 'the arm is not connected', 'Connect arm', connectArm);
        if (!document.getElementById('jog-leader').value) return apGuideShow('Leader', 'save an SO-107 leader profile first, then choose it under details, Teach', null, null);
        return apGuideShow('Leader', 'hold the leader arm where the task starts before pressing: the follower first moves to the leader\'s pose', 'Hand to leader', async () => { await jogLeaderToggle(); });
    }
    // 2. On the recording: the objects that matter, then the marks.
    const marksList = demo.keypoints || [];
    if (!marksList.length) {
        if (demo.stream_frames && !(demo.objects || []).length) return apGuideShow('Objects', `click each object that matters on the recording of "${demo.name}"`, 'Edit demo', async () => { apDetailsToggle(true); apSub('demo'); });
        return apGuideShow('Mark', `mark the pre-grasp and the grasp in "${demo.name}"`, 'Edit demo', async () => { apDetailsToggle(true); apSub('demo'); });
    }
    if (!apMarksObject && (!demo.taught || (demo.objects || []).length)) return apGuideShow('Mark', `the marks in "${demo.name}" do not say which object they follow: open the editor and save them`, 'Edit demo', async () => { apDetailsToggle(true); apSub('demo'); });
    // 3. The object, found live: from the demo's view of it, or taught the old way for a demo recorded after a teach.
    if (apMarksObject) {
        if (!ref || ref.object !== apMarksObject) return apGuideShow('Find', `click ${apMarksObject} in the camera view: it is found from the demo's view of it`, null, null);
        if (!ref.ok) return apGuideShow('Find', `${apMarksObject} was not found (${ref.reason}): click it again`, null, null);
        const loc = apPlaceObject ? (st.located || {})[apPlaceObject] : null;
        if (apPlaceObject && (!loc || !loc.ok || loc.strong === false)) {
            apClickFor = 'place';
            if (!loc) return apGuideShow('Find', `click ${apPlaceObject} in the camera view: the object to place onto, found from the demo's view of it`, null, null);
            if (!loc.ok) return apGuideShow('Find', `${apPlaceObject} was not found (${loc.reason}): click it again`, null, null);
            return apGuideShow('Find', `a weak find: ${apPlaceObject} matched ${loc.inliers} of the demo view's ${loc.card_points} points; turn it closer to how it lay in the demo, then click it again`, null, null);
        }
    } else {
        if (!st.teach) return apGuideShow('Teach', 'click the object in the camera view', null, null);
        // Tracking starts from the frame the object was taught on and follows the motion it sees; a jump from that frame
        // to wherever the object lies now has never been measured.
        if (!tr.on) return apGuideShow('Track', `"${st.teach.concept}" is not tracked: put it back where it was taught, start tracking, then move it while the dots follow it`, 'Start tracking', async () => { await pgPost('/api/pregrasp/track/start', pgTrackBody()); });
    }
    if (!st.arm_connected) return apGuideShow('Arm', `tracking "${st.teach.concept}" (${trackText}); the arm is not connected`, 'Connect arm', connectArm);
    // Only an object taught before the demo depends on the track that began at its teach. A designated object is
    // found again at Act wherever it was last seen, so a lost or hidden one does not hold the act back.
    if (!apMarksObject && tr.on && last.state === 'lost') {
        if (demo && demo.root) {
            return apGuideShow('Track', `the tracker lost "${st.teach.concept}": put it back where it was taught, then`, 'Load the demo again', async () => { await pgPost('/api/pregrasp/demo/load', {name: demo.name}); });
        }
        return apGuideShow('Track', `the tracker lost "${st.teach.concept}": click it in the camera view to teach it again`, null, null);
    }
    // One button: Act. The row says only what needs the operator now, in a few words: why the last act stopped (the
    // whole reason under "why"), or a weak find to turn closer. The finds' details are on the live view's badges, the
    // marks in the editor.
    const loc = apPlaceObject ? (st.located || {})[apPlaceObject] : null;
    const weak = [[apMarksObject, ref], [apPlaceObject, loc]].find(([o, f]) => o && f && f.ok && f.strong === false);
    let text = 'ready', why = '';
    if (act.ok === false && act.reason && !act.on) {
        text = `last act stopped: ${act.reason.split(': ')[0]}`;
        why = act.reason;
    } else if (weak) {
        text = `${weak[0]}: a weak find (${weak[1].inliers} of ${weak[1].card_points} points); turn it closer to how it lay in the demo, then click it again`;
    } else if (!apMarksObject) {
        text = `move and turn "${st.teach.concept}" while it is tracked (${trackText})`;
    }
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return apGuideShow('Act', text, 'Act', async () => { await actGo(); },
        `<label style="color:#888;">speed <input id="ap-guide-speed" type="number" step="0.25" min="0.1" max="2" value="${document.getElementById('act-speed').value || 0.5}" style="width:52px;" onchange="document.getElementById('act-speed').value=this.value"></label>` +
        (why ? `<details id="ap-guide-why"><summary style="cursor:pointer; color:#888;">why</summary><span style="color:#aaa;">${esc(why)}</span></details>` : ''));
}

function apDetailsToggle(force) {
    const on = typeof force === 'boolean' ? force : document.getElementById('ap-subtabs').style.display === 'none';
    document.getElementById('ap-subtabs').style.display = on ? '' : 'none';
    for (const el of document.querySelectorAll('.ap-sub')) el.style.display = on ? '' : 'none';
    let sub = 'setup';
    try { sub = localStorage.getItem('ap-sub') || sub; } catch (e) { /* storage may be unavailable */ }
    if (on && typeof apSub === 'function') apSub(sub);
    try { localStorage.setItem('ap-details', on ? '1' : '0'); } catch (e) { /* storage may be unavailable */ }
    document.getElementById('ap-details-btn').textContent = on ? 'hide details' : 'show details';
}


// ── demo editor: play the recording, mark the pre-grasp points and the end of the grasp ──
const de = {curve: null, i: 0, playing: false, timer: null, kps: [], loadedFor: null, frameBusy: false, framePending: null, reachTimer: null, dirty: false, objects: [], objTimer: null, pathArm: null};
const DE_PRE = '#ffaa00', DE_GRASP = '#00c8ff', DE_PLACE = '#7ddc6f';
const DE_GRASP_KINDS = ['pregrasp', 'grasp_end'], DE_PLACE_KINDS = ['preplace', 'place_end'];

function deStatus(text, isError = false) {
    const el = document.getElementById('de-status');
    if (!el) return;
    el.textContent = text; el.style.color = isError ? '#e55' : '#888';
}

function deVisible() {
    const el = document.getElementById('ap-sub-demo');
    return !!el && el.style.display !== 'none' && document.getElementById('tab-approach').classList.contains('active');
}

function deKey(name, n, kps) { return `${name}:${n}:${JSON.stringify(kps)}`; }
function dePre() { return de.kps.filter(k => k.kind === 'pregrasp').sort((a, b) => a.t - b.t); }
function deEnd() { return de.kps.find(k => k.kind === 'grasp_end') || null; }
function dePrePlace() { return de.kps.filter(k => k.kind === 'preplace').sort((a, b) => a.t - b.t); }
function dePlaceEnd() { return de.kps.find(k => k.kind === 'place_end') || null; }
function deStageObject(stage) {
    // The object a stage's marks follow: theirs once marked, else what its chooser says.
    const kinds = stage === 'place' ? DE_PLACE_KINDS : DE_GRASP_KINDS;
    const k = de.kps.find(m => kinds.includes(m.kind));
    if (k) return k.object || '';
    return (document.getElementById(stage === 'place' ? 'de-place-for' : 'de-marks-for') || {}).value || '';
}

async function deLoad(force = false) {
    if (!document.getElementById('ap-sub-demo')) return;
    try {
        const r = await fetch('/api/pregrasp/demo/curve');
        if (!r.ok) {
            de.curve = null; de.kps = []; de.loadedFor = null; de.dirty = false;
            deStatus(r.status === 404 ? 'no demo yet: record one in Teach, or load a saved one' : 'the server did not answer');
            document.getElementById('de-time').textContent = 'no demo';
            deRenderList(); deDrawStrip(); deDrawOverlay();
            return;
        }
        const c = await r.json();
        const key = deKey(c.name, c.n, c.keypoints);
        if (!force && key === de.loadedFor) return;
        if (de.dirty && !force && de.curve && de.curve.name === c.name) return; // unsaved edits stay
        de.curve = c; de.loadedFor = key; de.i = Math.min(de.i, c.n - 1); de.dirty = false;
        de.kps = c.keypoints.map(k => ({...k}));
        const landing = document.getElementById('de-landing');
        if (landing) landing.value = c.landing || 'exact';
        deStatus(c.keypoints.length ? 'saved with the demo' : 'nothing marked yet: scrub to a moment and add it');
        const sl = document.getElementById('de-slider'); sl.max = c.n - 1; sl.value = de.i;
        if (!c.has_frames) document.getElementById('de-frame').removeAttribute('src');
        document.getElementById('de-frame').style.cursor = c.recording ? 'crosshair' : '';
        deRenderList(); deSeek(de.i, true); deObjRefresh();
    } catch (e) { deStatus(e.message, true); }
}

function deSync(st) {
    // Called by the guide's poll: a new or reloaded demo replaces the editor's copy when the editor is open.
    if (!deVisible() || !st.demo) return;
    de.hold = st.demo.hold || null;
    if (deKey(st.demo.name, st.demo.n, st.demo.keypoints || []) !== de.loadedFor && !de.dirty) { deLoad(); return; }
    // The fingertip path is drawn through the connected arm's camera calibration: fetch it again when an arm comes
    // or goes, or a demo opened without the arm never shows it.
    const arm = !!st.arm_connected;
    if (de.curve && de.pathArm !== arm) { de.pathArm = arm; deRefreshPath(); }
}

async function deRefreshPath() {
    try {
        const r = await fetch('/api/pregrasp/demo/curve');
        if (!r.ok || !de.curve) return;
        const c = await r.json();
        if (c.name !== de.curve.name) return;
        de.curve.uv = c.uv; de.curve.image_size = c.image_size; de.curve.seen = c.seen; de.curve.pose_t = c.pose_t; de.curve.place_pose_t = c.place_pose_t; de.curve.grip_t = c.grip_t;  // the marks stay as edited
        deDrawStrip(); deSeek(de.i, true); deReach();
    } catch (e) { /* no server */ }
}

function deSeek(i, force = false) {
    const c = de.curve;
    if (!c) return;
    de.i = Math.max(0, Math.min(c.n - 1, i));
    document.getElementById('de-slider').value = de.i;
    // A fixed-width label: text that grew and shrank with the frame squeezed the slider, so the timeline jumped.
    document.getElementById('de-time').textContent = `${c.t[de.i].toFixed(2)} s · ${de.i + 1}/${c.n}`;
    deDrawStrip(); deDrawOverlay();
    if (!c.has_frames) return;
    if (de.frameBusy && !force) { de.framePending = de.i; return; }
    de.frameBusy = true;
    const img = document.getElementById('de-frame');
    img.onload = img.onerror = () => {
        de.frameBusy = false; deDrawOverlay();
        if (de.framePending !== null) { const n = de.framePending; de.framePending = null; deSeek(n); }
    };
    img.src = `/api/pregrasp/demo/frame.jpg?i=${de.i}`;
}

function deTogglePlay() {
    de.playing = !de.playing;
    document.getElementById('de-play').innerHTML = de.playing ? '&#10074;&#10074;' : '&#9654;';
    if (de.timer) { clearInterval(de.timer); de.timer = null; }
    if (!de.playing || !de.curve) return;
    const c = de.curve;
    const period = c.n > 1 ? Math.max(20, 1000 * (c.t[c.n - 1] - c.t[0]) / (c.n - 1)) : 100;
    de.timer = setInterval(() => {
        if (!de.curve) return deTogglePlay();
        if (de.frameBusy) return; // the next frame waits for this one
        if (de.i + 1 >= de.curve.n) return deTogglePlay();
        deSeek(de.i + 1);
    }, period);
}

function deIndexAt(t) { // the sample nearest a time
    const c = de.curve; let best = 0, d = Infinity;
    for (let i = 0; i < c.n; i++) { const e = Math.abs(c.t[i] - t); if (e < d) { d = e; best = i; } }
    return best;
}

function deAdd(kind) {
    if (!de.curve) return deStatus('no demo to mark', true);
    const t = de.curve.t[de.i], pre = dePre(), end = deEnd(), pp = dePrePlace(), pend = dePlaceEnd();
    const placing = DE_PLACE_KINDS.includes(kind);
    if (kind === 'grasp_end') {
        if (!pre.length) return deStatus('add a pre-grasp first: the grasp is replayed from the last one', true);
        if (t <= pre[pre.length - 1].t) return deStatus('the grasp ends after the last pre-grasp: scrub past it first', true);
        if (pp.length && t >= pp[0].t) return deStatus('the grasp ends before the first pre-place', true);
        de.kps = de.kps.filter(k => k.kind !== 'grasp_end');
    } else if (kind === 'pregrasp') {
        if (end && t >= end.t) return deStatus('a pre-grasp comes before the grasp end', true);
        if (pre.some(k => k.t === t)) return deStatus('there is a pre-grasp at this moment already', true);
    } else if (kind === 'preplace') {
        if (!end) return deStatus('set the grasp end first: the place comes after the grasp', true);
        if (t <= end.t) return deStatus('a pre-place comes after the grasp end: scrub past it first', true);
        if (pend && t >= pend.t) return deStatus('a pre-place comes before the place end', true);
        if (pp.some(k => k.t === t)) return deStatus('there is a pre-place at this moment already', true);
    } else {
        if (!pp.length) return deStatus('add a pre-place first: the place is replayed from the last one', true);
        if (t <= pp[pp.length - 1].t) return deStatus('the place ends after the last pre-place: scrub past it first', true);
        de.kps = de.kps.filter(k => k.kind !== 'place_end');
    }
    const forObject = (document.getElementById(placing ? 'de-place-for' : 'de-marks-for') || {}).value || '';
    if (placing) {
        const picked = deStageObject('grasp');
        if (!forObject) return deStatus('click the object it is placed onto on the recording first and let it track', true);
        if (!picked) return deStatus('a place needs the picked object clicked on the recording too: choose it under Pick', true);
        if (forObject === picked) return deStatus('the place goes onto another object than the one picked: choose it under place onto', true);
    } else if (!forObject && !de.curve.taught) return deStatus('click the object on the recording first and let it track: the marks follow it', true);
    const kinds = placing ? DE_PLACE_KINDS : DE_GRASP_KINDS;
    if (de.kps.some(k => kinds.includes(k.kind) && (k.object || '') !== forObject)) {
        return deStatus(`the ${placing ? 'pre-places and the place' : 'pre-grasps and the grasp'} follow one object: change the existing marks first`, true);
    }
    de.kps.push(forObject ? {t, kind, object: forObject} : {t, kind});
    de.kps.sort((a, b) => a.t - b.t);
    de.dirty = true;
    deStatus('not saved yet: press Save when the list is right');
    deRenderList(); deDrawStrip(); deDrawOverlay(); deReach();
}

function dePoseOf(o) {
    // Where an object's demo pose is read: a pose mark not yet saved wins; otherwise what the server worked out from
    // the saved marks (set, the frame it was clicked on, or the last frame seen by its stage's first mark).
    const k = de.kps.find(m => m.kind === 'pose' && m.object === o.name);
    if (k) return {t: k.t, from: 'set'};
    return {t: o.pose_t == null ? null : o.pose_t, from: o.pose_from || 'clicked'};
}

function dePoseT(stage) {
    // The time the act reads a stage's object pose at, for the strip and the frame's corner: the unsaved edit if any.
    const c = de.curve, obj = deStageObject(stage);
    const k = obj ? de.kps.find(m => m.kind === 'pose' && m.object === obj) : null;
    if (k) return k.t;
    return stage === 'grasp' ? c.pose_t : c.place_pose_t;
}

function dePoseHere(name) {
    if (!de.curve) return;
    const t = de.curve.t[de.i];
    // Read before the arm can have moved the object: no later than the last mark its stage reaches before replaying.
    const kinds = deStageObject('grasp') === name ? ['pregrasp', 'pre-grasp'] : deStageObject('place') === name ? ['preplace', 'pre-place'] : null;
    if (kinds) {
        const last = de.kps.filter(k => k.kind === kinds[0]).reduce((m, k) => Math.max(m, k.t), -Infinity);
        if (last > -Infinity && t > last) return deStatus(`read ${name}'s pose no later than its last ${kinds[1]} (${last.toFixed(2)} s), before the arm can have moved it`, true);
    }
    de.kps = de.kps.filter(k => !(k.kind === 'pose' && k.object === name));
    de.kps.push({t, kind: 'pose', object: name});
    de.kps.sort((a, b) => a.t - b.t);
    de.dirty = true;
    deStatus(`${name}'s pose is read at ${t.toFixed(2)} s: press Save to keep that`);
    deObjShow(de.objects); deDrawStrip(); deDrawOverlay(); deReach();
}

function dePoseReset(name) {
    de.kps = de.kps.filter(k => !(k.kind === 'pose' && k.object === name));
    de.dirty = true;
    deStatus(`${name}'s pose is read where it was clicked again: press Save to keep that`);
    deObjShow(de.objects); deDrawStrip(); deDrawOverlay(); deReach();
}

function deBindStage(stage, obj) {
    // Each stage follows one object: choosing another re-binds that stage's marks, kept on Save. The picked object's
    // list leaves out nothing; the place's lists every tracked object but the one picked.
    const kinds = stage === 'place' ? DE_PLACE_KINDS : DE_GRASP_KINDS;
    if (de.kps.some(k => kinds.includes(k.kind))) {
        de.kps = de.kps.map(k => {
            if (!kinds.includes(k.kind)) return k;
            const {object, ...rest} = k;
            return obj ? {...rest, object: obj} : rest;
        });
        de.dirty = true;
        deStatus(`the ${stage === 'place' ? 'pre-places and the place' : 'pre-grasps and the grasp'} now follow ${obj || 'the object taught before the demo'}: press Save to keep that`);
    }
    if (stage === 'grasp') deObjShow(de.objects); // the place's list leaves out the object now picked
    deRenderList(); deDrawStrip(); deDrawOverlay(); deReach();
}

function deRemove(i) { de.kps.splice(i, 1); de.dirty = true; deStatus('not saved yet'); deRenderList(); deDrawStrip(); deDrawOverlay(); deReach(); }
function deGo(i) { deSeek(deIndexAt(de.kps[i].t)); }

function deRenderList() {
    const tbl = document.getElementById('de-list');
    if (!tbl) return;
    const pre = dePre(), end = deEnd();
    if (!pre.length && !end) { tbl.innerHTML = '<tr><td style="color:#666; padding:4px 0;">nothing marked</td></tr>'; return; }
    const cell = 'padding:4px 8px 4px 0; vertical-align:top;';
    const del = i => `<td style="padding:4px 0; text-align:right; vertical-align:top;"><button class="btn-small secondary" onclick="deRemove(${i})" title="remove">&#x2715;</button></td>`;
    const rows = pre.map((k, n) => {
        const i = de.kps.indexOf(k);
        return `<tr style="border-top:1px solid #333;"><td style="${cell} color:${DE_PRE};">${n + 1}</td>` +
            `<td style="${cell} white-space:nowrap;"><a href="#" onclick="deGo(${i}); return false;" style="color:${DE_PRE}; text-decoration:none;" title="show this moment">${k.t.toFixed(2)} s</a></td>` +
            `<td style="${cell} white-space:nowrap;">pre-grasp</td><td style="${cell} color:#aaa;">straight line from ${n === 0 ? 'wherever the arm is' : 'pre-grasp ' + n}, following the object</td>${del(i)}</tr>`;
    });
    if (end) {
        const i = de.kps.indexOf(end);
        const from = pre.length ? `${pre[pre.length - 1].t.toFixed(2)}–` : '';
        rows.push(`<tr style="border-top:1px solid #333;"><td style="${cell}"></td>` +
            `<td style="${cell} white-space:nowrap;"><a href="#" onclick="deGo(${i}); return false;" style="color:${DE_GRASP}; text-decoration:none;" title="show the grasp's end">${from}${end.t.toFixed(2)} s</a></td>` +
            `<td style="${cell} white-space:nowrap;">grasp</td><td style="${cell} color:#aaa;">from the last pre-grasp, replayed exactly as shown, turned with the object where it was last clearly seen; the arm holds at the end</td>${del(i)}</tr>`);
    } else {
        rows.push(`<tr style="border-top:1px solid #333;"><td></td><td colspan="4" style="color:#777; padding:4px 0;">no grasp end: the arm stops at pre-grasp ${pre.length}</td></tr>`);
    }
    const pp = dePrePlace(), pend = dePlaceEnd();
    pp.forEach((k, n) => {
        const i = de.kps.indexOf(k);
        rows.push(`<tr style="border-top:1px solid #333;"><td style="${cell} color:${DE_PLACE};">${n + 1}</td>` +
            `<td style="${cell} white-space:nowrap;"><a href="#" onclick="deGo(${i}); return false;" style="color:${DE_PLACE}; text-decoration:none;" title="show this moment">${k.t.toFixed(2)} s</a></td>` +
            `<td style="${cell} white-space:nowrap;">pre-place</td><td style="${cell} color:#aaa;">straight line from ${n === 0 ? 'the grasp end' : 'pre-place ' + n}, carried with ${k.object || 'the object placed onto'}${n === pp.length - 1 ? '; the held object is found in the gripper here, the arm standing still' : ''}</td>${del(i)}</tr>`);
    });
    if (pend) {
        const i = de.kps.indexOf(pend);
        const from = pp.length ? `${pp[pp.length - 1].t.toFixed(2)}–` : '';
        rows.push(`<tr style="border-top:1px solid #333;"><td style="${cell}"></td>` +
            `<td style="${cell} white-space:nowrap;"><a href="#" onclick="deGo(${i}); return false;" style="color:${DE_PLACE}; text-decoration:none;" title="show the place's end">${from}${pend.t.toFixed(2)} s</a></td>` +
            `<td style="${cell} white-space:nowrap;">place</td><td style="${cell} color:#aaa;">replayed exactly as shown, moved with ${pend.object || 'the object placed onto'} and corrected for how the held object sits in the gripper; release included</td>${del(i)}</tr>`);
    } else if (pp.length) {
        rows.push(`<tr style="border-top:1px solid #333;"><td></td><td colspan="4" style="color:#777; padding:4px 0;">no place end: the arm stops at pre-place ${pp.length}, still holding</td></tr>`);
    }
    tbl.innerHTML = rows.join('');
}

function deDrawStrip() {
    const cv = document.getElementById('de-strip');
    if (!cv) return;
    const w = cv.clientWidth || 600; if (cv.width !== w) cv.width = w;
    const h = cv.height, ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#141414'; ctx.fillRect(0, 0, w, h);
    const c = de.curve; if (!c || c.n < 2) return;
    const t0 = c.t[0], t1 = c.t[c.n - 1], x = t => (t - t0) / (t1 - t0 || 1) * (w - 1);
    ctx.fillStyle = '#4a2433'; // the object hidden from the camera; the legend under the strip uses this colour
    for (let i = 0; i < c.n; i++) if (!c.seen[i]) ctx.fillRect(x(c.t[i]), 0, Math.max(1, w / c.n), h);
    const pre = dePre(), end = deEnd();
    if (pre.length && end) { ctx.fillStyle = 'rgba(0,200,255,0.18)'; const a = x(pre[pre.length - 1].t); ctx.fillRect(a, 0, Math.max(2, x(end.t) - a), h); }
    let gmin = Infinity, gmax = -Infinity; for (const g of c.gripper) { gmin = Math.min(gmin, g); gmax = Math.max(gmax, g); }
    const y = g => h - 4 - (gmax > gmin ? (g - gmin) / (gmax - gmin) : 0.5) * (h - 8);
    ctx.strokeStyle = '#9ad'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < c.n; i++) { const px = x(c.t[i]), py = y(c.gripper[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
    ctx.stroke();
    const line = (t, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x(t), 0); ctx.lineTo(x(t), h); ctx.stroke(); };
    pre.forEach(k => line(k.t, DE_PRE));
    if (end) line(end.t, DE_GRASP);
    const pp = dePrePlace(), pend = dePlaceEnd();
    if (pp.length && pend) { ctx.fillStyle = 'rgba(125,220,111,0.18)'; const a = x(pp[pp.length - 1].t); ctx.fillRect(a, 0, Math.max(2, x(pend.t) - a), h); }
    pp.forEach(k => line(k.t, DE_PLACE));
    if (pend) line(pend.t, DE_PLACE);
    // The frames the act reads the objects' demo poses from, the picked one's and the one placed onto; labelled on the frames too.
    for (const [pt, label] of [[dePoseT('grasp'), 'pose'], [dePoseT('place'), 'onto']]) {
        if (pt == null) continue;
        ctx.setLineDash([3, 3]); line(pt, '#ff00ff'); ctx.setLineDash([]);
        ctx.font = '10px sans-serif'; ctx.fillStyle = '#ff00ff'; ctx.fillText(label, Math.min(x(pt) + 3, w - 26), 10);
    }
    if (c.grip_t != null && dePrePlace().length) { // where the grip became firm: a place measures the hold from here
        ctx.setLineDash([2, 2]); line(c.grip_t, DE_GRASP); ctx.setLineDash([]);
        ctx.font = '10px sans-serif'; ctx.fillStyle = DE_GRASP; ctx.fillText('grip', Math.min(x(c.grip_t) + 3, w - 22), h - 18);
    }
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(c.t[de.i]) + 0.5, 0); ctx.lineTo(x(c.t[de.i]) + 0.5, h); ctx.stroke();
    ctx.font = '10px sans-serif';
    const label = 'gripper, up is closed', lw = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(20,20,20,0.85)'; ctx.fillRect(w - lw - 10, h - 15, lw + 8, 13);
    ctx.fillStyle = '#888'; ctx.fillText(label, w - lw - 6, h - 5);
}

function deDrawOverlay() {
    const img = document.getElementById('de-frame'), cv = document.getElementById('de-over');
    if (!img || !cv) return;
    const w = img.clientWidth, h = img.clientHeight;
    if (!w || !h) return;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, w, h);
    const c = de.curve; if (!c || !c.uv || !c.image_size) return deBadges(ctx);
    const sx = w / c.image_size[0], sy = h / c.image_size[1];
    const P = i => c.uv[i] ? [c.uv[i][0] * sx, c.uv[i][1] * sy] : null;
    const pre = dePre(), end = deEnd(), pp = dePrePlace(), pend = dePlaceEnd();
    const g0 = pre.length && end ? deIndexAt(pre[pre.length - 1].t) : -1, g1 = end ? deIndexAt(end.t) : -1;
    const p0 = pp.length && pend ? deIndexAt(pp[pp.length - 1].t) : -1, p1 = pend ? deIndexAt(pend.t) : -1;
    for (let i = 1; i < c.n; i++) { // the recorded path: cyan where the grasp replays it, green the place, dim elsewhere
        const a = P(i - 1), b = P(i); if (!a || !b) continue;
        const grasp = g0 >= 0 && i - 1 >= g0 && i <= g1, place = p0 >= 0 && i - 1 >= p0 && i <= p1;
        ctx.strokeStyle = grasp ? DE_GRASP : place ? DE_PLACE : 'rgba(180,180,180,0.4)'; ctx.lineWidth = grasp || place ? 2.5 : 1;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; // the straight lines between pre-grasps, and the carry
    const lines = [...pre, ...(end && pp.length ? [end, ...pp] : [])];
    for (let n = 1; n < lines.length; n++) {
        if (lines[n - 1] === pre[pre.length - 1] && lines[n] === end) continue; // the grasp itself is replayed, not a line
        const a = P(deIndexAt(lines[n - 1].t)), b = P(deIndexAt(lines[n].t)); if (!a || !b) continue;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.font = '12px sans-serif';
    const ring = (p, col, label) => { if (!p) return; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 2 * Math.PI); ctx.stroke(); ctx.fillStyle = col; ctx.fillText(label, p[0] + 10, p[1] - 6); };
    pre.forEach((k, n) => ring(P(deIndexAt(k.t)), DE_PRE, `pre-grasp ${n + 1}`));
    if (end) ring(P(g1), DE_GRASP, 'grasp end');
    pp.forEach((k, n) => ring(P(deIndexAt(k.t)), DE_PLACE, `pre-place ${n + 1}`));
    if (pend) ring(P(p1), DE_PLACE, 'place end');
    const q = P(de.i);
    if (q) { ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(q[0], q[1], 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 2 * Math.PI); ctx.fill(); }
    deBadges(ctx);
}

function deBadges(ctx) {
    // In the frame's corner: beside the slider they moved the timeline, beside the object the marks' labels hid them.
    const c = de.curve;
    if (!c) return;
    const badges = [];
    const pickT = dePoseT('grasp'), ontoT = dePoseT('place');
    if (pickT != null && de.i === deIndexAt(pickT)) badges.push(['the act reads the object\u2019s pose in this frame', '#ff66ff']);
    if (ontoT != null && de.i === deIndexAt(ontoT)) badges.push(['the act reads the pose of the object it places onto in this frame', '#ff66ff']);
    if (!c.seen[de.i]) badges.push(['object hidden', '#e5c07b']);
    ctx.font = '12px sans-serif';
    badges.forEach(([text, colour], n) => {
        const y = 6 + 24 * n, w = ctx.measureText(text).width + 12;
        ctx.fillStyle = 'rgba(0,0,0,0.65)'; ctx.fillRect(6, y, w, 20);
        ctx.fillStyle = colour; ctx.fillText(text, 12, y + 14);
    });
}

async function deSave() {
    if (!de.curve) return deStatus('no demo', true);
    try {
        const d = await pgPost('/api/pregrasp/demo/keypoints', {keypoints: de.kps.map(k => (k.object ? {t: k.t, kind: k.kind, object: k.object} : {t: k.t, kind: k.kind}))});
        de.dirty = false; de.kps = d.keypoints.map(k => ({...k})); de.curve.keypoints = d.keypoints.map(k => ({...k}));
        de.loadedFor = deKey(d.name, d.n, d.keypoints);
        deStatus(d.keypoints.length ? `saved${d.root ? ' beside the demo' : ' (save the demo to keep it)'}` : 'cleared');
        deRenderList(); deDrawStrip(); deDrawOverlay(); deReach(); deRefreshPath(); deObjRefresh();  // the marks' object sets what counts as seen
        if (typeof apGuideTick === 'function') apGuideTick();
    } catch (e) { deStatus(e.message, true); }
}

async function deLanding(value) {
    try {
        const d = await pgPost('/api/pregrasp/demo/landing', {landing: value});
        if (de.curve) de.curve.landing = d.landing;
        const onto = deStageObject('place') || 'its object';
        deStatus(value === 'turn' ? `the place may land turned any way about the middle of ${onto}: each act takes the turn the arm reaches with joints nearest the demo's`
            : value === 'symmetry' ? `the place may land at any of ${onto}'s symmetric turns: each act takes the one the arm reaches with joints nearest the demo's`
            : 'the place lands as shown');
        deReach();
    } catch (e) { deStatus(e.message, true); }
}

function deLandingNote(landing) {
    if (!landing) return '';
    if (landing.turn_deg === null || landing.turn_deg === undefined) return ' · no landing turn the arm reaches';
    const turn = ((landing.turn_deg + 180) % 360 + 360) % 360 - 180;
    return ` · lands turned ${turn.toFixed(0)}° about ${deStageObject('place') || 'its object'} (${landing.reachable} of the allowed turns within reach)`;
}

async function deReach() {
    const el = document.getElementById('de-reach');
    if (!el || !deVisible()) return;
    const notes = [];
    if (de.curve && !de.curve.uv) notes.push('connect the arm to see the fingertip path over the frames');
    if (!de.curve || !de.curve.keypoints.length || de.dirty) {
        if (de.dirty) notes.push('save to check whether the arm can reach these');
        el.textContent = notes.join(' · ');
        return;
    }
    try {
        const r = await fetch('/api/pregrasp/demo/reach');
        const d = await r.json();
        if (!r.ok) { el.textContent = [...notes, `reach: ${d.detail || 'unknown'}`].join(' · '); return; }
        const hold = de.hold ? ` · the demo's hold, measured on ${de.hold.n} still views: within ${de.hold.spread_mm.toFixed(1)} mm and ${de.hold.spread_deg.toFixed(1)}°` : '';
        el.innerHTML = 'as the objects lie now: ' + d.marks.map(m => `<span style="color:${m.ok ? '#6c6' : '#e55'};">${m.label} ${m.ok ? '&#10003;' : '&#10007; ' + m.residual_mm.toFixed(0) + ' mm short'}</span>`).join(' · ') +
            `<span style="color:#777;"> · ${d.summary.seconds.toFixed(1)} s of motion at speed 1${deLandingNote(d.landing)}${d.ok ? '' : ' · ' + d.reason}${d.place_problem ? ' · the place: ' + d.place_problem : ''}${hold}</span>`;
    } catch (e) { el.textContent = ''; }
}
function deReachStart() { deReachStop(); deReach(); de.reachTimer = setInterval(deReach, 3000); }
function deReachStop() { if (de.reachTimer) { clearInterval(de.reachTimer); de.reachTimer = null; } }

// ── objects designated on the recording: a click on the frame, then tracked through the whole stream ──
function deNextObjectName() {
    // The server keeps letters, digits and . _ - only, so the default is already in its stored form.
    const used = new Set(de.objects.map(o => o.name));
    let k = 1;
    while (used.has(`object_${k}`)) k++;
    return `object_${k}`;
}

async function deObjPick(ev) {
    if (!de.curve || !de.curve.recording) return;
    const img = document.getElementById('de-frame');
    const x = ev.offsetX / img.clientWidth * img.naturalWidth, y = ev.offsetY / img.clientHeight * img.naturalHeight;
    const name = document.getElementById('de-obj-name').value.trim() || deNextObjectName();
    try {
        const d = await pgPost('/api/pregrasp/demo/objects', {i: de.i, x, y, name});
        document.getElementById('de-obj-name').value = '';
        deObjShow(d.objects);
        deStatus(`tracking "${name}" through the recording…`);
    } catch (e) { deStatus(e.message, true); }
}

async function deObjRemove(name) {
    try { deObjShow((await pgPost('/api/pregrasp/demo/objects/remove', {name})).objects); deSeek(de.i, true); } catch (e) { deStatus(e.message, true); }
}

async function deObjSymmetry(name, order) {
    try {
        deObjShow((await pgPost('/api/pregrasp/demo/objects/symmetry', {name, order: Number(order)})).objects);
        deStatus(Number(order) > 1 ? `"${name}" reads the same turned by ${(360 / Number(order)).toFixed(0)}°: its next find reports the turn nearest the demo's` : `"${name}" has no symmetry: its finds report the turn as fitted`);
    } catch (e) { deStatus(e.message, true); }
}

async function deObjRefresh() {
    try {
        const r = await fetch('/api/pregrasp/demo/objects');
        if (r.ok) deObjShow((await r.json()).objects);
    } catch (e) { /* no server */ }
}

function deObjShow(objects) {
    const wasTracking = de.objects.some(o => o.status === 'tracking');
    de.objects = objects || [];
    const sel = document.getElementById('de-marks-for'), psel = document.getElementById('de-place-for');
    if (sel && psel) {
        const done = de.objects.filter(o => o.status === 'done').map(o => o.name);
        // Unnamed marks follow the object taught before the demo, which only a demo recorded after a teach has. Once an
        // object is clicked on the recording it is no choice: a teach left over from an earlier demo made it the
        // default, and marks saved by an operator who never taught anything followed it.
        const options = [...(de.curve && de.curve.taught && !done.length ? [['', 'the object taught before the demo']] : []), ...done.map(n => [n, n])];
        const values = options.map(([v]) => v);
        const graspMark = de.kps.find(k => DE_GRASP_KINDS.includes(k.kind));
        const named = graspMark ? (graspMark.object || '') : null;
        const current = named !== null && values.includes(named) ? named : (values.includes(sel.value) ? sel.value : (values.length ? values[0] : ''));
        sel.innerHTML = options.map(([v, label]) => `<option value="${v}">${label}</option>`).join('');
        sel.value = current;
        const placeMark = de.kps.find(k => DE_PLACE_KINDS.includes(k.kind));
        const placeNamed = placeMark ? (placeMark.object || '') : null;
        const others = done.filter(n => n !== current);
        const placeCurrent = placeNamed !== null && others.includes(placeNamed) ? placeNamed : (others.includes(psel.value) ? psel.value : (others[0] || ''));
        psel.innerHTML = others.map(n => `<option value="${n}">${n}</option>`).join('');
        psel.value = placeCurrent;
        document.getElementById('de-place-for-wrap').style.display = others.length ? '' : 'none';
        document.getElementById('de-marks-for-row').style.display = options.length > 1 ? '' : 'none';
        if (named !== null && named !== current && values.length) deBindStage('grasp', current);
        else if (placeNamed !== null && placeNamed !== placeCurrent && others.length) deBindStage('place', placeCurrent);
    }
    const tbl = document.getElementById('de-obj-list');
    if (tbl) {
        tbl.innerHTML = de.objects.length ? de.objects.map(o => {
            const state = o.status === 'tracking' ? `tracking ${(100 * o.progress).toFixed(0)}%` : o.status === 'done' ? `seen in ${(100 * o.seen_fraction).toFixed(0)}%` : `failed: ${o.reason}`;
            const pose = dePoseOf(o);
            const poseCell = o.status !== 'done' ? '<td></td>' :
                `<td style="padding:4px 8px 4px 0; color:#aaa; white-space:nowrap;" title="the frame the act reads where this object was in the demo">pose read at ${pose.t == null ? '?' : pose.t.toFixed(2) + ' s'} <span style="color:#777;">(${pose.from})</span> ` +
                `<button class="btn-small secondary" onclick="dePoseHere('${o.name}')" title="read this object's pose on the frame shown: one where it is in full view">pose here</button>` +
                (pose.from === 'set' ? ` <button class="btn-small secondary" onclick="dePoseReset('${o.name}')" title="back to the frame it was clicked on">reset</button>` : '') + '</td>';
            const symCell = o.status !== 'done' ? '<td></td>' :
                `<td style="padding:4px 8px 4px 0; color:#aaa; white-space:nowrap;" title="rotational symmetry about the axis it rests on: turned by 360/order degrees it looks and acts the same (2: a shape that reads the same turned end to end; 4: a plain cube). Its finds then report the turn nearest the demo's">symmetry ` +
                `<select onchange="deObjSymmetry('${o.name}', this.value)">${[1, 2, 3, 4, 6, 8].map(n => `<option value="${n}"${n === (o.symmetry || 1) ? ' selected' : ''}>${n === 1 ? 'none' : 'order ' + n}</option>`).join('')}</select></td>`;
            return `<tr style="border-top:1px solid #333;"><td style="padding:4px 8px 4px 0; color:${o.colour}; white-space:nowrap;">${o.name}</td>` +
                `<td style="padding:4px 8px 4px 0; color:#888; white-space:nowrap;">clicked at ${o.t == null ? '?' : o.t.toFixed(2) + ' s'}</td>` +
                `<td style="padding:4px 8px 4px 0; color:${o.status === 'failed' ? '#e55' : '#aaa'}; white-space:nowrap;">${state}</td>` + poseCell + symCell +
                `<td style="padding:4px 0; text-align:right;"><button class="btn-small secondary" onclick="deObjRemove('${o.name}')" title="remove">&#x2715;</button></td></tr>`;
        }).join('') : '<tr><td style="color:#666; padding:2px 0;">no objects yet</td></tr>';
    }
    const tracking = de.objects.some(o => o.status === 'tracking');
    if (tracking && !de.objTimer) de.objTimer = setInterval(deObjRefresh, 1000);
    if (!tracking && de.objTimer) { clearInterval(de.objTimer); de.objTimer = null; }
    if (wasTracking && !tracking) { deSeek(de.i, true); deStatus('tracked: the outlines show where it is in each frame'); }
}

(function deWire() {
    const strip = document.getElementById('de-strip');
    if (!strip) return;
    document.getElementById('de-frame').addEventListener('click', deObjPick);
    strip.addEventListener('click', ev => {
        const c = de.curve; if (!c) return;
        const r = strip.getBoundingClientRect();
        deSeek(Math.round((ev.clientX - r.left) / r.width * (c.n - 1)));
    });
    window.addEventListener('resize', () => { deDrawStrip(); deDrawOverlay(); });
})();


// Start-up runs last: a top-level call that reaches a `const` declared below it throws, and a throw here
// kills the rest of this script, the guided row with it. The guide starts before the optional restore of
// the details fold, so the fold can never keep the operator from the flow's entry point.
(function apGuideStart() {
    if (!document.getElementById('ap-guide')) return;
    const speed = document.getElementById('act-speed'); if (speed && Number(speed.value) === 1) speed.value = '0.5';
    apGuide.timer = setInterval(apGuideTick, 700);
    apGuideTick();
    let show = false;
    try { show = localStorage.getItem('ap-details') === '1'; } catch (e) { /* storage may be unavailable */ }
    apDetailsToggle(show);
    apInjectFill();
    let injectOpen = false;
    try { injectOpen = localStorage.getItem('ap-inject-open') === '1'; } catch (e) { /* storage may be unavailable */ }
    const fold = document.getElementById('ap-inject');
    if (fold) fold.open = injectOpen;
})();
