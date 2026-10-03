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
        try { localStorage.setItem('jog-profile', document.getElementById('jog-profile').value); } catch (e) { /* storage may be unavailable */ }
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
        const body = stopping ? undefined : JSON.stringify({profile: document.getElementById('jog-leader').value || 'blue', arm: document.getElementById('jog-arm').value});
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
            if (!pgLive.on) document.getElementById('pg-frame').src = `/api/pregrasp/${st.test ? 'test' : 'teach'}.jpg?t=${Date.now()}`;
            if (st.test && !st.test.ok) pgSet(`not found: ${st.test.reason}`, true);
            else if (st.test) pgSet(`object found — ${st.test.n_inliers} of ${st.test.n_matches} matches agree, rms ${(st.test.rms_m * 1000).toFixed(1)} mm; the camera view shows the path the arm would follow`);
            else if (st.teach) pgSet(`taught by SAM3 + DINO: ${st.teach.n_points} points on "${st.teach.concept}"${st.teach.face_usable ? `, a flat face toward the camera (${(st.teach.face_planarity * 100).toFixed(0)}% of its cloud)` : `, no single flat face${st.teach.face_planarity != null ? ` (${(st.teach.face_planarity * 100).toFixed(0)}% on the largest plane)` : ''}`} — start tracking, then record a demo in Teach or load one`);
            else pgSet('teach failed — see the worker log', true);
        }
        if (w.log && (st.teach_pending || st.find_pending || !w.ready)) lines.push('worker: ' + w.log.split('\n').slice(-3).join(' | '));
        const flat = document.getElementById('pg-flat'); if (flat && document.activeElement !== flat) flat.checked = !!st.flat;
        if (st.teach) lines.push(`taught ${st.teach.at} (${st.teach.mode}): ` + (st.teach.mode === 'features' ? `${st.teach.n_points} DINO points on "${st.teach.concept}", radius ${st.teach.radius_mm.toFixed(0)} mm, visible cloud ${st.teach.shape_class === 'disc' ? 'thin from this view' : st.teach.shape_class}${st.teach.face_planarity != null ? `, ${(st.teach.face_planarity * 100).toFixed(0)}% of the cloud on its face${st.teach.face_usable ? '' : ' (not usable as an axis)'}` : ''}` : st.teach.mode === 'texture' ? `${st.teach.n_with_depth} of ${st.teach.n_keypoints} keypoints have depth` : `${st.teach.n_points} depth points above the table, ${st.teach.height_mm.toFixed(0)} mm tall${st.teach.colour_cue ? ', colour is a usable cue' : ', colour not distinctive'}`) + (st.teach.tip_mm ? ` · demo starts at (${st.teach.tip_mm.map(v => v.toFixed(0)).join(', ')}) mm, gripper ${st.teach.gripper == null ? '?' : st.teach.gripper.toFixed(0)}` : ' · no demo loaded'));
        if (st.test) {
            const armTxt = st.test.arm_turn_deg != null ? ` · the gripper will turn ${st.test.arm_turn_deg.toFixed(0)}° about vertical and lean ${st.test.arm_lean_deg.toFixed(0)}°` : '';
            if (st.test.ok && st.test.mode === 'features') {
                const ax = st.test.axis_source;
                const turn = ax === 'face' ? `turned ${st.test.yaw_deg.toFixed(0)}° about its face, and the face tipped ${st.test.face_tilt_deg.toFixed(0)}°; the raw fit's axis was ${st.test.fit_axis_tilt_deg.toFixed(0)}° off`
                    : ax === 'surface' ? `turned ${(st.test.yaw_deg || 0).toFixed(0)}° about the surface it rests on${st.test.surface_tilt_deg > 1 ? `, which tilted ${st.test.surface_tilt_deg.toFixed(0)}°` : ''}${st.test.face_tilt_deg != null ? ` (face tilt ${st.test.face_tilt_deg.toFixed(0)}° reported only)` : ''}; raw fit's axis ${(st.test.fit_axis_tilt_deg || 0).toFixed(0)}° off${st.test.turn_source === 'footprint' ? ` · turn from the footprint${st.test.footprint_symmetric ? ' (symmetric: no turn measurable)' : ''}, features said ${(st.test.yaw_deg + (st.test.turn_disagreement_deg || 0)).toFixed(0)}°` : ''}`
                    : ax === 'table' ? `turned ${(st.test.yaw_deg || 0).toFixed(0)}° about the table normal${st.test.face_tilt_deg != null ? ` (face tilt ${st.test.face_tilt_deg.toFixed(0)}° ignored: objects stay on the table)` : ''}${st.test.fit_axis_tilt_deg != null ? `; raw fit's axis ${st.test.fit_axis_tilt_deg.toFixed(0)}° off` : ''}`
                    : `turned ${st.test.motion.rotation_deg.toFixed(0)}° in 6-DoF (raw fit)`;
                lines.push(`found ${st.test.at} (SAM3 + DINO): ${st.test.n_inliers} of ${st.test.n_matches} matches agree · rms ${(st.test.rms_m * 1000).toFixed(1)} mm · scale ${st.test.scale.toFixed(3)} · ${turn}${st.test.transported_tip_mm ? ` · go to (${st.test.transported_tip_mm.map(v => v.toFixed(0)).join(', ')}) mm` : ''}${armTxt}`);
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

async function pgTeachAt(x, y) {
    try {
        await pgPost('/api/pregrasp/teach/capture', {mode: 'features', concept: document.getElementById('pg-concept').value, click: [x, y]});
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
        document.getElementById('pg-frame').src = `/api/pregrasp/teach.jpg?t=${Date.now()}`;
    } catch (e) { pgSet(e.message, true); }
    pgState();
}




let pgTrialsShown = -1;
async function pgTrialsRefresh(force = false) {
    try {
        const r = await fetch('/api/pregrasp/trials');
        if (!r.ok) return;
        const rows = (await r.json()).rows || [];
        if (!force && rows.length === pgTrialsShown && !rows.some(x => x.verdict == null)) return;
        pgTrialsShown = rows.length;
        const box = document.getElementById('pg-trials');
        if (!rows.length) { box.innerHTML = ''; return; }
        const f = (v, d = 0) => (v == null ? '–' : Number(v).toFixed(d));
        const last = rows.slice(-12);
        const start = rows.length - last.length;
        box.innerHTML = `<table style="border-collapse:collapse; width:100%;"><thead><tr style="color:#aaa; text-align:left;">
            <th>#</th><th>time</th><th>object</th><th>source</th><th>moved mm</th><th>turned °</th><th>axis</th><th>agree</th><th>gripper turn/lean °</th><th>result</th><th>closed at</th><th>verdict</th></tr></thead><tbody>` +
            last.map((x, k) => {
                const i = start + k;
                const verdict = x.verdict ? x.verdict : ['lifted', 'missed', 'collided', 'other'].map(v => `<button class="btn-small" onclick="pgVerdict(${i}, '${v}')">${v}</button>`).join(' ');
                return `<tr style="border-top:1px solid #333;"><td>${i}</td><td>${x.at.slice(11)}</td><td>${x.object}</td><td>${x.source || ''}</td><td>${f(x.centre_shift_mm)}</td><td>${f(x.yaw_deg)}</td><td>${x.axis_source || ''}</td><td>${x.n_inliers == null ? '–' : x.n_inliers + '/' + x.n_matches}</td><td>${f(x.arm_turn_deg)}/${f(x.arm_lean_deg)}</td><td style="color:${x.result === 'lifted' ? '#7c7' : '#e55'}">${x.result}${x.reason ? ': ' + x.reason : ''}</td><td>${f(x.grip_at_close)} (taught ${f(x.grip_taught)})</td><td>${verdict}</td></tr>`;
            }).join('') + '</tbody></table>';
    } catch (e) { /* no server */ }
}

async function pgVerdict(index, verdict) {
    try { await pgPost('/api/pregrasp/trials/verdict', {index, verdict}); } catch (e) { pgSet(e.message, true); }
    pgTrialsRefresh(true);
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
        await pgPost('/api/pregrasp/act', {speed: Number(document.getElementById('act-speed').value || 1)});
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

// ---- the guided flow: one step at a time -------------------------------------------------------
// Every 700 ms the server's state is read and reduced to ONE step with ONE primary action. The
// existing handlers do the work; the guide only decides which of them is next.
const apGuide = {timer: null, action: null, busy: false, pendingVerdict: null, lastActSeen: null};

function apGuideShow(step, text, label, action, extraHtml = '') {
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

async function apGuideVerdict(index, verdict) {
    await pgVerdict(index, verdict);
    apGuide.pendingVerdict = null;
    apGuideTick();
}

async function apGuideTick() {
    const el = document.getElementById('ap-guide');
    if (!el || !document.getElementById('tab-approach').classList.contains('active')) return;
    let st, jg;
    try {
        st = await (await fetch('/api/pregrasp/state')).json();
        jg = st.arm_connected ? await (await fetch('/api/jog/state')).json() : {};
    } catch (e) { apGuideShow('offline', 'the server is not answering', null, null); return; }
    const w = st.worker || {}, tr = st.track || {}, act = st.act || {}, demo = st.demo;
    const last = tr.last || {};
    if (typeof deSync === 'function') deSync(st);
    const trackText = tr.on ? `${last.state || 'starting'} at ${(tr.fps || 0).toFixed(0)} fps` : 'not tracking';
    // The act that just finished asks for its verdict once; the trials table keeps the history.
    if (act.ok !== null && act.ok !== undefined && !act.on && apGuide.lastActSeen !== act.reason + act.step + st.test?.at) {
        apGuide.lastActSeen = act.reason + act.step + st.test?.at;
        try {
            const rows = (await (await fetch('/api/pregrasp/trials')).json()).rows || [];
            const i = rows.length - 1;
            if (i >= 0 && !rows[i].verdict) apGuide.pendingVerdict = i;
        } catch (e) { /* no trials yet */ }
    }
    if (!st.camera_live) {
        return apGuideShow('Camera', 'the camera is off', 'Start camera', async () => {
            const sel = document.getElementById('ap-camera');
            if (sel && !sel.value && sel.options.length) sel.selectedIndex = sel.options.length - 1;
            await apCameraToggle();
        });
    }
    if (!w.running) return apGuideShow('Worker', 'the tracking worker is off', 'Start worker', async () => { await pgPost('/api/pregrasp/worker/start'); });
    if (!w.ready) return apGuideShow('Worker', 'the worker is loading its models…', null, null);
    if (st.teach_pending) return apGuideShow('Teach', 'teaching the object…', null, null);
    if (!st.teach) return apGuideShow('Teach', 'click the object in the camera view (a saved demo loads under details → Teach and teaches from its own frame)', document.getElementById('pg-concept').value.trim() ? 'Teach by name' : null, async () => { await pgTeach(); });
    // Tracking starts from the frame the object was taught on (a loaded demo teaches from its own frame) and follows the motion
    // it sees; a jump from that frame to wherever the object lies now has never been measured.
    if (!tr.on) return apGuideShow('Track', `"${st.teach.concept}" is not tracked: put it back where it was taught, start tracking, then move it while the dots follow it`, 'Start tracking', async () => { await pgPost('/api/pregrasp/track/start', pgTrackBody()); });
    // A saved demo without marks is marked before anything else: the arm is not needed for it.
    if (demo && demo.root && !(demo.keypoints || []).length) return apGuideShow('Mark', `mark the pre-grasp and the grasp in "${demo.name}"`, 'Edit demo', async () => { apDetailsToggle(true); apSub('demo'); });
    if (!st.arm_connected) {
        return apGuideShow('Arm', `tracking "${st.teach.concept}" (${trackText}); the arm is not connected`, 'Connect arm', async () => {
            const prof = document.getElementById('jog-profile'); if (prof && [...prof.options].some(o => o.value === 'white')) prof.value = 'white';
            const arm = document.getElementById('jog-arm'); if (arm) arm.value = 'left';
            await jogToggle();
        });
    }
    if (act.on) return apGuideShow('Act', `acting: ${act.step} ${(100 * (act.progress || 0)).toFixed(0)}%`, 'Stop', async () => { await actStop(); });
    if (apGuide.pendingVerdict !== null) {
        const i = apGuide.pendingVerdict;
        return apGuideShow('Result', `the act ended: ${act.ok ? 'done' : (act.reason || 'aborted')}. What happened?`, null, null,
            ['lifted', 'missed', 'collided'].map(v => `<button class="btn-small" onclick="apGuideVerdict(${i}, '${v}')">${v}</button>`).join(''));
    }
    const leader = jg.mode === 'leader';
    if (st.recording) return apGuideShow('Record', `recording the demo (${jg.record_n || 0} samples); do the whole grasp with the leader, then`, 'Stop recording', async () => { await demoRecordToggle(); });
    if (demo && !demo.root) return apGuideShow('Save', `recorded ${demo.n} samples; name it on the right if you like, then`, 'Save demo', async () => { await demoSave(); });
    if (leader && demo) return apGuideShow('Hand back', 'the demo is saved; raise the arm clear of the tray with the leader, then', 'Hand the arm back', async () => { await jogLeaderToggle(); });
    if (leader) return apGuideShow('Record', 'the leader drives the arm; bring it above the object with the gripper open, then', 'Record demo', async () => { await demoRecordToggle(); });
    if (!demo) return apGuideShow('Leader', 'hold the leader arm above the tray with its gripper open before pressing: the follower first moves to the leader\'s pose', 'Hand to leader', async () => {
        const l = document.getElementById('jog-leader'); if (l && !l.value) l.value = 'blue';
        await jogLeaderToggle();
    });
    // One button: Act. A new demo for the same object is recorded from the Teach panel under details.
    const marks = (demo && demo.keypoints) || [];
    const npre = marks.filter(k => k.kind === 'pregrasp').length, grasp = marks.some(k => k.kind === 'grasp_end');
    const span = npre ? ` the arm follows it to ${npre === 1 ? 'the pre-grasp' : npre + ' pre-grasp points'}${grasp ? ', waits for it to hold still, then replays the grasp' : ' and stops'};` : '';
    const refused = act.ok === false && act.reason && !act.on ? `last act: ${act.reason}. ` : '';
    return apGuideShow('Act', `${refused}move and turn "${st.teach.concept}" while it is tracked (${trackText});${span} then`, 'Act', async () => { await actGo(); },
        `<label style="color:#888;">speed <input id="ap-guide-speed" type="number" step="0.25" min="0.1" max="2" value="${document.getElementById('act-speed').value || 0.5}" style="width:52px;" onchange="document.getElementById('act-speed').value=this.value"></label>`);
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
const de = {curve: null, i: 0, playing: false, timer: null, kps: [], loadedFor: null, frameBusy: false, framePending: null, reachTimer: null, dirty: false};
const DE_PRE = '#ffaa00', DE_GRASP = '#00c8ff';

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
        deStatus(c.keypoints.length ? 'saved with the demo' : 'nothing marked yet: scrub to a moment and add it');
        const sl = document.getElementById('de-slider'); sl.max = c.n - 1; sl.value = de.i;
        if (!c.has_frames) document.getElementById('de-frame').removeAttribute('src');
        deRenderList(); deSeek(de.i, true);
    } catch (e) { deStatus(e.message, true); }
}

function deSync(st) {
    // Called by the guide's poll: a new or reloaded demo replaces the editor's copy when the editor is open.
    if (!deVisible() || !st.demo) return;
    if (deKey(st.demo.name, st.demo.n, st.demo.keypoints || []) !== de.loadedFor && !de.dirty) deLoad();
}

function deSeek(i, force = false) {
    const c = de.curve;
    if (!c) return;
    de.i = Math.max(0, Math.min(c.n - 1, i));
    document.getElementById('de-slider').value = de.i;
    document.getElementById('de-time').textContent = `${c.t[de.i].toFixed(2)} s · sample ${de.i + 1}/${c.n}${c.seen[de.i] ? '' : ' · object hidden'}`;
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
    const t = de.curve.t[de.i], pre = dePre(), end = deEnd();
    if (kind === 'grasp_end') {
        if (!pre.length) return deStatus('add a pre-grasp first: the grasp is replayed from the last one', true);
        if (t <= pre[pre.length - 1].t) return deStatus('the grasp ends after the last pre-grasp: scrub past it first', true);
        de.kps = de.kps.filter(k => k.kind !== 'grasp_end');
    } else {
        if (end && t >= end.t) return deStatus('a pre-grasp comes before the grasp end', true);
        if (pre.some(k => k.t === t)) return deStatus('there is a pre-grasp at this moment already', true);
    }
    de.kps.push({t, kind});
    de.kps.sort((a, b) => a.t - b.t);
    de.dirty = true;
    deStatus('not saved yet: press Save when the list is right');
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
            `<td style="${cell} white-space:nowrap;">grasp</td><td style="${cell} color:#aaa;">once the object holds still, replayed exactly as shown, turned with the object; the arm holds at the end</td>${del(i)}</tr>`);
    } else {
        rows.push(`<tr style="border-top:1px solid #333;"><td></td><td colspan="4" style="color:#777; padding:4px 0;">no grasp end: the arm stops at pre-grasp ${pre.length}</td></tr>`);
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
    ctx.fillStyle = '#2a1a1a'; // the object hidden from the camera
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
    const c = de.curve; if (!c || !c.uv || !c.image_size) return;
    const sx = w / c.image_size[0], sy = h / c.image_size[1];
    const P = i => c.uv[i] ? [c.uv[i][0] * sx, c.uv[i][1] * sy] : null;
    const pre = dePre(), end = deEnd();
    const g0 = pre.length && end ? deIndexAt(pre[pre.length - 1].t) : -1, g1 = end ? deIndexAt(end.t) : -1;
    for (let i = 1; i < c.n; i++) { // the recorded path: cyan where the grasp replays it, dim elsewhere
        const a = P(i - 1), b = P(i); if (!a || !b) continue;
        const grasp = g0 >= 0 && i - 1 >= g0 && i <= g1;
        ctx.strokeStyle = grasp ? DE_GRASP : 'rgba(180,180,180,0.4)'; ctx.lineWidth = grasp ? 2.5 : 1;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; // the straight lines between pre-grasps
    for (let n = 1; n < pre.length; n++) {
        const a = P(deIndexAt(pre[n - 1].t)), b = P(deIndexAt(pre[n].t)); if (!a || !b) continue;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.font = '12px sans-serif';
    const ring = (p, col, label) => { if (!p) return; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 2 * Math.PI); ctx.stroke(); ctx.fillStyle = col; ctx.fillText(label, p[0] + 10, p[1] - 6); };
    pre.forEach((k, n) => ring(P(deIndexAt(k.t)), DE_PRE, `pre-grasp ${n + 1}`));
    if (end) ring(P(g1), DE_GRASP, 'grasp end');
    const q = P(de.i);
    if (q) { ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(q[0], q[1], 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 2 * Math.PI); ctx.fill(); }
}

async function deSave() {
    if (!de.curve) return deStatus('no demo', true);
    try {
        const d = await pgPost('/api/pregrasp/demo/keypoints', {keypoints: de.kps.map(k => ({t: k.t, kind: k.kind}))});
        de.dirty = false; de.kps = d.keypoints.map(k => ({...k})); de.curve.keypoints = d.keypoints.map(k => ({...k}));
        de.loadedFor = deKey(d.name, d.n, d.keypoints);
        deStatus(d.keypoints.length ? `saved${d.root ? ' beside the demo' : ' (save the demo to keep it)'}` : 'cleared');
        deRenderList(); deDrawStrip(); deDrawOverlay(); deReach();
        if (typeof apGuideTick === 'function') apGuideTick();
    } catch (e) { deStatus(e.message, true); }
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
        el.innerHTML = 'as the object lies now: ' + d.marks.map(m => `<span style="color:${m.ok ? '#6c6' : '#e55'};">${m.label} ${m.ok ? '&#10003;' : '&#10007; ' + m.residual_mm.toFixed(0) + ' mm short'}</span>`).join(' · ') +
            `<span style="color:#777;"> · ${d.summary.seconds.toFixed(1)} s of motion at speed 1 · tracker tilt ignored ${d.summary.tilt_ignored_deg.toFixed(1)}°${d.ok ? '' : ' · ' + d.reason}</span>`;
    } catch (e) { el.textContent = ''; }
}
function deReachStart() { deReachStop(); deReach(); de.reachTimer = setInterval(deReach, 3000); }
function deReachStop() { if (de.reachTimer) { clearInterval(de.reachTimer); de.reachTimer = null; } }

(function deWire() {
    const strip = document.getElementById('de-strip');
    if (!strip) return;
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
})();
