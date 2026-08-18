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
    document.getElementById('ss-live-btn').style.display = info.live ? '' : 'none';
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
