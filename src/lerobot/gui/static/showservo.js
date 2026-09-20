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
    jogRefreshProfiles();
    calibRefresh();
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
            tile.src = '/static/urdf_viz.html?mode=jog&v=4';
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
        jogStatus(`gap ${st.err_mm.toFixed(1)} mm / ${st.err_deg.toFixed(1)}°` +
                  (hottest !== null ? ` · hottest motor ${hottest} °C` : '') +
                  (st.halted ? ` · FROZEN: ${st.reason}` : ''), !!st.halted);
    } catch (e) { /* transient */ }
}


// ── Touch calibration: fingertip (tool point), then camera to base ───────────
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

async function calibRefresh() {
    let st;
    try { st = await (await fetch('/api/calib/state')).json(); } catch (e) { return; }
    const mm = (v) => (v * 1000).toFixed(1);
    // fingertip
    const tool = st.tool, saved = st.saved || {};
    let lines = tool.touches.map((t, i) => {
        const res = tool.result ? ` residual ${mm(tool.result.residuals_m[i])} mm` : '';
        return `#${i + 1} ${t.at}  tip (${t.tip_m.map(mm).join(', ')}) mm${res}`;
    });
    if (tool.result) {
        lines.push(`offset from wrist link (${tool.result.offset_m.map(mm).join(', ')}) mm · rms ${mm(tool.result.rms_m)} · max ${mm(tool.result.max_m)} mm`);
    }
    if (saved.tool_point) lines.push(`saved: (${saved.tool_point.offset_m.map(mm).join(', ')}) mm, rms ${mm(saved.tool_point.rms_m)} mm, ${saved.saved_at}`);
    document.getElementById('calib-tool-list').textContent = lines.join('\n') || (st.arm_connected ? 'no touches yet' : 'connect the jog arm');
    // camera
    const cam = st.camera;
    lines = cam.touches.map((t, i) => {
        const res = cam.result && cam.result.touch_ids.includes(`${t.marker_id}.${t.corner}`)
            ? ` residual ${mm(cam.result.residuals_m[cam.result.touch_ids.indexOf(`${t.marker_id}.${t.corner}`)])} mm` : '';
        const depth = t.cam_depth_m ? `depth z ${t.cam_depth_m[2].toFixed(3)} m` : 'no depth';
        const pnp = t.cam_pnp_m ? ` · pnp z ${t.cam_pnp_m[2].toFixed(3)} m` : '';
        return `marker ${t.marker_id} corner ${t.corner} ${t.at}  ${depth}${pnp}  base (${t.base_m.map(mm).join(', ')}) mm${res}`;
    });
    if (cam.result) {
        lines.push(`fit (${cam.result.source}, ${cam.result.n} touches): rms ${mm(cam.result.rms_m)} · max ${mm(cam.result.max_m)} mm · similarity scale ${cam.result.scale.toFixed(4)}`);
    }
    if (saved.camera) lines.push(`saved: ${saved.camera.source}, rms ${mm(saved.camera.rms_m)} mm over ${saved.camera.n}, ${saved.saved_at}`);
    document.getElementById('calib-cam-list').textContent = lines.join('\n') || 'no corner touches yet';
    if (st.markers) {
        const sel = document.getElementById('calib-marker');
        const cur = sel.value;
        sel.innerHTML = st.markers.markers.map(m =>
            `<option value="${m.id}">${m.id}${m.depth_ok ? '' : ' (no depth)'}</option>`).join('') || '<option value="">none found</option>';
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    }
}

async function calibTool(action) {
    try {
        const r = await calibPost(`/api/calib/tool/${action}`);
        if (action === 'save') calibSet('calib-tool-status', `saved to ${r.path} — reconnect the jog to use it`);
        else if (action === 'solve') calibSet('calib-tool-status', `rms ${(r.rms_m * 1000).toFixed(1)} mm over ${r.n} touches`);
        else calibSet('calib-tool-status', `${r.n} touch${r.n === 1 ? '' : 'es'}`);
    } catch (e) { calibSet('calib-tool-status', e.message, true); }
    calibRefresh();
}

async function calibDetect() {
    const side = document.getElementById('calib-side').value;
    try {
        const r = await calibPost('/api/calib/markers', {
            dictionary: document.getElementById('calib-dict').value, side_mm: side === '' ? null : Number(side),
        });
        calibSet('calib-cam-status', `${r.n} marker${r.n === 1 ? '' : 's'}: ${r.ids.join(', ') || 'none'} (${r.at})`);
        const img = document.getElementById('calib-markers');
        img.src = `/api/calib/markers.jpg?t=${Date.now()}`;
        img.style.display = '';
    } catch (e) { calibSet('calib-cam-status', e.message, true); }
    calibRefresh();
}

async function calibCamera(action) {
    const body = {};
    if (action === 'touch') {
        body.marker_id = Number(document.getElementById('calib-marker').value);
        body.corner = Number(document.getElementById('calib-corner').value);
        if (Number.isNaN(body.marker_id) || document.getElementById('calib-marker').value === '') {
            calibSet('calib-cam-status', 'detect markers and pick one first', true); return;
        }
    }
    if (action === 'solve') body.source = document.getElementById('calib-source').value;
    try {
        const r = await calibPost(`/api/calib/camera/${action}`, body);
        if (action === 'save') calibSet('calib-cam-status', `saved to ${r.path}`);
        else if (action === 'solve') calibSet('calib-cam-status', `rms ${(r.rms_m * 1000).toFixed(1)} mm, max ${(r.max_m * 1000).toFixed(1)} mm, scale ${r.scale.toFixed(4)}`);
        else calibSet('calib-cam-status', `${r.n} corner touch${r.n === 1 ? '' : 'es'}`);
    } catch (e) { calibSet('calib-cam-status', e.message, true); }
    calibRefresh();
}
