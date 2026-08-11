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
    ssTeach = new Set(ssScenes.length ? [0] : []);
    document.getElementById('ss-setup').style.display = 'none';
    document.getElementById('ss-session').style.display = '';
    document.getElementById('ss-session-name').textContent = info.name + (info.live ? ' (live)' : ' (reopened)');
    document.getElementById('ss-capture-btn').style.display = info.live ? '' : 'none';
    document.getElementById('ss-preview-wrap').style.display = info.live ? '' : 'none';
    ssSetStatus(info.live ? 'camera live' : 'session reopened — bind only');
    ssRenderScenes();
    if (info.live) {
        ssPreviewTimer = setInterval(() => {
            document.getElementById('ss-preview').src = '/api/showservo/preview.jpg?t=' + Date.now();
        }, 250);
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
        if (ssScenes.length === 1) ssTeach.add(0);
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
}
