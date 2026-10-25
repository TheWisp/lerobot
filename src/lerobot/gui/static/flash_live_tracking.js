/* Perception-only FLASH debugger. Images and geometry arrive in one frame-tagged packet. */
let flashTrackingTimer = null;
let flashTrackingSnapshot = null;
let flashTrackingDefaults = null;
let flashTrackingBusy = false;
let flashTrackingYaw = 0, flashTrackingPitch = 0, flashTrackingZoom = 1;

function flashTrackingForm() {
    return `<section id="flash-tracking-setup">
        <h2>Live tracking</h2><p class="form-hint">Camera only · no robot motion</p>
        <label for="flash-tracking-camera">RGB-D camera</label>
        <select id="flash-tracking-camera"><option value="">Auto (one RealSense connected)</option></select>
        <button type="button" class="btn-small secondary" onclick="flashTrackingFindCameras()">Find cameras</button>
        <button type="button" class="flash-live-entry" id="flash-tracking-open" onclick="flashTrackingStart()" disabled>Open camera</button>
        <label for="flash-tracking-name">Object name</label>
        <input id="flash-tracking-name" placeholder="Describe one object" autocomplete="off">
        <button type="button" class="flash-live-entry" onclick="flashTrackingControl('select')">Track by name</button>
        <button type="button" class="flash-live-entry" onclick="flashTrackingControl('freeze')">Freeze image & select by click</button>
        <p class="form-hint">Each selection starts a new fixed surface reference. Keep the object visible while it initializes.</p>
        <button type="button" class="btn-small secondary" onclick="flashTrackingControl('clear')">Clear tracking</button>
        <button type="button" class="btn-small secondary" onclick="flashTrackingStop()">Close camera</button>
        <details><summary>Camera & tracking parameters</summary><p class="form-hint">Applied when opening the camera. Defaults reproduce the surface experiment.</p><textarea id="flash-tracking-config" rows="15" spellcheck="false"></textarea></details>
        <button type="button" class="btn-small secondary" onclick="flashTrackingSave()">Save diagnostic snapshot</button>
        <div id="flash-tracking-message" role="status"></div>
    </section>`;
}

async function flashTrackingApi(path, body) {
    const response = await fetch('/api/run/flash/tracking/' + path, body === undefined ? {} : {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    return data;
}

function flashTrackingError(error) {
    document.getElementById('flash-tracking-message').textContent = error.message;
}

async function showFlashTracking() {
    clearInterval(flashReplayTimer); flashReplayTimer = null;
    stopFlashTelemetry();
    flashUiMode = 'tracking';
    _flashUpdateWorkspace();
    if (!document.getElementById('flash-surface-view')) {
        const view = document.createElement('section');
        view.id = 'flash-surface-view';
        view.innerHTML = `<header><strong>Live surface tracking</strong><span id="flash-tracking-status">Open a camera to begin</span></header>
            <div class="flash-surface-grid">
                <figure><figcaption>Live camera</figcaption><canvas id="flash-surface-preview"></canvas></figure>
                <figure><figcaption id="flash-surface-caption">Fitted pose · source image</figcaption><canvas id="flash-surface-estimate" aria-label="Frozen image for object selection"></canvas></figure>
                <figure><figcaption>3D surface · drag to orbit, scroll to zoom</figcaption><canvas id="flash-surface-cloud"></canvas></figure>
                <figure class="flash-surface-diagnostics"><figcaption>Fit diagnostics</figcaption><p id="flash-surface-metrics"></p><p>Surface agreement is not calibrated confidence. This view uses vision only.</p></figure>
            </div>
            <div class="flash-surface-legend"><label><input type="checkbox" id="flash-surface-mask" checked onchange="flashTrackingRender()">Cyan: observed mask / depth</label><label><input type="checkbox" id="flash-surface-reference" checked onchange="flashTrackingRender()">White: fitted reference</label><label><input type="checkbox" id="flash-surface-axes" checked onchange="flashTrackingRender()">A / B / C and axes</label><button type="button" onclick="flashTrackingYaw=flashTrackingPitch=0;flashTrackingZoom=1;flashTrackingRender()">Reset 3D view</button></div>
            <p class="form-hint">A / B / C and the axes stay attached to the initial partial surface. They do not mark the object's center of mass.</p>`;
        document.getElementById('run-main').prepend(view);
        document.getElementById('flash-surface-estimate').onclick = event => {
            const estimate = flashTrackingSnapshot?.estimate;
            if (estimate?.status !== 'frozen') return;
            const canvas = event.currentTarget, rect = canvas.getBoundingClientRect();
            if (canvas.dataset.frameId !== String(estimate.frame.frame_id)) return;
            flashTrackingControl('select', {
                point: [(event.clientX - rect.left) * canvas.width / rect.width,
                        (event.clientY - rect.top) * canvas.height / rect.height],
                frame_id: estimate.frame.frame_id,
            });
        };
        const cloud = document.getElementById('flash-surface-cloud');
        cloud.onpointerdown = event => cloud.setPointerCapture(event.pointerId);
        cloud.onpointermove = event => {
            if (!cloud.hasPointerCapture(event.pointerId)) return;
            flashTrackingYaw += event.movementX * 0.01;
            flashTrackingPitch += event.movementY * 0.01;
            flashTrackingDrawCloud();
        };
        cloud.onwheel = event => {
            event.preventDefault();
            flashTrackingZoom = Math.max(0.1, Math.min(10, flashTrackingZoom * Math.exp(-event.deltaY * 0.001)));
            flashTrackingDrawCloud();
        };
    }
    try {
        flashTrackingDefaults ??= await flashTrackingApi('defaults');
        const config = document.getElementById('flash-tracking-config');
        if (!config.value) config.value = JSON.stringify(flashTrackingDefaults, null, 2);
        document.getElementById('flash-tracking-open').disabled = _isRunning;
    } catch (error) { flashTrackingError(error); }
    if (!flashTrackingTimer) flashTrackingPoll();
}

async function flashTrackingFindCameras() {
    try {
        const data = await flashTrackingApi('cameras', {});
        const select = document.getElementById('flash-tracking-camera');
        select.replaceChildren(new Option('Auto (one RealSense connected)', ''));
        for (const camera of data.cameras) select.add(new Option(`${camera.name} · ${camera.id}`, camera.id));
        if (data.cameras.length === 1) select.value = data.cameras[0].id;
        document.getElementById('flash-tracking-message').textContent = `${data.cameras.length} RGB-D camera(s) found`;
    } catch (error) { flashTrackingError(error); }
}

async function flashTrackingStart() {
    if (flashTrackingBusy) return;
    flashTrackingBusy = true;
    try {
        const config = JSON.parse(document.getElementById('flash-tracking-config').value);
        config.serial = document.getElementById('flash-tracking-camera').value;
        await flashTrackingApi('start', config);
        flashTrackingSnapshot = null;
        document.getElementById('flash-tracking-message').textContent = 'Opening RGB-D camera…';
        await pollRunStatus();
    } catch (error) { flashTrackingError(error); }
    finally { flashTrackingBusy = false; }
}

async function flashTrackingControl(action, extra = {}) {
    try {
        await flashTrackingApi('control', {action, id: crypto.randomUUID(),
            prompt: document.getElementById('flash-tracking-name').value.trim(), ...extra});
        document.getElementById('flash-tracking-message').textContent = action === 'freeze' ? 'Freezing an image for selection…' : 'Processing selection…';
    } catch (error) { flashTrackingError(error); }
}

async function flashTrackingStop() {
    try { await flashTrackingApi('stop', {}); await pollRunStatus(); }
    catch (error) { flashTrackingError(error); }
}

async function flashTrackingPoll() {
    flashTrackingTimer = null;
    if (flashUiMode !== 'tracking' || selectedWorkflow !== 'flash') return;
    try {
        flashTrackingSnapshot = await flashTrackingApi('state');
        flashTrackingRender();
    } catch (error) { flashTrackingError(error); }
    flashTrackingTimer = setTimeout(flashTrackingPoll, 200);
}

function flashTrackingPaint(id, frame, draw) {
    const canvas = document.getElementById(id);
    const generation = (canvas._generation || 0) + 1;
    canvas._generation = generation;
    if (!frame) {
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        delete canvas.dataset.frameId;
        return;
    }
    const image = new Image();
    image.onload = () => {
        if (canvas._generation !== generation) return;
        canvas.width = frame.width; canvas.height = frame.height;
        canvas.style.aspectRatio = `${frame.width} / ${frame.height}`;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        canvas.dataset.frameId = String(frame.frame_id);
        if (draw) draw(ctx);
    };
    image.src = frame.image;
}

function flashTrackingRender() {
    const data = flashTrackingSnapshot;
    if (!data || !document.getElementById('flash-surface-view')) return;
    const estimate = data.estimate || {};
    const config = document.getElementById('flash-tracking-config');
    if (data.config && config.dataset.session !== data.session_id) {
        config.value = JSON.stringify(data.config, null, 2);
        config.dataset.session = data.session_id;
        const camera = document.getElementById('flash-tracking-camera');
        const serial = data.config.serial || data.preview?.camera_serial;
        if (serial && !Array.from(camera.options).some(option => option.value === serial)) camera.add(new Option(serial, serial));
        if (serial) camera.value = serial;
    }
    const name = document.getElementById('flash-tracking-name');
    if (estimate.object_name && !name.value && document.activeElement !== name) name.value = estimate.object_name;
    const states = {opening_camera: 'Opening camera', preview: 'Choose an object', frozen: 'Click the object in the frozen image', loading_models: 'Loading tracking models', reference: 'Reference initialized', fitted: 'Experimental pose fit', fitted_without_rgb_motion: 'Surface fit; RGB motion unavailable', no_mask: 'Mask missing — no pose', insufficient_depth: 'Insufficient depth — no pose', error: 'Tracking error'};
    document.getElementById('flash-tracking-status').textContent = data.running ? (states[estimate.status] || 'Starting…') : 'Camera closed';
    document.getElementById('flash-tracking-message').textContent = estimate.message || states[estimate.status] || '';
    document.getElementById('flash-tracking-open').disabled = data.running || !flashTrackingDefaults;
    flashTrackingPaint('flash-surface-preview', data.preview);
    const visible = id => document.getElementById(id).checked;
    flashTrackingPaint('flash-surface-estimate', estimate.frame, ctx => {
        ctx.lineWidth = 2;
        if (visible('flash-surface-mask')) {
            ctx.strokeStyle = '#42e4ec';
            for (const points of estimate.contours || []) {
                ctx.beginPath(); points.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath(); ctx.stroke();
            }
        }
        if (visible('flash-surface-reference')) {
            ctx.fillStyle = 'white';
            for (const [x,y] of estimate.reference_px || []) ctx.fillRect(x-1,y-1,2,2);
        }
        if (visible('flash-surface-axes') && estimate.anchors_px) flashTrackingDrawAxes(ctx, estimate.anchors_px, estimate.axes_px);
    });
    document.getElementById('flash-surface-estimate').style.cursor = estimate.status === 'frozen' ? 'crosshair' : 'default';
    const age = estimate.frame ? Math.max(0, (data.server_time_s - estimate.frame.captured_s) * 1000) : null;
    document.getElementById('flash-surface-caption').textContent = estimate.status === 'frozen' ? `Frozen frame ${estimate.frame.frame_id} · click the object` : `Fitted pose · frame ${estimate.frame?.frame_id ?? '—'} · ${age === null ? '—' : Math.round(age) + ' ms old'}`;
    const d = estimate.diagnostics || {}, t = estimate.timings || {};
    const fmt = value => Number.isFinite(value) ? value.toFixed(1) : '—';
    document.getElementById('flash-surface-metrics').textContent = `Pose proposal: ${estimate.accepted === true ? 'accepted' : 'unverified candidate'} · Camera ${fmt(data.preview?.capture_fps)} fps · Processing ${fmt(t.processed_fps)} fps · SAM ${fmt(t.mask_ms)} ms · RGB match ${fmt(d.match_ms)} ms · Surface fit ${fmt(d.fit_ms)} ms\nSurface gap median / p90: ${fmt(d.surface_median_mm)} / ${fmt(d.surface_p90_mm)} mm · ${d.surface_points ?? '—'} measured samples · ${d.inliers ?? '—'} RGB-D matches\nDepth pixels withheld: boundary ${d.boundary_count ?? '—'}, discontinuity ${d.edge_count ?? '—'} · Fixed reference frame ${d.reference_frame_id ?? '—'}`;
    flashTrackingDrawCloud();
}

function flashTrackingDrawAxes(ctx, anchors, axes) {
    ctx.font = 'bold 18px sans-serif'; ctx.lineWidth = 3;
    anchors.forEach(([x,y], i) => {
        ctx.fillStyle = 'white'; ctx.strokeStyle = '#111'; ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fill(); ctx.stroke();
        ctx.strokeText('ABC'[i],x+8,y-8); ctx.fillText('ABC'[i],x+8,y-8);
    });
    axes.forEach(([x,y], i) => {
        ctx.strokeStyle = ['#ff5555','#58e280','#699bff'][i];
        ctx.beginPath(); ctx.moveTo(...anchors[0]); ctx.lineTo(x,y); ctx.stroke();
    });
}

function flashTrackingDrawCloud() {
    const canvas = document.getElementById('flash-surface-cloud');
    if (!canvas) return;
    canvas.width = Math.max(400, canvas.clientWidth); canvas.height = Math.max(300, canvas.clientHeight);
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#0a1320'; ctx.fillRect(0,0,canvas.width,canvas.height);
    const e = flashTrackingSnapshot?.estimate || {};
    const all = [...(e.observed_m || []), ...(e.reference_m || [])];
    if (!all.length) return;
    const center = [0,1,2].map(i => all.reduce((sum,p) => sum+p[i],0)/all.length);
    const radius = Math.max(...all.map(p=> Math.hypot(...p.map((v,i)=>v-center[i]))), 0.001);
    const scale = Math.min(canvas.width,canvas.height) * 0.4 / radius * flashTrackingZoom;
    const project = p => {
        const [x,y,z] = p.map((v,i)=>v-center[i]);
        const X = Math.cos(flashTrackingYaw)*x + Math.sin(flashTrackingYaw)*z;
        const Z = -Math.sin(flashTrackingYaw)*x + Math.cos(flashTrackingYaw)*z;
        const Y = Math.cos(flashTrackingPitch)*y - Math.sin(flashTrackingPitch)*Z;
        return [canvas.width/2+X*scale, canvas.height/2+Y*scale];
    };
    for (const [key,color,toggle] of [['observed_m','#42e4ec','flash-surface-mask'],['reference_m','#ffffff','flash-surface-reference']]) {
        if (!document.getElementById(toggle).checked) continue;
        ctx.fillStyle = color;
        for (const point of e[key] || []) { const [x,y]=project(point); ctx.fillRect(x-1,y-1,2,2); }
    }
    if (e.anchors_m && document.getElementById('flash-surface-axes').checked) flashTrackingDrawAxes(ctx,e.anchors_m.map(project),e.axes_m.map(project));
}

function flashTrackingSave() {
    if (!flashTrackingSnapshot) return;
    const blob = new Blob([JSON.stringify(flashTrackingSnapshot, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href=url; link.download=`flash-tracking-frame-${flashTrackingSnapshot.estimate?.frame?.frame_id ?? 'preview'}.json`; link.click(); URL.revokeObjectURL(url);
}
