/* Perception-only FLASH debugger. Images and geometry arrive in one frame-tagged packet. */
let flashTrackingTimer = null;
let flashTrackingSnapshot = null;
let flashTrackingDefaults = null;
let flashTrackingBusy = false;
let flashTrackingFraming = null;
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
        <details open id="flash-sampling-controls"><summary>Surface sampling</summary>
            <label>Minimum point spacing (mm)<input type="number" id="flash-sampling-min" step="0.1" onchange="flashTrackingSamplingChanged()"></label>
            <label>Maximum point spacing (mm)<input type="number" id="flash-sampling-max" step="0.1" onchange="flashTrackingSamplingChanged()"></label>
            <label>Samples across object<input type="number" id="flash-sampling-across" step="1" onchange="flashTrackingSamplingChanged()"></label>
            <label>Fit sample budget per surface<input type="number" id="flash-sampling-budget" step="1" onchange="flashTrackingSamplingChanged()"></label>
            <p class="form-hint">Smaller objects get finer spacing, limited by available depth pixels. The fit budget limits computation; the full measured cloud still checks distances and the outline. Apply changes by closing and reopening the camera.</p>
        </details>
        <details><summary>Camera & tracking parameters</summary><p class="form-hint">Applied when opening the camera. Shared with demo extraction and live inference.</p><textarea id="flash-tracking-config" rows="15" spellcheck="false" onchange="flashTrackingSamplingSync()"></textarea></details>
        <button type="button" class="btn-small secondary" onclick="flashTrackingSave()">Save diagnostic snapshot</button>
        <div id="flash-tracking-message" role="status"></div>
    </section>`;
}

function flashTrackingSamplingSync() {
    try {
        const p = JSON.parse(document.getElementById('flash-tracking-config').value).vision_pipeline;
        const values = {min:p.observation.min_surface_spacing_m * 1000,
            max:p.observation.surface_spacing_m * 1000, across:p.observation.samples_across_object,
            budget:p.shape.max_fit_points};
        for (const [key,value] of Object.entries(values)) document.getElementById('flash-sampling-' + key).value = value ?? '';
    } catch (error) { flashTrackingError(error); }
}

function flashTrackingSamplingChanged() {
    try {
        const field = document.getElementById('flash-tracking-config'), config = JSON.parse(field.value);
        const value = key => Number(document.getElementById('flash-sampling-' + key).value);
        config.vision_pipeline.observation.min_surface_spacing_m = value('min') / 1000;
        config.vision_pipeline.observation.surface_spacing_m = value('max') / 1000;
        config.vision_pipeline.observation.samples_across_object = value('across');
        config.vision_pipeline.shape.max_fit_points = value('budget');
        field.value = JSON.stringify(config, null, 2);
    } catch (error) { flashTrackingError(error); }
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
                <figure><figcaption id="flash-surface-local-caption">Object close-up · select an object to zoom</figcaption><canvas id="flash-surface-local"></canvas></figure>
                <figure><figcaption id="flash-surface-cloud-caption">3D object view · drag to orbit, scroll to zoom</figcaption><canvas id="flash-surface-cloud"></canvas></figure>
                <figure class="flash-surface-diagnostics"><figcaption>Fit diagnostics</figcaption><p id="flash-surface-metrics"></p><p>Surface agreement is not calibrated confidence. This view uses vision only.</p></figure>
            </div>
            <div class="flash-surface-legend"><label><input type="checkbox" id="flash-surface-mask" checked onchange="flashTrackingRender()">Cyan: observed mask / depth</label><label><input type="checkbox" id="flash-surface-reference" checked onchange="flashTrackingRender()">White: fitted reference</label><label><input type="checkbox" id="flash-surface-axes" checked onchange="flashTrackingRender()">A / B / C and axes</label><label><input type="checkbox" id="flash-surface-follow" onchange="flashTrackingDrawCloud()">Follow scene pose</label><button type="button" onclick="flashTrackingYaw=flashTrackingPitch=0;flashTrackingZoom=1;flashTrackingRender()">Reset 3D view</button></div>
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
        flashTrackingSamplingSync();
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

function flashTrackingPaint(id, frame, draw, crop = null) {
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
        canvas.width = crop ? Math.min(frame.width, 640) : frame.width;
        canvas.height = crop ? Math.round(canvas.width * frame.height / frame.width) : frame.height;
        canvas.style.aspectRatio = `${frame.width} / ${frame.height}`;
        const ctx = canvas.getContext('2d');
        if (crop) {
            ctx.drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, canvas.width, canvas.height);
            ctx.scale(canvas.width / crop.width, canvas.height / crop.height);
            ctx.translate(-crop.x, -crop.y);
        } else ctx.drawImage(image, 0, 0);
        canvas.dataset.frameId = String(frame.frame_id);
        if (draw) draw(ctx);
    };
    image.src = frame.image;
}

function flashTrackingObjectCrop(estimate) {
    const frame = estimate.frame;
    if (!frame) return null;
    const observed = (estimate.contours || []).flat();
    const points = observed.length ? observed : (estimate.reference_px || []);
    let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
    for (const [x,y] of points) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        left = Math.min(left,x); right = Math.max(right,x);
        top = Math.min(top,y); bottom = Math.max(bottom,y);
    }
    if (!Number.isFinite(left)) return null;
    const aspect = frame.width / frame.height;
    // Leave context around the observed object; limit magnification of tiny masks.
    const width = Math.min(frame.width, Math.max(96, (right-left) * 1.5, (bottom-top) * 1.5 * aspect));
    const height = width / aspect;
    return {
        x: Math.max(0, Math.min(frame.width-width, (left+right-width)/2)),
        y: Math.max(0, Math.min(frame.height-height, (top+bottom-height)/2)),
        width, height, source: observed.length ? 'observed' : 'reference',
    };
}

function flashTrackingDrawEstimate(ctx, estimate) {
    const visible = id => document.getElementById(id).checked;
    const scale = ctx.getTransform().a;
    ctx.lineWidth = 2 / scale;
    if (visible('flash-surface-mask')) {
        ctx.strokeStyle = '#42e4ec';
        for (const points of estimate.contours || []) {
            ctx.beginPath(); points.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath(); ctx.stroke();
        }
    }
    if (visible('flash-surface-reference')) {
        ctx.fillStyle = 'white';
        for (const [x,y] of estimate.reference_px || []) ctx.fillRect(x-1/scale,y-1/scale,2/scale,2/scale);
    }
    if (visible('flash-surface-axes') && estimate.anchors_px) flashTrackingDrawAxes(ctx, estimate.anchors_px, estimate.axes_px);
}

function flashTrackingRender() {
    const data = flashTrackingSnapshot;
    if (!data || !document.getElementById('flash-surface-view')) return;
    const estimate = data.estimate || {};
    const config = document.getElementById('flash-tracking-config');
    if (data.config && config.dataset.session !== data.session_id) {
        config.value = JSON.stringify(data.config, null, 2);
        config.dataset.session = data.session_id;
        flashTrackingSamplingSync();
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
    const crop = flashTrackingObjectCrop(estimate);
    flashTrackingPaint('flash-surface-estimate', estimate.frame, ctx => {
        flashTrackingDrawEstimate(ctx, estimate);
        if (crop) {
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.setLineDash([8,6]);
            ctx.strokeRect(crop.x, crop.y, crop.width, crop.height);
        }
    });
    flashTrackingPaint('flash-surface-local', crop ? estimate.frame : null,
        ctx => flashTrackingDrawEstimate(ctx, estimate), crop);
    document.getElementById('flash-surface-local-caption').textContent = crop
        ? `Object close-up · frame ${estimate.frame.frame_id}${crop.source === 'reference' ? ' · no current mask' : ''}`
        : 'Object close-up · select an object to zoom';
    document.getElementById('flash-surface-estimate').style.cursor = estimate.status === 'frozen' ? 'crosshair' : 'default';
    const age = estimate.frame ? Math.max(0, (data.server_time_s - estimate.frame.captured_s) * 1000) : null;
    document.getElementById('flash-surface-caption').textContent = estimate.status === 'frozen' ? `Frozen frame ${estimate.frame.frame_id} · click the object` : `Fitted pose · frame ${estimate.frame?.frame_id ?? '—'} · ${age === null ? '—' : Math.round(age) + ' ms old'}`;
    const d = estimate.diagnostics || {}, t = estimate.timings || {};
    const fmt = value => Number.isFinite(value) ? value.toFixed(1) : '—';
    document.getElementById('flash-surface-metrics').textContent = `Pose proposal: ${estimate.accepted === true ? 'accepted' : 'unverified candidate'} · Camera ${fmt(data.preview?.capture_fps)} fps · Processing ${fmt(t.processed_fps)} fps · SAM ${fmt(t.mask_ms)} ms · RGB match ${fmt(d.match_ms)} ms · Surface fit ${fmt(d.fit_ms)} ms · Prepare ${fmt(d.preprocess_ms)} ms\nSurface gap median / p90: ${fmt(d.surface_median_mm)} / ${fmt(d.surface_p90_mm)} mm · ${d.surface_points ?? '—'} measured samples at ${fmt(d.sample_spacing_mm)} mm · Fit samples ${d.fit_reference_points ?? '—'} / ${d.fit_target_points ?? '—'} · ${d.inliers ?? '—'} RGB-D matches\nRotation this update: RGB ${fmt(d.rgb_rotation_deg)}° · surface correction ${fmt(d.surface_correction_deg)}° · final ${fmt(d.final_rotation_deg)}°\nDepth pixels withheld: boundary ${d.boundary_count ?? '—'}, discontinuity ${d.edge_count ?? '—'} · Fixed reference frame ${d.reference_frame_id ?? '—'}`;
    flashTrackingDrawCloud();
}

function flashTrackingDrawAxes(ctx, anchors, axes) {
    const scale = ctx.getTransform().a;
    ctx.font = `bold ${18/scale}px sans-serif`; ctx.lineWidth = 3/scale;
    anchors.forEach(([x,y], i) => {
        ctx.fillStyle = 'white'; ctx.strokeStyle = '#111'; ctx.beginPath(); ctx.arc(x,y,5/scale,0,Math.PI*2); ctx.fill(); ctx.stroke();
        ctx.strokeText('ABC'[i],x+8/scale,y-8/scale); ctx.fillText('ABC'[i],x+8/scale,y-8/scale);
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
    if (!e.reference_m?.length || !e.camera_from_reference) return;
    const follow = document.getElementById('flash-surface-follow').checked;
    document.getElementById('flash-surface-cloud-caption').textContent = follow
        ? '3D scene pose · drag to orbit, scroll to zoom'
        : '3D object view · pose removed · drag to orbit, scroll to zoom';
    const pose = e.camera_from_reference;
    const local = p => [0,1,2].map(i => [0,1,2].reduce((sum,j) => sum + pose[j][i] * (p[j] - pose[j][3]), 0));
    const identity = `${flashTrackingSnapshot.session_id}:${e.command_id}:${e.diagnostics?.reference_frame_id}`;
    if (flashTrackingFraming?.identity !== identity) {
        const points = e.reference_m.map(local);
        const center = [0,1,2].map(i => points.reduce((sum,p) => sum+p[i],0)/points.length);
        let radius = 0.001;
        for (const p of points) radius = Math.max(radius, Math.hypot(p[0]-center[0],p[1]-center[1],p[2]-center[2]));
        flashTrackingFraming = {identity, center, radius};
    }
    // Framing belongs to the initial reference, never the changing visible patch.
    const {center,radius} = flashTrackingFraming;
    const scale = Math.min(canvas.width,canvas.height) * 0.4 / radius * flashTrackingZoom;
    const project = p => {
        const [x,y,z] = (follow ? p : local(p)).map((v,i)=>v-center[i]);
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
