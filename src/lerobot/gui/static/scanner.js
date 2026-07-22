// 3D Scanner tab: detect a camera, capture a frame (or use the static test image), run the
// SAM3 + SAM 3D pipeline, and show the reconstructed mesh in the scanner viewer (iframe).
// The output mesh feeds FoundationPose (next step). Multi-view = capture at varying angles + re-scan
// until the mesh looks good (single-view SAM 3D per scan today; true fusion is a follow-up).

async function scannerDetectCameras() {
    const sel = document.getElementById('scanner-camera');
    sel.innerHTML = '<option>detecting…</option>';
    try {
        const cams = await (await fetch('/api/robot/detect-cameras', { method: 'POST' })).json();
        sel.innerHTML = '';
        (cams || []).forEach((c) => {
            const o = document.createElement('option');
            o.value = c.id;
            o.textContent = `${c.name || c.type || 'camera'} (${c.id})`;
            sel.appendChild(o);
        });
        if (!sel.options.length) sel.innerHTML = '<option value="">no cameras found</option>';
    } catch (e) {
        sel.innerHTML = '<option value="">detect failed</option>';
    }
}

async function scannerCapture() {
    const id = document.getElementById('scanner-camera').value;
    const st = document.getElementById('scanner-status');
    if (!id) { st.textContent = 'pick a camera first (Detect)'; return; }
    st.textContent = 'capturing…';
    try {
        const r = await (await fetch('/api/run/scan3d/capture', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: id }),
        })).json();
        if (r.ok) {
            document.getElementById('scanner-preview').src = '/api/run/scan3d/capture/preview?t=' + Date.now();
            st.textContent = 'captured frame ' + (r.shape ? r.shape.join('×') : '');
        } else {
            st.textContent = 'capture failed: ' + (r.error || '');
        }
    } catch (e) {
        st.textContent = 'capture error: ' + e;
    }
}

async function scannerAddToSet() {
    const id = document.getElementById('scanner-camera').value;
    const st = document.getElementById('scanner-status');
    if (!id) { st.textContent = 'pick a camera first (Detect)'; return; }
    try {
        const r = await (await fetch('/api/run/scan3d/captures', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: id }),
        })).json();
        if (r.ok) {
            scannerRenderSet(r.captures);
            st.textContent = `added ${r.name} to the capture set`;
        } else {
            st.textContent = 'add failed: ' + (r.error || '');
        }
    } catch (e) {
        st.textContent = 'add error: ' + e;
    }
}

async function scannerClearSet() {
    const r = await (await fetch('/api/run/scan3d/captures', { method: 'DELETE' })).json();
    scannerRenderSet(r.captures || []);
}

async function scannerRefreshSet() {
    try {
        const r = await (await fetch('/api/run/scan3d/captures')).json();
        scannerRenderSet(r.captures || []);
    } catch (e) { /* GUI booting; the next action refreshes */ }
}

function scannerRenderSet(names) {
    const grid = document.getElementById('scanner-set');
    document.getElementById('scanner-set-count').textContent = names.length ? `(${names.length})` : '';
    grid.innerHTML = '';
    names.forEach((n) => {
        const img = document.createElement('img');
        img.src = `/api/run/scan3d/captures/${n}?t=${Date.now()}`;
        img.title = n;
        img.style.cssText = 'width:100%; aspect-ratio:4/3; object-fit:cover; border:1px solid var(--border,#333); border-radius:3px; background:#111;';
        grid.appendChild(img);
    });
}

document.addEventListener('DOMContentLoaded', scannerRefreshSet);

function scannerScan() {
    _scannerScan({ prompt: document.getElementById('scanner-prompt').value });
}

function scannerScanStatic() {
    _scannerScan({ prompt: document.getElementById('scanner-prompt').value, use_static: true });
}

async function _scannerScan(body) {
    const btn = document.getElementById('scanner-scan-btn');
    const st = document.getElementById('scanner-status');
    btn.disabled = true;
    st.textContent = 'starting scan (SAM3 segment → SAM 3D geometry, ~30–90 s)…';
    try {
        await fetch('/api/run/scan3d/scan', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
    } catch (e) {
        st.textContent = 'failed to start: ' + e; btn.disabled = false; return;
    }
    const poll = setInterval(async () => {
        try {
            const s = await (await fetch('/api/run/scan3d/scan/status')).json();
            if (s.log) st.textContent = s.log.slice(-1400);
            if (s.done && !s.running) {
                clearInterval(poll);
                btn.disabled = false;
                st.textContent += s.error ? ('\n\n✗ ERROR: ' + s.error) : '\n\n✓ scan complete — mesh in the viewer →';
            }
        } catch (e) { /* keep polling */ }
    }, 1000);
}
