/* LeRobot GUI — Preprocess tab.
 *
 * Thin frontend for /api/dataset-preparation: pick a source dataset, launch
 * an HVLA 224x224 H.264 preparation job, poll it once a second until it
 * reaches a terminal state. All heavy lifting lives in
 * lerobot.datasets.hvla_preparation; this file only renders job state.
 */

let _prepJobId = null;
let _prepPollTimer = null;

function _esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

async function preprocessTabInit() {
    await refreshPrepSources();
}

async function refreshPrepSources() {
    const sel = document.getElementById('prep-source-select');
    if (!sel) return;
    try {
        const res = await fetch('/api/datasets/sources');
        if (!res.ok) throw new Error('Failed to list sources');
        const sources = await res.json();
        const options = [];
        for (const source of sources) {
            const enc = encodeURIComponent(source.path);
            const dsRes = await fetch(`/api/datasets/sources/${enc}/datasets`);
            if (!dsRes.ok) continue;
            const datasets = await dsRes.json();
            for (const d of datasets) {
                options.push(
                    `<option value="${_esc(d.root)}" data-name="${_esc(d.name)}">` +
                    `${_esc(d.name)} (${d.total_episodes} ep, ${d.fps} fps)</option>`
                );
            }
        }
        sel.innerHTML = options.length
            ? options.join('')
            : '<option value="">No datasets found</option>';
        prepSourceChanged();
    } catch (e) {
        showToast('Preprocess', e.message || 'Failed to load datasets', 'error');
    }
}

function prepSourceChanged() {
    const sel = document.getElementById('prep-source-select');
    const out = document.getElementById('prep-output-repo');
    if (!sel || !out) return;
    const name = sel.selectedOptions[0]?.dataset.name || '';
    out.value = name ? `${name}_hvla224` : '';
    _prepUpdateStartButton();
}

function _prepUpdateStartButton() {
    const sel = document.getElementById('prep-source-select');
    const button = document.getElementById('prep-start-btn');
    if (!button) return;
    button.disabled = !sel?.value || _prepJobId !== null;
}

async function startPreprocess() {
    const sel = document.getElementById('prep-source-select');
    const out = document.getElementById('prep-output-repo');
    const sourceRoot = sel?.value;
    const sourceName = sel?.selectedOptions[0]?.dataset.name || '';
    if (!sourceRoot) {
        showToast('Preprocess', 'Pick a source dataset first.', 'warning');
        return;
    }
    const body = {
        source_repo_id: sourceName,
        source_root: sourceRoot,
        output_repo_id: out.value.trim() || null,
    };
    let res;
    try {
        res = await fetch('/api/dataset-preparation/hvla', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    } catch (e) {
        showToast('Preprocess', 'Server unreachable', 'error');
        return;
    }
    if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: res.statusText }));
        showToast('Preprocess', data.detail || 'Failed to start preparation', 'error');
        return;
    }
    const data = await res.json();
    _prepJobId = data.job_id;
    document.getElementById('prep-empty').style.display = 'none';
    document.getElementById('prep-status-panel').style.display = 'block';
    document.getElementById('prep-start-btn').disabled = true;
    _prepRender({ status: 'pending', done: 0, total: 0, current_file: '' });
    _prepPoll();
}

async function _prepPoll() {
    if (!_prepJobId) return;
    try {
        const res = await fetch(`/api/dataset-preparation/jobs/${_prepJobId}`);
        if (res.ok) {
            const job = await res.json();
            _prepRender(job);
            if (job.status === 'complete') {
                showToast('Preprocess', `Done: ${job.output_repo_id}`, 'success');
                _prepFinish();
                if (typeof window.refreshExpandedSources === 'function') {
                    window.refreshExpandedSources();
                }
                // The training panel caches its dataset list page-locally;
                // refresh it too or the new derivative only shows up after a
                // full page reload.
                if (typeof window.trainingLoadDatasets === 'function') {
                    window.trainingLoadDatasets();
                }
                return;
            }
            if (job.status === 'failed') {
                showToast('Preprocess failed', job.error || 'Unknown error', 'error', 0);
                _prepFinish();
                return;
            }
        }
    } catch (e) {
        // Network blip — keep polling.
    }
    _prepPollTimer = setTimeout(_prepPoll, 1000);
}

function _prepFinish() {
    _prepJobId = null;
    if (_prepPollTimer) clearTimeout(_prepPollTimer);
    _prepPollTimer = null;
    _prepUpdateStartButton();
}

function _prepRender(job) {
    const line = document.getElementById('prep-status-line');
    const bar = document.getElementById('prep-progress-bar');
    const cur = document.getElementById('prep-current-file');
    const out = document.getElementById('prep-output-path');
    const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
    line.textContent = `${job.status} — ${job.done}/${job.total} files (${pct}%)`;
    bar.style.width = `${pct}%`;
    cur.textContent = job.current_file || '';
    out.textContent = job.output_root ? `Output: ${job.output_root}` : '';
}
