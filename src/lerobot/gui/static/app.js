/* LeRobot Dataset GUI - Application Logic */

let datasets = {};
let episodes = {};
// Bridge highlight state — repo_id → Set<episode_id>. Wired by
// bridge_consumers.js when the AI calls highlight_in_viewer; renderTree()
// reads from it so the cyan outline survives re-renders. Tab-scope (lost on
// reload), intentionally not persisted.
const bridgeHighlights = new Map();
window.bridgeHighlights = bridgeHighlights;
let expandedNodes = new Set();
let currentDataset = null;
let currentEpisode = null;
let currentFrame = 0;
let totalFrames = 0;
let isPlaying = false;
let playInterval = null;
let fps = 30;
let playbackSpeed = 1;
let isDragging = false;

// One browser-wide choice for every live camera surface. Transport selection
// is a display preference, so it belongs to this global shell rather than to a
// particular Run workflow or robot profile.
const CameraVideoMode = (() => {
    const STORAGE_KEY = 'lerobot.cameraVideoMode';
    const MODES = new Set(['auto', 'full-quality', 'low-bandwidth']);
    let preference = 'auto';
    let recommendedMode = 'full-quality';

    function readStoredPreference() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            return MODES.has(stored) ? stored : 'auto';
        } catch (_) {
            return 'auto';
        }
    }

    function getPreference() {
        return preference;
    }

    function getRecommendedMode() {
        return recommendedMode;
    }

    function getEffectiveMode() {
        return preference === 'auto' ? recommendedMode : preference;
    }

    function snapshot(reason) {
        return {
            preference,
            recommendedMode,
            effectiveMode: getEffectiveMode(),
            reason,
        };
    }

    function renderControl() {
        const select = document.getElementById('camera-video-mode');
        if (select && select.value !== preference) select.value = preference;
        const control = document.getElementById('camera-video-control');
        if (!control) return;
        const effectiveLabel = getEffectiveMode() === 'low-bandwidth' ? 'Low Bandwidth' : 'Full Quality';
        control.dataset.preference = preference;
        control.dataset.effectiveMode = getEffectiveMode();
        control.title = preference === 'auto'
            ? `Choose how live camera images are sent to this browser. Auto currently uses ${effectiveLabel}.`
            : `Live camera images currently use ${effectiveLabel}.`;
    }

    function emitChange(reason) {
        window.dispatchEvent(new CustomEvent('camera-video-mode-change', { detail: snapshot(reason) }));
    }

    function setPreference(value, { persist = true, emit = true } = {}) {
        if (!MODES.has(value)) return false;
        const changed = preference !== value;
        preference = value;
        if (persist) {
            try {
                localStorage.setItem(STORAGE_KEY, value);
            } catch (_) {
                // The current page still honors the choice when storage is disabled.
            }
        }
        renderControl();
        if (emit && changed) emitChange('preference');
        return true;
    }

    async function loadRecommendation() {
        try {
            const response = await fetch('/api/run/camera-video-mode', { cache: 'no-store' });
            if (!response.ok) return;
            const payload = await response.json();
            const next = payload?.recommended_mode;
            if (!MODES.has(next) || next === 'auto' || next === recommendedMode) return;
            recommendedMode = next;
            renderControl();
            if (preference === 'auto') emitChange('recommendation');
        } catch (_) {
            // Full Quality is the compatibility default when recommendation fails.
        }
    }

    function init() {
        preference = readStoredPreference();
        const select = document.getElementById('camera-video-mode');
        if (select) select.addEventListener('change', () => setPreference(select.value));
        renderControl();
        loadRecommendation();
    }

    window.addEventListener('storage', (event) => {
        if (event.key !== STORAGE_KEY) return;
        const next = MODES.has(event.newValue) ? event.newValue : 'auto';
        setPreference(next, { persist: false, emit: true });
    });

    return { init, getPreference, getRecommendedMode, getEffectiveMode, setPreference };
})();
window.CameraVideoMode = CameraVideoMode;

// Editing state
let pendingEdits = [];
let contextMenuTarget = null;  // {datasetId, episodeIndex}

// Trim state
let trimStart = 0;  // Frame index
let trimEnd = 0;    // Frame index (exclusive, like end_frame in API)
let isDraggingTrimLeft = false;
let isDraggingTrimRight = false;
let justFinishedTrimDrag = false;

// Dataset sources (folder browser)
let sources = [];
let sourceDatasets = {};  // {sourcePath: [{name, root, total_episodes, ...}]}
let expandedSources = new Set();
let _sourcesLoaded = false;

// `let` at script-scope is NOT visible on `window` — sibling scripts
// (feature_editing.js, etc.) can read these via bare names but not via
// `window.X`. Mirror the shared state via getters so cross-file readers
// work either way. Read-only — sibling scripts must not assign these.
Object.defineProperties(window, {
    datasets: { get: () => datasets, configurable: true },
    episodes: { get: () => episodes, configurable: true },
    currentDataset: { get: () => currentDataset, configurable: true },
    currentEpisode: { get: () => currentEpisode, configurable: true },
    currentFrame: { get: () => currentFrame, configurable: true },
    totalFrames: { get: () => totalFrames, configurable: true },
    fps: { get: () => fps, configurable: true },
    trimStart: { get: () => trimStart, configurable: true },
    trimEnd: { get: () => trimEnd, configurable: true },
    pendingEdits: { get: () => pendingEdits, configurable: true },
});

async function loadSources() {
    try {
        const res = await fetch('/api/datasets/sources');
        if (!res.ok) return;
        sources = await res.json();
        // Restore expansion state
        expandedSources.clear();
        for (const s of sources) {
            if (s.expanded) expandedSources.add(s.path);
        }
        renderSources();
        // Scan expanded sources
        for (const s of sources) {
            if (s.expanded) {
                scanSource(s.path);
            }
        }
        _sourcesLoaded = true;
    } catch (e) {
        console.error('Failed to load sources:', e);
    }
}

async function scanSource(sourcePath) {
    const container = document.getElementById(`source-children-${_sourceId(sourcePath)}`);
    if (container) container.innerHTML = '<div class="source-loading">Scanning...</div>';
    try {
        const res = await fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}/datasets`);
        if (!res.ok) throw new Error(await res.text());
        sourceDatasets[sourcePath] = await res.json();
        renderSources();
    } catch (e) {
        console.error(`Failed to scan source ${sourcePath}:`, e);
        if (container) container.innerHTML = '<div class="source-empty">Scan failed</div>';
    }
}

function _sourceId(path) {
    // Create a safe DOM id from a path
    return path.replace(/[^a-zA-Z0-9]/g, '_');
}

async function toggleSource(sourcePath) {
    if (expandedSources.has(sourcePath)) {
        expandedSources.delete(sourcePath);
    } else {
        expandedSources.add(sourcePath);
        // Scan if not yet loaded
        if (!sourceDatasets[sourcePath]) {
            scanSource(sourcePath);
        }
    }
    // Persist expansion state
    fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}/expanded?expanded=${expandedSources.has(sourcePath)}`, { method: 'PUT' });
    renderSources();
}

async function addSource() {
    const path = prompt('Enter folder path to scan for datasets:');
    if (!path) return;
    try {
        const res = await fetch('/api/datasets/sources', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({ detail: 'Failed to add source' }));
            throw new Error(data.detail || 'Failed to add source');
        }
        await loadSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

async function removeSource(sourcePath, e) {
    e.stopPropagation();
    if (!confirm(`Remove source folder?\n${sourcePath}`)) return;
    try {
        const res = await fetch(`/api/datasets/sources/${encodeURIComponent(sourcePath)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to remove source');
        delete sourceDatasets[sourcePath];
        expandedSources.delete(sourcePath);
        await loadSources();
    } catch (e) {
        showToast('Error', e.message, 'error');
    }
}

function openDatasetFromSource(root) {
    openDataset(root);
}

// Replacing a panel's innerHTML scrolls it back to the top, however small the
// data change was. Preserve the offset of whichever ancestor actually scrolls,
// so re-rendering after a copy or a delete does not lose the user's place.
function _withScrollPreserved(el, render) {
    const scroller = el && el.closest('.sources-section, .tree-container');
    const top = scroller ? scroller.scrollTop : 0;
    render();
    if (scroller && scroller.scrollTop !== top) scroller.scrollTop = top;
}

function renderSources() {
    const container = document.getElementById('sources-container');
    if (!container) return;

    if (sources.length === 0) {
        container.innerHTML = '<div class="source-empty">No sources configured</div>';
        return;
    }

    let html = '';
    for (const source of sources) {
        const isExpanded = expandedSources.has(source.path);
        const sid = _sourceId(source.path);
        const datasets = sourceDatasets[source.path] || [];
        const countText = datasets.length > 0 ? `${datasets.length}` : '';
        // Show last two path segments for readability
        const parts = source.path.split('/').filter(Boolean);
        const displayPath = parts.length > 2 ? '.../' + parts.slice(-2).join('/') : source.path;

        html += `<div class="source-folder">`;
        html += `<div class="source-folder-header" onclick="toggleSource('${source.path.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${source.path.replace(/'/g, "\\'")}')" title="${source.path}">`;
        html += `<span class="source-folder-toggle">${isExpanded ? '▼' : '▶'}</span>`;
        html += `<span class="source-folder-path">${displayPath}</span>`;
        html += `<span class="source-folder-count">${countText}</span>`;
        if (source.removable) {
            html += `<span class="source-folder-remove" onclick="removeSource('${source.path.replace(/'/g, "\\'")}', event)" title="Remove source">&times;</span>`;
        }
        html += `</div>`;

        html += `<div class="source-folder-children ${isExpanded ? 'expanded' : ''}" id="source-children-${sid}">`;
        if (isExpanded) {
            const rows = sourceRowsFor(source.path, datasets, pendingCopies);
            if (rows.length === 0 && !sourceDatasets[source.path]) {
                html += '<div class="source-loading">Scanning...</div>';
            } else if (rows.length === 0) {
                html += '<div class="source-empty">No datasets found</div>';
            } else {
                for (const ds of rows) {
                    if (ds.copying) {
                        html += `<div class="source-dataset copying" title="Copying ${ds.source} → ${ds.root}">`;
                        html += `<span class="source-dataset-name">${ds.name}</span>`;
                        html += `<span class="source-dataset-meta">copying…</span>`;
                        html += `</div>`;
                        continue;
                    }
                    const isOpen = Object.keys(window.datasets || {}).some(id => {
                        const d = window.datasets[id];
                        return d && d.root === ds.root;
                    });
                    html += `<div class="source-dataset${isOpen ? ' active' : ''}" onclick="openDatasetFromSource('${ds.root.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}', false, true)" title="${ds.root}\n${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames">`;
                    html += `<span class="source-dataset-name">${ds.name}</span>`;
                    html += `<span class="source-dataset-meta">${ds.total_episodes} ep</span>`;
                    html += notesAddButton(ds.root);
                    html += `</div>`;
                    html += notesLine(ds.root);
                }
            }
        }
        html += `</div></div>`;
    }
    _withScrollPreserved(container, () => { container.innerHTML = html; });

    // Notes arrive after the tree; the fetch is batched over every visible
    // dataset and re-renders only if any of them actually has one.
    const visible = sources
        .filter(s => expandedSources.has(s.path))
        .flatMap(s => (sourceDatasets[s.path] || []).map(d => d.root));
    notesEnsure(visible, renderSources);
}

notesOnRerender(renderSources);

async function openDataset(path) {
    if (!path) return;

    setStatus('Opening dataset...');
    try {
        const body = path.startsWith('/') ? { local_path: path } : { repo_id: path };
        const res = await fetch('/api/datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        // 409 = incomplete local cache. Hand off to the Hub modal in
        // 'open-sync' mode — the modal owns the download + open flow from
        // here. We bail out of openDataset; the modal calls _completeOpen()
        // on success.
        if (res.status === 409) {
            const payload = await res.json();
            const detail = payload && payload.detail;
            if (detail && detail.code === 'incomplete_local_cache') {
                openHubModal(null, 'open-sync', { body, detail });
                return;
            }
        }

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        await _completeOpen(data);
    } catch (e) {
        let errorMsg = e.message;
        try {
            const parsed = JSON.parse(errorMsg);
            if (parsed.detail) errorMsg = parsed.detail;
        } catch (_) {}
        setStatus('Error: ' + errorMsg);
        showToast('Failed to open dataset', errorMsg, 'error', 10000);
    }
}

// Shared post-open flow: surface errors/warnings, load episodes, expand the
// tree, refresh edits. Called from both the normal openDataset path and the
// Hub-modal 'open-sync' path.
async function _completeOpen(data) {
    datasets[data.id] = data;

    if (data.errors && data.errors.length > 0) {
        showToast('Dataset Error', data.errors.join('\n'), 'error', 0);
    }
    if (data.warnings && data.warnings.length > 0) {
        const actionable = data.warnings.filter(w => !w.startsWith('stats.json mismatch'));
        if (actionable.length > 0) {
            showToast('Dataset Warning', actionable.join('\n'), 'warning', 8000);
        }
    }

    const epRes = await fetch(`/api/datasets/${encodeURIComponent(data.id)}/episodes`);
    episodes[data.id] = await epRes.json();
    datasets[data.id].total_episodes = episodes[data.id].length;

    expandedNodes.add(data.id);
    await refreshPendingEdits();

    renderTree();
    renderSources();
    if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
    if (window.FeatureEditing) window.FeatureEditing.onDatasetOpened(data.id);
    setStatus(`Opened: ${data.repo_id}`);
}

function isEpisodeDeleted(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'delete');
}

function isEpisodeTrimmed(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'trim');
}

// Derive episode quality flags from the per-component action stats the
// API exposes (min, max, mean, std arrays — pre-computed at record time).
// Backend just surfaces the raw characteristics; consumers decide what
// counts as a problem. New checks can land here without touching the API.
function _episodeActionFlags(stats) {
    if (!stats) return { allZero: false };  // unknown — render no badges
    const absMaxOfMin = Math.max(...stats.min.map(Math.abs));
    const absMaxOfMax = Math.max(...stats.max.map(Math.abs));
    return {
        allZero: absMaxOfMin === 0 && absMaxOfMax === 0,
        // Future: static = Math.max(...stats.std) === 0
        // Future: jittery = mean(stats.std) > some_threshold
    };
}

function renderTree() {
    const container = document.getElementById('tree-container');
    // Copies of an opened dataset will themselves be opened when they land, so
    // they belong here too. Without this the row appears only under Sources and
    // the Opened area gives no sign that anything is happening.
    const pendingOpens = [...pendingCopies.entries()].filter(([, c]) => c.wasOpen);
    if (Object.keys(datasets).length === 0 && pendingOpens.length === 0) {
        container.innerHTML = '<div style="padding: 8px 12px; color: #666; font-size: 12px;">No datasets opened</div>';
        return;
    }

    let html = '';
    for (const [id, ds] of Object.entries(datasets)) {
        const isExpanded = expandedNodes.has(id);
        const dsEpisodes = episodes[id] || [];
        const dsEditCount = pendingEdits.filter(e => e.dataset_id === id).length;
        const totalFrames = dsEpisodes.reduce((sum, ep) => sum + ep.length, 0);
        const tooltip = `${ds.repo_id}\n${ds.total_episodes} episodes, ${totalFrames.toLocaleString()} frames\nPath: ${ds.root}`;

        html += `
            <div class="tree-node">
                <div class="tree-header" onclick="toggleDataset('${id}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}', false, true)" title="${tooltip}">
                    <span class="tree-toggle">${isExpanded ? '▼' : '▶'}</span>
                    <span class="tree-icon">${ds.errors && ds.errors.length > 0 ? '⚠️' : '📁'}</span>
                    <span class="tree-label">${ds.repo_id}</span>
                    <span class="tree-meta">${dsEditCount > 0 ? `${dsEditCount}✎ ` : ''}${ds.total_episodes} ep</span>
                    ${notesAddButton(ds.root)}
                    <span class="tree-close" onclick="closeDataset('${id}', event)" title="Close">&times;</span>
                </div>
                ${notesLine(ds.root, 'note-opened')}
                <div class="tree-children ${isExpanded ? 'expanded' : ''}">
        `;

        // The dominant video profile, so a row can be flagged for differing from
        // it. A dataset built by merging carries more than one, and which
        // episodes came from where is otherwise invisible.
        const _profileCount = new Map();
        for (const ep of dsEpisodes) {
            const streams = ep.video_streams || {};
            const keys = Object.keys(streams).sort();
            if (!keys.length) continue;
            const sig = keys.map((k) => `${k}:${streams[k].codec}:${streams[k].width}x${streams[k].height}`).join('|');
            _profileCount.set(sig, (_profileCount.get(sig) || 0) + 1);
        }
        let _dominantProfile = null;
        let _dominantCount = 0;
        for (const [sig, n] of _profileCount) {
            if (n > _dominantCount) { _dominantProfile = sig; _dominantCount = n; }
        }
        const _profilesDiffer = _profileCount.size > 1;

        for (const ep of dsEpisodes) {
            const isActive = currentDataset === id && currentEpisode === ep.episode_index;
            const isDeleted = isEpisodeDeleted(id, ep.episode_index);
            const isTrimmed = isEpisodeTrimmed(id, ep.episode_index);
            // What the files actually are, which for a merged dataset is not
            // one answer and is not what info.json claims.
            const streams = ep.video_streams || {};
            const streamKeys = Object.keys(streams);
            let videoTitle = "";
            if (streamKeys.length) {
                const codecs = [...new Set(streamKeys.map((k) => streams[k].codec))];
                const res = [...new Set(streamKeys.map((k) => `${streams[k].width}x${streams[k].height}`))];
                videoTitle = streamKeys
                    .map((k) => {
                        const v = streams[k];
                        return `${k.split(".").pop()}: ${v.codec} ${v.width}x${v.height} `
                            + `${v.pix_fmt} ${v.fps}fps ${v.bitrate_kbps}kbps`;
                    })
                    .join("\n");
                ep._codecSummary = codecs.join("/");
                ep._resSummary = res.join(" ");
                const sig = streamKeys.slice().sort()
                    .map((k) => `${k}:${streams[k].codec}:${streams[k].width}x${streams[k].height}`)
                    .join('|');
                ep._videoOdd = _profilesDiffer && sig !== _dominantProfile;
            }
            ep._videoTitle = videoTitle;
            const hasVideoMismatch = ep.video_extra_frames !== 0;
            // Derive action-quality flags from the raw per-component stats
            // exposed by the API. New checks (static, saturated, jittery)
            // can be added here without touching the backend — the API just
            // surfaces the raw characteristics.
            const actionFlags = _episodeActionFlags(ep.action_stats);
            const hasZeroActions = actionFlags.allZero;
            // "quality-warning" is the unified visual state for any per-episode
            // quality issue. The tooltip distinguishes the cause.
            const hasQualityWarning = hasVideoMismatch || hasZeroActions;
            const isBridgeHighlight = bridgeHighlights.get(id)?.has(ep.episode_index);
            const classes = ['tree-header'];
            if (isActive) classes.push('active');
            if (isDeleted) classes.push('deleted');
            if (isTrimmed) classes.push('trimmed');
            if (hasQualityWarning) classes.push('quality-warning');
            if (isBridgeHighlight) classes.push('bridge-highlight');

            let icon = '🎬';
            if (isDeleted) icon = '🗑️';
            else if (hasQualityWarning) icon = '⚠️';

            let meta = `${ep.length} frames`;
            if (hasVideoMismatch) {
                const sign = ep.video_extra_frames > 0 ? '+' : '';
                meta += ` (${sign}${ep.video_extra_frames})`;
            }
            if (hasZeroActions) meta += ' (zero actions)';
            // Only the minority profile is called out; badging all 274 rows of a
            // uniform dataset would be noise carrying no information.
            if (ep._videoOdd) meta += ` · ${ep._codecSummary} ${ep._resSummary}`;

            // Compose tooltip across all warnings on this episode.
            const tipParts = [];
            if (hasVideoMismatch) {
                tipParts.push(
                    ep.video_extra_frames > 0
                        ? `Video-data mismatch: ${ep.video_extra_frames} extra frames (re-recording artifact)`
                        : `Video-data mismatch: ${Math.abs(ep.video_extra_frames)} missing frames (truncated video)`
                );
            }
            if (hasZeroActions) {
                tipParts.push(
                    'Action column is identically zero across every frame — almost always a recording-flow bug ' +
                    '(intervention flag never engaged during teleop). Episode is useless for training/replay.'
                );
            }
            // The video profile is on every row's tooltip, not just flagged ones:
            // it is what the files actually are, which for a merged dataset is
            // not one answer and is not what info.json claims.
            if (ep._videoTitle) {
                tipParts.push(
                    (ep._videoOdd
                        ? 'Video differs from the rest of this dataset:\n'
                        : 'Video:\n') + ep._videoTitle
                );
            }
            const titleAttr = tipParts.length ? `title="${tipParts.join('\n\n').replace(/"/g, '&quot;')}"` : '';

            html += `
                <div class="${classes.join(' ')}"
                     data-episode-row
                     data-dataset-id="${id}"
                     data-episode-id="${ep.episode_index}"
                     onclick="selectEpisode('${id}', ${ep.episode_index}, ${ep.video_length || ep.length})"
                     oncontextmenu="showContextMenu(event, '${id}', ${ep.episode_index})"
                     ${titleAttr}>
                    <span class="tree-toggle"></span>
                    <span class="tree-icon">${icon}</span>
                    <span class="tree-label">Episode ${ep.episode_index}</span>
                    <span class="tree-meta">${meta}</span>
                </div>
            `;
        }

        html += '</div></div>';
    }
    for (const [dstPath, copy] of pendingOpens) {
        html += `<div class="tree-node"><div class="tree-header copying" title="Copying ${copy.source} → ${dstPath}">`;
        html += `<span class="tree-toggle"></span>`;
        html += `<span class="tree-icon">📁</span>`;
        html += `<span class="tree-label">${openedLabelFor(dstPath)}</span>`;
        html += `<span class="tree-meta">copying…</span>`;
        html += `</div></div>`;
    }

    _withScrollPreserved(container, () => { container.innerHTML = html; });
    updateEditsBar();
    notesEnsure(Object.values(datasets).map(d => d.root), renderTree);
}

notesOnRerender(renderTree);

function toggleDataset(id) {
    if (expandedNodes.has(id)) {
        expandedNodes.delete(id);
    } else {
        expandedNodes.add(id);
    }
    renderTree();
}

// Suggested name for a copy: the source folder with a suffix, the same shape
// "Process dataset…" uses for its output. Kept pure so it can be unit-tested.
function duplicateNameFor(path) {
    const base = String(path || '').split(/[\\/]/).filter(Boolean).pop() || 'dataset';
    return `${base}_copy`;
}

// Copies in flight, keyed by destination path. The client knows what it asked
// for, so the tree can show the copy where its result will appear without the
// server tracking a job. Cleared when the request settles; a reload drops the
// row while the copy carries on server-side, and the next scan picks it up.
const pendingCopies = new Map();

// How the Opened panel labels a dataset: `owner/name`, the repo_id form, not the
// bare folder. A pending copy has no dataset object yet, so its label is derived
// from the destination path's last two components.
function openedLabelFor(dstPath) {
    const parts = String(dstPath || '').split('/').filter(Boolean);
    return parts.slice(-2).join('/') || String(dstPath || '');
}

// Rows to draw under one source: everything scanned there, plus any copy this
// client started that will land there. Copies carry the same relative-path name
// a scanned row does and sort among them — a row appended after 150 datasets
// under a different naming convention is not findable. Computed for every
// source state, because a copy into an empty or still-scanning source is
// exactly when the placeholder matters most.
function sourceRowsFor(sourcePath, scanned, pending) {
    const rows = [...(scanned || [])];
    for (const [dstPath, copy] of pending || []) {
        if (!dstPath.startsWith(sourcePath + '/')) continue;
        rows.push({
            name: dstPath.slice(sourcePath.length + 1),
            root: dstPath,
            copying: true,
            source: copy.source,
        });
    }
    return rows.sort((a, b) => a.name.localeCompare(b.name));
}

// Which repo kind the Hub dialog is currently open for. Set when it opens; the
// preview link and the started job both follow it.
let _hubRepoType = 'dataset';

// Hub URL for a repo. Models sit at the Hub root, datasets under /datasets —
// the tray hardcoded the dataset prefix, so a model transfer linked nowhere.
// A model run has no repo_id of its own, so the suggested one is derived from
// the run folder under the logged-in owner, matching how a dataset copy is named.
// Falls back to `me` only when the auth probe has not answered or reports
// logged out. A wrong owner is not a cosmetic default: accepting it passes the
// pre-flight whoami and fails minutes later inside the worker on create_repo,
// which `classify_error` reports as an expired token.
/** Repo id for a dataset directory: <owner>/<name>, the on-disk layout.
 *
 * Mirrors what the server derives when it is handed a path, so the field is
 * prefilled with the same thing the transfer would default to. Falls back to
 * the signed-in user when the path is too shallow to name an owner.
 */
function defaultDatasetRepoId(datasetPath) {
    const parts = String(datasetPath || '').replace(/\/+$/, '').split('/').filter(Boolean);
    const name = parts.pop() || 'dataset';
    const owner = parts.pop() || window.hfUser || 'me';
    return `${owner}/${name}`;
}

function defaultModelRepoId(runPath) {
    const name = String(runPath || '').replace(/\/+$/, '').split('/').filter(Boolean).pop() || 'model';
    return `${window.hfUser || 'me'}/${name}`;
}

function hubRepoUrl(repoId, repoType) {
    return `https://huggingface.co/${repoType === 'model' ? '' : 'datasets/'}${repoId}`;
}

async function duplicateDatasetAt(path) {
    const name = prompt(
        `Copy this dataset to a new folder beside it.\n\nSource: ${path}\n\nNew folder name:`,
        duplicateNameFor(path),
    );
    if (name === null) return;
    const parent = path.replace(/\/+$/, '').split('/').slice(0, -1).join('/');
    const dstPath = `${parent}/${name}`;
    const wasOpen = !!datasets[path];
    pendingCopies.set(dstPath, { name, source: path, wasOpen, startedAt: Date.now() });
    renderSources();
    renderTree();
    setStatus(`Copying to ${name}...`);
    try {
        // Path travels in the body, not the URL: a `{id:path}` route would be
        // ambiguous against the catch-all DELETE used for close.
        const res = await fetch('/api/datasets/duplicate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, new_name: name }),
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
            setStatus(`Copy failed: ${body.detail || res.status}`);
            alert(`Copy failed:\n${body.detail || res.status}`);
            return;
        }
        setStatus(`Copied to ${body.root}`);
        // Insert the new row rather than rescanning every expanded source: a
        // rescan rebuilds the panel and drops its scroll position. The copy is
        // byte-identical to its source, so cloning the source's row gives exact
        // episode and frame counts without asking the server again.
        for (const [srcPath, list] of Object.entries(sourceDatasets)) {
            const origin = list.find(d => d.root === path);
            if (!origin) continue;
            list.push({ ...origin, root: body.root, name: body.root.slice(srcPath.length + 1) });
            list.sort((a, b) => a.name.localeCompare(b.name));
        }
        // Opened only if the original was — duplicating from the Sources list
        // is browsing, duplicating something you have open is working on it.
        if (wasOpen) await openDataset(body.root);
    } catch (e) {
        setStatus(`Copy failed: ${e.message}`);
    } finally {
        pendingCopies.delete(dstPath);
        renderSources();
        renderTree();
    }
}

async function deleteDatasetFilesAt(path) {
    const open = datasets[path];
    const scale = open ? `${open.total_episodes} episodes, ${open.total_frames.toLocaleString()} frames` : '';
    if (!confirm(
        `Delete this dataset from disk?\n\n${path}\n${scale}\n\n`
        + 'The files are removed permanently — there is no trash, and this cannot be undone.'
    )) return;
    setStatus('Deleting dataset...');
    try {
        // Path as a query parameter, not a path segment: `{dataset_id:path}`
        // is greedy, and a suffix route under it would capture close.
        const res = await fetch(
            `/api/datasets/files?path=${encodeURIComponent(path)}`,
            { method: 'DELETE' },
        );
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
            setStatus(`Delete failed: ${body.detail || res.status}`);
            alert(`Delete failed:\n${body.detail || res.status}`);
            return;
        }
        // The server already dropped it from the registry; drop every client
        // trace too, or the tree, the camera grid and the Inspector go on
        // describing a directory that no longer exists.
        // Drop the row from the cached listing rather than rescanning every
        // expanded source from disk. We know exactly which path went away, and
        // a rescan rebuilds the whole panel — which resets its scroll position
        // for the sake of one removed row.
        for (const list of Object.values(sourceDatasets)) {
            const i = list.findIndex(d => d.root === path);
            if (i >= 0) list.splice(i, 1);
        }
        forgetDatasetInClient(path);
        setStatus('Dataset deleted');
    } catch (e) {
        setStatus(`Delete failed: ${e.message}`);
    }
}

// Drop every client-side trace of a dataset. Shared by close and delete —
// deleting is strictly more than closing, so the two teardowns must not drift.
function forgetDatasetInClient(id) {
    delete datasets[id];
    delete episodes[id];
    expandedNodes.delete(id);
    if (currentDataset === id) {
        currentDataset = null;
        currentEpisode = null;
        window.currentDataset = null;
        window.currentEpisode = null;
        // Through renderCameraGrid, not a direct innerHTML write: it is the
        // one place that clears the tile signature. Writing the empty state
        // behind its back would leave the stale signature stamped, and
        // re-opening this same dataset would then match it and skip the
        // rebuild — leaving "Select an episode to view" where the tiles go.
        renderCameraGrid();
    }
    // The Inspector keeps the schema and per-episode cards of whatever it last
    // rendered. Without this it goes on describing a dataset that is closed —
    // or, after a delete, one whose directory no longer exists.
    window.FeatureEditing?.onDatasetClosed?.(id);
    renderTree();
    if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
    renderSources();
}

async function closeDataset(id, e) {
    e.stopPropagation();
    try {
        await fetch(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
        forgetDatasetInClient(id);
    } catch (err) {
        showToast('Error', 'Failed to close dataset: ' + err.message, 'error');
    }
}

function selectEpisode(datasetId, epIdx, length) {
    const datasetChanged = currentDataset !== datasetId;
    currentDataset = datasetId;
    currentEpisode = epIdx;
    totalFrames = length;
    currentFrame = 0;
    fps = datasets[datasetId].fps || 30;

    // Initialize trim to full episode
    trimStart = 0;
    trimEnd = totalFrames;

    // Stop playback
    if (isPlaying) {
        togglePlay();
    }

    renderTree();
    renderCameraGrid();
    loadAllFrames(0);
    loadTrimForCurrentEpisode();
    if (window.FeatureEditing) window.FeatureEditing.onEpisodeSelected(datasetId, epIdx);
    // A dataset switch changes the camera set — rebuild the overlay panel's camera list (dropping
    // selections the new dataset lacks) and re-sync the worker to it.
    if (datasetChanged && window.Overlays && window.Overlays.refreshCameras) window.Overlays.refreshCameras();
}

// Tile identity of the observation grid: the camera set plus the robot the
// URDF tile resolves to. Rebuilding the grid means `grid.innerHTML = …`, which
// destroys the URDF iframe — and an iframe cannot be carried across that (nor
// re-parented; detaching a nested browsing context reloads it). So the grid is
// rebuilt only when this signature actually changes. An episode switch inside
// one dataset leaves it identical: the URDF is resolved from the dataset's
// `observation.state` motor names, which no episode can change.
//
// The robot half reads `pending` until the probe lands; _probeAndAttachUrdfViz
// re-stamps the signature once it knows, so the next episode switch matches
// instead of paying one more rebuild.
function _tileSignature(datasetId) {
    const info = _urdfVizInfo[datasetId];
    const robot = info === undefined ? 'pending' : (info.available ? info.robot : 'none');
    return `${datasets[datasetId].camera_keys.join('\u0000')}\u0001${robot}`;
}

function renderCameraGrid() {
    const grid = document.getElementById('camera-grid');
    if (!currentDataset || currentEpisode === null) {
        grid.innerHTML = '<div class="empty-state">Select an episode to view</div>';
        delete grid.dataset.tileSig;
        return;
    }

    const sig = _tileSignature(currentDataset);
    if (grid.dataset.tileSig === sig) {
        // Same tiles, same robot — keep the DOM. The <img> srcs are rewritten by
        // the loadAllFrames call that follows, and the URDF iframe drops its own
        // per-episode caches when that call's frame message names a new episode.
        // So it keeps its parsed meshes, its orbit camera and its ghost toggle
        // instead of cold-booting once per episode click.
        return;
    }

    const ds = datasets[currentDataset];
    const cameras = ds.camera_keys;
    // The URDF tile counts as one cell in the grid; treat it as a virtual
    // camera for layout purposes (and append it physically below). Before the
    // first probe of a dataset we don't yet know whether it has one, so the
    // placeholder is emitted and _probeAndAttachUrdfViz removes it if this
    // dataset's motor set has no vendored URDF.
    const _info = _urdfVizInfo[currentDataset];
    const hasUrdfTile = _info === undefined || _info.available;
    const tileCount = cameras.length + (hasUrdfTile ? 1 : 0);

    let cols = 1;
    if (tileCount === 2) cols = 2;
    else if (tileCount >= 3 && tileCount <= 4) cols = 2;
    else if (tileCount >= 5) cols = 3;

    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    let html = '';
    for (const cam of cameras) {
        const camName = cam.split('.').pop();
        html += `
            <div class="camera-panel" data-cam-cell="${cam}">
                <div class="camera-title">${camName}</div>
                <div class="camera-frame">
                    <img id="frame-${cam.replace(/\./g, '-')}" src="" alt="${camName}">
                    <video id="video-${cam.replace(/\./g, '-')}" class="camera-video" muted playsinline
                           preload="auto" style="display:none"></video>
                    <img class="overlay-layer" id="overlay-${cam.replace(/\./g, '-')}" src="" alt="">
                    <canvas class="overlay-layer mask-layer" id="mask-${cam.replace(/\./g, '-')}"></canvas>
                    <button class="obs-cam-zoom" data-zoom="${cam}" type="button"
                            title="Enlarge this camera (click again to restore)">⤢</button>
                </div>
            </div>
        `;
    }
    if (hasUrdfTile) {
        html += `
            <div class="camera-panel" id="urdf-viz-panel" style="display: none;">
                <div class="camera-title">visualizer</div>
                <div class="camera-frame">
                    <iframe id="urdf-viz-iframe" src="" title="URDF state visualization"
                            style="width: 100%; height: 100%; border: none; background: #1a1a1a;"></iframe>
                </div>
            </div>
        `;
    }
    grid.innerHTML = html;
    grid.dataset.tileSig = sig;
    _installCameraZoom(grid);
    _probeAndAttachUrdfViz(currentDataset, currentEpisode);
}

// Enlarge one camera to fill the grid (click again to restore) — the control the run tab
// has had since it shipped, which the data tab lacked. Reviewing a mask on a 4-camera grid
// means squinting at quarter-panel tiles.
//
// Delegated to the GRID and installed idempotently, because this grid element OUTLIVES its
// tiles: every episode change replaces its innerHTML, so a per-tile listener would bind to
// elements that no longer exist, and a per-rebuild grid listener would pile up. Hides the
// other panels rather than restyling each, so restore is exact; the grid template is read
// back off the element so the column layout returns as it was. Deliberately no Esc hotkey —
// several dialogs already bind Escape (the reasoning is recorded in run.js).
function _installCameraZoom(grid) {
    if (grid.dataset.zoomWired) return;
    grid.dataset.zoomWired = '1';
    let focused = null;
    grid.addEventListener('click', (e) => {
        const btn = e.target.closest('.obs-cam-zoom');
        if (!btn) return;
        e.stopPropagation();  // never let the zoom press read as a click on the tile itself
        const key = btn.dataset.zoom;
        focused = focused === key ? null : key;
        const panels = grid.querySelectorAll(':scope > .camera-panel');
        if (focused) {
            if (grid.dataset.zoomCols === undefined) grid.dataset.zoomCols = grid.style.gridTemplateColumns || '';
            // Remember each panel's OWN display before hiding it. The visualizer tile is
            // display:none until its probe succeeds (and removed outright when it fails),
            // so blanket-restoring to '' would reveal a tile that was meant to stay hidden.
            for (const panel of panels) {
                if (panel.dataset.preZoom === undefined) panel.dataset.preZoom = panel.style.display;
                panel.style.display = (panel.dataset.camCell === focused) ? '' : 'none';
            }
            grid.style.gridTemplateColumns = '1fr';
        } else {
            for (const panel of panels) {
                panel.style.display = panel.dataset.preZoom || '';
                delete panel.dataset.preZoom;
            }
            grid.style.gridTemplateColumns = grid.dataset.zoomCols || '';
            delete grid.dataset.zoomCols;
        }
    });
}

// dataset_id -> {available, robot}, cached after the first probe. ``robot`` is
// the resolved description name (``spec.name`` server-side); it is a property
// of the dataset's motor set, so every episode of a dataset — and every dataset
// recorded on the same arm — shares one value. That is what _tileSignature
// keys the grid on.
let _urdfVizInfo = {};

// Per-tab persisted preference for the data-tab URDF ghost / trajectory
// toggle. Backed by sessionStorage so it survives the iframe reloads that do
// still happen: a full page reload, or a dataset switch whose camera set or
// robot differs (both rebuild the grid and lose the iframe's module-level
// ``_ghostOn``). Falls back to the parent's ``?urdfGhost=on`` URL param (the
// bookmarkable initial state, also what the screenshot script keys off).
function _urdfGhostPref() {
    const stored = sessionStorage.getItem('urdfGhost');
    if (stored !== null) return stored === 'on';
    return new URLSearchParams(location.search).get('urdfGhost') === 'on';
}

// One-time install: iframe postMessages ``urdfGhostChanged`` when the
// user clicks the toggle inside it. We update sessionStorage so the
// next iframe initializes with the remembered value via _urdfGhostPref above.
(function _wireUrdfGhostPersistence() {
    window.addEventListener('message', (ev) => {
        if (ev.data && ev.data.type === 'urdfGhostChanged') {
            sessionStorage.setItem('urdfGhost', ev.data.on ? 'on' : 'off');
        }
    });
})();

async function _probeAndAttachUrdfViz(datasetId, episodeIdx) {
    const panel = document.getElementById('urdf-viz-panel');
    const iframe = document.getElementById('urdf-viz-iframe');
    if (!panel || !iframe) return;

    let info = _urdfVizInfo[datasetId];
    if (info === undefined) {
        try {
            // The meta endpoint answers both questions in one round trip: is
            // there a vendored description for this motor set, and which one.
            // (The iframe fetches the same endpoint on boot, so this is warm.)
            const url = `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/urdf-viz/meta`;
            const r = await fetch(url);
            const d = await r.json();
            info = { available: !!d.available, robot: d.name || null };
        } catch (e) {
            info = { available: false, robot: null };
        }
        _urdfVizInfo[datasetId] = info;
    }
    // Bail if the user has navigated away while we were probing — a later
    // selectEpisode call has re-rendered the grid and a new probe is in
    // flight for the new episode.
    if (currentDataset !== datasetId || currentEpisode !== episodeIdx) return;
    // The robot is known now, so the signature stamped by the rebuild (which
    // said ``pending``) is stale. Re-stamp it, otherwise the very next episode
    // switch would see a mismatch and rebuild the grid one extra time.
    const gridEl = document.getElementById('camera-grid');
    if (gridEl) gridEl.dataset.tileSig = _tileSignature(datasetId);
    if (!info.available) {
        // Only reachable on a dataset's first probe — later rebuilds already
        // know not to emit the placeholder at all.
        panel.remove();
        // Drop the empty cell back out of the column count.
        const cams = datasets[datasetId].camera_keys.length;
        let cols = 1;
        if (cams === 2) cols = 2;
        else if (cams >= 3 && cams <= 4) cols = 2;
        else if (cams >= 5) cols = 3;
        if (gridEl) gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        return;
    }
    panel.style.display = '';
    // mode=dataset means the iframe waits for postMessage frame updates from
    // the parent (this page), driven by the scrubber via _postFrameToUrdfViz.
    // ``_urdfGhostPref()`` reads sessionStorage first (sticky across the
    // reloads that remain) then falls back to the parent URL's
    // ``?urdfGhost=on`` (bookmarkable initial state, used by the screenshot
    // script). Bump the version any time this seams (URL param contract or
    // postMessage protocol) changes so an old cached iframe doesn't stick.
    const ghostInit = _urdfGhostPref() ? '&ghost=on' : '';
    iframe.src = `/static/urdf_viz.html?mode=dataset&v=2${ghostInit}`;
    // Fast path: iframe.onload fires when the document is parsed, which is
    // usually before the module script has registered its message listener
    // but in practice fast enough for an idle main thread. Belt:
    // `urdfVizReady` from the iframe's module script (see urdf_viz.html)
    // arrives once the listener IS registered. Heavy main-thread load
    // (Playwright video recording, dev tools, etc.) can race the fast path;
    // the ready signal re-posts after the listener is guaranteed live.
    iframe.addEventListener('load', () => _postFrameToUrdfViz(currentFrame), { once: true });
}

// Re-post the frame on the iframe's ready signal — robust to the
// module-script-deferred-registration race the fast iframe.onload path
// can hit under heavy main-thread load.
window.addEventListener('message', (e) => {
    const msg = e.data;
    if (!msg || typeof msg !== 'object' || msg.type !== 'urdfVizReady') return;
    if (currentDataset == null || currentEpisode == null) return;
    _postFrameToUrdfViz(currentFrame);
});

function _postFrameToUrdfViz(frameIdx) {
    const iframe = document.getElementById('urdf-viz-iframe');
    if (!iframe || !iframe.contentWindow || !currentDataset || currentEpisode === null) return;
    iframe.contentWindow.postMessage(
        { type: 'frame', dataset: currentDataset, episode: currentEpisode, frame: frameIdx },
        '*',
    );
}

// Per-camera loader state: at most one request in flight, plus a single slot
// for the most recently requested frame. A scrub overwrites the slot instead
// of queueing, so the link delivers what it can and always chases the cursor.
const _frameLoad = {};

function _frameUrl(cam, frame) {
    return `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}`
        + `/frame/${frame}?camera=${encodeURIComponent(cam)}&profile=${_videoProfile()}`;
}

function _pumpFrame(cam, img) {
    const st = _frameLoad[cam];
    if (st.want === null) {
        st.inflight = false;
        // Nothing outstanding: whoever was waiting on this camera is done.
        const waiters = st.waiters;
        st.waiters = [];
        waiters.forEach((r) => r());
        return;
    }
    const frame = st.want;
    st.want = null;
    st.inflight = true;
    const loader = new Image();
    const done = () => _pumpFrame(cam, img);
    loader.onload = () => {
        // Swap only on decode, so the previous frame stays up rather than the
        // element blanking while bytes are in flight.
        img.src = loader.src;
        done();
    };
    loader.onerror = done;   // leave the previous frame visible
    loader.src = _frameUrl(cam, frame);
}

function _requestFrame(cam, img, frame) {
    const st = (_frameLoad[cam] = _frameLoad[cam] || { inflight: false, want: null, waiters: [] });
    st.want = frame;
    const settled = new Promise((resolve) => st.waiters.push(resolve));
    if (!st.inflight) _pumpFrame(cam, img);
    return settled;
}

function loadAllFrames(idx) {
    // A manual frame request while the stream plays is a scrub: leave stream
    // playback and serve the requested still. The stream's own playhead updates
    // go through __streamSetPlayhead and never arrive here.
    if (window.OverlayStream && window.OverlayStream.streaming) window.OverlayStream.stop({resume: false});
    if (!currentDataset || currentEpisode === null) return Promise.resolve();
    currentFrame = Math.max(0, Math.min(idx, totalFrames - 1));

    const ds = datasets[currentDataset];
    const promises = [];

    for (const cam of ds.camera_keys) {
        const img = document.getElementById(`frame-${cam.replace(/\./g, '-')}`);
        if (img) promises.push(_requestFrame(cam, img, currentFrame));
    }

    updateFrameUI();
    // Whoever asked for a specific frame wants the still, not the stream.
    if (!isPlaying) _showVideo(false);
    return Promise.all(promises);
}

// Everything the playhead drives except fetching stills. Called by the JPEG
// path and by the video clock alike.
window.loadAllFrames = loadAllFrames;
window.__streamSetPlayhead = (f) => {
    currentFrame = Math.max(0, Math.min(f, totalFrames - 1));
    updateFrameUI();
};

function updateFrameUI() {
    document.getElementById('frame-info').textContent = `${currentFrame + 1} / ${totalFrames}`;
    const pct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0;
    document.getElementById('timeline-progress').style.width = `${pct}%`;
    document.getElementById('timeline-scrubber').style.left = `${pct}%`;

    // Update time display
    const currentTime = formatTime(currentFrame / fps);
    const totalTime = formatTime(totalFrames / fps);
    document.getElementById('time-info').textContent = `${currentTime} / ${totalTime}`;

    // Mirror playhead state to window so sibling scripts (overlays.js) can read it
    // (these are top-level `let`s, which are NOT window properties on their own).
    window.currentDataset = currentDataset;
    window.currentEpisode = currentEpisode;
    window.currentFrame = currentFrame;
    if (window.FeatureEditing) window.FeatureEditing.onPlayheadChanged();
    if (window.Overlays) window.Overlays.onFrame();
    if (window.MaskOverlay) window.MaskOverlay.onPlayheadChanged();
    _postFrameToUrdfViz(currentFrame);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// --- streamed playback -----------------------------------------------------
// The old loop fetched one JPEG per camera per frame: ~324 KB/frame for three
// cameras, 78 Mbit/s at 30 fps, which no remote link carries. The same footage
// is already H.264 on disk, so playing streams it (1.6 Mbit/s) and lets the
// browser keep time.

function _camVideoEls() {
    const ds = datasets[currentDataset];
    if (!ds) return [];
    return ds.camera_keys
        .map((cam) => document.getElementById(`video-${cam.replace(/\./g, '-')}`))
        .filter(Boolean);
}

function _videoProfile() {
    // 'low' is 640px/500kbps; 'medium' is 1280px/1.5Mbps. Source-quality
    // 'full' is a stream copy and available on the URL, but it is 12 Mbit/s.
    return (window.CameraVideoMode?.getEffectiveMode?.() === 'low-bandwidth') ? 'low' : 'medium';
}

function _videoUrl(cam) {
    return `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}`
        + `/video?camera=${encodeURIComponent(cam)}&profile=${_videoProfile()}`;
}

function _showVideo(on) {
    const ds = datasets[currentDataset];
    if (!ds) return;
    for (const cam of ds.camera_keys) {
        const id = cam.replace(/\./g, '-');
        const img = document.getElementById(`frame-${id}`);
        const vid = document.getElementById(`video-${id}`);
        if (img) img.style.display = on ? 'none' : '';
        if (vid) vid.style.display = on ? '' : 'none';
    }
}

async function _startStreamedPlayback() {
    const ds = datasets[currentDataset];
    if (!ds) return;
    const start = (currentFrame >= trimEnd - 1 || currentFrame < trimStart) ? trimStart : currentFrame;
    currentFrame = start;

    await Promise.all(ds.camera_keys.map((cam) => new Promise((resolve) => {
        const vid = document.getElementById(`video-${cam.replace(/\./g, '-')}`);
        if (!vid) return resolve();
        const want = _videoUrl(cam);
        if (vid.dataset.src !== want) {
            vid.dataset.src = want;
            vid.src = want;
            vid.addEventListener('loadeddata', () => resolve(), { once: true });
            vid.addEventListener('error', () => resolve(), { once: true });
            vid.load();
        } else {
            resolve();
        }
    })));

    // Seek before revealing, for the same reason: an un-seeked <video> shows
    // the frame it was left on, then jumps once the seek completes.
    await Promise.all(_camVideoEls().map((vid) => new Promise((resolve) => {
        vid.playbackRate = playbackSpeed;
        if (Math.abs(vid.currentTime - start / fps) < 1 / fps) return resolve();
        vid.addEventListener('seeked', () => resolve(), { once: true });
        setTimeout(resolve, 600);  // never hang the button on a stalled seek
        vid.currentTime = start / fps;
    })));

    _showVideo(true);
    for (const vid of _camVideoEls()) vid.play().catch(() => {});
    _followVideoClock();
}

function _followVideoClock() {
    const primary = _camVideoEls()[0];
    if (!primary) return;
    const step = () => {
        if (!isPlaying) return;
        const frame = Math.round(primary.currentTime * fps);
        if (frame >= trimEnd - 1) {
            for (const vid of _camVideoEls()) vid.currentTime = trimStart / fps;
            currentFrame = trimStart;
        } else {
            currentFrame = Math.max(trimStart, Math.min(frame, totalFrames - 1));
        }
        updateFrameUI();
        if (primary.requestVideoFrameCallback) primary.requestVideoFrameCallback(step);
        else requestAnimationFrame(step);
    };
    if (primary.requestVideoFrameCallback) primary.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
}

function _stopStreamedPlayback() {
    // Just stop. The paused <video> is already displaying the exact frame, so
    // there is nothing to fetch and nothing to swap; the still only has to
    // appear when the user moves to a different frame, which goes through
    // loadAllFrames and reveals it there.
    for (const vid of _camVideoEls()) vid.pause();
}

function changeSpeed(speed) {
    playbackSpeed = parseFloat(speed);
    for (const vid of _camVideoEls()) vid.playbackRate = playbackSpeed;
}

function togglePlay() {
    // When the SAM3 overlay is live, Play means the server-composited stream:
    // one H.264 atlas of every selected camera instead of per-frame stills +
    // overlay pulls. Pause and manual scrubs land back on the still path.
    if (window.OverlayStream && window.OverlayStream.eligible()) { window.OverlayStream.toggle(); return; }
    if (!currentDataset || currentEpisode === null) return;

    isPlaying = !isPlaying;
    document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';

    if (isPlaying) {
        _startStreamedPlayback();
    } else {
        _stopStreamedPlayback();
    }
}

async function launchRerun() {
    if (!currentDataset || currentEpisode === null) {
        setStatus('Select an episode first');
        return;
    }

    setStatus('Launching Rerun...');
    try {
        const res = await fetch(
            `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}/visualize`,
            { method: 'POST' }
        );
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setStatus(data.message);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

function getFrameFromTimelineEvent(e) {
    const timeline = document.getElementById('timeline');
    const rect = timeline.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    return Math.floor(pct * (totalFrames - 1));
}

function seekTimeline(frame) {
    if (!currentDataset || currentEpisode === null) return;
    loadAllFrames(frame);
}

function updateHoverPreview(e) {
    if (!currentDataset || currentEpisode === null) return;
    const frame = getFrameFromTimelineEvent(e);
    const time = formatTime(frame / fps);
    const hover = document.getElementById('timeline-hover');
    const timeline = document.getElementById('timeline');
    const rect = timeline.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    hover.style.left = `${pct * 100}%`;
    hover.textContent = `${time} / Frame ${frame + 1}`;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function showToast(title, message, type = 'info', duration = 5000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="toast-title">${title}</div><div class="toast-message">${message}</div>`;
    // Click to dismiss
    toast.style.cursor = 'pointer';
    toast.addEventListener('click', () => {
        toast.style.animation = 'toast-out 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    });
    container.appendChild(toast);
    if (duration > 0) {
        setTimeout(() => {
            toast.style.animation = 'toast-out 0.3s ease-out forwards';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

// Timeline interaction
document.addEventListener('DOMContentLoaded', () => {
    // --- Resizable sidebars ---
    document.querySelectorAll('.sidebar-resize-handle').forEach((handle) => {
        const sidebar = handle.previousElementSibling;
        if (!sidebar || !sidebar.classList.contains('sidebar')) return;
        let startX, startW;
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startX = e.clientX;
            startW = sidebar.offsetWidth;
            handle.classList.add('dragging');
            function onMove(e) {
                const w = Math.max(240, Math.min(startW + e.clientX - startX, window.innerWidth * 0.5));
                sidebar.style.width = w + 'px';
            }
            function onUp() {
                handle.classList.remove('dragging');
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    });

    const timelineContainer = document.getElementById('timeline-container');
    const timeline = document.getElementById('timeline');
    const scrubber = document.getElementById('timeline-scrubber');

    // Click to seek (but not if we were dragging trim handles)
    timeline.addEventListener('click', (e) => {
        if (!isDragging && !justFinishedTrimDrag) {
            // Check if click was on a trim handle
            if (e.target.classList.contains('trim-handle')) return;
            seekTimeline(getFrameFromTimelineEvent(e));
        }
    });

    // Hover preview
    timelineContainer.addEventListener('mousemove', updateHoverPreview);

    // Drag scrubber
    scrubber.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDragging = true;
        document.body.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging && currentDataset && currentEpisode !== null) {
            seekTimeline(getFrameFromTimelineEvent(e));
        }
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            document.body.style.cursor = '';
        }
        if (isDraggingTrimLeft || isDraggingTrimRight) {
            isDraggingTrimLeft = false;
            isDraggingTrimRight = false;
            justFinishedTrimDrag = true;
            document.body.style.cursor = '';
            // Auto-save the trim
            saveTrim();
            // Reset after a short delay to allow click event to check
            setTimeout(() => { justFinishedTrimDrag = false; }, 50);
        }
    });

    // Trim handle drag
    const trimHandleLeft = document.getElementById('trim-handle-left');
    const trimHandleRight = document.getElementById('trim-handle-right');

    trimHandleLeft.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingTrimLeft = true;
        document.body.style.cursor = 'ew-resize';
    });

    trimHandleRight.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingTrimRight = true;
        document.body.style.cursor = 'ew-resize';
    });

    document.addEventListener('mousemove', (e) => {
        if (!currentDataset || currentEpisode === null) return;

        if (isDraggingTrimLeft) {
            const frame = getFrameFromTimelineEvent(e);
            trimStart = Math.max(0, Math.min(frame, trimEnd - 1));
            updateTrimDisplay();
        } else if (isDraggingTrimRight) {
            const frame = getFrameFromTimelineEvent(e);
            // trimEnd is exclusive, so we add 1 to the clicked frame
            trimEnd = Math.max(trimStart + 1, Math.min(frame + 1, totalFrames));
            updateTrimDisplay();
        }
    });
});

// Keyboard controls
// Tab switching
function switchTab(tabName) {
    // Captured before the active classes are cleared below — the camera release
    // hook at the end of this function needs to know which tab we are leaving.
    const previousTab = document.querySelector('.tab.active')?.dataset.tab || null;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    // Re-scan whatever this tab lists from disk.
    if (typeof window.refreshTabFromDisk === 'function') {
        window.refreshTabFromDisk(tabName);
    }
    // Notify robot tab
    if (tabName === 'robot' && typeof robotTabInit === 'function') {
        robotTabInit();
    }
    // Notify run tab
    if (tabName === 'run' && typeof runTabInit === 'function') {
        runTabInit();
    }
    // Notify model tab
    if (tabName === 'model' && typeof modelTabInit === 'function') {
        modelTabInit();
    }
    // Notify preprocess tab
    if (tabName === 'preprocess' && typeof preprocessTabInit === 'function') {
        preprocessTabInit();
    }
    // Leaving the robot tab must RELEASE the cameras, not just stop drawing them.
    // stopCameraPreview() only clears the polling interval and hides the Stop
    // Preview button; the backend keeps a V4L2 / librealsense handle per camera,
    // invisibly, and a run launched from another tab then competes with the GUI
    // for its own devices. stopAllCameras() also POSTs /api/robot/stop-cameras.
    // Guarded on something actually being held so a normal tab switch does not
    // fire a POST on every click.
    // The backend decides what it actually holds — do not gate this on frontend
    // state like detectedCameras.length, which desyncs (page reload, direct API
    // call) and then silently skips the release.
    if (typeof CameraRelease !== 'undefined'
        && CameraRelease.shouldReleaseCameras(previousTab, tabName)
        && typeof stopAllCameras === 'function') {
        stopAllCameras();
    } else if (tabName !== 'robot' && typeof stopCameraPreview === 'function') {
        // Fallback if the module failed to load: at least stop the polling loop.
        stopCameraPreview();
    }
    // Disconnect SSE when leaving run tab (but don't kill process)
    if (tabName !== 'run' && typeof disconnectOutputSSE === 'function') {
        disconnectOutputSSE();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    // Only handle data-tab shortcuts when data tab is active
    const activeTab = document.querySelector('.tab.active')?.dataset.tab;
    if (activeTab !== 'data') return;
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        loadAllFrames(currentFrame - (e.shiftKey ? 10 : 1));
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        loadAllFrames(currentFrame + (e.shiftKey ? 10 : 1));
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        navigateEpisode(-1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        navigateEpisode(1);
    } else if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
    } else if (e.key === 'Home') {
        e.preventDefault();
        loadAllFrames(0);
    } else if (e.key === 'End') {
        e.preventDefault();
        loadAllFrames(totalFrames - 1);
    } else if (e.key === 'Delete' && currentDataset && currentEpisode !== null) {
        e.preventDefault();
        deleteCurrentEpisode();
    } else if (e.key === 'r' && currentDataset && currentEpisode !== null) {
        e.preventDefault();
        resetTrim();
    } else if (e.key === 'Escape') {
        hideContextMenu();
    }
});

function navigateEpisode(direction) {
    if (!currentDataset || currentEpisode === null) return;
    const dsEpisodes = episodes[currentDataset] || [];
    const newIndex = currentEpisode + direction;
    if (newIndex >= 0 && newIndex < dsEpisodes.length) {
        const ep = dsEpisodes.find(e => e.episode_index === newIndex);
        if (ep) selectEpisode(currentDataset, newIndex, ep.video_length || ep.length);
    }
}

// Context menu
// Position a context menu so it stays inside the viewport. The menu must
// already be `visible` (or have its dimensions otherwise readable) before
// calling — we measure with getBoundingClientRect after a forced layout.
function _positionContextMenu(menu, clientX, clientY) {
    const margin = 4;
    const rect = menu.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    let left = clientX;
    let top = clientY;
    if (left + rect.width + margin > vw) left = Math.max(margin, vw - rect.width - margin);
    if (top + rect.height + margin > vh) top = Math.max(margin, vh - rect.height - margin);
    menu.style.left = left + 'px';
    menu.style.top = top + 'px';
}

function showContextMenu(e, datasetId, episodeIndex) {
    e.preventDefault();
    e.stopPropagation();

    contextMenuTarget = { datasetId, episodeIndex };
    const menu = document.getElementById('context-menu');
    const isDeleted = isEpisodeDeleted(datasetId, episodeIndex);
    const isTrimmed = isEpisodeTrimmed(datasetId, episodeIndex);

    // Show/hide appropriate menu items
    menu.querySelectorAll('.context-menu-item').forEach(item => {
        const action = item.getAttribute('onclick').match(/contextAction\('(\w+)'\)/)?.[1];
        if (action === 'delete') item.style.display = isDeleted ? 'none' : 'block';
        if (action === 'undelete') item.style.display = isDeleted ? 'block' : 'none';
        if (action === 'cleartrim') item.style.display = isTrimmed ? 'block' : 'none';
    });

    // Make visible BEFORE measuring so getBoundingClientRect returns real dims.
    menu.classList.add('visible');
    _positionContextMenu(menu, e.clientX, e.clientY);
}

function hideContextMenu() {
    document.getElementById('context-menu').classList.remove('visible');
    document.getElementById('folder-context-menu').classList.remove('visible');
    contextMenuTarget = null;
    _folderContextPath = null;
}

document.addEventListener('click', hideContextMenu);

// Folder context menu (source folders + datasets + model runs)
let _folderContextPath = null;
let _folderContextIsModelRun = false;
let _folderContextIsDataset = false;

function showFolderContextMenu(e, path, isModelRun, isDataset) {
    e.preventDefault();
    e.stopPropagation();
    _folderContextPath = path;
    _folderContextIsModelRun = !!isModelRun;
    _folderContextIsDataset = !!isDataset;
    const menu = document.getElementById('folder-context-menu');
    // Show/hide model-run-specific items
    const testItem = document.getElementById('folder-ctx-test-on-robot');
    const testSep = document.getElementById('folder-ctx-test-separator');
    if (testItem) testItem.style.display = _folderContextIsModelRun ? '' : 'none';
    if (testSep) testSep.style.display = _folderContextIsModelRun ? '' : 'none';
    // Show/hide merge-into for opened datasets with 2+ datasets open
    const isOpenedDataset = !!datasets[path];
    const hasMultipleDatasets = Object.keys(datasets).length >= 2;
    const mergeItem = document.getElementById('folder-ctx-merge-into');
    const mergeSep = document.getElementById('folder-ctx-merge-separator');
    if (mergeItem) mergeItem.style.display = (isOpenedDataset && hasMultipleDatasets) ? '' : 'none';
    if (mergeSep) mergeSep.style.display = (isOpenedDataset && hasMultipleDatasets) ? '' : 'none';
    // Stereo split: any opened dataset. Whether a camera can actually be split
    // needs the feature shapes, which the client copy does not carry and a
    // context menu cannot wait for — so the modal reports it instead.
    const splitItem = document.getElementById('folder-ctx-split-stereo');
    if (splitItem) splitItem.style.display = isOpenedDataset ? '' : 'none';
    // Hub transfers: anything that is a dataset on disk, or any model run. The
    // gate asks what the node IS, not whether the server happens to hold it in
    // memory — those diverge after a GUI restart, and the tree already knows the
    // answer. Asking "is it open" hid the action from a dataset sitting right
    // there in the tree, which reads as missing rather than unavailable.
    const canTransfer = _folderContextIsDataset || isOpenedDataset || _folderContextIsModelRun;
    const hubUpload = document.getElementById('folder-ctx-hub-upload');
    const hubDownload = document.getElementById('folder-ctx-hub-download');
    const hubSep = document.getElementById('folder-ctx-hub-separator');
    if (hubUpload) hubUpload.style.display = canTransfer ? '' : 'none';
    if (hubDownload) hubDownload.style.display = canTransfer ? '' : 'none';
    if (hubSep) hubSep.style.display = canTransfer ? '' : 'none';
    // Copy and delete act on a dataset directory. The caller says whether this
    // path is one — a source folder and a model run share this menu, and
    // neither can be duplicated or deleted through these routes.
    for (const id of ['folder-ctx-duplicate', 'folder-ctx-delete-separator', 'folder-ctx-delete']) {
        const el = document.getElementById(id);
        if (el) el.style.display = _folderContextIsDataset ? '' : 'none';
    }
    menu.classList.add('visible');
    _positionContextMenu(menu, e.clientX, e.clientY);
}

function folderContextAction(action) {
    if (!_folderContextPath) return;
    if (action === 'duplicate') {
        duplicateDatasetAt(_folderContextPath);
    } else if (action === 'delete-files') {
        deleteDatasetFilesAt(_folderContextPath);
    } else if (action === 'open-in-files') {
        fetch('/api/datasets/open-in-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: _folderContextPath })
        }).catch(e => console.error('Failed to open file manager:', e));
    } else if (action === 'test-on-robot') {
        if (typeof testModelOnRobot === 'function') {
            testModelOnRobot(_folderContextPath);
        }
    } else if (action === 'merge-into') {
        openMergeModal(_folderContextPath);
    } else if (action === 'split-stereo') {
        openSplitStereoModal(_folderContextPath);
    } else if (action === 'hub-upload') {
        hubUploadDataset(_folderContextPath, _folderContextIsModelRun ? 'model' : 'dataset');
    } else if (action === 'hub-download') {
        hubDownloadDataset(_folderContextPath, _folderContextIsModelRun ? 'model' : 'dataset');
    }
    hideContextMenu();
}

// --- Split Stereo modal ---

async function openSplitStereoModal(id) {
    // The folder context menu passes the dataset id, the same value openMergeModal
    // receives and indexes `datasets` with.
    if (!datasets[id]) { alert('Open the dataset first.'); return; }
    let cams;
    try {
        const r = await fetch(`/api/process/stereo-candidates/${encodeURIComponent(id)}`);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        cams = (await r.json()).cameras || [];
    } catch (e) {
        alert(`Could not read cameras: ${e.message}`);
        return;
    }
    const splittable = cams.filter((c) => c.splittable);
    if (!splittable.length) { alert('No camera in this dataset has an even width, so none can be a side-by-side pair.'); return; }

    const suffix = '_split';
    const base = (datasets[id]?.repo_id || '').split('/').pop() || 'dataset';
    const rows = splittable.map((c) => `
        <label class="split-cam">
            <input type="checkbox" value="${c.name}" ${c.likely_stereo ? 'checked' : ''}>
            <span class="split-cam-name">${c.name}</span>
            <span class="split-cam-dims">${c.width}&times;${c.height} &rarr; ${c.channels[0]}, ${c.channels[1]} @ ${c.width / 2}&times;${c.height}</span>
            ${c.likely_stereo ? '' : '<span class="split-cam-note">not obviously stereo</span>'}
        </label>`).join('');

    const modal = document.createElement('div');
    modal.className = 'proc-modal';
    modal.style.display = 'flex';
    modal.innerHTML = `
        <div class="proc-box">
            <div class="proc-head"><span class="proc-title">Split stereo camera</span>
                <button class="proc-close" title="close (Esc)">&times;</button></div>
            <div class="proc-body">
                <div class="proc-hint">Each selected camera is replaced by two channels, one per eye.
                    The source dataset is not modified. Videos are re-encoded, so this takes roughly
                    a minute per 800 frames.</div>
                <div class="split-cams">${rows}</div>
                <div class="proc-row"><div class="proc-grow">
                    <label class="proc-label">New dataset name</label>
                    <input class="split-name" type="text" value="${base}${suffix}"></div></div>
                <div class="proc-error"></div>
                <div class="proc-actions-row">
                    <button class="proc-start">Split</button>
                </div>
            </div>
        </div>`;
    document.body.appendChild(modal);
    const close = () => modal.remove();
    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
    modal.querySelector('.proc-close').addEventListener('click', close);
    modal.querySelector('.proc-start').addEventListener('click', async () => {
        const picked = [...modal.querySelectorAll('.split-cam input:checked')].map((i) => i.value);
        const err = modal.querySelector('.proc-error');
        if (!picked.length) { err.textContent = 'Select at least one camera.'; return; }
        const btn = modal.querySelector('.proc-start');
        btn.disabled = true; btn.textContent = 'Starting…';
        try {
            const r = await fetch('/api/process/split-stereo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: id, cameras: picked,
                    out_name: modal.querySelector('.split-name').value.trim() || null,
                }),
            });
            const body = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : r.statusText);
            close();
            window.ProcessData?.refreshJobs?.();
        } catch (e) {
            err.textContent = e.message;
            btn.disabled = false; btn.textContent = 'Split';
        }
    });
}

// --- Merge Into modal ---
let _mergeSourceId = null;

function openMergeModal(sourceDatasetId) {
    _mergeSourceId = sourceDatasetId;
    const sourceDs = datasets[sourceDatasetId];
    if (!sourceDs) return;

    document.getElementById('merge-source-name').textContent = sourceDs.repo_id;
    document.getElementById('merge-status').textContent = '';
    document.getElementById('merge-execute-btn').disabled = false;

    // Populate target dropdown with other opened datasets
    const select = document.getElementById('merge-target-select');
    select.innerHTML = '';
    for (const [id, ds] of Object.entries(datasets)) {
        if (id === sourceDatasetId) continue;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${ds.repo_id} (${ds.total_episodes} ep)`;
        select.appendChild(opt);
    }

    updateMergePreview();
    const overlay = document.getElementById('merge-modal-overlay');
    overlay.style.display = 'flex';
}

let _mergeForce = false;

function updateMergePreview() {
    const preview = document.getElementById('merge-preview');
    const diffPanel = document.getElementById('merge-diff-panel');
    const targetId = document.getElementById('merge-target-select').value;
    const sourceDs = _mergeSourceId ? datasets[_mergeSourceId] : null;
    const targetDs = targetId ? datasets[targetId] : null;
    if (!sourceDs || !targetDs) { preview.textContent = ''; diffPanel.style.display = 'none'; return; }

    const srcEps = sourceDs.total_episodes;
    const tgtEps = targetDs.total_episodes;
    preview.innerHTML =
        `<strong>${targetDs.repo_id}</strong> will go from ${tgtEps} to ${tgtEps + srcEps} episodes ` +
        `(+${srcEps} from ${sourceDs.repo_id}).<br>` +
        `This modifies the target dataset on disk.`;

    // Reset diff panel and force state on target change
    diffPanel.style.display = 'none';
    _mergeForce = false;
    const btn = document.getElementById('merge-execute-btn');
    btn.textContent = 'Merge (modifies target)';
    btn.style.background = '#c24038';
    document.getElementById('merge-status').textContent = '';

    // Run validation in background
    _validateMerge(sourceDs, targetDs, targetId);
}

async function _validateMerge(sourceDs, targetDs, targetId) {
    try {
        const res = await fetch('/api/edits/merge-into/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dataset_id: _mergeSourceId,
                target_dataset_id: targetId,
            })
        });
        if (!res.ok) return;
        const data = await res.json();
        _renderMergeDiff(data);
    } catch (e) { /* ignore */ }
}

function _renderMergeDiff(validation) {
    const diffPanel = document.getElementById('merge-diff-panel');
    const btn = document.getElementById('merge-execute-btn');

    if (validation.compatible) {
        diffPanel.style.display = 'none';
        _mergeForce = false;
        btn.textContent = 'Merge (modifies target)';
        btn.style.background = '#c24038';
        btn.disabled = false;
        return;
    }

    // Build diff HTML
    let html = '';
    for (const m of validation.mismatches) {
        if (m.field === 'features') {
            if (m.target_only.length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Only in target:</span>`;
                for (const k of m.target_only)
                    html += `<span class="merge-diff-removed">${_esc(k)}</span>`;
                html += `</div>`;
            }
            if (m.source_only.length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Only in source:</span>`;
                for (const k of m.source_only)
                    html += `<span class="merge-diff-added">${_esc(k)}</span>`;
                html += `</div>`;
            }
            if (Object.keys(m.shared_diff).length) {
                html += `<div class="merge-diff-section"><span class="merge-diff-label">Different definitions:</span>`;
                for (const [k, v] of Object.entries(m.shared_diff)) {
                    html += `<details><summary>${_esc(k)}</summary>` +
                        `<div class="merge-diff-json">` +
                        `<div class="merge-diff-removed"><strong>target:</strong><pre>${_esc(JSON.stringify(v.target, null, 2))}</pre></div>` +
                        `<div class="merge-diff-added"><strong>source:</strong><pre>${_esc(JSON.stringify(v.source, null, 2))}</pre></div>` +
                        `</div></details>`;
                }
                html += `</div>`;
            }
        } else {
            html += `<div class="merge-diff-section">` +
                `<span class="merge-diff-label">${_esc(m.field)}:</span> ` +
                `<span class="merge-diff-removed">${_esc(String(m.target))}</span> (target) vs ` +
                `<span class="merge-diff-added">${_esc(String(m.source))}</span> (source)` +
                `</div>`;
        }
    }

    diffPanel.innerHTML = `<div class="merge-diff-header">Mismatches found</div>${html}`;
    diffPanel.style.display = 'block';

    // Switch button to force mode
    _mergeForce = true;
    btn.textContent = 'Force merge (skip validation)';
    btn.style.background = '#8b4513';
    btn.disabled = false;
}

function _esc(s) { const d = document.createElement('span'); d.textContent = s; return d.innerHTML; }

function closeMergeModal() {
    document.getElementById('merge-modal-overlay').style.display = 'none';
    document.getElementById('merge-diff-panel').style.display = 'none';
    _mergeSourceId = null;
    _mergeForce = false;
}

async function executeMerge() {
    const targetId = document.getElementById('merge-target-select').value;
    if (!_mergeSourceId || !targetId) return;

    const sourceDs = datasets[_mergeSourceId];
    const targetDs = datasets[targetId];

    const forceLabel = _mergeForce ? '\n\nWARNING: Skipping validation - features/metadata may differ!' : '';
    if (!confirm(
        `Merge ${sourceDs.total_episodes} episodes from "${sourceDs.repo_id}" ` +
        `into "${targetDs.repo_id}"?\n\n` +
        `This will modify "${targetDs.repo_id}" on disk.${forceLabel}`
    )) return;

    const btn = document.getElementById('merge-execute-btn');
    const status = document.getElementById('merge-status');
    btn.disabled = true;
    status.textContent = 'Merging...';

    try {
        const res = await fetch('/api/edits/merge-into', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dataset_id: _mergeSourceId,
                target_dataset_id: targetId,
                force: _mergeForce,
            })
        });

        if (res.status === 423) {
            status.textContent = 'Dataset is busy, please wait.';
            btn.disabled = false;
            return;
        }

        const data = await res.json();
        if (!res.ok) {
            status.textContent = data.detail || 'Merge failed';
            btn.disabled = false;
            return;
        }

        closeMergeModal();
        showToast('Merge Complete', data.message, 'info');

        // Refresh the target dataset's episodes in the tree
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(targetId)}/episodes`);
            if (epRes.ok) {
                episodes[targetId] = await epRes.json();
                datasets[targetId].total_episodes = episodes[targetId].length;
                datasets[targetId].total_frames = episodes[targetId].reduce((s, e) => s + e.length, 0);
            }
        } catch (e) { /* ignore */ }
        renderTree();
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        btn.disabled = false;
    }
}

// Close merge modal on overlay click
document.addEventListener('click', (e) => {
    const overlay = document.getElementById('merge-modal-overlay');
    if (e.target === overlay) closeMergeModal();
});

function contextAction(action) {
    if (!contextMenuTarget) return;
    const { datasetId, episodeIndex } = contextMenuTarget;

    if (action === 'view') {
        const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
        if (ep) selectEpisode(datasetId, episodeIndex, ep.video_length || ep.length);
    } else if (action === 'rerun') {
        const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
        if (ep) {
            selectEpisode(datasetId, episodeIndex, ep.video_length || ep.length);
            launchRerun();
        }
    } else if (action === 'replay') {
        // Switch to Run tab → Replay workflow with this episode pre-selected
        if (typeof selectWorkflow === 'function') selectWorkflow('replay');
        switchTab('run');
        // After tab init renders the form, select the right episode
        setTimeout(() => {
            const sel = document.getElementById('run-replay-episode');
            if (sel) {
                const val = `${datasetId}:${episodeIndex}`;
                sel.value = val;
                if (typeof _onReplayEpisodeChange === 'function') _onReplayEpisodeChange();
            }
        }, 50);
    } else if (action === 'delete') {
        markEpisodeDeleted(datasetId, episodeIndex);
    } else if (action === 'undelete') {
        unmarkEpisodeDeleted(datasetId, episodeIndex);
    } else if (action === 'cleartrim') {
        clearEpisodeTrim(datasetId, episodeIndex);
    }

    hideContextMenu();
}

async function clearEpisodeTrim(datasetId, episodeIndex) {
    // Find and remove the trim edit
    const editIndex = pendingEdits.findIndex(
        e => e.dataset_id === datasetId && e.episode_index === episodeIndex && e.edit_type === 'trim'
    );
    if (editIndex >= 0) {
        try {
            const res = await fetch(`/api/edits/${editIndex}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(await res.text());
            await refreshPendingEdits();
            setStatus(`Trim cleared for episode ${episodeIndex}`);
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }
}

// Edit operations
async function markEpisodeDeleted(datasetId, episodeIndex) {
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }
    try {
        const res = await fetch('/api/edits/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId, episode_index: episodeIndex })
        });
        if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus(`Episode ${episodeIndex} marked for deletion`);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

async function unmarkEpisodeDeleted(datasetId, episodeIndex) {
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }
    // Find and remove the delete edit
    const editIndex = pendingEdits.findIndex(
        e => e.dataset_id === datasetId && e.episode_index === episodeIndex && e.edit_type === 'delete'
    );
    if (editIndex >= 0) {
        try {
            const res = await fetch(`/api/edits/${editIndex}`, { method: 'DELETE' });
            if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
            if (!res.ok) throw new Error(await res.text());
            await refreshPendingEdits();
            setStatus(`Episode ${episodeIndex} restored`);
        } catch (e) {
            setStatus('Error: ' + e.message);
        }
    }
}

function deleteCurrentEpisode() {
    if (currentDataset && currentEpisode !== null) {
        if (isEpisodeDeleted(currentDataset, currentEpisode)) {
            unmarkEpisodeDeleted(currentDataset, currentEpisode);
        } else {
            markEpisodeDeleted(currentDataset, currentEpisode);
        }
    }
}

async function refreshPendingEdits() {
    try {
        const res = await fetch('/api/edits');
        const data = await res.json();
        pendingEdits = data.edits;
        renderTree();
        loadTrimForCurrentEpisode();
        if (window.FeatureEditing) window.FeatureEditing.onPendingEditsChanged();
    } catch (e) {
        console.error('Failed to refresh edits:', e);
    }
}

function updateEditsBar() {
    const bar = document.getElementById('edits-bar');
    const count = document.getElementById('edits-count');
    if (pendingEdits.length > 0) {
        bar.classList.add('visible');
        count.textContent = `${pendingEdits.length} pending edit${pendingEdits.length > 1 ? 's' : ''}`;
    } else {
        bar.classList.remove('visible');
    }
}

let datasetBusy = false;

function setEditingEnabled(enabled) {
    datasetBusy = !enabled;
    const bar = document.getElementById('edits-bar');
    if (bar) {
        bar.querySelectorAll('button').forEach(btn => btn.disabled = !enabled);
    }
}

async function discardEdits() {
    if (!confirm('Discard all pending edits?')) return;
    setEditingEnabled(false);
    try {
        const res = await fetch('/api/edits/discard', { method: 'POST' });
        if (res.status === 423) {
            setStatus('Dataset is busy, please wait');
            return;
        }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus('All edits discarded');
    } catch (e) {
        setStatus('Error: ' + e.message);
    } finally {
        setEditingEnabled(true);
    }
}

async function applyEdits() {
    if (!currentDataset) {
        setStatus('No dataset selected');
        return;
    }
    if (!confirm(
        `Apply ${pendingEdits.length} edit(s) to disk? This cannot be undone.\n\n` +
        `Pause any training jobs reading this dataset before continuing — ` +
        `the GUI server serializes its own writes, but external readers see ` +
        `torn state across shards mid-Save.`
    )) return;

    setEditingEnabled(false);
    setStatus('Applying edits...');
    try {
        const res = await fetch(`/api/edits/apply?dataset_id=${encodeURIComponent(currentDataset)}`, {
            method: 'POST'
        });
        if (res.status === 423) {
            setStatus('Dataset is busy, please wait');
            return;
        }
        const data = await res.json();
        if (data.status === 'ok' || data.status === 'partial') {
            // Reload dataset episodes
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(currentDataset)}/episodes`);
            episodes[currentDataset] = await epRes.json();
            // Sync episode count after edits may have changed it
            if (datasets[currentDataset]) {
                datasets[currentDataset].total_episodes = episodes[currentDataset].length;
            }
            await refreshPendingEdits();
            if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();

            // Re-select current episode (or nearest neighbour if deleted)
            if (currentDataset && currentEpisode !== null) {
                const epList = episodes[currentDataset] || [];
                const stillExists = epList.find(e => e.episode_index === currentEpisode);
                if (stillExists) {
                    // Re-select to refresh view (e.g. after trim changed length)
                    selectEpisode(currentDataset, currentEpisode, stillExists.video_length || stillExists.length);
                } else if (epList.length > 0) {
                    // Select nearest neighbour
                    const nearest = epList.reduce((best, e) =>
                        Math.abs(e.episode_index - currentEpisode) < Math.abs(best.episode_index - currentEpisode) ? e : best
                    );
                    selectEpisode(currentDataset, nearest.episode_index, nearest.video_length || nearest.length);
                } else {
                    currentEpisode = null;
                    renderCameraGrid();
                }
            }

            setStatus(data.message);
        } else {
            throw new Error(data.message);
        }
    } catch (e) {
        setStatus('Error: ' + e.message);
    } finally {
        setEditingEnabled(true);
    }
}

// Trim functions
function updateTrimDisplay() {
    const cutLeft = document.getElementById('trim-cut-left');
    const cutRight = document.getElementById('trim-cut-right');

    if (!currentDataset || currentEpisode === null || totalFrames === 0) {
        document.getElementById('trim-region').classList.remove('visible');
        document.getElementById('trim-controls').classList.remove('visible');
        cutLeft.classList.remove('visible');
        cutRight.classList.remove('visible');
        return;
    }

    const region = document.getElementById('trim-region');
    const leftPct = (trimStart / (totalFrames - 1)) * 100;
    const rightPct = ((trimEnd - 1) / (totalFrames - 1)) * 100;
    const widthPct = rightPct - leftPct;

    region.style.left = `${leftPct}%`;
    region.style.width = `${widthPct}%`;
    region.classList.add('visible');

    // Show cut zones (red tint for regions that will be removed)
    if (trimStart > 0) {
        cutLeft.style.width = `${leftPct}%`;
        cutLeft.classList.add('visible');
    } else {
        cutLeft.classList.remove('visible');
    }

    if (trimEnd < totalFrames) {
        cutRight.style.width = `${100 - rightPct}%`;
        cutRight.classList.add('visible');
    } else {
        cutRight.classList.remove('visible');
    }

    // Show trim controls if trim is different from full range
    const controls = document.getElementById('trim-controls');
    const info = document.getElementById('trim-info');
    if (trimStart > 0 || trimEnd < totalFrames) {
        const framesKept = trimEnd - trimStart;
        info.textContent = `Keep: frames ${trimStart}-${trimEnd - 1} (${framesKept} of ${totalFrames})`;
        controls.classList.add('visible');
    } else {
        controls.classList.remove('visible');
    }
}

function resetTrim() {
    trimStart = 0;
    trimEnd = totalFrames;
    updateTrimDisplay();
}

async function saveTrim() {
    if (!currentDataset || currentEpisode === null) return;
    if (datasetBusy) { setStatus('Dataset is busy, please wait'); return; }

    try {
        const res = await fetch('/api/edits/trim', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: currentDataset,
                episode_index: currentEpisode,
                start_frame: trimStart,
                end_frame: trimEnd
            })
        });
        if (res.status === 423) { setStatus('Dataset is busy, please wait'); return; }
        if (!res.ok) throw new Error(await res.text());
        await refreshPendingEdits();
        setStatus(`Trim set: keeping frames ${trimStart}-${trimEnd - 1} of episode ${currentEpisode}`);
    } catch (e) {
        setStatus('Error: ' + e.message);
    }
}

function loadTrimForCurrentEpisode() {
    if (!currentDataset || currentEpisode === null) return;

    // Check if there's an existing trim edit for this episode
    const existingTrim = pendingEdits.find(
        e => e.dataset_id === currentDataset && e.episode_index === currentEpisode && e.edit_type === 'trim'
    );

    if (existingTrim) {
        trimStart = existingTrim.params.start_frame;
        trimEnd = existingTrim.params.end_frame;
    } else {
        trimStart = 0;
        trimEnd = totalFrames;
    }
    updateTrimDisplay();
}

// Make state accessible for other scripts (run.js, etc.). `datasets` and `episodes`
// are NOT assigned here: they are getter-only window props (see the defineProperties
// near the top), so an assignment is silently dropped in sloppy mode.
window.sourceDatasets = sourceDatasets;
// Awaits its scans: without that the promise resolves before any fetch lands,
// so callers cannot tell a refresh is still running and issue another.
window.refreshExpandedSources = async function() {
    await Promise.all([...expandedSources].map(sourcePath => scanSource(sourcePath)));
};

// Each tree caches a directory listing and nothing invalidates it, so a
// dataset or checkpoint written elsewhere stays invisible until a reload.
// Per tab, not global: a shared timestamp made switching data -> model inside
// the window skip the model refresh.
const _lastRefreshAt = {};
const SOURCE_RESCAN_MIN_INTERVAL_MS = 2000;

const REFRESH_BY_TAB = {
    data: () => window.refreshExpandedSources?.(),
    model: () => window.refreshExpandedModelSources?.(),
    robot: () => window.refreshRobotProfiles?.(),
};

// Selecting a tab always re-reads — it is the user asking to see it. Focus
// fires on every alt-tab without anyone asking, so it throttles.
window.refreshTabFromDisk = function (tabName, { throttle = false } = {}) {
    const refresh = REFRESH_BY_TAB[tabName];
    if (!refresh) return;
    const now = Date.now();
    if (throttle && now - (_lastRefreshAt[tabName] || 0) < SOURCE_RESCAN_MIN_INTERVAL_MS) return;
    _lastRefreshAt[tabName] = now;
    return refresh();
};

function rescanSourcesOnFocus() {
    const active = document.querySelector('.tab.active')?.dataset.tab;
    window.refreshTabFromDisk(active, { throttle: true });
}

window.addEventListener('focus', rescanSourcesOnFocus);
// focus misses a tab switched back inside an already-focused window.
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) rescanSourcesOnFocus();
});
window.refreshOpenedDatasets = async function() {
    for (const id of Object.keys(datasets)) {
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(id)}/episodes`);
            if (epRes.ok) {
                episodes[id] = await epRes.json();
                datasets[id].total_episodes = episodes[id].length;
            }
        } catch (e) { /* ignore per-dataset errors */ }
    }
    renderTree();
    if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
};


// Restore previously opened datasets
async function restoreOpenedDatasets() {
    try {
        const res = await fetch('/api/datasets/previously-opened');
        if (!res.ok) return;
        const items = await res.json();
        for (const item of items) {
            try {
                await openDataset(item.root);
            } catch (e) {
                console.warn(`Failed to restore dataset ${item.root}:`, e);
            }
        }
    } catch (e) {
        console.warn('Failed to restore opened datasets:', e);
    }
}

// --- HuggingFace Hub operations ---

async function checkHubAuth() {
    try {
        const res = await fetch('/api/datasets/hub/auth-status');
        const data = await res.json();
        // The owner half of a suggested repo id comes from here. This probe is
        // the only place the GUI learns who it is logged in as, so dropping the
        // username into the indicator's text and nowhere else left the suggested
        // id permanently owned by a namespace nobody has.
        window.hfUser = data.logged_in ? data.username : null;
        const el = document.getElementById('hf-auth-indicator');
        if (el) {
            // There is no login UI yet, so the indicator carries the command
            // itself: without it a logged-out operator sees only grey text and
            // has no route forward. It must run on the GUI *host* — the token
            // is read server-side, so logging in on the browser's machine does
            // nothing.
            el.textContent = data.logged_in
                ? `HF: @${data.username}`
                : 'HF: not logged in — run `huggingface-cli login` on the GUI host';
            el.title = data.logged_in
                ? `Logged in to HuggingFace Hub as @${data.username}. `
                  + 'Run `huggingface-cli login` on the GUI host to switch accounts, then reload.'
                : 'Not logged in to HuggingFace Hub. Run `huggingface-cli login` in a terminal on '
                  + 'the machine serving this GUI (not this browser\'s machine), then reload. '
                  + 'Gated repos additionally need access granted on their model page.';
            el.style.color = data.logged_in ? 'var(--text-secondary, #888)' : 'var(--text-tertiary, #555)';
        }
    } catch (e) { /* ignore */ }
}

let _hubDatasetId = null;
let _hubAction = null;  // 'upload' | 'download' | 'open-sync'
let _hubOpenSyncCtx = null;  // { body, detail } for 'open-sync' mode
let _hubRepoInfoTimer = null;

// Sentinel: the body was not JSON and the caller has already been told.
const HUB_RESPONSE_NOT_JSON = Symbol('hub-response-not-json');

/** Parse a Hub response body, surfacing a non-JSON error rather than a parse failure. */
async function hubParseResponse(res, status, btn) {
    const raw = await res.text();
    try {
        return raw ? JSON.parse(raw) : {};
    } catch {
        const first = raw.split('\n')[0].slice(0, 200) || `HTTP ${res.status}`;
        if (status) status.textContent = `Server error (${res.status}): ${first}`;
        if (btn) btn.disabled = false;
        return HUB_RESPONSE_NOT_JSON;
    }
}

function hubUploadDataset(datasetId, repoType) { openHubModal(datasetId, 'upload', { repoType }); }
function hubDownloadDataset(datasetId, repoType) { openHubModal(datasetId, 'download', { repoType }); }

// Enable/disable the Hub modal's primary button with a *visible* disabled
// state — the inline accent background overrides the browser's greyed-out
// styling, so a bare `disabled` looks identical to enabled. Dim + not-allowed
// cursor make a gated 'Download & Open' read as inert.
function setHubExecuteEnabled(enabled) {
    const b = document.getElementById('hub-execute-btn');
    b.disabled = !enabled;
    b.style.opacity = enabled ? '1' : '0.5';
    b.style.cursor = enabled ? 'pointer' : 'not-allowed';
}

// Transfer-path selector state. Kept in the DOM rather than a module var so
// the modal's reset-on-open path has a single source of truth.
// "repo" rather than "dataset": the same selector serves model uploads, where
// naming the wrong kind of thing reads as the dialog not knowing what it is
// about to send.
const _HUB_PATH_HINTS = {
    xet: 'Re-uploading an edited repo sends only what actually changed.',
    lfs: 'Try this if uploads stall. Re-uploading an edited repo sends whole files again.',
};

function setHubTransferPath(path) {
    const seg = document.getElementById('hub-path-seg');
    if (!seg) return;
    for (const b of seg.querySelectorAll('.hub-seg-btn')) {
        b.classList.toggle('sel', b.dataset.path === path);
    }
    const hint = document.getElementById('hub-path-hint');
    if (hint) hint.textContent = _HUB_PATH_HINTS[path] || '';
}

function hubTransferPath() {
    const sel = document.querySelector('#hub-path-seg .hub-seg-btn.sel');
    return sel ? sel.dataset.path : 'xet';
}

function openHubModal(datasetId, action, ctx) {
    _hubDatasetId = datasetId;
    _hubAction = action;
    _hubOpenSyncCtx = action === 'open-sync' ? (ctx || null) : null;

    const ds = datasetId != null ? datasets[datasetId] : null;
    // A model run is a checkpoint directory, not an opened LeRobotDataset, so it
    // is absent from `datasets` by construction, and the modal cannot infer the
    // kind from `datasets[datasetId]` alone. The caller says which it is.
    //
    // Deriving it from the context-menu global instead let an unrelated earlier
    // click decide: that flag latches on a model interaction and is only ever
    // cleared by a later right-click, while `open-sync` always passes a null id
    // — so a dataset repair dialog opened after any model action queried the
    // model namespace, found nothing, and disabled its own download button.
    _hubRepoType = (ctx && ctx.repoType) || 'dataset';
    // A dataset that is not open has no client-side record, but a transfer needs
    // only its directory and a repo id, and the tree supplied the directory.
    // Returning here made the menu item inert — a click that did nothing at all,
    // which is worse than the hidden item it replaced.
    if (action === 'open-sync' && !_hubOpenSyncCtx) return;

    const titleEl = document.getElementById('hub-modal-title');
    const btn = document.getElementById('hub-execute-btn');
    const statusEl = document.getElementById('hub-status');
    const repoInput = document.getElementById('hub-repo-input');
    const localInfoEl = document.getElementById('hub-local-info');
    const repoInfoEl = document.getElementById('hub-repo-info');

    btn.disabled = false;
    statusEl.textContent = '';
    repoInfoEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Loading remote info...</span>';

    // Restore reusable modal chrome: the metadata-inconsistent open-sync state
    // hides the repo input, remote-info panel, and execute button; the
    // missing-files state dims the button until the repo is confirmed.
    document.getElementById('hub-repo-input-row').style.display = '';
    repoInfoEl.style.display = '';
    btn.style.display = '';
    setHubExecuteEnabled(true);

    // Upload-only. Downloads honour the same HF flag, but the Xet download
    // route is CDN-backed and measured fast even on links where the Xet
    // *upload* endpoints stall, so exposing it there would be a knob with
    // no known use. Reset each open — this is a per-transfer choice, not a
    // sticky preference.
    const xetRow = document.getElementById('hub-xet-row');
    if (xetRow) {
        xetRow.style.display = action === 'upload' ? '' : 'none';
        setHubTransferPath('xet');
    }

    if (action === 'upload') {
        titleEl.textContent = 'Upload to Hub';
        btn.textContent = 'Upload';
        btn.style.background = 'var(--accent, #0e639c)';
        repoInput.value = ds
            ? ds.repo_id
            : _hubRepoType === 'model'
              ? defaultModelRepoId(datasetId)
              : defaultDatasetRepoId(datasetId);
        localInfoEl.innerHTML =
            (ds
                ? `<strong>Local:</strong> ${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames<br>`
                : `<strong>Local:</strong> model checkpoint<br>`) +
            `<span style="color:var(--text-tertiary,#666)">${ds ? ds.root : datasetId}</span>`;
    } else if (action === 'download') {
        titleEl.textContent = 'Download from Hub';
        btn.textContent = 'Download';
        btn.style.background = '#c24038';
        repoInput.value = ds
            ? ds.repo_id
            : _hubRepoType === 'model'
              ? defaultModelRepoId(datasetId)
              : defaultDatasetRepoId(datasetId);
        localInfoEl.innerHTML =
            (ds
                ? `<strong>Local:</strong> ${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames<br>`
                : `<strong>Local:</strong> model checkpoint<br>`) +
            `<span style="color:var(--text-tertiary,#666)">${ds ? ds.root : datasetId}</span>`;
    } else if (action === 'open-sync') {
        const { detail } = _hubOpenSyncCtx;
        const probs = (detail.problems || []).slice(0, 5)
            .map(p => `<div style="color:var(--text-tertiary,#666); font-size:11px;">• ${p}</div>`).join('');
        const more = (detail.problems || []).length > 5
            ? `<div style="color:var(--text-tertiary,#666); font-size:11px;">• (and ${detail.problems.length - 5} more)</div>`
            : '';

        if (detail.kind === 'metadata_inconsistent') {
            // The metadata contradicts itself — not a download problem. State it
            // faithfully and drop the Hub chrome; only the Cancel button remains.
            titleEl.textContent = "Couldn't open dataset — metadata is inconsistent";
            document.getElementById('hub-repo-input-row').style.display = 'none';
            repoInfoEl.style.display = 'none';
            btn.style.display = 'none';
            repoInput.value = '';  // nothing to look up; keeps fetchHubRepoInfo a no-op
            localInfoEl.innerHTML =
                `<span style="color:var(--text-tertiary,#666)">${detail.local_path}</span>` +
                `<div style="margin-top:6px;"><strong>Problem:</strong></div>${probs}${more}` +
                `<div style="margin-top:8px; color:var(--text-tertiary,#666); font-size:11px;">` +
                `The dataset's <code>info.json</code> and its episode metadata table disagree, so it can't be ` +
                `opened. This isn't a missing-files problem — there's nothing to download.</div>`;
        } else {
            // Missing files — re-downloadable when a Hub copy exists. Gate the
            // button until fetchHubRepoInfo() confirms the repo exists.
            titleEl.textContent = 'Open dataset — local cache is incomplete';
            btn.textContent = 'Download & Open';
            btn.style.background = 'var(--accent, #0e639c)';
            setHubExecuteEnabled(false);  // gated until fetchHubRepoInfo confirms the repo
            repoInput.value = detail.repo_id || '';
            localInfoEl.innerHTML =
                `<strong>Local cache:</strong> incomplete<br>` +
                `<span style="color:var(--text-tertiary,#666)">${detail.local_path}</span>` +
                `<div style="margin-top:6px;"><strong>Missing:</strong></div>${probs}${more}` +
                `<div style="margin-top:8px; color:var(--text-tertiary,#666); font-size:11px;">` +
                `If this dataset is on the Hub, <em>Download &amp; Open</em> fetches the missing files into the path ` +
                `above, then opens it. Progress prints to the server terminal — this dialog stays open until done.</div>`;
        }
    }

    document.getElementById('hub-modal-overlay').style.display = 'flex';
    fetchHubRepoInfo();
    if (action !== 'open-sync') {
        // A model has no episode or shard layout to diff, so it gets the one
        // comparison that does mean something for a checkpoint: which side was
        // written more recently.
        if (_hubRepoType === 'model') fetchModelFreshness();
        else fetchHubDiff();
    }
}

function closeHubModal() {
    document.getElementById('hub-modal-overlay').style.display = 'none';
    _hubRepoType = 'dataset';
    _hubDatasetId = null;
    _hubAction = null;
    _hubOpenSyncCtx = null;
}

function fetchHubRepoInfo() {
    clearTimeout(_hubRepoInfoTimer);
    _hubRepoInfoTimer = setTimeout(async () => {
        const repoId = document.getElementById('hub-repo-input').value.trim();
        const infoEl = document.getElementById('hub-repo-info');
        if (!repoId) { infoEl.innerHTML = ''; return; }

        infoEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Loading...</span>';
        try {
            const res = await fetch(
                `/api/datasets/hub/repo-info?repo_id=${encodeURIComponent(repoId)}` +
                `&repo_type=${encodeURIComponent(_hubRepoType)}`,
            );
            const data = await res.json();
            const hubUrl = hubRepoUrl(repoId, _hubRepoType);
            const linkHtml = `<a href="${hubUrl}" target="_blank" rel="noopener noreferrer" style="color:#61afef; text-decoration:none;" title="Open on HuggingFace Hub">${repoId} ↗</a>`;
            if (!data.exists) {
                infoEl.innerHTML = _hubAction === 'upload'
                    ? `<span style="color:#e5c07b">New repo — will be created on upload</span><br><span style="color:var(--text-tertiary,#666)">URL: ${linkHtml}</span>`
                    : '<span style="color:#e06c75">Repo not found on Hub</span>';
                if (_hubAction === 'open-sync') {
                    // Not on the Hub → nothing to download. Keep the action
                    // blocked instead of offering a misleading 'Download & Open'.
                    setHubExecuteEnabled(false);
                    document.getElementById('hub-execute-btn').title =
                        "This repo isn't on the Hub — nothing to download.";
                    document.getElementById('hub-status').innerHTML =
                        '<span style="color:#e06c75">Not on the Hub — nothing to download.</span>';
                }
                return;
            }
            // Repo exists: for open-sync this makes 'Download & Open' a valid
            // action, so lift the gate set when the modal opened.
            if (_hubAction === 'open-sync') {
                setHubExecuteEnabled(true);
                document.getElementById('hub-execute-btn').title = '';
                document.getElementById('hub-status').textContent = '';
            }
            const epInfo = data.total_episodes != null
                ? `${data.total_episodes} episodes, ${data.total_frames?.toLocaleString() || '?'} frames`
                : `${data.files} files`;
            infoEl.innerHTML =
                `<strong>Remote:</strong> ${linkHtml}<br>` +
                `${epInfo}, ${data.total_size_mb} MB` +
                `${data.private ? ' (private)' : ''}<br>` +
                `Last modified: ${data.last_modified || 'unknown'}<br>` +
                `Downloads: ${data.downloads} | SHA: ${data.sha || '?'}`;
        } catch (e) {
            infoEl.innerHTML = `<span style="color:#e06c75">Failed to fetch info</span>`;
        }
        // Also refresh the comparison when the repo changes. Which comparison
        // depends on the repo kind: `/hub/diff` is a dataset route and 404s for
        // a run path, and its error handler blanks the shared status line — so
        // calling it unconditionally erased the freshness verdict a few hundred
        // milliseconds after it appeared, and again on every keystroke here.
        if (_hubRepoType === 'model') fetchModelFreshness(); else fetchHubDiff();
    }, 400);
}

// Which side is newer. The file-by-file diff is a dataset notion; for a
// checkpoint the useful question is simply whether the Hub copy is behind.
async function fetchModelFreshness() {
    const repoId = document.getElementById('hub-repo-input').value.trim();
    const statusEl = document.getElementById('hub-status');
    if (!repoId || !_hubDatasetId) { statusEl.textContent = ''; return; }
    try {
        const [localRes, remoteRes] = await Promise.all([
            fetch(`/api/models/run-mtime?path=${encodeURIComponent(_hubDatasetId)}`),
            fetch(`/api/datasets/hub/repo-info?repo_id=${encodeURIComponent(repoId)}&repo_type=model`),
        ]);
        const local = await localRes.json();
        const remote = await remoteRes.json();
        if (!remote.exists) { statusEl.textContent = ''; return; }  // handled by repo-info
        if (!remote.last_modified || !local.mtime) { statusEl.textContent = ''; return; }

        const localDate = new Date(local.mtime * 1000);
        const remoteDate = new Date(remote.last_modified);
        const d = (x) => x.toISOString().slice(0, 10);
        // Compared at the granularity shown. Comparing timestamps while
        // printing dates lets "Local is newer — 2025-10-09 vs 2025-10-09"
        // through, which reads as a bug in the dialog rather than a fact.
        if (d(localDate) === d(remoteDate)) {
            statusEl.innerHTML = `<span style="color:#98c379">Same date — ${d(localDate)}</span>`;
        } else if (localDate > remoteDate) {
            statusEl.innerHTML =
                `<span style="color:#e5c07b">Local is newer — ${d(localDate)} vs ${d(remoteDate)} on the Hub</span>`;
        } else {
            statusEl.innerHTML =
                `<span style="color:#e5c07b">Hub is newer — ${d(remoteDate)} vs ${d(localDate)} locally</span>`;
        }
    } catch {
        statusEl.textContent = '';
    }
}

async function fetchHubDiff() {
    if (!_hubDatasetId) return;
    const repoId = document.getElementById('hub-repo-input').value.trim();
    const statusEl = document.getElementById('hub-status');
    if (!repoId) { statusEl.textContent = ''; return; }

    statusEl.innerHTML = '<span style="color:var(--text-tertiary,#666)">Comparing...</span>';
    try {
        const res = await fetch(`/api/datasets/${encodeURIComponent(_hubDatasetId)}/hub/diff?repo_id=${encodeURIComponent(repoId)}`);
        const data = await res.json();
        if (data.status === 'error') {
            statusEl.textContent = data.message;
            return;
        }
        if (data.in_sync) {
            statusEl.innerHTML = '<span style="color:#98c379">In sync — no differences</span>';
            return;
        }
        let parts = [];
        if (data.modified.length > 0) parts.push(`${data.modified.length} modified`);
        if (data.local_only.length > 0) parts.push(`${data.local_only.length} local only`);
        if (data.remote_only.length > 0) parts.push(`${data.remote_only.length} remote only`);
        statusEl.innerHTML = `<span style="color:#e5c07b">${parts.join(', ')} (${data.unchanged} unchanged)</span>`;
    } catch (e) {
        statusEl.textContent = '';
    }
}

async function executeHubAction() {
    // open-sync uses _hubOpenSyncCtx instead of _hubDatasetId
    if (!_hubAction) return;
    if (_hubAction !== 'open-sync' && !_hubDatasetId) return;
    if (_hubAction === 'open-sync' && !_hubOpenSyncCtx) return;

    const repoId = document.getElementById('hub-repo-input').value.trim();
    if (!repoId) return;

    const btn = document.getElementById('hub-execute-btn');
    const status = document.getElementById('hub-status');
    btn.disabled = true;

    // 'open-sync' is a separate, synchronous flow — different endpoint
    // (the dataset-open one, not /hub/{upload,download}) and the response
    // body IS the opened dataset. Keep it blocking; promoting it to the
    // background-job pattern would require touching the open codepath.
    if (_hubAction === 'open-sync') {
        status.textContent = 'Downloading & opening…';
        try {
            const res = await fetch('/api/datasets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ..._hubOpenSyncCtx.body,
                    repo_id: repoId,
                    confirm_hub_sync: true,
                }),
            });
            if (!res.ok) {
                let msg = 'Open failed';
                try {
                    const data = await res.json();
                    msg = (data && data.detail) || msg;
                    if (typeof msg === 'object') msg = msg.message || JSON.stringify(msg);
                } catch (_) {}
                status.textContent = msg;
                btn.disabled = false;
                return;
            }
            const data = await res.json();
            closeHubModal();
            await _completeOpen(data);
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
            btn.disabled = false;
        }
        return;
    }

    // Upload / download: kick off a background job, close the modal
    // immediately, surface progress in the top-bar Transfers tray.
    // Models have their own routes and take the path in the body; the dataset
    // ones resolve an opened LeRobotDataset, which a checkpoint is not.
    const endpoint = _hubRepoType === 'model'
        ? `/api/models/hub/${_hubAction}`
        : `/api/datasets/${encodeURIComponent(_hubDatasetId)}/hub/${_hubAction}`;
    const body = { repo_id: repoId };
    // The model routes carry the run directory in the body — they take no path
    // segment, the router having greedy catch-all routes a suffix would capture.
    if (_hubRepoType === 'model') body.path = _hubDatasetId;
    if (_hubAction === 'upload' && hubTransferPath() === 'lfs') body.disable_xet = true;
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (res.status === 401) {
            status.textContent = 'Not logged in. Run `huggingface-cli login` in terminal.';
            btn.disabled = false;
            return;
        }
        if (res.status === 423) {
            status.textContent = 'Dataset is busy, please wait.';
            btn.disabled = false;
            return;
        }

        // An unhandled server fault returns a plain-text body, so parsing it as
        // JSON throws and buries the real failure under a syntax error about the
        // letter I. Read the text first and report it verbatim when it is not
        // JSON: a 500 should still say what went wrong.
        const data = await hubParseResponse(res, status, btn);
        if (data === HUB_RESPONSE_NOT_JSON) return;
        if (!res.ok) {
            // 409 with job_id = a Hub transfer is already running for this dataset.
            if (res.status === 409 && data?.detail?.job_id) {
                closeHubModal();
                Transfers.refreshNow();
                Transfers.openPopover();
                showToast('Transfer already running', 'See the Transfers tray for progress.', 'info', 4000);
                return;
            }
            // 409 with code=incomplete_local_state = the upload-time completeness
            // check found files present on the remote but missing locally. This is
            // the download-fail-then-upload guardrail. Ask the user before proceeding.
            if (res.status === 409 && data?.detail?.code === 'incomplete_local_state') {
                const missing = (data.detail.missing_locally || []).slice(0, 5);
                const incomplete = (data.detail.incomplete_locally || []).slice(0, 5);
                const detailLines = [];
                if (missing.length) detailLines.push('Missing: ' + missing.join(', '));
                if (incomplete.length) detailLines.push('Incomplete: ' + incomplete.join(', '));
                const ok = confirm(
                    'Your local copy is missing files that exist on the remote ' +
                    '(likely from an interrupted download). Uploading would push a ' +
                    'worse-than-remote state, but HF history preserves the old commit ' +
                    'so the prior state remains recoverable.\n\n' +
                    detailLines.join('\n') +
                    '\n\nUpload anyway?'
                );
                if (!ok) {
                    status.textContent = 'Cancelled. Re-download first to restore the missing files.';
                    btn.disabled = false;
                    return;
                }
                // Re-issue with confirm_force=true to bypass the guardrail.
                const force = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...body, confirm_force: true }),
                });
                if (!force.ok) {
                    const fd = await force.json().catch(() => ({}));
                    status.textContent = fd?.detail?.message || fd?.detail || 'Upload failed';
                    btn.disabled = false;
                    return;
                }
                closeHubModal();
                Transfers.refreshNow();
                showToast('Upload started', 'Progress in the Transfers tray (top right).', 'info', 4000);
                return;
            }
            status.textContent = (data.detail && data.detail.message) || data.detail || 'Operation failed';
            btn.disabled = false;
            return;
        }

        // Job kicked off. Close modal, ping the tray, point the user at it.
        // The verb is read before closing: closeHubModal() clears _hubAction,
        // so reading it afterwards made every upload announce "Download started".
        const verb = _hubAction === 'upload' ? 'Upload' : 'Download';
        closeHubModal();
        Transfers.refreshNow();
        showToast(`${verb} started`, 'Progress in the Transfers tray (top right).', 'info', 4000);
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        btn.disabled = false;
    }
}

// Hub modal: dismissal is intentionally Cancel-button-only. The transfer
// settings (repo id, sync direction) are easy to misclick away from when
// click-on-overlay closes the dialog, so we don't bind a backdrop handler.

// ── Hub Transfers tray ─────────────────────────────────────────────────
//
// Top-bar indicator + popover that lists every active or recently-finished
// Hub transfer. The single polling loop covers all transfers for the whole
// app — no more per-modal polling. Poll cadence:
//   - while any job is active: 1 Hz (counters tick once per file, faster
//     polling would just re-render the same snapshot)
//   - while idle (only-finished or empty): off — refreshNow() restarts it
//     when a fresh transfer is kicked off via executeHubAction
//
// Keyed by Transfers.* as a tiny module so executeHubAction + listeners
// don't have to know the internal state names. Globally exposed so the
// inline `onclick` handlers in the tab bar can call into it.

const Transfers = (function () {
    let _pollTimer = null;
    let _jobs = [];               // latest snapshot from /hub/jobs
    let _completionShown = new Set();  // job_ids we've already toasted
    let _popoverOpen = false;

    function _fmtBytes(n) {
        if (!n) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        let i = 0;
        while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
        return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function _isActive(j) {
        return j.status === 'pending' || j.status === 'running' || j.status === 'cancelling';
    }

    function _fmtRate(bps) {
        if (!bps || bps <= 0) return '';
        return `${_fmtBytes(bps)}/s`;
    }

    function _fmtDuration(s) {
        s = Math.round(s);
        if (s < 60) return `${s}s`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m ${s % 60}s`;
        return `${Math.floor(m / 60)}h ${m % 60}m`;
    }

    // Mirrors hub_jobs.STALL_THRESHOLD_S. A transfer with no observed byte
    // movement for this long gets an explicit warning rather than leaving
    // the user to guess whether a static number means "slow" or "hung".
    const STALL_THRESHOLD_S = 90;

    function _renderIndicator() {
        const ind = document.getElementById('transfers-indicator');
        const label = document.getElementById('transfers-indicator-label');
        if (!ind || !label) return;
        const active = _jobs.filter(_isActive);
        // Indicator is always present (the HTML omits `hidden`); we only
        // toggle the active styling + count badge based on job state. The
        // user can always find the popover entry point.
        ind.hidden = false;
        if (active.length > 0) {
            ind.classList.add('active');
            label.textContent = `Transfers · ${active.length}`;
        } else {
            ind.classList.remove('active');
            label.textContent = `Transfers`;
        }
    }

    // ── Past outcomes ──────────────────────────────────────────────────
    // Read from the durable history file, not the job registry: the registry
    // drops a job 30 minutes after it finishes and loses everything on a
    // server restart, so a long upload could complete and leave the user no
    // way to tell success from failure. Loaded lazily — most opens of the
    // tray are to watch something live, not to audit last week.
    let _history = null;      // null = not fetched yet
    let _historyOpen = false;

    function _fmtWhen(ts) {
        if (!ts) return '';
        const secs = Math.max(0, Date.now() / 1000 - ts);
        if (secs < 90) return 'just now';
        if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
        if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
        return new Date(ts * 1000).toLocaleDateString();
    }

    function _historyCardHtml(h) {
        const dir = h.direction === 'upload' ? '▲' : '▼';
        // State the outcome in a word, then the evidence for it. "Complete"
        // with no numbers is the readout that started all this.
        const cls = h.status === 'complete' ? 'complete' : (h.status === 'cancelled' ? 'cancelled' : 'failed');
        const size = h.bytes_total > 0 ? _fmtBytes(h.bytes_total) : '';
        const files = h.files_total > 0 ? `${h.files_total} files` : '';
        const took = h.duration_s > 0 ? _fmtDuration(h.duration_s) : '';
        const facts = [size, files, took].filter(Boolean).join(' · ');
        const link = h.pr_url || hubRepoUrl(h.repo_id, h.repo_type);
        const why = h.status !== 'complete' && h.error
            ? `<div class="transfer-msg ${cls}" title="${h.error.replace(/"/g, '&quot;')}">${h.error.slice(0, 140)}</div>`
            : '';
        return (
            `<div class="transfer-card ${cls}">` +
              `<div class="transfer-card-head">` +
                `<span class="transfer-direction">${dir}</span>` +
                `<a class="transfer-repo" href="${link}" target="_blank" rel="noopener noreferrer" title="${h.repo_id}">${h.repo_id}</a>` +
                `<span style="margin-left:auto; font-size:10px; color:var(--text-tertiary,#888);">${_fmtWhen(h.ts)}</span>` +
              `</div>` +
              `<div class="transfer-stats">${h.status}${facts ? ' · ' + facts : ''}</div>` +
              why +
            `</div>`
        );
    }

    async function _loadHistory() {
        try {
            const res = await fetch('/api/datasets/hub/history?limit=20');
            const data = await res.json();
            _history = data.transfers || [];
        } catch (e) {
            _history = [];
        }
        _renderHistory();
    }

    function _renderHistory() {
        const section = document.getElementById('transfers-history-section');
        const list = document.getElementById('transfers-history-list');
        const btn = document.getElementById('transfers-history-toggle');
        if (!section || !list || !btn) return;
        // Only offer it when there is something to show that isn't already
        // on screen as a live card.
        const liveIds = new Set(_jobs.map(j => j.job_id));
        const past = (_history || []).filter(h => !liveIds.has(h.job_id));
        section.hidden = past.length === 0;
        btn.textContent = _historyOpen ? 'Hide' : 'Show';
        list.hidden = !_historyOpen;
        if (_historyOpen) list.innerHTML = past.map(_historyCardHtml).join('');
    }

    function toggleHistory() {
        _historyOpen = !_historyOpen;
        if (_historyOpen && _history === null) _loadHistory();
        else _renderHistory();
    }

    function _renderPopover() {
        const list = document.getElementById('transfers-list');
        if (!list) return;
        if (_jobs.length === 0) {
            list.innerHTML =
                '<div class="transfers-empty" style="padding:14px 16px; color:var(--text-tertiary,#888); font-size:12px;">' +
                'No Hub transfers. Start one from a dataset\'s right-click menu &rarr; Upload / Download.' +
                '</div>';
            const clearBtn = document.querySelector('.transfers-clear-btn');
            if (clearBtn) clearBtn.disabled = true;
            // Still render Earlier: an empty live list is the *most* likely
            // moment to want it. Returning before this left a just-cleared
            // transfer invisible in both places until a page reload.
            _renderHistory();
            return;
        }
        list.innerHTML = _jobs.map(_cardHtml).join('');
        _renderHistory();
        const clearBtn = document.querySelector('.transfers-clear-btn');
        if (clearBtn) {
            const hasFinished = _jobs.some(j => !_isActive(j));
            clearBtn.disabled = !hasFinished;
        }
    }

    function _errorClassMessage(j) {
        // Prefer HF's own error text when present — for the fail-fast
        // cases (429, 401, 403) it's far more actionable than a canned
        // hint (e.g. carries the exact retry-after seconds and the
        // documented rate-limit caps). Fall back to a generic remediation
        // string when the worker didn't capture a specific message.
        if (!j.error && j.error_class !== 'cancelled') return '';
        switch (j.error_class) {
            case 'auth':
                return j.error || 'Authentication failed. Your HF token may be expired or lacks write permission. Run `huggingface-cli login` and click Retry.';
            case 'rate_limit':
                return j.error || 'Rate-limited by the Hub. Wait a few minutes and click Retry.';
            case 'bad_request':
                // Common cause we've observed: local HF upload cache
                // claims blobs are uploaded but HF doesn't have them
                // (state from a prior aborted upload). Verbatim message
                // first; if blank, point at the cache as the likely fix.
                return j.error || 'HF rejected the request. If this is a re-upload of a dataset that previously failed mid-flight, try clearing the dataset\'s `.cache/huggingface/upload/` directory before retrying.';
            case 'network':
                return `Network error: ${j.error}. Click Retry to resume.`;
            case 'cancelled':
                return 'Cancelled by user.';
            case 'unresponsive':
                return j.error || 'The transfer stopped responding and was ended. Click Retry — it continues from where it stopped.';
            default:
                return j.error;
        }
    }

    function _cardHtml(j) {
        const dir = j.direction === 'upload' ? '▲ Upload' : '▼ Download';
        // Link to the PR for uploads (when one exists) so the user can
        // inspect the staged state. Falls back to the repo URL otherwise.
        const linkUrl = j.pr_url
            ? j.pr_url
            : hubRepoUrl(j.repo_id, j.repo_type);
        const filesDone = j.files_done_estimate ?? 0;
        const filesTotal = j.files_total ?? 0;
        const bytesDone = j.bytes_done_estimate ?? 0;
        const bytesTotal = j.bytes_total ?? 0;
        // Bytes drive the bar, not file counts. A dataset is a handful of
        // large video files, so the file counter sits on 0 / 1 for the
        // entire multi-GB transfer while the byte counter moves steadily —
        // the file-count bar is what made a healthy upload read as hung.
        const pct = bytesTotal > 0
            ? Math.min(100, Math.round(100 * bytesDone / bytesTotal))
            : (filesTotal > 0 ? Math.min(100, Math.round(100 * filesDone / filesTotal)) : 0);
        const stalledFor = j.stalled_for_s ?? 0;
        const isStalled = _isActive(j) && stalledFor > STALL_THRESHOLD_S;

        // Action buttons depend on terminal-vs-active state. Cancel and
        // Discard get text labels because they affect remote state (kill
        // worker / close draft PR); Hide is icon-only because it's pure UI
        // dismissal with no consequences.
        let actions = '';
        let extra = '';
        if (_isActive(j)) {
            // While cancelling, the button becomes an explicit escalation:
            // the first click asked politely, this one force-kills. Naming
            // it so beats a disabled spinner that gives the user nothing to
            // do while a wedged worker keeps uploading.
            actions = j.status === 'cancelling'
                ? `<button class="transfer-action-btn danger" type="button"
                    title="Still stopping — click again to stop it immediately"
                    onclick="Transfers.cancel('${j.job_id}')">Force stop</button>`
                : `<button class="transfer-action-btn danger" type="button"
                    onclick="Transfers.cancel('${j.job_id}')">Cancel</button>`;
            const stageLine = j.milestone
                ? `<div class="transfer-milestone">${j.milestone}</div>`
                : '';
            const curFile = j.current_file
                ? `<div class="transfer-current-file" title="${j.current_file}">${j.current_file}</div>`
                : '';
            const stallLine = isStalled
                ? `<div class="transfer-msg failed">⚠ No data transferred for ${_fmtDuration(stalledFor)}` +
                  ` — the transfer may be stuck.</div>`
                : '';
            extra = stageLine + curFile + stallLine;
        } else if (j.status === 'complete') {
            // Complete: clear the card. A merged upload has no draft PR,
            // so this is list-only either way.
            actions = `<button class="transfer-action-btn hide-btn" type="button"
                onclick="Transfers.clear('${j.job_id}')"
                title="Clear from this list. Nothing is deleted — your files stay and the outcome stays under Earlier.">✕</button>`;
            const bytesText = bytesDone > 0 ? ` · ${_fmtBytes(bytesDone)}` : '';
            extra = `<div class="transfer-msg complete">Done${bytesText}</div>`;
        } else {
            // Three verbs, three tiers — the separation browser download
            // managers keep: clearing an entry from the list never destroys
            // what it refers to. Without the ✕ here, tidying a failed card
            // out of the tray meant Discard, which closes the draft PR the
            // transfer would have resumed from.
            //
            // Discard is offered only when it has something to destroy: a
            // draft PR on HF, which only an upload has. On a download it
            // would have been a second button doing exactly what ✕ does,
            // under a name that implies otherwise.
            const canDiscard = j.direction === 'upload' && j.pr_num != null;
            // Three texts, because ✕ does three different amounts of thing.
            //
            // Say "upload it again", not "Retry": ✕ removes the card, and the
            // Retry button lives on the card. What survives is the draft PR,
            // which the ordinary Upload action picks up — so naming the
            // button the user no longer has described a route that is gone.
            //
            // And name what has to match for that to happen. "Resumes" alone
            // invites the reading that any later upload continues this one;
            // what continues is this dataset to this repo, because that is
            // what the PR lookup keys on.
            const clearTitle = canDiscard
                ? 'Clear from this list. Nothing is deleted — your local files stay, the outcome stays under Earlier, and the draft PR is kept: uploading this dataset to this repo again continues from the files that already reached it, instead of re-sending them.'
                : j.direction === 'download'
                    ? 'Clear from this list. Nothing is deleted — whatever downloaded stays on disk, and the outcome stays under Earlier.'
                    : 'Clear from this list. Nothing is deleted — your local files stay and the outcome stays under Earlier.';
            actions =
                `<button class="transfer-action-btn" type="button"
                    onclick="Transfers.retry('${j.job_id}')">Retry</button>` +
                (canDiscard
                    ? `<button class="transfer-action-btn danger" type="button"
                        onclick="Transfers.discard('${j.job_id}')"
                        title="Closes the draft PR on HF and drops the partially uploaded data there. Your local files are untouched, but Retry can no longer resume.">Discard</button>`
                    : '') +
                `<button class="transfer-action-btn hide-btn" type="button"
                    onclick="Transfers.clear('${j.job_id}')"
                    title="${clearTitle}">✕</button>`;
            const msgClass = j.status === 'failed' ? 'failed' : 'cancelled';
            // Fall back on the status, not on a fixed string. A failed job
            // whose worker never captured an error message would otherwise
            // be labelled "Cancelled" — telling the user they stopped
            // something that actually broke, and hiding that anything went
            // wrong at all.
            const fallback = j.status === 'cancelled' ? 'Cancelled' : 'Transfer failed for an unknown reason.';
            extra = `<div class="transfer-msg ${msgClass}">${_errorClassMessage(j) || fallback}</div>`;
        }

        const showBar = filesTotal > 0 || bytesTotal > 0 || _isActive(j);
        // Rate is the single most useful "is this alive?" signal, so it
        // leads the stats line whenever we have one.
        const rateText = _isActive(j) && !isStalled ? _fmtRate(j.transfer_rate_bps) : '';
        const progressLine = showBar
            ? `<div class="transfer-stats">` +
              (bytesTotal > 0
                  ? `${_fmtBytes(bytesDone)} / ${_fmtBytes(bytesTotal)}`
                  : `${filesDone} / ${filesTotal} files`) +
              (rateText ? ` · ${rateText}` : '') +
              (bytesTotal > 0 && filesTotal > 0 ? ` · ${filesTotal} files` : '') +
              `<span class="pct">${pct}%</span></div>` +
              `<progress value="${pct}" max="100"></progress>`
            : '';

        // Short job-id prefix so a user clicking "Open log folder" can
        // identify which <job_id>.log file is theirs in the directory
        // listing. Full id is in the title attribute for copy/paste.
        // Which transfer path this job took. Only shown when it is the
        // non-default one, so the tray doesn't carry a badge on every card.
        const xetChip = j.disable_xet
            ? `<span class="transfer-xet" title="This transfer is using classic LFS instead of Xet" style="font-size:10px; padding:1px 5px; border-radius:3px; background:var(--bg-primary,#252526); color:var(--text-tertiary,#888);">LFS</span>`
            : '';
        const jobIdShort = (j.job_id || '').slice(0, 8);
        const jobIdChip = jobIdShort
            ? `<span class="transfer-jobid" title="job_id=${j.job_id}\nLog file: ${jobIdShort}…log" style="margin-left:auto; font-family:monospace; font-size:10px; color:var(--text-tertiary,#888);">${jobIdShort}</span>`
            : '';

        return (
            `<div class="transfer-card ${j.status}">` +
              `<div class="transfer-card-head">` +
                `<span class="transfer-direction">${dir}</span>` +
                `<a class="transfer-repo" href="${linkUrl}" target="_blank" rel="noopener noreferrer" title="${j.repo_id}">${j.repo_id}</a>` +
                xetChip +
                jobIdChip +
                `<span class="transfer-actions">${actions}</span>` +
              `</div>` +
              progressLine +
              extra +
            `</div>`
        );
    }

    function _onJobsUpdated(prevJobs, jobs) {
        // Surface a one-shot toast for each transition into a terminal
        // state — but only when the popover is closed. When the popover is
        // open the card itself shows the new status, and an overlapping
        // toast on the same screen edge is redundant + visually noisy
        // (they share the top-right corner). We still record the job_id
        // in _completionShown so a later open/close doesn't re-toast.
        for (const j of jobs) {
            if (_isActive(j) || _completionShown.has(j.job_id)) continue;
            _completionShown.add(j.job_id);
            // Download still needs the post-completion refresh regardless
            // of toast visibility — it's not a notification, it's state sync.
            if (j.status === 'complete' && j.direction === 'download' && datasets[j.dataset_id]) {
                _refreshAfterDownload(j.dataset_id);
            }
            if (_popoverOpen) continue;
            const verb = j.direction === 'upload' ? 'Upload' : 'Download';
            const filesDone = j.files_done_estimate ?? 0;
            const bytesDone = j.bytes_done_estimate ?? 0;
            if (j.status === 'complete') {
                const bytesText = bytesDone > 0 ? `, ${_fmtBytes(bytesDone)}` : '';
                showToast(`${verb} complete`, `${j.repo_id} — ${filesDone} files${bytesText}`, 'info');
            } else if (j.status === 'failed') {
                showToast(`${verb} failed`, `${j.repo_id}: ${_errorClassMessage(j)}`, 'error', 8000);
            } else if (j.status === 'cancelled') {
                showToast(`${verb} cancelled`, j.repo_id, 'warning');
            }
        }
    }

    async function _refreshAfterDownload(datasetId) {
        try {
            const epRes = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/episodes`);
            if (epRes.ok) {
                episodes[datasetId] = await epRes.json();
                datasets[datasetId].total_episodes = episodes[datasetId].length;
                datasets[datasetId].total_frames = episodes[datasetId].reduce((s, e) => s + e.length, 0);
                renderTree();
            }
        } catch (e) { /* non-critical refresh */ }
    }

    async function poll() {
        try {
            const res = await fetch('/api/datasets/hub/jobs');
            if (!res.ok) return;
            const data = await res.json();
            const prev = _jobs;
            _jobs = data.jobs || [];
            _onJobsUpdated(prev, _jobs);
            _renderIndicator();
            if (_popoverOpen) _renderPopover();
        } catch (e) {
            // Network blip — keep last snapshot, try again next tick.
        }
        // Schedule next poll only if there's work to watch. When all jobs
        // are terminal, the indicator stays visible until dismissed but
        // we stop hammering the server.
        if (_pollTimer) clearTimeout(_pollTimer);
        _pollTimer = null;
        if (_jobs.some(_isActive)) {
            _pollTimer = setTimeout(poll, 1000);
        }
    }

    function refreshNow() {
        // Called when a new transfer is kicked off — restarts the poll
        // loop unconditionally (so a job initiated when the tray was
        // idle gets immediate attention).
        if (_pollTimer) clearTimeout(_pollTimer);
        _pollTimer = null;
        poll();
    }

    function openPopover() {
        _popoverOpen = true;
        const pop = document.getElementById('transfers-popover');
        if (pop) pop.hidden = false;
        // Fetch once per page load so the "Earlier" affordance can appear at
        // all; the list itself stays collapsed until asked for.
        if (_history === null) _loadHistory();
        _renderPopover();
    }

    function closePopover() {
        _popoverOpen = false;
        const pop = document.getElementById('transfers-popover');
        if (pop) pop.hidden = true;
    }

    function toggle() {
        if (_popoverOpen) closePopover();
        else openPopover();
    }

    async function cancel(jobId) {
        // Confirmation only if the transfer has actually started moving
        // bytes (active mid-flight). For a "still starting" / "just queued"
        // job the cancel is free of regret. A second click on an
        // already-cancelling job is the force-kill escalation — the user
        // has confirmed once already, so don't ask again.
        const j = _jobs.find(x => x.job_id === jobId);
        if (j && j.status !== 'cancelling' && (j.bytes_done_estimate ?? 0) > 0 && j.direction === 'upload') {
            const ok = confirm(
                'Cancel this upload?\n\n' +
                'Nothing already uploaded is lost — Retry continues from where it stopped.'
            );
            if (!ok) return;
        }
        try {
            await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
            refreshNow();
        } catch (e) { /* ignored */ }
    }

    async function retry(jobId) {
        const j = _jobs.find(x => x.job_id === jobId);
        if (!j) return;
        // Retry is just re-POSTing the upload/download endpoint with the
        // same dataset+repo. The server detects the existing draft PR and
        // resumes into it via the reuse_pr_num path (transferring pr_num
        // ownership off the old terminal entry as a side effect, so the
        // follow-up dismiss below does NOT close the resumed PR).
        // A model job's id is a run directory, which the dataset route rejects
        // with 404. The tray renders Retry on every terminal card and its copy
        // tells the user to click it, so the route has to follow repo_type.
        const isModel = j.repo_type === 'model';
        const endpoint = isModel
            ? `/api/models/hub/${j.direction}`
            : `/api/datasets/${encodeURIComponent(j.dataset_id)}/hub/${j.direction}`;
        const post = (body) => fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(isModel ? { ...body, path: j.dataset_id } : body),
        });
        // Carry the transfer-path choice across a retry. A retry of a job
        // the user deliberately put on the LFS path must not silently
        // revert to Xet — that is the path they retried to get away from.
        const retryBody = { repo_id: j.repo_id };
        if (j.disable_xet) retryBody.disable_xet = true;
        try {
            let res = await post(retryBody);
            if (res.status === 409) {
                const data = await res.json().catch(() => ({}));
                const detail = data && data.detail;
                // Two distinct 409 shapes: "active job exists" carries job_id;
                // the completeness guardrail carries code: 'incomplete_local_state'.
                if (detail && detail.code === 'incomplete_local_state') {
                    const missing = detail.missing_locally || [];
                    const incomplete = detail.incomplete_locally || [];
                    const lines = [];
                    if (missing.length) lines.push('Missing: ' + missing.join(', '));
                    if (incomplete.length) lines.push('Incomplete: ' + incomplete.join(', '));
                    const ok = confirm(
                        'Local copy is missing files that exist on the remote.\n\n' +
                        lines.join('\n') +
                        '\n\nRetry the upload anyway?'
                    );
                    if (!ok) return;
                    res = await post({ ...retryBody, confirm_force: true });
                    if (!res.ok) {
                        const fd = await res.json().catch(() => ({}));
                        showToast('Retry failed', fd?.detail?.message || fd?.detail || 'Could not restart transfer', 'error');
                        return;
                    }
                } else {
                    // Already running (concurrent retry); attach to the existing job.
                    refreshNow();
                    return;
                }
            } else if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                showToast('Retry failed', data.detail?.message || data.detail || 'Could not restart transfer', 'error');
                return;
            }
            // Drop the old terminal entry so the tray shows only the new attempt.
            // Safe: server already transferred pr_num ownership; this dismiss
            // will not close the resumed PR.
            await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss`, { method: 'POST' });
            refreshNow();
        } catch (e) {
            showToast('Retry failed', e.message, 'error');
        }
    }

    async function discard(jobId) {
        const j = _jobs.find(x => x.job_id === jobId);
        const isUpload = j && j.direction === 'upload';
        const hasPR = j && j.pr_num != null;
        if (isUpload && hasPR) {
            const ok = confirm(
                'Discard upload? The pending HF PR will be closed and ' +
                'partially uploaded data will be cleaned up. Resume will ' +
                'no longer be possible. Use Retry to resume instead.\n\n' +
                'The record of how it ended is kept under Earlier.'
            );
            if (!ok) return;
        }
        try {
            const res = await fetch(`/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss`, { method: 'POST' });
            // The card is about to leave the live list, so the only place it
            // still exists is Earlier — fetched once per page load, so it needs
            // re-reading or the outcome is invisible until a reload.
            if (res.ok) { _history = null; _loadHistory(); refreshNow(); }
        } catch (e) { /* ignored */ }
    }

    async function clear(jobId) {
        // Clears the card and nothing else. `close_pr=false` makes that true
        // for a failed or cancelled job too, where the same endpoint would
        // otherwise close the draft PR the transfer could resume from —
        // browser download managers draw exactly this line: clearing an entry
        // from the list never deletes the file. The outcome itself survives
        // in the transfer history, under Earlier.
        try {
            const res = await fetch(
                `/api/datasets/hub/progress/${encodeURIComponent(jobId)}/dismiss?close_pr=false`,
                { method: 'POST' });
            // The card is about to leave the live list, so the only place it
            // still exists is Earlier — fetched once per page load, so it needs
            // re-reading or the outcome is invisible until a reload.
            if (res.ok) { _history = null; _loadHistory(); refreshNow(); }
        } catch (e) { /* ignored */ }
    }

    async function dismissAllFinished() {
        // Bulk form of the per-card ✕, and it must mean the same thing.
        // It used to loop the destructive dismiss, so "Clear finished" closed
        // the draft PR of every failed card — the same word doing two
        // different things depending on which control you reached for. It now
        // clears the list and nothing else, which is why the alarming
        // confirmation it needed is gone: nothing is destroyed, and the
        // outcomes stay under Earlier.
        //
        // Destroying remote state stays deliberate and per-card: Discard.
        const targets = _jobs.filter(j => !_isActive(j));
        for (const j of targets) {
            try {
                await fetch(
                    `/api/datasets/hub/progress/${encodeURIComponent(j.job_id)}/dismiss?close_pr=false`,
                    { method: 'POST' });
            } catch (e) { /* ignored */ }
        }
        _history = null;
        _loadHistory();
        refreshNow();
    }

    return { poll, refreshNow, openPopover, closePopover, toggle, cancel, retry, discard, clear, dismissAllFinished, toggleHistory };
})();

// Global handles for the inline onclick attributes in index.html.
window.toggleTransfersPopover = () => Transfers.toggle();
window.dismissAllFinishedTransfers = () => Transfers.dismissAllFinished();
window.Transfers = Transfers;

// Opens the per-job log directory on the GUI host machine. Same constraint
// as the dataset "open in files" buttons: this is the host's filesystem,
// not the frontend's — fine when the GUI is running locally, degrades to
// a clear error toast when xdg-open isn't available (e.g. headless server).
window.openHubJobFolder = async () => {
    try {
        const res = await fetch('/api/datasets/hub/open-job-folder', { method: 'POST' });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            showToast('Couldn\'t open folder', data?.detail || 'xdg-open failed on the GUI host', 'error', 6000);
        }
    } catch (e) {
        showToast('Couldn\'t open folder', e.message, 'error', 6000);
    }
};

// Click-outside closes the popover. Anchored to the indicator: if the
// click is on the indicator or inside the popover, leave it alone.
document.addEventListener('click', (e) => {
    const pop = document.getElementById('transfers-popover');
    const ind = document.getElementById('transfers-indicator');
    if (!pop || pop.hidden) return;
    if (ind && ind.contains(e.target)) return;
    if (pop.contains(e.target)) return;
    Transfers.closePopover();
});

// Initialize
CameraVideoMode.init();
refreshPendingEdits();
loadSources();
restoreOpenedDatasets();
checkHubAuth();
// Pick up any in-flight transfers from a prior session (the server keeps
// them until they finish + 30 min). One probe at startup is enough — if
// it returns active jobs the poll loop schedules itself thereafter.
Transfers.refreshNow();
