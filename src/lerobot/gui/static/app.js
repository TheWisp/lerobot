/* LeRobot Dataset GUI - Application Logic */

let datasets = {};
let episodes = {};
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
            if (datasets.length === 0 && !sourceDatasets[source.path]) {
                html += '<div class="source-loading">Scanning...</div>';
            } else if (datasets.length === 0) {
                html += '<div class="source-empty">No datasets found</div>';
            } else {
                for (const ds of datasets) {
                    const isOpen = Object.keys(window.datasets || {}).some(id => {
                        const d = window.datasets[id];
                        return d && d.root === ds.root;
                    });
                    html += `<div class="source-dataset${isOpen ? ' active' : ''}" onclick="openDatasetFromSource('${ds.root.replace(/'/g, "\\'")}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}')" title="${ds.root}\n${ds.total_episodes} episodes, ${ds.total_frames.toLocaleString()} frames">`;
                    html += `<span class="source-dataset-name">${ds.name}</span>`;
                    html += `<span class="source-dataset-meta">${ds.total_episodes} ep</span>`;
                    html += `</div>`;
                }
            }
        }
        html += `</div></div>`;
    }
    container.innerHTML = html;
}

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
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        datasets[data.id] = data;

        // Show toast for any warnings
        if (data.warnings && data.warnings.length > 0) {
            showToast(
                'Dataset Warning',
                data.warnings.join('\n'),
                'warning',
                8000
            );
        }

        // Load episodes
        const epRes = await fetch(`/api/datasets/${encodeURIComponent(data.id)}/episodes`);
        episodes[data.id] = await epRes.json();

        // Sync episode count (backend may have reloaded metadata with new episodes)
        datasets[data.id].total_episodes = episodes[data.id].length;

        // Expand this dataset by default
        expandedNodes.add(data.id);

        // Refresh pending edits (backend restores persisted edits on open)
        await refreshPendingEdits();

        renderTree();
        renderSources();  // Update source list to show open state
        if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
        setStatus(`Opened: ${data.repo_id}`);
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

function isEpisodeDeleted(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'delete');
}

function isEpisodeTrimmed(datasetId, epIdx) {
    return pendingEdits.some(e => e.dataset_id === datasetId && e.episode_index === epIdx && e.edit_type === 'trim');
}

function renderTree() {
    const container = document.getElementById('tree-container');
    if (Object.keys(datasets).length === 0) {
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
                <div class="tree-header" onclick="toggleDataset('${id}')" oncontextmenu="showFolderContextMenu(event, '${ds.root.replace(/'/g, "\\'")}')" title="${tooltip}">
                    <span class="tree-toggle">${isExpanded ? '▼' : '▶'}</span>
                    <span class="tree-icon">📁</span>
                    <span class="tree-label">${ds.repo_id}</span>
                    <span class="tree-meta">${dsEditCount > 0 ? `${dsEditCount}✎ ` : ''}${ds.total_episodes} ep</span>
                    <span class="tree-close" onclick="closeDataset('${id}', event)" title="Close">&times;</span>
                </div>
                <div class="tree-children ${isExpanded ? 'expanded' : ''}">
        `;

        for (const ep of dsEpisodes) {
            const isActive = currentDataset === id && currentEpisode === ep.episode_index;
            const isDeleted = isEpisodeDeleted(id, ep.episode_index);
            const isTrimmed = isEpisodeTrimmed(id, ep.episode_index);
            const hasQualityWarning = ep.video_extra_frames !== 0;
            const classes = ['tree-header'];
            if (isActive) classes.push('active');
            if (isDeleted) classes.push('deleted');
            if (isTrimmed) classes.push('trimmed');
            if (hasQualityWarning) classes.push('quality-warning');

            let icon = '🎬';
            if (isDeleted) icon = '🗑️';
            else if (hasQualityWarning) icon = '⚠️';

            let meta = `${ep.length} frames`;
            if (hasQualityWarning) {
                const sign = ep.video_extra_frames > 0 ? '+' : '';
                meta += ` (${sign}${ep.video_extra_frames})`;
            }

            html += `
                <div class="${classes.join(' ')}"
                     onclick="selectEpisode('${id}', ${ep.episode_index}, ${ep.video_length || ep.length})"
                     oncontextmenu="showContextMenu(event, '${id}', ${ep.episode_index})"
                     ${hasQualityWarning ? `title="Video-data mismatch: ${ep.video_extra_frames > 0 ? ep.video_extra_frames + ' extra frames (re-recording artifact)' : Math.abs(ep.video_extra_frames) + ' missing frames (truncated video)'}"` : ''}>
                    <span class="tree-toggle"></span>
                    <span class="tree-icon">${icon}</span>
                    <span class="tree-label">Episode ${ep.episode_index}</span>
                    <span class="tree-meta">${meta}</span>
                </div>
            `;
        }

        html += '</div></div>';
    }
    container.innerHTML = html;
    updateEditsBar();
}

function toggleDataset(id) {
    if (expandedNodes.has(id)) {
        expandedNodes.delete(id);
    } else {
        expandedNodes.add(id);
    }
    renderTree();
}

async function closeDataset(id, e) {
    e.stopPropagation();
    try {
        await fetch(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
        delete datasets[id];
        delete episodes[id];
        expandedNodes.delete(id);
        if (currentDataset === id) {
            currentDataset = null;
            currentEpisode = null;
            document.getElementById('camera-grid').innerHTML = '<div class="empty-state">Select an episode to view</div>';
        }
        renderTree();
        if (typeof refreshRunDatasetSelects === 'function') refreshRunDatasetSelects();
        renderSources();
    } catch (err) {
        showToast('Error', 'Failed to close dataset: ' + err.message, 'error');
    }
}

function selectEpisode(datasetId, epIdx, length) {
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
}

function renderCameraGrid() {
    const grid = document.getElementById('camera-grid');
    if (!currentDataset || currentEpisode === null) {
        grid.innerHTML = '<div class="empty-state">Select an episode to view</div>';
        return;
    }

    const ds = datasets[currentDataset];
    const cameras = ds.camera_keys;
    const numCameras = cameras.length;

    // Determine grid layout
    let cols = 1;
    if (numCameras === 2) cols = 2;
    else if (numCameras >= 3 && numCameras <= 4) cols = 2;
    else if (numCameras >= 5) cols = 3;

    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    let html = '';
    for (const cam of cameras) {
        const camName = cam.split('.').pop();
        html += `
            <div class="camera-panel">
                <div class="camera-title">${camName}</div>
                <div class="camera-frame">
                    <img id="frame-${cam.replace(/\./g, '-')}" src="" alt="${camName}">
                </div>
            </div>
        `;
    }
    grid.innerHTML = html;
}

function loadAllFrames(idx) {
    if (!currentDataset || currentEpisode === null) return Promise.resolve();
    currentFrame = Math.max(0, Math.min(idx, totalFrames - 1));

    const ds = datasets[currentDataset];
    const promises = [];

    for (const cam of ds.camera_keys) {
        const url = `/api/datasets/${encodeURIComponent(currentDataset)}/episodes/${currentEpisode}/frame/${currentFrame}?camera=${encodeURIComponent(cam)}`;
        const imgId = `frame-${cam.replace(/\./g, '-')}`;
        const img = document.getElementById(imgId);
        if (img) {
            const promise = new Promise((resolve) => {
                img.onload = resolve;
                img.onerror = resolve; // Don't block on errors
            });
            img.src = url;
            promises.push(promise);
        }
    }

    // Update UI
    document.getElementById('frame-info').textContent = `${currentFrame + 1} / ${totalFrames}`;
    const pct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0;
    document.getElementById('timeline-progress').style.width = `${pct}%`;
    document.getElementById('timeline-scrubber').style.left = `${pct}%`;

    // Update time display
    const currentTime = formatTime(currentFrame / fps);
    const totalTime = formatTime(totalFrames / fps);
    document.getElementById('time-info').textContent = `${currentTime} / ${totalTime}`;

    return Promise.all(promises);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function playLoop() {
    while (isPlaying) {
        const frameTime = 1000 / (fps * playbackSpeed);
        const startTime = performance.now();

        // Constrain playback to trim region
        const playStart = trimStart;
        const playEnd = trimEnd - 1;  // trimEnd is exclusive

        if (currentFrame >= playEnd) {
            currentFrame = playStart;
        } else if (currentFrame < playStart) {
            currentFrame = playStart;
        } else {
            currentFrame++;
        }

        await loadAllFrames(currentFrame);

        // Wait remaining time to maintain fps (if frames loaded fast enough)
        const elapsed = performance.now() - startTime;
        const sleepTime = frameTime - elapsed;
        if (sleepTime > 0) {
            await new Promise(r => setTimeout(r, sleepTime));
        }
    }
}

function changeSpeed(speed) {
    playbackSpeed = parseFloat(speed);
}

function togglePlay() {
    if (!currentDataset || currentEpisode === null) return;

    isPlaying = !isPlaying;
    document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';

    if (isPlaying) {
        playLoop();
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
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toast-out 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Timeline interaction
document.addEventListener('DOMContentLoaded', () => {
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
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    // Re-scan sources when switching to data tab (picks up newly recorded datasets)
    if (tabName === 'data' && typeof window.refreshExpandedSources === 'function') {
        window.refreshExpandedSources();
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
    // Stop camera preview when leaving robot tab
    if (tabName !== 'robot' && typeof stopCameraPreview === 'function') {
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

    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.add('visible');
}

function hideContextMenu() {
    document.getElementById('context-menu').classList.remove('visible');
    document.getElementById('folder-context-menu').classList.remove('visible');
    contextMenuTarget = null;
    _folderContextPath = null;
}

document.addEventListener('click', hideContextMenu);

// Folder context menu (source folders + datasets)
let _folderContextPath = null;

function showFolderContextMenu(e, path) {
    e.preventDefault();
    e.stopPropagation();
    _folderContextPath = path;
    const menu = document.getElementById('folder-context-menu');
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.add('visible');
}

function folderContextAction(action) {
    if (!_folderContextPath) return;
    if (action === 'open-in-files') {
        // Use whichever open-in-files endpoint is available (both do the same thing)
        fetch('/api/datasets/open-in-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: _folderContextPath })
        }).catch(e => console.error('Failed to open file manager:', e));
    }
    hideContextMenu();
}

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
    if (!confirm(`Apply ${pendingEdits.length} edit(s) to disk? This cannot be undone.`)) return;

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

// Make state accessible for other scripts (run.js, etc.)
window.datasets = datasets;
window.episodes = episodes;
window.sourceDatasets = sourceDatasets;
window.refreshExpandedSources = async function() {
    for (const sourcePath of expandedSources) {
        scanSource(sourcePath);
    }
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

// Initialize
refreshPendingEdits();
loadSources();
restoreOpenedDatasets();
