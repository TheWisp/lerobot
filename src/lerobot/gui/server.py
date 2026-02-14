# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI server for the LeRobot Dataset GUI."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from lerobot.gui.api import datasets, edits, playback
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LeRobot Dataset GUI",
    description="Web-based visualization and editing tool for LeRobot datasets",
    version="0.1.0",
)

# Global app state (initialized on startup)
_app_state: AppState | None = None


def parse_cache_size(size_str: str) -> int:
    """Parse cache size string like '500MB' or '1GB' to bytes."""
    size_str = size_str.strip().upper()
    # Check longer suffixes first to avoid "MB" matching "B"
    multipliers = [
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
        ("B", 1),
    ]
    for suffix, mult in multipliers:
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup."""
    global _app_state
    # Default cache size, can be overridden via CLI
    cache_size = getattr(app.state, "cache_size", 500_000_000)
    _app_state = AppState(frame_cache=FrameCache(max_bytes=cache_size))
    datasets.set_app_state(_app_state)
    playback.set_app_state(_app_state)
    edits.set_app_state(_app_state)
    logger.info(f"Initialized frame cache with {cache_size / 1_000_000:.0f} MB budget")


# Include API routers
app.include_router(datasets.router)
app.include_router(playback.router)
app.include_router(edits.router)


# Minimal HTML viewer for testing
MINIMAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeRobot Dataset GUI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; overflow: hidden; }

        .layout { display: flex; height: 100vh; }

        /* Sidebar */
        .sidebar { width: 280px; min-width: 280px; background: #16213e; border-right: 1px solid #0f3460; display: flex; flex-direction: column; }
        .sidebar-header { padding: 16px; border-bottom: 1px solid #0f3460; }
        .sidebar-header h1 { font-size: 18px; color: #4fc3f7; margin-bottom: 12px; }
        .open-dataset { display: flex; gap: 8px; }
        .open-dataset input { flex: 1; padding: 8px; border-radius: 4px; border: none; background: #0f3460; color: #fff; font-size: 13px; }
        .open-dataset button { padding: 8px 12px; border-radius: 4px; border: none; background: #4fc3f7; color: #000; cursor: pointer; font-size: 13px; }
        .open-dataset button:hover { background: #81d4fa; }
        .recent-dropdown { position: relative; }
        .recent-btn { padding: 8px; border-radius: 4px; border: none; background: #0f3460; color: #888; cursor: pointer; font-size: 13px; }
        .recent-btn:hover { background: #1a4a7a; color: #fff; }
        .recent-btn:disabled { opacity: 0.5; cursor: default; }
        .recent-menu { position: absolute; top: 100%; right: 0; margin-top: 4px; background: #16213e; border: 1px solid #0f3460; border-radius: 4px; min-width: 250px; max-width: 350px; z-index: 100; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .recent-menu.visible { display: block; }
        .recent-menu-item { padding: 8px 12px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #0f3460; }
        .recent-menu-item:last-child { border-bottom: none; }
        .recent-menu-item:hover { background: #0f3460; }
        .recent-menu-item .recent-path { color: #888; font-size: 11px; display: block; margin-top: 2px; }

        .tree-container { flex: 1; overflow-y: auto; padding: 8px 0; }

        /* Tree view */
        .tree-node { user-select: none; }
        .tree-header { display: flex; align-items: center; padding: 6px 12px; cursor: pointer; gap: 6px; }
        .tree-header:hover { background: #0f3460; }
        .tree-header.active { background: #1a4a7a; }
        .tree-header.deleted { opacity: 0.5; text-decoration: line-through; }
        .tree-header.trimmed .tree-label::after { content: ' ✂'; color: #f39c12; }
        .tree-toggle { width: 16px; font-size: 10px; color: #888; flex-shrink: 0; }
        .tree-icon { width: 16px; text-align: center; flex-shrink: 0; }
        .tree-label { flex: 1; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .tree-meta { font-size: 11px; color: #666; flex-shrink: 0; }
        .tree-children { display: none; }
        .tree-children.expanded { display: block; }
        .tree-children .tree-header { padding-left: 28px; }

        /* Context menu */
        .context-menu { position: fixed; background: #16213e; border: 1px solid #0f3460; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 1000; min-width: 150px; display: none; }
        .context-menu.visible { display: block; }
        .context-menu-item { padding: 8px 16px; cursor: pointer; font-size: 13px; }
        .context-menu-item:hover { background: #0f3460; }
        .context-menu-item.danger { color: #e74c3c; }
        .context-menu-item.danger:hover { background: #c0392b; color: #fff; }
        .context-menu-separator { height: 1px; background: #0f3460; margin: 4px 0; }

        /* Edits panel */
        .edits-bar { background: #1a3a5c; border-top: 1px solid #0f3460; padding: 8px 16px; display: none; align-items: center; gap: 12px; }
        .edits-bar.visible { display: flex; }
        .edits-count { font-size: 13px; color: #f39c12; flex: 1; }
        .edits-bar button { padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 12px; }
        .edits-bar .btn-save { background: #27ae60; color: #fff; }
        .edits-bar .btn-save:hover { background: #2ecc71; }
        .edits-bar .btn-discard { background: #e74c3c; color: #fff; }
        .edits-bar .btn-discard:hover { background: #c0392b; }

        /* Toast notifications */
        .toast-container { position: fixed; top: 20px; right: 20px; z-index: 1000; display: flex; flex-direction: column; gap: 8px; }
        .toast { background: #1a3a5c; border: 1px solid #0f3460; border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 13px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); animation: toast-in 0.3s ease-out; max-width: 350px; }
        .toast.warning { border-left: 4px solid #f39c12; }
        .toast.info { border-left: 4px solid #4fc3f7; }
        .toast.success { border-left: 4px solid #27ae60; }
        .toast-title { font-weight: 600; margin-bottom: 4px; }
        .toast-message { color: #aaa; font-size: 12px; }
        @keyframes toast-in { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes toast-out { from { opacity: 1; } to { opacity: 0; } }

        /* Main content */
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        /* Camera grid */
        .camera-grid { flex: 1; display: grid; gap: 8px; padding: 16px; overflow: auto; }
        .camera-panel { background: #16213e; border-radius: 8px; display: flex; flex-direction: column; min-height: 200px; }
        .camera-title { padding: 8px 12px; font-size: 12px; color: #888; border-bottom: 1px solid #0f3460; }
        .camera-frame { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 0 0 8px 8px; overflow: hidden; }
        .camera-frame img { max-width: 100%; max-height: 100%; object-fit: contain; }

        /* Controls */
        .controls-bar { background: #16213e; border-top: 1px solid #0f3460; padding: 12px 16px; display: flex; align-items: center; gap: 12px; }
        .controls-bar button { padding: 8px 16px; border-radius: 4px; border: none; background: #4fc3f7; color: #000; cursor: pointer; font-size: 14px; }
        .controls-bar button:hover { background: #81d4fa; }
        .speed-select { padding: 6px 8px; border-radius: 4px; border: none; background: #0f3460; color: #fff; font-size: 12px; cursor: pointer; }
        .timeline-container { flex: 1; position: relative; padding: 8px 0; }
        .timeline { height: 8px; background: #0f3460; border-radius: 4px; cursor: pointer; position: relative; }
        .timeline-progress { height: 100%; background: #4fc3f7; border-radius: 4px; width: 0%; pointer-events: none; }
        .timeline-scrubber { position: absolute; top: 50%; width: 16px; height: 16px; background: #fff; border-radius: 50%; transform: translate(-50%, -50%); cursor: grab; box-shadow: 0 2px 4px rgba(0,0,0,0.3); pointer-events: auto; }
        .timeline-scrubber:active { cursor: grabbing; }
        .timeline-scrubber:hover { transform: translate(-50%, -50%) scale(1.2); }
        .timeline-hover { position: absolute; bottom: 100%; left: 0; background: #0f3460; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 11px; transform: translateX(-50%); white-space: nowrap; opacity: 0; pointer-events: none; margin-bottom: 8px; }
        .timeline-container:hover .timeline-hover { opacity: 1; }

        /* Trim handles */
        .trim-region { position: absolute; top: -4px; bottom: -4px; background: rgba(76, 175, 80, 0.25); border-left: 3px solid #4caf50; border-right: 3px solid #4caf50; pointer-events: none; display: none; z-index: 5; overflow: visible; }
        .trim-region.visible { display: block; }
        /* Cut zones - red tint for regions that will be removed */
        .trim-cut-left, .trim-cut-right { position: absolute; top: -4px; bottom: -4px; background: rgba(244, 67, 54, 0.3); pointer-events: none; display: none; z-index: 4; }
        .trim-cut-left.visible, .trim-cut-right.visible { display: block; }
        .trim-cut-left { left: 0; }
        .trim-cut-right { right: 0; }
        .trim-handle { position: absolute; top: 50%; width: 20px; height: 28px; cursor: ew-resize; z-index: 20; pointer-events: auto; background: transparent; transform: translateY(-50%); }
        .trim-handle::before { content: ''; position: absolute; top: 50%; left: 50%; width: 6px; height: 24px; background: #4caf50; border-radius: 3px; transform: translate(-50%, -50%); box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .trim-handle:hover::before { background: #66bb6a; transform: translate(-50%, -50%) scaleY(1.1); }
        .trim-handle.left { left: -10px; }
        .trim-handle.right { right: -10px; }
        .trim-controls { display: flex; align-items: center; gap: 8px; margin-left: 12px; visibility: hidden; min-width: 220px; }
        .trim-controls.visible { visibility: visible; }
        .trim-controls button { padding: 4px 10px; border-radius: 4px; border: none; font-size: 11px; cursor: pointer; background: #555; color: #fff; }
        .trim-controls button:hover { background: #666; }
        .trim-info { font-size: 11px; color: #f39c12; }

        .frame-info { font-size: 13px; color: #888; min-width: 140px; text-align: right; }
        .time-info { font-size: 11px; color: #666; }
        .status { font-size: 12px; color: #666; min-width: 150px; text-align: right; }

        /* Empty state */
        .empty-state { flex: 1; display: flex; align-items: center; justify-content: center; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="layout">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>LeRobot Dataset GUI</h1>
                <div class="open-dataset">
                    <input type="text" id="dataset-path" placeholder="Path or repo_id" onkeydown="if(event.key==='Enter')openDataset()">
                    <button onclick="openDataset()">Open</button>
                    <div class="recent-dropdown">
                        <button class="recent-btn" id="recent-btn" onclick="toggleRecentMenu()" title="Recent datasets">▾</button>
                        <div class="recent-menu" id="recent-menu"></div>
                    </div>
                </div>
            </div>
            <div class="tree-container" id="tree-container">
                <div style="padding: 16px; color: #666; font-size: 13px;">No datasets loaded</div>
            </div>
        </div>

        <div class="main">
            <div class="camera-grid" id="camera-grid">
                <div class="empty-state">Select an episode to view</div>
            </div>

            <div class="controls-bar">
                <button id="play-btn" onclick="togglePlay()">▶ Play</button>
                <select class="speed-select" id="speed-select" onchange="changeSpeed(this.value)">
                    <option value="0.25">0.25x</option>
                    <option value="0.5">0.5x</option>
                    <option value="1" selected>1x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2">2x</option>
                </select>
                <div class="timeline-container" id="timeline-container">
                    <div class="timeline-hover" id="timeline-hover">0:00 / Frame 0</div>
                    <div class="timeline" id="timeline">
                        <div class="timeline-progress" id="timeline-progress"></div>
                        <div class="trim-cut-left" id="trim-cut-left"></div>
                        <div class="trim-cut-right" id="trim-cut-right"></div>
                        <div class="trim-region" id="trim-region">
                            <div class="trim-handle left" id="trim-handle-left"></div>
                            <div class="trim-handle right" id="trim-handle-right"></div>
                        </div>
                        <div class="timeline-scrubber" id="timeline-scrubber"></div>
                    </div>
                </div>
                <div class="trim-controls" id="trim-controls">
                    <span class="trim-info" id="trim-info"></span>
                    <button class="btn-reset" onclick="resetTrim()">Reset</button>
                </div>
                <div class="frame-info">
                    <span id="frame-info">- / -</span>
                    <div class="time-info" id="time-info">0:00 / 0:00</div>
                </div>
                <div class="status" id="status">Ready</div>
            </div>

            <div class="edits-bar" id="edits-bar">
                <span class="edits-count" id="edits-count">0 pending edits</span>
                <button class="btn-discard" onclick="discardEdits()">Discard</button>
                <button class="btn-save" onclick="applyEdits()">Save Changes</button>
            </div>
        </div>
    </div>

    <!-- Context menu -->
    <div class="context-menu" id="context-menu">
        <div class="context-menu-item" onclick="contextAction('view')">View Episode</div>
        <div class="context-menu-item" onclick="contextAction('rerun')">Open in Rerun</div>
        <div class="context-menu-separator"></div>
        <div class="context-menu-item" onclick="contextAction('cleartrim')">Clear Trim</div>
        <div class="context-menu-separator"></div>
        <div class="context-menu-item" onclick="contextAction('undelete')">Restore Episode</div>
        <div class="context-menu-item danger" onclick="contextAction('delete')">Delete Episode</div>
    </div>

    <!-- Toast notifications -->
    <div class="toast-container" id="toast-container"></div>

    <script>
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

        // Recent datasets (localStorage)
        const RECENT_KEY = 'lerobot_recent_datasets';
        const MAX_RECENT = 10;

        function getRecentDatasets() {
            try {
                return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
            } catch { return []; }
        }

        function addRecentDataset(path, repoId) {
            const recent = getRecentDatasets().filter(r => r.path !== path);
            recent.unshift({ path, repoId, timestamp: Date.now() });
            localStorage.setItem(RECENT_KEY, JSON.stringify(recent.slice(0, MAX_RECENT)));
            updateRecentButton();
        }

        function updateRecentButton() {
            const btn = document.getElementById('recent-btn');
            const recent = getRecentDatasets();
            btn.disabled = recent.length === 0;
        }

        function toggleRecentMenu() {
            const menu = document.getElementById('recent-menu');
            if (menu.classList.contains('visible')) {
                menu.classList.remove('visible');
                return;
            }

            const recent = getRecentDatasets();
            if (recent.length === 0) return;

            menu.innerHTML = recent.map(r => `
                <div class="recent-menu-item" onclick="openRecentDataset('${r.path.replace(/'/g, "\\'")}')">
                    ${r.repoId || r.path.split('/').pop()}
                    <span class="recent-path">${r.path}</span>
                </div>
            `).join('');
            menu.classList.add('visible');
        }

        function openRecentDataset(path) {
            document.getElementById('recent-menu').classList.remove('visible');
            document.getElementById('dataset-path').value = path;
            openDataset();
        }

        // Close recent menu when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.recent-dropdown')) {
                document.getElementById('recent-menu').classList.remove('visible');
            }
        });

        async function openDataset() {
            const path = document.getElementById('dataset-path').value.trim();
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
                        data.warnings.join('\\n'),
                        'warning',
                        8000
                    );
                }

                // Load episodes
                const epRes = await fetch(`/api/datasets/${encodeURIComponent(data.id)}/episodes`);
                episodes[data.id] = await epRes.json();

                // Expand this dataset by default
                expandedNodes.add(data.id);

                // Save to recent
                addRecentDataset(path, data.repo_id);

                renderTree();
                setStatus(`Opened: ${data.repo_id}`);
                document.getElementById('dataset-path').value = '';
            } catch (e) {
                setStatus('Error: ' + e.message);
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
                container.innerHTML = '<div style="padding: 16px; color: #666; font-size: 13px;">No datasets loaded</div>';
                return;
            }

            let html = '';
            for (const [id, ds] of Object.entries(datasets)) {
                const isExpanded = expandedNodes.has(id);
                const dsEpisodes = episodes[id] || [];
                const dsEditCount = pendingEdits.filter(e => e.dataset_id === id).length;
                const totalFrames = dsEpisodes.reduce((sum, ep) => sum + ep.length, 0);
                const tooltip = `${ds.repo_id}\\n${ds.total_episodes} episodes, ${totalFrames.toLocaleString()} frames\\nPath: ${ds.root}`;

                html += `
                    <div class="tree-node">
                        <div class="tree-header" onclick="toggleDataset('${id}')" title="${tooltip}">
                            <span class="tree-toggle">${isExpanded ? '▼' : '▶'}</span>
                            <span class="tree-icon">📁</span>
                            <span class="tree-label">${ds.repo_id}</span>
                            <span class="tree-meta">${dsEditCount > 0 ? `${dsEditCount}✎ ` : ''}${ds.total_episodes} ep</span>
                        </div>
                        <div class="tree-children ${isExpanded ? 'expanded' : ''}">
                `;

                for (const ep of dsEpisodes) {
                    const isActive = currentDataset === id && currentEpisode === ep.episode_index;
                    const isDeleted = isEpisodeDeleted(id, ep.episode_index);
                    const isTrimmed = isEpisodeTrimmed(id, ep.episode_index);
                    const classes = ['tree-header'];
                    if (isActive) classes.push('active');
                    if (isDeleted) classes.push('deleted');
                    if (isTrimmed) classes.push('trimmed');

                    html += `
                        <div class="${classes.join(' ')}"
                             onclick="selectEpisode('${id}', ${ep.episode_index}, ${ep.length})"
                             oncontextmenu="showContextMenu(event, '${id}', ${ep.episode_index})">
                            <span class="tree-toggle"></span>
                            <span class="tree-icon">${isDeleted ? '🗑️' : '🎬'}</span>
                            <span class="tree-label">Episode ${ep.episode_index}</span>
                            <span class="tree-meta">${ep.length} frames</span>
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
                            <img id="frame-${cam.replace(/\\./g, '-')}" src="" alt="${camName}">
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
                const imgId = `frame-${cam.replace(/\\./g, '-')}`;
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
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
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
                if (ep) selectEpisode(currentDataset, newIndex, ep.length);
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
                const action = item.getAttribute('onclick').match(/contextAction\\('(\\w+)'\\)/)?.[1];
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
            contextMenuTarget = null;
        }

        document.addEventListener('click', hideContextMenu);

        function contextAction(action) {
            if (!contextMenuTarget) return;
            const { datasetId, episodeIndex } = contextMenuTarget;

            if (action === 'view') {
                const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
                if (ep) selectEpisode(datasetId, episodeIndex, ep.length);
            } else if (action === 'rerun') {
                const ep = episodes[datasetId]?.find(e => e.episode_index === episodeIndex);
                if (ep) {
                    selectEpisode(datasetId, episodeIndex, ep.length);
                    launchRerun();
                }
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
            try {
                const res = await fetch('/api/edits/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_id: datasetId, episode_index: episodeIndex })
                });
                if (!res.ok) throw new Error(await res.text());
                await refreshPendingEdits();
                setStatus(`Episode ${episodeIndex} marked for deletion`);
            } catch (e) {
                setStatus('Error: ' + e.message);
            }
        }

        async function unmarkEpisodeDeleted(datasetId, episodeIndex) {
            // Find and remove the delete edit
            const editIndex = pendingEdits.findIndex(
                e => e.dataset_id === datasetId && e.episode_index === episodeIndex && e.edit_type === 'delete'
            );
            if (editIndex >= 0) {
                try {
                    const res = await fetch(`/api/edits/${editIndex}`, { method: 'DELETE' });
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

        async function discardEdits() {
            if (!confirm('Discard all pending edits?')) return;
            try {
                const res = await fetch('/api/edits/discard', { method: 'POST' });
                if (!res.ok) throw new Error(await res.text());
                await refreshPendingEdits();
                setStatus('All edits discarded');
            } catch (e) {
                setStatus('Error: ' + e.message);
            }
        }

        async function applyEdits() {
            if (!currentDataset) {
                setStatus('No dataset selected');
                return;
            }
            if (!confirm(`Apply ${pendingEdits.length} edit(s) to disk? This cannot be undone.`)) return;

            setStatus('Applying edits...');
            try {
                const res = await fetch(`/api/edits/apply?dataset_id=${encodeURIComponent(currentDataset)}`, {
                    method: 'POST'
                });
                const data = await res.json();
                if (data.status === 'ok' || data.status === 'partial') {
                    // Reload dataset episodes
                    const epRes = await fetch(`/api/datasets/${encodeURIComponent(currentDataset)}/episodes`);
                    episodes[currentDataset] = await epRes.json();
                    await refreshPendingEdits();
                    setStatus(data.message);
                } else {
                    throw new Error(data.message);
                }
            } catch (e) {
                setStatus('Error: ' + e.message);
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

        // Initialize
        refreshPendingEdits();
        updateRecentButton();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the minimal HTML viewer."""
    return MINIMAL_HTML


def run_server(host: str = "127.0.0.1", port: int = 8000, cache_size: int = 500_000_000):
    """Run the GUI server."""
    import uvicorn

    app.state.cache_size = cache_size
    logger.info(f"Starting LeRobot Dataset GUI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, access_log=False)


def setup_logging(log_dir: Path | None = None) -> Path:
    """Configure persistent file logging for the GUI server.

    Only configures logging for lerobot.gui.* loggers, not the root logger.
    Called once at server startup.

    Args:
        log_dir: Directory for log files. Defaults to ~/.cache/huggingface/lerobot/gui/logs/

    Returns:
        Path to the log directory.
    """
    if log_dir is None:
        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with date in name
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"server_{date_str}.log"

    # Configure only the lerobot.gui logger (not root)
    gui_logger = logging.getLogger("lerobot.gui")
    gui_logger.setLevel(logging.INFO)
    gui_logger.propagate = False  # Don't propagate to root logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # File handler (rotating, max 10MB per file, keep 10 files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    gui_logger.addHandler(console_handler)
    gui_logger.addHandler(file_handler)

    gui_logger.info(f"Logging to {log_file}")

    return log_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LeRobot Dataset GUI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--cache-size",
        default="500MB",
        help="Frame cache size (default: 500MB). Examples: 500MB, 1GB, 2GB",
    )

    args = parser.parse_args()

    # Setup persistent logging before starting server
    setup_logging()

    cache_bytes = parse_cache_size(args.cache_size)
    run_server(host=args.host, port=args.port, cache_size=cache_bytes)


if __name__ == "__main__":
    main()
