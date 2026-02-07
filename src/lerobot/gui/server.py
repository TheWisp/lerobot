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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from lerobot.gui.api import datasets, playback
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
    logger.info(f"Initialized frame cache with {cache_size / 1_000_000:.0f} MB budget")


# Include API routers
app.include_router(datasets.router)
app.include_router(playback.router)


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

        .tree-container { flex: 1; overflow-y: auto; padding: 8px 0; }

        /* Tree view */
        .tree-node { user-select: none; }
        .tree-header { display: flex; align-items: center; padding: 6px 12px; cursor: pointer; gap: 6px; }
        .tree-header:hover { background: #0f3460; }
        .tree-header.active { background: #1a4a7a; }
        .tree-toggle { width: 16px; font-size: 10px; color: #888; flex-shrink: 0; }
        .tree-icon { width: 16px; text-align: center; flex-shrink: 0; }
        .tree-label { flex: 1; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .tree-meta { font-size: 11px; color: #666; flex-shrink: 0; }
        .tree-children { display: none; }
        .tree-children.expanded { display: block; }
        .tree-children .tree-header { padding-left: 28px; }

        /* Main content */
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        /* Camera grid */
        .camera-grid { flex: 1; display: grid; gap: 8px; padding: 16px; overflow: auto; }
        .camera-panel { background: #16213e; border-radius: 8px; display: flex; flex-direction: column; min-height: 200px; }
        .camera-title { padding: 8px 12px; font-size: 12px; color: #888; border-bottom: 1px solid #0f3460; }
        .camera-frame { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 0 0 8px 8px; overflow: hidden; }
        .camera-frame img { max-width: 100%; max-height: 100%; object-fit: contain; }

        /* Controls */
        .controls-bar { background: #16213e; border-top: 1px solid #0f3460; padding: 12px 16px; display: flex; align-items: center; gap: 16px; }
        .controls-bar button { padding: 8px 16px; border-radius: 4px; border: none; background: #4fc3f7; color: #000; cursor: pointer; font-size: 14px; }
        .controls-bar button:hover { background: #81d4fa; }
        .timeline { flex: 1; height: 8px; background: #0f3460; border-radius: 4px; cursor: pointer; position: relative; }
        .timeline-progress { height: 100%; background: #4fc3f7; border-radius: 4px; width: 0%; pointer-events: none; }
        .frame-info { font-size: 13px; color: #888; min-width: 120px; text-align: right; }
        .status { font-size: 12px; color: #666; min-width: 200px; }

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
                    <input type="text" id="dataset-path" placeholder="Path or repo_id">
                    <button onclick="openDataset()">Open</button>
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
                <div class="timeline" id="timeline" onclick="seekTimeline(event)">
                    <div class="timeline-progress" id="timeline-progress"></div>
                </div>
                <div class="frame-info" id="frame-info">- / -</div>
                <div class="status" id="status">Ready</div>
            </div>
        </div>
    </div>

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

                // Load episodes
                const epRes = await fetch(`/api/datasets/${encodeURIComponent(data.id)}/episodes`);
                episodes[data.id] = await epRes.json();

                // Expand this dataset by default
                expandedNodes.add(data.id);

                renderTree();
                setStatus(`Opened: ${data.repo_id}`);
                document.getElementById('dataset-path').value = '';
            } catch (e) {
                setStatus('Error: ' + e.message);
            }
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

                html += `
                    <div class="tree-node">
                        <div class="tree-header" onclick="toggleDataset('${id}')">
                            <span class="tree-toggle">${isExpanded ? '▼' : '▶'}</span>
                            <span class="tree-icon">📁</span>
                            <span class="tree-label" title="${ds.repo_id}">${ds.repo_id}</span>
                            <span class="tree-meta">${ds.total_episodes} ep</span>
                        </div>
                        <div class="tree-children ${isExpanded ? 'expanded' : ''}">
                `;

                for (const ep of dsEpisodes) {
                    const isActive = currentDataset === id && currentEpisode === ep.episode_index;
                    html += `
                        <div class="tree-header ${isActive ? 'active' : ''}" onclick="selectEpisode('${id}', ${ep.episode_index}, ${ep.length})">
                            <span class="tree-toggle"></span>
                            <span class="tree-icon">🎬</span>
                            <span class="tree-label">Episode ${ep.episode_index}</span>
                            <span class="tree-meta">${ep.length} frames</span>
                        </div>
                    `;
                }

                html += '</div></div>';
            }
            container.innerHTML = html;
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

            // Stop playback
            if (isPlaying) {
                togglePlay();
            }

            renderTree();
            renderCameraGrid();
            loadAllFrames(0);
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

            return Promise.all(promises);
        }

        async function playLoop() {
            const frameTime = 1000 / fps;

            while (isPlaying) {
                const startTime = performance.now();

                if (currentFrame >= totalFrames - 1) {
                    currentFrame = 0;
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

        function togglePlay() {
            if (!currentDataset || currentEpisode === null) return;

            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';

            if (isPlaying) {
                playLoop();
            }
        }

        function seekTimeline(e) {
            if (!currentDataset || currentEpisode === null) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            loadAllFrames(Math.floor(pct * totalFrames));
        }

        function setStatus(msg) {
            document.getElementById('status').textContent = msg;
        }

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') loadAllFrames(currentFrame - 1);
            else if (e.key === 'ArrowRight') loadAllFrames(currentFrame + 1);
            else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        });
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


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="LeRobot Dataset GUI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--cache-size",
        default="500MB",
        help="Frame cache size (default: 500MB). Examples: 500MB, 1GB, 2GB",
    )

    args = parser.parse_args()
    cache_bytes = parse_cache_size(args.cache_size)

    run_server(host=args.host, port=args.port, cache_size=cache_bytes)


if __name__ == "__main__":
    main()
