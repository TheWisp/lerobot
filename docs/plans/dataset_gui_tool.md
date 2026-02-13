# LeRobot Dataset Visualization and Editing GUI Tool

## Overview

A local desktop application for visualizing and editing LeRobot datasets, similar to rerun.io but with additional dataset management and editing features.

## Core Features

### Dataset Management
- Load one or more LeRobot datasets
- Connect to HuggingFace Hub (authentication required)
- Open local datasets (drag and drop support)
- Create new datasets from the UI
- Tree view showing multiple datasets and their episodes

### Episode Visualization
- Main panel displays all episode features:
  - Camera views (multiple simultaneous video streams)
  - Actions (plotted as time series)
  - States (plotted as time series)
  - Other features
- Horizontal timeline with timestamps
- Play/pause controls
- Click-to-seek on timeline
- Frame-by-frame navigation

### Episode Editing
- Mark episodes as deleted (with confirmation)
- Reorder episodes (swap)
- Duplicate episodes
- Copy/move episodes between datasets
  - Enables natural dataset merging
  - Enables dataset splitting (create new + move)
- Timeline trimming (iOS-style drag handles)
  - Shrink/expand start and end
  - Combined with duplicate: carve multiple segments from one episode

### Data Model
- Source of truth: locally cached dataset files
- In-memory working copy for edits
- Edits are local until explicit save
- UI navigation doesn't lose unsaved changes
- Global save operation (and later: undo/redo)

---

## Local vs Remote Dataset Philosophy

### Core Principle: Local-First Editing

The GUI operates on a **local-first** model where:

```
┌─────────────────────────────────────────────────────────┐
│                    HuggingFace Hub                       │
│                   (remote storage)                       │
└─────────────────────────────────────────────────────────┘
              ↑ Upload                    ↓ Download
              │ (explicit,                │ (explicit,
              │  overwrites remote)       │  overwrites local)
┌─────────────────────────────────────────────────────────┐
│                    Local Dataset                         │
│            (source of truth for editing)                 │
│                                                          │
│   • All edits happen here                                │
│   • Reload = re-read from local disk only                │
│   • User always knows the local path                     │
└─────────────────────────────────────────────────────────┘
```

### Why No Sync/Merge?

1. **No version control** - LeRobot datasets don't have commit history or checksums
2. **No conflict resolution** - We can't reliably detect what changed or merge edits
3. **Explicit is better** - User decides direction: "replace local" or "replace remote"
4. **Simplicity** - No complex sync state to track or debug

### Dataset Opening Modes

| Mode | Input | Behavior |
|------|-------|----------|
| **Open Local** | `/path/to/dataset` | Work directly on local files |
| **Open from Hub** | `user/repo_id` | Download to local path, then work locally |

When opening from Hub:
- If local cache exists: ask user "Use cached version or download fresh?"
- If no cache: download to `~/.cache/huggingface/lerobot/{repo_id}/`
- After download: work entirely on local copy

### Reload Semantics

**"Reload" always means re-read from local disk:**
- Used after applying edits (parquet/metadata changed on disk)
- Used when external tool modified the dataset
- Never fetches from HuggingFace Hub

```python
def reload_dataset_from_disk(dataset: LeRobotDataset) -> None:
    """Re-read all data from local disk. No network calls."""
    root = dataset.root
    dataset.meta.info = load_info(root)
    dataset.meta.episodes = load_episodes(root)
    dataset.meta.stats = load_stats(root)
    dataset.meta.tasks = load_tasks(root)
    dataset.hf_dataset = load_nested_dataset(root / "data", ...)
    # Invalidate caches
    frame_cache.invalidate_dataset(dataset_id)
    video_decoder_cache.clear()
```

### HuggingFace Hub Operations

Both operations are **destructive** and require user confirmation:

#### Download (Pull from Hub)
```
User clicks "Download from Hub"
→ Confirmation: "Replace local dataset with version from HuggingFace Hub?
                 Local changes will be lost."
→ Download all files from Hub to local path (overwrite)
→ Reload dataset from disk
→ Clear all pending edits
```

#### Upload (Push to Hub)
```
User clicks "Upload to Hub"
→ Confirmation: "Push local dataset to HuggingFace Hub?
                 Remote version will be overwritten."
→ Check HF authentication
→ Upload all local files to Hub (overwrite)
→ Show success/failure
```

### API Endpoints

```
# Dataset opening (existing, clarified)
POST /api/datasets
  body: { local_path: "/path/to/dataset" }           # Open local
  body: { repo_id: "user/dataset" }                   # Open from Hub (downloads)

# Hub operations (new)
POST /api/datasets/{id}/hub/download
  - Requires confirmation token from frontend
  - Downloads from Hub, overwrites local
  - Returns: { status, message, episodes_count }

POST /api/datasets/{id}/hub/upload
  - Requires confirmation token from frontend
  - Uploads local to Hub, overwrites remote
  - Returns: { status, message, url }

GET /api/hub/auth-status
  - Returns: { logged_in: bool, username: str | null }

POST /api/hub/login
  body: { token: "hf_..." }
  - Stores token using huggingface_hub.login()
```

### UI Elements

```
┌─────────────────────────────────────────────────────────┐
│ Dataset: my_robot_data                                  │
│ Local: /home/user/datasets/my_robot_data               │
│ Hub: lerobot/my_robot_data (linked)          [⟳] [↑]   │
│                                              DL  Upload │
└─────────────────────────────────────────────────────────┘

# Auth indicator in header:
│ HF: logged in as @username │  or  │ HF: not logged in [Login] │
```

### Implementation Details

#### Upload Implementation
```python
async def upload_to_hub(dataset_id: str, repo_id: str | None = None):
    """Push local dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi, upload_folder

    dataset = _app_state.datasets[dataset_id]
    repo_id = repo_id or dataset.repo_id

    # Ensure logged in
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        raise HTTPException(401, "Not logged in to HuggingFace Hub")

    # Upload entire dataset folder
    upload_folder(
        folder_path=str(dataset.root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update from LeRobot GUI",
    )

    return {"status": "ok", "url": f"https://huggingface.co/datasets/{repo_id}"}
```

#### Download Implementation
```python
async def download_from_hub(dataset_id: str):
    """Download fresh copy from HuggingFace Hub, replacing local."""
    from huggingface_hub import snapshot_download

    dataset = _app_state.datasets[dataset_id]
    repo_id = dataset.repo_id
    local_path = dataset.root

    # Clear pending edits first
    _app_state.clear_edits(dataset_id)
    clear_edits_file(local_path)

    # Download fresh (force re-download)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        force_download=True,
    )

    # Reload from disk
    reload_dataset_from_disk(dataset)

    return {"status": "ok", "message": f"Downloaded {dataset.meta.total_episodes} episodes"}
```

---

## Technical Analysis: LeRobot Dataset Format

### Dataset Structure
```
dataset_root/
├── meta/
│   ├── info.json              # fps, features, codec info
│   ├── tasks.parquet          # task definitions
│   └── episodes/chunk-*/      # episode metadata (parquet)
├── data/chunk-*/              # frame data (parquet, ~100MB chunks)
└── videos/{camera}/chunk-*/   # video files (mp4, ~200MB chunks)
```

### Performance Characteristics

| Component | Load Time | Notes |
|-----------|-----------|-------|
| Metadata (info.json) | <10ms | Loaded once at init |
| Episode info (parquet) | ~50ms | Memory-mapped, fast lookups |
| Frame data (parquet) | ~1ms/frame | Memory-mapped, snappy compression |
| **Video decode** | **50-500ms/frame** | **THE BOTTLENECK** |

### Video Decoding Bottleneck Analysis

**Current implementation (`LeRobotDataset.__getitem__`):**
1. Each frame request triggers `decode_video_frames()` synchronously
2. No frame caching - rewatching re-decodes same frames
3. DataLoader multiprocessing causes segfaults with video decoders
4. Default `num_workers=0` means single-threaded decoding

**Backend comparison:**
| Backend | Latency | Notes |
|---------|---------|-------|
| torchcodec | 50-100ms | Preferred, GPU support, approximate seeking |
| pyav | 200-500ms | Fallback, must decode from keyframes |

**Root cause:** Video containers (MP4) require seeking to keyframes (every 1-2s) and decoding forward. Random access is fundamentally expensive.

---

## Recommended Architecture

### Decision: **Python Backend (FastAPI) + Web Frontend**

**Rationale:**
1. **Must use LeRobot Python code** - No rewriting dataset logic
2. **Video decoding stays in Python** - torchcodec/pyav are Python libs
3. **Web UI is best for timeline/video UX** - Rich ecosystem for scrubbing, charts
4. **Can wrap in Tauri later** - For native feel without changing code

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Frontend (React)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ TreeView │  │ Timeline │  │ VideoGrid│  │ Action Charts   │  │
│  │(datasets)│  │(scrubber)│  │(cameras) │  │ (plotly/d3)     │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket + REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend (FastAPI)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Frame Cache (LRU)                        │ │
│  │  - Caches decoded frames by (dataset, episode, frame_idx)  │ │
│  │  - Configurable size (e.g., 500 frames = ~1GB for 720p)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Prefetch Worker (Thread)                   │ │
│  │  - Decodes next N frames in background                      │ │
│  │  - Triggered on playback or timeline seek                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Edit State Manager                         │ │
│  │  - In-memory pending edits (trim, delete, reorder)         │ │
│  │  - Applies edits on save via dataset_tools functions        │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              LeRobot Integration Layer                      │ │
│  │  - LeRobotDataset instances per opened dataset              │ │
│  │  - Calls trim_episode, delete_episodes, etc.                │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Optimizations

#### 1. Frame Cache with LRU Eviction
```python
class FrameCache:
    def __init__(self, max_frames=500):  # ~1GB for 720p RGB
        self.cache = OrderedDict()  # (dataset_id, ep_idx, frame_idx) -> jpeg_bytes
        self.max_frames = max_frames

    def get(self, key) -> bytes | None:
        if key in self.cache:
            self.cache.move_to_end(key)  # LRU update
            return self.cache[key]
        return None

    def put(self, key, frame_jpeg: bytes):
        self.cache[key] = frame_jpeg
        if len(self.cache) > self.max_frames:
            self.cache.popitem(last=False)  # Evict oldest
```

#### 2. Background Prefetching
```python
class PrefetchWorker:
    def __init__(self, cache: FrameCache, lookahead=30):
        self.queue = Queue()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def request_prefetch(self, dataset, ep_idx, start_frame, direction=1):
        """Called on seek or playback start"""
        for i in range(self.lookahead):
            frame_idx = start_frame + (i * direction)
            self.queue.put((dataset, ep_idx, frame_idx))

    def _worker(self):
        while True:
            dataset, ep_idx, frame_idx = self.queue.get()
            key = (id(dataset), ep_idx, frame_idx)
            if self.cache.get(key) is None:
                frame = dataset[frame_idx]  # Decode
                jpeg = encode_jpeg(frame)
                self.cache.put(key, jpeg)
```

#### 3. JPEG Encoding for Transfer
Send JPEG over WebSocket instead of raw RGB:
- Raw 720p frame: 720 × 1280 × 3 = 2.7 MB
- JPEG (quality=85): ~50-100 KB
- 30 fps playback: 1.5-3 MB/s vs 81 MB/s

#### 4. Thumbnail Strip for Timeline
Pre-generate 1 thumbnail per second for timeline scrubbing preview:
```python
def generate_thumbnail_strip(dataset, ep_idx, thumb_height=60):
    """Generate one thumbnail per second for timeline preview"""
    fps = dataset.fps
    length = dataset.meta.episodes[ep_idx]["length"]
    thumbnails = []
    for frame_idx in range(0, length, fps):  # One per second
        frame = dataset[frame_idx]["observation.images.front"]
        thumb = resize(frame, height=thumb_height)
        thumbnails.append(encode_jpeg(thumb, quality=60))
    return thumbnails
```

---

## API Design

### REST Endpoints

```
GET  /api/datasets                     # List opened datasets
POST /api/datasets                     # Open dataset (local path or HF repo_id)
DELETE /api/datasets/{id}              # Close dataset

GET  /api/datasets/{id}/episodes       # List episodes with metadata
GET  /api/datasets/{id}/episodes/{ep}  # Episode details (length, tasks, etc.)
GET  /api/datasets/{id}/episodes/{ep}/thumbnails  # Thumbnail strip for timeline

GET  /api/datasets/{id}/episodes/{ep}/frame/{idx}  # Single frame (JPEG)
GET  /api/datasets/{id}/episodes/{ep}/data         # Parquet data (action, state)

POST /api/edits/trim                   # Queue trim edit
POST /api/edits/delete                 # Queue delete edit
POST /api/edits/reorder                # Queue reorder edit
GET  /api/edits/pending                # List pending edits
POST /api/edits/save                   # Apply all pending edits to disk
POST /api/edits/discard                # Discard pending edits
```

### WebSocket for Playback

```
WS /ws/playback/{dataset_id}/{episode_idx}

Client -> Server:
  { "action": "play", "from_frame": 0 }
  { "action": "pause" }
  { "action": "seek", "frame": 150 }
  { "action": "set_speed", "fps": 30 }

Server -> Client:
  { "type": "frame", "frame_idx": 0, "timestamp": 0.0,
    "cameras": { "front": "<base64 jpeg>", "wrist": "<base64 jpeg>" } }
  { "type": "buffering", "progress": 0.5 }
```

---

## Edit State Model

```python
@dataclass
class PendingEdit:
    edit_type: Literal["trim", "delete", "reorder", "duplicate", "move"]
    dataset_id: str
    params: dict
    created_at: datetime

@dataclass
class EditState:
    pending_edits: list[PendingEdit]

    # Virtual view: how episodes appear with pending edits applied
    # (without modifying files)
    def get_virtual_episodes(self, dataset_id) -> list[VirtualEpisode]:
        ...

    def apply_all(self):
        """Apply edits to disk using dataset_tools functions"""
        for edit in self.pending_edits:
            if edit.edit_type == "trim":
                trim_episode(dataset, **edit.params)
            elif edit.edit_type == "delete":
                delete_episodes(dataset, **edit.params)
            # etc.
        self.pending_edits.clear()
```

---

## HuggingFace Authentication

Use existing `huggingface_hub` token:
```python
from huggingface_hub import HfApi, login

def get_hf_token():
    """Check for existing token or prompt login"""
    api = HfApi()
    try:
        api.whoami()  # Throws if not logged in
        return api.token
    except:
        return None  # Frontend should show login prompt

def hf_login(token: str):
    """Store token using huggingface_hub"""
    login(token=token, add_to_git_credential=True)
```

---

## File Structure

```
lerobot/
├── src/lerobot/
│   ├── datasets/
│   │   └── dataset_tools.py      # Edit functions (trim, delete, verify)
│   └── gui/                       # GUI package
│       ├── __init__.py
│       ├── __main__.py            # Entry point: python -m lerobot.gui
│       ├── server.py              # FastAPI app + inline HTML/JS frontend
│       ├── frame_cache.py         # LRU frame cache
│       ├── state.py               # AppState + PendingEdit + persistence
│       └── api/
│           ├── __init__.py
│           ├── datasets.py        # Dataset endpoints
│           ├── playback.py        # WebSocket handler
│           └── edits.py           # Edit endpoints
└── tests/gui/                     # GUI tests
    └── test_episode_indexing.py
```

**Note:** Frontend is currently inline in `server.py`. See Refactoring TODOs for extraction + React migration plan.

---

## Development Phases

### Phase 1: Read-Only Viewer
- [x] FastAPI server with dataset loading
- [x] Frame cache + prefetching
- [x] WebSocket playback
- [x] Frontend: tree view, video grid, timeline (inline HTML/JS, not React)
- [ ] Parquet data display (action/state charts)
- [ ] **(P1) Monitor local dataset changes** - Detect new episodes recorded while GUI is open; auto-refresh UI to show new episodes in tree view

### Phase 2: Basic Editing
- [x] Trim episode (iOS-style handles)
- [x] Delete episode
- [x] Pending edits model
- [x] Save/discard UI
- [x] Pending edits persistence (survives server restart)
- [ ] **(P0) Dataset locking during operations** - Lock dataset (not editable) while server is processing. For now, lock entire dataset during any operation since all operations are fast. Later can be per-episode granularity.

### Phase 3: Advanced Editing
- [ ] Duplicate episode
- [ ] Copy/move between datasets
- [ ] Reorder episodes
- [ ] Create new dataset

### Phase 4: Polish
- [ ] Undo/redo
- [ ] HuggingFace Hub sync (see "Local vs Remote Dataset Philosophy" section):
  - [ ] `GET /api/hub/auth-status` - check login state
  - [ ] `POST /api/hub/login` - store HF token
  - [ ] `POST /api/datasets/{id}/hub/download` - pull from Hub (overwrites local)
  - [ ] `POST /api/datasets/{id}/hub/upload` - push to Hub (overwrites remote)
  - [ ] Frontend: auth indicator in header
  - [ ] Frontend: download/upload buttons per dataset with confirmation dialogs
- [ ] Drag-drop dataset opening
- [ ] Optional Tauri wrapper for native feel

---

## Refactoring TODOs

### High Priority

1. **Extract frontend to separate files (then migrate to React)**
   - Phase A: Move ~1000 lines of inline HTML/CSS/JS from `server.py` to:
     - `frontend/index.html`
     - `frontend/styles.css`
     - `frontend/app.js`
   - Serve with FastAPI `StaticFiles`
   - Phase B: Migrate to React for better component structure
     - Use Vite for build tooling
     - Split into components: TreeView, Timeline, VideoGrid, EditsBar
     - Add TypeScript for type safety

2. **Extract `reload_dataset_from_disk()` function**
   - Currently duplicated in `edits.py:apply_edits()` and `datasets.py:_check_and_reload_metadata()`
   - Should be single function in `state.py` or new `reload.py`
   - Handles: clear caches, disable HF caching, reload hf_dataset, reload metadata

3. **Extract episode loading helper**
   - Pattern `if dataset.meta.episodes is None: load_episodes()` appears 4+ times
   - Should be `ensure_episodes_loaded(dataset)` helper

### Medium Priority

4. **Use FastAPI dependency injection for AppState**
   - Replace global `_app_state` with `Depends(get_app_state)`
   - More testable, cleaner architecture

5. **Consolidate module-level caches**
   - `_episode_start_indices`, `_dataset_info_mtime` scattered in `datasets.py`
   - Move into `AppState` or dedicated `CacheManager`

---

## Future Features

### 3D Robot Visualization (URDF-based)

**Motivation:** Data synchronization issues between camera streams and action/proprioception data are common in robotics datasets. Human reviewers need to verify that video frames align with recorded joint positions and commanded actions.

**Proposed feature:**
- Load robot URDF model from dataset or user-provided file
- Render 3D robot visualization synchronized with timeline
- Display current joint positions from `observation.state`
- Optionally overlay commanded actions from `action`
- Side-by-side comparison: video | 3D model | action/state plots

**Technical considerations:**
- Use Three.js or similar for WebGL rendering
- URDF parser: `urdf-loader` npm package or custom Python parser
- Joint state mapping: need to handle different robot configurations
- Performance: 3D rendering should not block video playback

**Use cases:**
1. Verify camera-to-robot time alignment
2. Debug teleop latency issues
3. Identify dropped frames or sensor glitches
4. Quality check before training

### Robot Integration (requires local robot setup)

**Prerequisite:** User must configure local robot connection (robot type, IP/port, etc.). This configuration could be stored per-dataset or globally.

**Proposed features:**

1. **Replay episode on robot** (`lerobot-replay`)
   - Select episode in GUI → "Replay on Robot" button
   - Robot executes recorded actions in real-time
   - Live camera feed comparison: recorded vs current
   - Useful for verifying data quality and robot calibration

2. **Record new episodes**
   - Start/stop recording from GUI
   - Live preview of camera streams
   - Episode appears in tree view after recording completes
   - Integrates with existing `lerobot-record` functionality

3. **Run inference/policy testing**
   - Load trained policy checkpoint
   - Run policy on robot with live visualization
   - Record inference episodes to dataset
   - Compare policy behavior to demonstration episodes

**Technical considerations:**
- Robot configuration: YAML file or GUI settings panel
- Safety: Emergency stop button, speed limits during replay
- Network: Robot may be on different machine (SSH tunnel, ROS bridge)
- Dependencies: Only load robot-related code when features are used

**Implementation approach:**
- Start with replay (simplest, read-only)
- Add recording (requires robot config)
- Add inference (requires policy loading)

---

## Open Questions

1. **Multiple camera sync:** Should we allow independent scrubbing per camera, or always sync?
2. **Large datasets:** For datasets with 1000+ episodes, need virtual scrolling in tree view
3. **Video re-encoding on trim:** Show progress bar? Allow cancel?
4. **Concurrent editing:** Lock dataset while editing, or allow multiple sessions?
5. **URDF loading:** Should URDF be stored in dataset metadata, or loaded separately?
