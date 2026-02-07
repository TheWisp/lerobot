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
│   │   └── dataset_tools.py      # Existing edit functions
│   └── gui/                       # NEW: GUI package
│       ├── __init__.py
│       ├── server.py              # FastAPI app
│       ├── frame_cache.py         # LRU frame cache
│       ├── prefetch.py            # Background prefetcher
│       ├── edit_state.py          # Pending edits manager
│       ├── api/
│       │   ├── datasets.py        # Dataset endpoints
│       │   ├── playback.py        # WebSocket handler
│       │   └── edits.py           # Edit endpoints
│       └── frontend/              # React app (built artifacts)
│           ├── index.html
│           └── assets/
└── scripts/
    └── lerobot_gui.py             # Entry point: python -m lerobot.gui
```

---

## Development Phases

### Phase 1: Read-Only Viewer
- [ ] FastAPI server with dataset loading
- [ ] Frame cache + prefetching
- [ ] WebSocket playback
- [ ] React frontend: tree view, video grid, timeline
- [ ] Parquet data display (action/state charts)

### Phase 2: Basic Editing
- [ ] Trim episode (iOS-style handles)
- [ ] Delete episode
- [ ] Pending edits model
- [ ] Save/discard UI

### Phase 3: Advanced Editing
- [ ] Duplicate episode
- [ ] Copy/move between datasets
- [ ] Reorder episodes
- [ ] Create new dataset

### Phase 4: Polish
- [ ] Undo/redo
- [ ] HuggingFace Hub integration (push)
- [ ] Drag-drop dataset opening
- [ ] Optional Tauri wrapper for native feel

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

---

## Open Questions

1. **Multiple camera sync:** Should we allow independent scrubbing per camera, or always sync?
2. **Large datasets:** For datasets with 1000+ episodes, need virtual scrolling in tree view
3. **Video re-encoding on trim:** Show progress bar? Allow cancel?
4. **Concurrent editing:** Lock dataset while editing, or allow multiple sessions?
5. **URDF loading:** Should URDF be stored in dataset metadata, or loaded separately?
