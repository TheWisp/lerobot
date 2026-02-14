# Plan: Async Video Compression in lerobot_record Pipeline

## Current Architecture Analysis

### Where Blocking Occurs

The blocking wait happens in `save_episode()` at [lerobot_dataset.py:1235-1270](src/lerobot/datasets/lerobot_dataset.py#L1235-L1270):

```
record_loop() completes
    ↓
save_episode() called at line 555 in lerobot_record.py
    ↓
_wait_image_writer() - blocks until async PNG writes finish (line 1228)
    ↓
_save_episode_data() - saves parquet (fast, ~milliseconds)
    ↓
[BLOCKING] Video encoding via ProcessPoolExecutor (lines 1240-1267)
    - Single camera: sequential _save_episode_video()
    - Multi-camera: parallel encoding but still blocks main thread
    ↓
meta.save_episode() - MUST run after encoding (line 1273)
    - Needs video metadata: timestamps, chunk/file indices
```

### Video Encoding Pipeline

1. **Storage during recording**: PNG frames written asynchronously by `AsyncImageWriter`
2. **Encoding trigger**: `save_episode()` calls `encode_video_frames()` via PyAV
3. **Codec**: libsvtav1 (AV1) default, h264/hevc available
4. **Output**: Concatenated MP4 files in `videos/{camera_key}/chunk-{N}/file-{M}.mp4`

### Batched Encoding (Existing Feature)

When `batch_encoding_size > 1`, encoding is deferred:
- Episodes accumulate without immediate encoding
- Every N episodes, `_batch_save_episode_video()` encodes all at once
- This reduces per-episode blocking but creates periodic burst blocking
- `VideoEncodingManager.__exit__()` handles final cleanup

---

## Re-record Compatibility Analysis

### Current Re-record Flow ([lerobot_record.py:548-553](src/lerobot/scripts/lerobot_record.py#L548-L553))

```python
if events["rerecord_episode"]:
    log_say("Re-record episode", cfg.play_sounds)
    events["rerecord_episode"] = False
    events["exit_early"] = False
    dataset.clear_episode_buffer()  # Deletes PNGs, resets buffer
    continue  # Loop back to record_loop()
```

### Why Re-record is Safe with Async Encoding

Re-record operates on the **current episode in progress**, while async encoding would process **previously saved episodes**:

| State | Current Episode (re-recordable) | Previous Episodes (encoding) |
|-------|--------------------------------|------------------------------|
| PNG frames | In `images/episode_N/` | Already consumed by encoder |
| Episode buffer | In memory | Already saved to parquet |
| Parquet data | Not yet written | Written before encoding started |
| Video | Not started | Being encoded or queued |

**Conclusion**: Re-record is compatible because it never touches episodes that have already called `save_episode()`. The episode indices are distinct.

---

## Queue Buildup Concerns

### The Problem

If episodes are short (e.g., 3-5 seconds) and encoding takes longer than episode duration:
- Episode 5 ends → queued for encoding
- Episode 6 ends → queued for encoding
- Episode 7 ends → queue keeps growing
- Memory and resource exhaustion possible

### Mitigation Strategies

1. **Bounded Queue with Backpressure**
   - Limit queue to N pending encoding jobs (e.g., 3-5 episodes)
   - Block if queue full until a slot opens
   - Trade-off: introduces occasional blocking

2. **Priority Queue with Dropping Old Work**
   - Not viable - all episodes must be encoded for dataset integrity

3. **Adaptive Batching**
   - If queue > threshold, switch to batch mode dynamically
   - Encode multiple episodes in one burst

4. **Monitoring + Warning**
   - Track queue depth, warn user if falling behind
   - Log encoding time vs recording time ratio

---

## Implementation Approach

### Option A: Background Encoding Thread/Process Pool (Recommended)

```
save_episode() changes:
    ↓
_wait_image_writer()  [still blocking - needed for stats]
    ↓
_save_episode_data()  [parquet - fast]
    ↓
Submit encoding job to persistent ThreadPoolExecutor/ProcessPoolExecutor
    ↓
Return immediately  [non-blocking!]
    ↓
meta.save_episode() called with partial metadata
    - Video metadata updated asynchronously when encoding completes
```

**Key Components:**

1. **EncodingJobQueue** class:
   - Persistent executor (not created per-episode)
   - Tracks pending jobs: `{episode_idx: Future}`
   - Bounded queue with configurable max_pending
   - Callback on completion to update metadata

2. **Metadata Handling**:
   - Split `meta.save_episode()` into two phases:
     - Phase 1 (immediate): episode_length, tasks, stats
     - Phase 2 (async callback): video timestamps, chunk/file indices
   - Or: Write placeholder video metadata, update on completion

3. **Finalization**:
   - `dataset.finalize()` must wait for all pending encoding jobs
   - `VideoEncodingManager.__exit__()` already handles cleanup

### Option B: Async/Await Pattern

Use Python asyncio with async executor:
- More complex, requires async context throughout
- Better for I/O bound, but encoding is CPU-bound
- Not recommended for this use case

### Option C: Separate Encoding Process

Spawn dedicated encoding process that watches a queue:
- Most isolation, can survive crashes
- More complex IPC
- Overkill for this use case

---

## Detailed Implementation Plan (Option A)

### Step 1: Create AsyncVideoEncoder Class

Location: `src/lerobot/datasets/video_utils.py`

```python
class AsyncVideoEncoder:
    def __init__(self, max_pending: int = 5, num_workers: int = None):
        self.executor = ProcessPoolExecutor(max_workers=num_workers or cpu_count() - 1)
        self.pending_jobs: dict[int, Future] = {}
        self.max_pending = max_pending
        self.completion_callbacks: list[Callable] = []
        self._lock = threading.Lock()

    def submit(self, episode_idx: int, video_key: str, ...):
        # Block if queue full (backpressure)
        while len(self.pending_jobs) >= self.max_pending:
            self._wait_one()

        future = self.executor.submit(_encode_video_worker, ...)
        future.add_done_callback(lambda f: self._on_complete(episode_idx, video_key, f))
        self.pending_jobs[episode_idx] = future

    def wait_all(self):
        """Wait for all pending jobs - called during finalization."""
        concurrent.futures.wait(self.pending_jobs.values())

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown()
```

### Step 2: Modify save_episode() for Async Path

Location: `src/lerobot/datasets/lerobot_dataset.py`

```python
def save_episode(self, ..., async_encoding: bool = True):
    ...
    if has_video_keys and async_encoding and not use_batched_encoding:
        # Submit async encoding jobs
        for video_key in self.meta.video_keys:
            self.async_encoder.submit(
                episode_index, video_key, self.root, self.fps, self.vcodec,
                callback=lambda meta: self._update_video_metadata(episode_index, meta)
            )
        # Save episode with placeholder video metadata
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats,
                               ep_metadata, video_pending=True)
    else:
        # Existing synchronous path
        ...
```

### Step 3: Handle Metadata Updates

When async encoding completes, update the episode metadata:

```python
def _update_video_metadata(self, episode_index: int, video_meta: dict):
    """Called when async video encoding completes."""
    # Update parquet with video file paths/timestamps
    # Or: design metadata schema to not require video info upfront
```

### Step 4: Modify finalize() to Wait for Encoding

```python
def finalize(self):
    if self.async_encoder:
        logging.info("Waiting for pending video encoding jobs...")
        self.async_encoder.wait_all()
    # ... existing finalization
```

### Step 5: Configuration

Add to config:
```yaml
async_video_encoding: true
max_pending_encoding_jobs: 5
encoding_workers: null  # Auto-detect from CPU count
```

---

## Testing Considerations

1. **Unit Tests**:
   - AsyncVideoEncoder queue behavior
   - Backpressure mechanism
   - Completion callbacks

2. **Integration Tests**:
   - Record multiple episodes, verify all encoded
   - Test re-record during async encoding
   - Test graceful shutdown with pending jobs

3. **Edge Cases**:
   - Encoding failure mid-recording
   - Very short episodes (< 1 second)
   - Keyboard interrupt during encoding queue
   - Resume recording with pending encodes from previous session (not supported - always finalize first)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Encoding fails after episode saved | Incomplete dataset | Transactional metadata; mark episodes as "pending encoding"; recover on resume |
| Queue overflow | Memory exhaustion | Bounded queue with backpressure |
| Race conditions in metadata updates | Corrupted metadata | Use locks; design schema to avoid updates |
| Complexity increase | Maintenance burden | Keep sync path as fallback; feature flag |

---

## Summary

**Feasibility**: Yes, async video compression is implementable and compatible with re-recording.

**Recommended Approach**: Option A - Background ProcessPoolExecutor with bounded queue and backpressure.

**Key Changes**:
1. New `AsyncVideoEncoder` class in video_utils.py
2. Modified `save_episode()` with async path
3. Split or deferred metadata updates
4. Modified `finalize()` to wait for completion
5. Configuration options for queue size and worker count

**Compatibility**: Re-record remains safe because it only affects the current in-progress episode, which hasn't been submitted for encoding yet.
