# LeRobot GUI

Browser-based tool for reviewing and editing robot training datasets.

![LeRobot GUI](screenshot.png)

## TLDR

```bash
python -m lerobot.gui --port 8000
```

Open http://127.0.0.1:8000/, enter a repo ID (e.g. `lerobot/pusht`) or local path, then:

- **Play** episodes across all cameras with timeline scrubbing
- **Delete** episodes you don't want (mark → save)
- **Trim** episodes to keep only the useful frames (drag handles → save)
- **Visualize** any episode in Rerun via right-click

All edits are non-destructive until you hit **Save Changes**.

## Usage

```bash
python -m lerobot.gui [--host HOST] [--port PORT] [--cache-size SIZE]
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` for network access. |
| `--port` | `8000` | TCP port. |
| `--cache-size` | `500MB` | In-memory frame cache budget (`500MB`, `1GB`, etc.). |

## Dataset Playback

- Enter a HuggingFace repo ID or absolute local path to open a dataset.
- Episodes are listed in the sidebar; click one to load it.
- Multi-camera views auto-arrange in a responsive grid (1–3 columns depending on camera count).
- Playback controls: play/pause, speed (0.25x–2x), timeline scrubber with hover preview.
- Frames are decoded once per index across all cameras and served as JPEG with an LRU cache and background prefetch.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Previous / next frame |
| `Shift+←` / `Shift+→` | Jump 10 frames |
| `↑` / `↓` | Previous / next episode |
| `Home` / `End` | First / last frame |
| `Delete` | Toggle episode deletion |
| `r` | Reset trim |

## Episode Deletion

1. Select an episode and press `Delete` (or right-click → Delete Episode).
2. The episode shows strikethrough in the sidebar.
3. Press `Delete` again or right-click → Restore to undo.
4. Click **Save Changes** to apply — removes metadata and parquet rows without re-encoding video.

## Episode Trimming

1. Drag the green trim handles on the timeline to set the keep region.
2. Red-tinted zones show frames that will be cut.
3. The trim info bar shows the kept range (e.g. `Keep: frames 12–85 (74 of 100)`).
4. Right-click → Clear Trim to undo before saving.
5. Click **Save Changes** to apply — re-encodes only the trimmed segment using a streaming video encoder.

## Rerun Integration

Right-click any episode → **Open in Rerun** to launch the [Rerun](https://rerun.io/) visualizer for that episode. This spawns a separate process independent of the web server.
