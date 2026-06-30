# Data Editing — segment + effect → augmented dataset

**Status:** prototype (2026-06-30). Builds on the Overlays SAM3 path
([overlays.md](overlays.md)) and the Hub-transfer job model
([hub_transfers.md](hub_transfers.md)).

## What it is

Camera-side **visual domain randomization** for imitation-learning data. The user
segments the task-relevant objects (the same SAM3 the data-tab overlay already
previews), and an offline pass rewrites every frame's **background** (or, for
global effects, the whole frame) and writes the result as a **new
LeRobotDataset**. Only camera pixels change — actions, states, tasks, and timing
are copied verbatim, so the augmented dataset trains exactly like the original.

This is the GreenAug / RoboEngine recipe: keeping a few objects and randomizing
the background gives the largest measured robustness gain, beating both
no-augmentation and (more expensive) generative backgrounds. The segmented
objects + anything the user marks are the **protected foreground**.

## Flow

```
Data tab → Overlays panel (pick SAM3, name objects)  ──►  "⚙ Process dataset…"
   │                                                          │
   │  objects = protected foreground                          ▼
   │                                              ProcessData modal (process.js)
   │                                              effect · params · apply-mode ·
   │                                              copies · output name
   ▼                                                          │ POST /api/process/start
GUI server (api/process.py)                                   ▼
   • frees the live overlay (VRAM)                  spawn detached worker subprocess
   • registers ProcessJobState              ──►     python -m lerobot.gui.process_worker
   • polls <job>.json for progress                          │
                                                            ▼
                                            dataset_postprocess.process_dataset:
                                            for each episode/camera/frame →
                                              SAM3 segment → feathered alpha →
                                              apply effect → add_frame
                                            → save_episode → new LeRobotDataset
```

The worker writes a per-job progress JSON (`~/.cache/lerobot/gui/process_jobs/`)
~2 Hz; the GUI's `GET /api/process/jobs` merges it and renders frame-count
progress cards (Cancel / Dismiss / Open dataset). "Open dataset" calls
`window.openDataset(out_root)`, so the augmented dataset lands in the tree.

## Effects

Background (foreground protected, soft-feathered seam):

- **Randomize background (color)** — random solid colour, per episode.
- **Randomize background (texture)** — random blobby colour texture, per episode.
- **Solid background color** — fixed colour (param).
- **Blur background** — Gaussian defocus (param: sigma).

Global (whole frame, mask ignored):

- **Jitter brightness / contrast** — per-episode random within ±amount.

**Apply mode** (`per_episode` default · `per_frame` · `static`) controls how often
a randomized effect re-samples. Per-episode is the right default for trajectory
data — per-frame flicker corrupts the motion cues a policy learns from. **Copies
per episode** writes N independently-randomized variants of each source episode.

## Layers

| Layer          | File                                              | Role                                                                                                     |
| -------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Core transform | `datasets/dataset_postprocess.py`                 | `process_dataset` + effect registry + compositing (pure, GPU-agnostic; SAM adapter injectable for tests) |
| Segmentation   | `overlays/adapters.py`                            | `Sam3TrackByDetectionAdapter.segment()` — raw per-object masks (shared body with `infer()`)              |
| Job IPC        | `gui/process_jobs.py`                             | `ProcessJobConfig` / `State` / `Paths` (reuses `hub_jobs` pid/atomic-write helpers)                      |
| Worker         | `gui/process_worker.py`                           | subprocess entry; progress writer thread; SIGTERM = graceful cancel                                      |
| API            | `gui/api/process.py`                              | `/effects`, `/start`, `/jobs`, `/{id}/cancel`, `/{id}/dismiss`                                           |
| UI             | `gui/static/process.js` + button in `overlays.js` | modal + job tray                                                                                         |

## Notes / limits

- One processing job per source dataset at a time (409 otherwise); the output
  path must not already exist (no clobber).
- SAM3 tracks one instance per concept — a two-arm scene protects one arm unless
  the user adds a second object row.
- Starting a job tears down the live data overlay to avoid double-loading SAM3 on
  the GPU.

Follow-ups are tracked in [../TODO.md](../TODO.md) (Data Editing section).
