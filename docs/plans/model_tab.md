# Model Tab Plan

## Overview

The Model tab manages local training outputs and HF Hub models. Unlike the Data tab (rich visual content per episode), models are primarily metadata + metrics + config. The key actions are: browse models, inspect training details, launch/monitor training, and feed models into the Run tab for policy evaluation.

---

## What We Know

### Local checkpoint structure (`outputs/` directory)

```
outputs/
  act_training_pickup_socket_feb_20/
    checkpoints/
      020000/
        pretrained_model/
          config.json              # policy architecture config
          model.safetensors        # weights (206MB for ACT, 7.3GB for pi05)
          train_config.json        # full training config (dataset, optimizer, wandb, etc.)
          policy_preprocessor.json
          policy_postprocessor.json
          *.safetensors            # normalizer states
        training_state/
          training_step.json       # {"step": 20000}
          optimizer_state.safetensors
          scheduler_state.json
          rng_state.safetensors
      last -> 020000               # symlink to latest
```

Each `train_config.json` contains: dataset repo_id, policy type + architecture params, optimizer/scheduler config, wandb run_id, batch_size, total steps, save_freq, etc.

### HF Hub models

Downloaded to `~/.cache/huggingface/` (standard HF cache). Policy loading via `PreTrainedPolicy.from_pretrained()` accepts both local paths and HF repo IDs (e.g. `lerobot/pi05_base`).

### Policy types

ACT, diffusion, pi0, pi0_fast, pi05, vqbet, tdmpc, sac, smolvla, groot, xvla, wall_x, reward_classifier. Registry in `policies/factory.py`.

### Training commands

```bash
# Fresh training
lerobot-train \
    --dataset.repo_id=thewisp/pickup_socket_merged_head \
    --policy.type=act \
    --policy.pretrained_path=outputs/.../pretrained_model \
    --output_dir=outputs/act_training_mar_01 \
    --batch_size=4 --steps=100000

# Resume
lerobot-train \
    --config_path=outputs/.../pretrained_model/train_config.json \
    --resume=true
```

### WandB

`WandBConfig` fields: enable, project, entity, run_id, mode. Training logs loss, grad_norm, lr, eval metrics at `log_freq` intervals. The `run_id` in `train_config.json` links back to the wandb run.

---

## Design

### Layout: Source Folders + Model List + Detail Panel

Same pattern as the Data tab:

```
+------------------+------------------------------------------------+
| Source Folders    |  Model Detail                                  |
|                  |                                                 |
| [+ Add folder]   |  act_training_pickup_socket_feb_20             |
|                  |                                                 |
| ~/Documents/     |  [Overview] [Config] [Checkpoints] [Logs]      |
|   lerobot/       |                                                 |
|   outputs/       |  Policy: ACT                                   |
|   > act_train..  |  Dataset: thewisp/pickup_socket_merged_head    |
|   > pi05_trai..  |  Steps: 20000 / 100000 (20%)                  |
|     > 002500     |  Batch size: 4                                 |
|     > 005000     |  Model size: 206 MB                            |
|     > last       |  WandB: lerobot/iqm4fb0y                       |
|                  |                                                 |
| ~/.cache/        |  [Train] [Resume] [Use in Run tab]             |
|   huggingface/   |                                                 |
+------------------+------------------------------------------------+
```

### Source folders

- Default sources: `outputs/` (relative to cwd), `~/.cache/huggingface/lerobot/` (HF cache)
- User can add custom folders
- Each source is scanned for directories containing `checkpoints/*/pretrained_model/config.json`
- HF cache has a different structure — scan for `config.json` + `model.safetensors` pairs
- Persist source list to `~/.config/lerobot/settings.json`

### Model tree

- Top level: output directories (training runs)
- Expandable: individual checkpoints (002500, 005000, last)
- Show policy type icon/badge, training status (completed / in-progress / failed)
- A running training job shows a spinner and live step count

### Detail panel — Overview subtab

For a selected training run (top-level):
- Policy type, dataset repo_id
- Training progress: current step / total steps, completion %
- Key hyperparams: batch_size, lr, optimizer type
- Model size (model.safetensors file size)
- WandB link (clickable, opens in browser) if run_id exists
- Created date (from directory mtime)

For a selected checkpoint:
- Step number
- File sizes (model weights, optimizer state, total)
- Whether training_state/ exists (can resume or not)

### Detail panel — Config subtab

- Render `train_config.json` as a formatted, collapsible key-value tree
- Grouped by section: dataset, policy, optimizer, scheduler, wandb, eval
- Read-only — this is the config that was used, not editable

### Detail panel — Checkpoints subtab

- Table of all checkpoints for this training run
- Columns: step, model size, has training state, date
- Actions per checkpoint: "Use in Run tab", "Resume training", "Open folder"
- Highlight the `last` symlink target

### Detail panel — Metrics subtab (stretch goal)

Two approaches, in order of feasibility:

**Option A: Embed WandB (simplest)**
- If wandb run_id exists in train_config.json, embed the WandB run page in an iframe
- Similar to how we embed Rerun viewer
- Requires user to be logged into wandb in their browser
- Fallback: clickable link to open in browser

**Option B: Parse wandb local logs (offline)**
- WandB stores local run data in `wandb/` directory inside the output dir
- Parse the local event files to extract loss curves
- Render with a simple JS charting library (Chart.js or similar)
- More work but works offline and without WandB account

**Option C: Read training logs from stdout (minimal)**
- Training logs metrics to stdout at `log_freq` intervals
- For running jobs, we already capture stdout (like the Run tab)
- Parse log lines for loss values, plot them
- Simplest for live training, but no historical data for completed runs

### Actions

**"Use in Run tab"**
- Sets the selected checkpoint as the policy for Run tab's policy workflow
- Passes `pretrained_model/` path to the Run tab
- This is the main integration point between Model and Run tabs

**"Train" (new training)**
- Form with fields:
  - Dataset: selector from opened datasets (same as Run tab) or type repo_id
  - Policy type: dropdown from registry
  - Base checkpoint: optional, select from model tree or type HF repo_id
  - Output directory: auto-generated default, editable
  - Key hyperparams: batch_size, steps, lr, save_freq
  - Advanced (collapsible): gradient_checkpointing, dtype, peft config, wandb enable
  - Device: cuda / cpu dropdown
- Launches `lerobot-train` as subprocess (same pattern as Run tab)
- Streams stdout to a terminal panel
- Training appears as an active job in the model tree

**"Resume"**
- One-click: constructs `--config_path=.../train_config.json --resume=true`
- Optionally allow overriding `--steps` (to extend training)
- Same subprocess management as new training

**"Open folder"**
- Opens the checkpoint directory in the system file manager (same as Data tab's context menu)

---

## Backend API

### `src/lerobot/gui/api/models.py`

```
GET  /api/models/sources          — list source folders
POST /api/models/sources          — add source folder
DELETE /api/models/sources         — remove source folder
GET  /api/models/list             — scan sources, return model tree
GET  /api/models/{path}/config    — return train_config.json contents
GET  /api/models/{path}/info      — return computed metadata (sizes, step, status)
POST /api/models/train            — launch training subprocess
POST /api/models/train/stop       — stop training subprocess
GET  /api/models/train/status     — training job status
GET  /api/models/train/output     — SSE stream of training stdout
```

The training subprocess management reuses the same pattern as `run.py` (one active subprocess, SSE output streaming). Initially only one training job at a time (same as Run tab). Later: multiple concurrent jobs.

### Model scanning

Scan each source folder for training runs:
1. Look for `checkpoints/` subdirectories
2. For each checkpoint, read `pretrained_model/config.json` and `pretrained_model/train_config.json`
3. Extract: policy type, dataset, steps, wandb info
4. Check for `training_state/` to determine if resumable
5. Check `training_step.json` for current step

For HF cache: different scan pattern — look for `config.json` + `model.safetensors` pairs.

Cache scan results in memory, re-scan on request or after training completes.

---

## Frontend

### `src/lerobot/gui/static/model.js`

- Source folder management (add/remove, persisted via API)
- Model tree rendering (collapsible, with checkpoint sub-items)
- Detail panel with subtabs (Overview, Config, Checkpoints, Metrics)
- Training form (similar to Run tab's record form)
- Training output terminal (reuse terminal styling from Run tab)

### Shared patterns with existing tabs

- Source folder tree: same UI pattern as Data tab's source browser
- Subprocess management: same SSE + terminal pattern as Run tab
- "Open folder" context menu: same as Data tab

---

## Implementation Order

### Phase 1: Browse & Inspect
1. Backend: model source scanning, list/info/config endpoints
2. Frontend: source tree, model list, detail panel (Overview + Config + Checkpoints)
3. Source folder persistence

### Phase 2: Training
4. Backend: train subprocess launch/stop/status/output endpoints
5. Frontend: training form, terminal output, live status in model tree
6. Resume training support

### Phase 3: Run Tab Integration
7. "Use in Run tab" action — passes checkpoint path to Run tab policy workflow
8. Run tab policy workflow reads selected model and launches `lerobot-eval` or policy-based teleoperate

### Phase 4: Metrics & Polish
9. WandB iframe embed (or local log parsing)
10. Training curve visualization
11. Model comparison (side-by-side config diff, metric comparison)
12. HF Hub: download models, push trained models

---

## Open Questions

- Should training and Run tab share the same subprocess slot, or can they run independently? (Training is GPU-bound, teleop/record is robot-bound — they could coexist)
- How to handle multi-GPU training? (Low priority — most lerobot users train on single GPU)
- Should we support cloud training launch (Nebius etc.) from the GUI? (Mentioned in workbench plan but low priority)
