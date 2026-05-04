# Feature Editing Design

Per-frame feature view + edit in the Data tab. A schema-driven way to display and edit any per-frame feature (`reward`, `success`, `subtask`, …) without leaving the GUI. V1 is one primitive — drag-select a range, edit values in the Inspector — built so feature-specific tooling (curve editors, segment-brushes) layers on top later.

![Feature editing layout sketch](feature_editing_layout.svg)

---

## Context

Motivated by RECAP-style labeling ([π\*0.6](https://www.pi.website/blog/pistar06)): per-frame `reward`, dense per-frame `subtask` language, per-episode `success` — relabeled iteratively. The GUI has no view/edit for these today.

### LeRobot format primer

- **Schema is per-dataset, values are per-frame.** `info.json` declares one `features` dict; every frame in every episode shares it. Adding or removing a feature is a structural change: the data parquet shards (`data/chunk-*/file-*.parquet`) get rewritten with the new column set, and per-feature `stats/<feature>/*` columns in `meta/episodes/*.parquet` are added or dropped. Videos are untouched unless the feature being added/removed is itself `image` or `video`. Per-frame value edits never touch any of this — they overwrite cells in the existing data shards.
- **Adding a feature, properly.** Two paths: declare it in the `features` dict at `LeRobotDataset.create(...)` time, or call [`modify_features(add_features=...)`](../../datasets/dataset_tools.py) on an existing dataset. There's no in-place "just edit `info.json` and append" path — the shards must agree with the schema, and stats need to be computed.
- **`task` is fully first-class.** Auto-deduped at write ([dataset_metadata.py:373](../../datasets/dataset_metadata.py#L373)) — strings in `meta/tasks.parquet`, frames carry `task_index`.
- **`subtask` is officially supported on the read side.** Same dedup pattern (`meta/subtasks.parquet` + per-frame `subtask_index`, decoded by `dataset[i]`); see the [official subtask guide](https://huggingface.co/docs/lerobot/dataset_subtask). What's missing is an `add_frame(subtask=...)` writer; today subtasks are written via the [lerobot-annotate Space](https://huggingface.co/spaces/lerobot/annotate). V1 treats `subtask_index` as a normal per-frame `int64` feature and updates `meta/subtasks.parquet` on Save when new strings appear.
- **Per-frame features are extensible; per-episode features are not.** The per-frame `features` dict can hold any user-defined column. But `meta/episodes/*.parquet` has a hardcoded schema (episode_index, length, tasks list, file pointers, stats) — there's no `episode_features` slot in `info.json`. So per-episode logical fields like `success` live as per-frame `bool[1]` columns broadcast across the episode.

---

## What's preserved from the existing GUI

Unchanged: the global tab bar, the dataset-tree sidebar, the camera grid, the existing `controls-bar` (Play / speed dropdown / frame info / time info / Reset-trim / status), the timeline scrubber + playhead, the trim handles, and the bottom edits-bar (Discard / Save). This design slots **per-feature timeline rows** under the existing scrubber, adds the **right-hand Inspector**, and reshapes the edits-bar from flat chips into a grouped/expandable list with a "Show pending edits" toggle.

---

## Editing model

**Selection is a vertical slice — a frame range, not a feature.** Drag on any row produces `{ episodeIndex, frameFrom, frameTo }`; the band extends through all feature rows. Inspector then shows per-feature edit cards scoped to that range.

**Origin-row focus**: when the user drags on a specific row, the Inspector auto-scrolls that feature's card into view and gives it the focus ring. With ~10–15 cards in the column, this skips the hunt and lets the user start typing immediately.

In V1 the only on-row primitive is **drag to select a range**.

### Selection semantics

- **Half-open** `[from, to)` — matches LeRobot's `dataset_from_index`/`dataset_to_index` (where `to - from == length`). Single-frame = `[N, N+1)`. Empty `[N, N)` = no selection.
- **No multi-range selection.** Holding shift to select non-contiguous frame ranges (e.g. 120–129 ∪ 200–211) isn't supported in V1.
- **Boundary frames are selectable.** Half-open trim `[trim_from, trim_to)` includes `trim_from`, so the first selectable frame is `trim_from` and the last is `trim_to - 1`. Hit-testing must keep feature-row click zones full-width (including the columns under the trim handles), since trim handles live on the time axis row, not on feature rows — there's no overlap.
- **Trim ≠ selection.** Trim adjusts only the outer envelope `[trim_from, trim_to)` (V1 doesn't support internal cuts — that would force episode splits + metadata/hash rewrites). Trim handles live on the time axis; gestures don't conflict with feature-row drag.

### Click semantics — one rule, with an escape hatch

Clicking anywhere inside the trim envelope (on the time axis or any feature row) seeks the playhead AND produces a selection. Single rule for the entire editable area; no special "this row is for playback only" carve-out.

- **Click inside the trim envelope at frame N** → playhead → N, selection = `[N, N+1)`.
- **Drag inside the trim envelope from A to B** → while dragging, the playhead tracks the drag-end so the camera shows the frame at the current "to" edge. On release, selection = `[min(A,B), max(A,B) + 1)`.
- **Drag the existing playhead thumb** → scrub only; selection is unchanged. This is the explicit "I just want to navigate without re-selecting" gesture. Also covers the existing playback workflow (`Play` / frame-step buttons in the controls bar) which doesn't touch selection either.
- **Click outside the trim envelope** (the dimmed region) → no-op. The dim region behaves like a wall — you can't put a selection there because those frames won't ship after Save.
- **Esc** → clear selection.

The accepted cost: a misclick during inspection overwrites the selection ROI. Mitigations are the playhead-thumb drag, the existing Play / frame-step controls, and the fact that any staged edits are still recoverable per-chip in the edits bar — only the ROI is lost, not committed work.

### Boundary visualization — frame cells, not ticks

Each frame N occupies cell `[N, N+1)` on the timeline. Selection handles sit on cell _boundaries_, not on cells — same as DaVinci Resolve, Premiere, Audacity, Aegisub.

```
   │ 119 │ 120 │ 121 │ 122 │ 123 │ 124 │ 125 │ 126 │ 127 │ 128 │ 129 │ 130 │
         ╞═══════════════════════════════════════════════════════════╡
         ↑                                                           ↑
         from = 120                                                  to = 130
```

Coverage = cells `from`..`to-1`, exactly `to - from` cells. Human label: "frames N…M (K frames)" with M = `to-1`. Click snaps to cell boundaries; drag uses `from = min(a,b)`, `to = max(a,b) + 1`.

---

## V1 edit affordances per type

| Feature                                           | Row display                               | Inspector widget                                                                                                  |
| ------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `bool[1]`                                         | on/off band                               | **checkbox** (project convention — not toggle)                                                                    |
| numeric scalar (e.g. `float32[1]`, `int64[1]`)    | line                                      | **slider + number input** (slider range from per-feature stats min/max; number input wins for out-of-range)       |
| `string`                                          | colored stripe + text labels              | single-line input or textarea                                                                                     |
| `subtask_index` (with `meta/subtasks.parquet`)    | colored stripe (decoded `subtask` string) | dropdown over existing subtask strings + "+ new subtask…" — writes the new index, appends to lookup table on Save |
| numeric vector, small (e.g. `float32[N]`, N ≤ ~8) | mini multi-line                           | row of N inputs                                                                                                   |
| numeric vector, large (e.g. `float32[N]`, N > ~8) | line plot                                 | "Edit as JSON…" textarea                                                                                          |
| `action`, `observation.*` (any dtype)             | hidden by default in V1; pin to show      | **read-only** — recorded data, multi-dim viz design deferred (see Follow-ups)                                     |
| `image`, `video`, 2D+                             | not on timeline                           | **read-only**                                                                                                     |
| `DEFAULT_FEATURES`                                | hidden by default                         | **read-only always**                                                                                              |

---

## Inspector behavior

Schema-driven by `(dtype, shape)`. Always visible when a dataset is open.

- **Empty state** (no selection): compact dataset summary.
- **In-selection state**: scrollable feature-card column scoped to `[from, to)`. Origin-row card auto-focused. Per card: name + dtype, summary of values in the range (min/max for scalars, unique values for strings, "uniform/mixed" for bools), edit widget or read-only badge.
- **Edits stage automatically** as the user changes a card. No per-card Apply. The card shows `● pending`; a chip appears in the edits bar; bottom **Save** is the single confirmation, **Discard** drops everything.

### Schema discoverability

Schema is read-only and surfaces in two always-on places once a dataset is open:

- **Inline on each timeline row** — name + dtype label at the row's left edge gives an at-a-glance overview.
- **Inspector cards on selection** — full per-feature detail (dtype, shape, range stats, edit widget).

There's no separate schema modal. The dataset row in the sidebar is not itself a selectable / inspectable target today, so a "right-click → Info…" gesture would target a thing with no other selection semantics. If "peek schema before opening" ever becomes a real need (e.g. browsing 50 datasets in a folder), the right answer is search/filter, not a schema viewer.

---

## Edit application pipeline

Staging extends `PendingEdit` ([state.py](../state.py)) with `edit_type: "feature_set"`, params `{ feature, episode_index, frame_from, frame_to, value }`. Persists to `.lerobot_gui_edits.json`. Discard / Save / per-chip removal work for free.

**Applying on Save needs a new LeRobot API** — value editing is a missing function at the dataset level. The available `dataset_tools` operations are all schema-level (`add_features`, `remove_feature`, `rename_feature`, `swap_features`) or episode-level (`delete_episodes`, `split_dataset`, `merge_*`); none overwrite values in existing columns. `modify_features` re-encodes the whole dataset — right tool for schema changes, wrong for value edits (slow, touches videos, forks the dataset on every Save).

This work adds `set_feature_values()` to [`dataset_tools.py`](../../datasets/dataset_tools.py), as a peer to `modify_features`:

```python
def set_feature_values(
    dataset: LeRobotDataset,
    edits: list[dict],          # [{feature, from_index, to_index, value}, ...]
    *,
    in_place: bool = True,      # rewrite shards in place; if False, fork to output_dir
    output_dir: Path | None = None,
) -> None:
    """Overwrite values of EXISTING features for specific frame ranges.
    Schema is unchanged. Stats for affected episodes are recomputed.
    Calls dataset.finalize() before returning."""
```

Behavior:

- Group edits by the parquet shard they fall into (using global `index`). For each shard, read → overwrite the listed cells → atomically replace via `tmp` + `os.replace`.
- For each touched episode, recompute `meta/episodes/*.parquet` `stats/<feature>/*` columns via the existing `_recompute_episode_stats_from_data`.
- For features mapping to a sidecar lookup (`subtask_index` → `meta/subtasks.parquet`): when new strings appear, append to the lookup table and resolve to the new indices.
- Always call `dataset.finalize()` before returning ([v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3#always-call-finalize-before-pushing)) — without it, parquet files end up corrupt.

This belongs at the LeRobot data level, not buried in the GUI: any tool (CLI, notebook, GUI) that needs to relabel a dataset benefits from one canonical entry point. The GUI's `_apply_feature_set_edits()` in [edits.py](../api/edits.py) just translates staged `PendingEdit`s into a `set_feature_values()` call. Reuses existing helpers (`_write_parquet`, `_recompute_episode_stats_from_data`, `_copy_episodes_metadata_and_stats`) and path templates (`DATA_DIR`, `DEFAULT_DATA_PATH` from [utils.py](../../datasets/utils.py)).

### Safety rails

- Validate every staged edit against schema (dtype + shape) via [`validate_feature_dtype_and_shape`](../../datasets/feature_utils.py).
- Block edits on read-only features (`DEFAULT_FEATURES`, `action`, `observation.*`, image/video).
- Confirmation dialog if a single Save touches > 10 000 frames.
- **No edit-edit dependencies in V1.** All editable V1 features are independent primitives. When derived features arrive (e.g. RECAP `reward = f(success, step_count)`), they'll be read-only with a "Recompute from inputs" button — dependencies declared at the feature-definition level, never between staged edits.

---

## Default visibility & scrolling

Heuristic without a format change:

- **Hidden by default**: `image`/`video`, `DEFAULT_FEATURES`, `task_index` (replaced by `task` string), and `action` / `observation.*` (multi-dim numeric vectors — read-only in V1, viz design deferred).
- **Visible by default**: everything else — `reward`, `success`, `subtask`, scalar custom features, short-string custom features.
- **Pin / hide** any feature individually; preference persisted in localStorage.

---

## V1 design decisions

| Concern              | Decision                                                                                                                                                                                                 | Reason                                                                                                                                                       |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Selection model      | Vertical slice through all rows                                                                                                                                                                          | Frames are tuples; one mental model; faster multi-feature labeling                                                                                           |
| Trim semantics       | Outer-bounds-only; handles on time axis                                                                                                                                                                  | Internal cuts force episode splits + metadata rewrites (out of scope)                                                                                        |
| Click semantics      | One rule: click inside the trim envelope seeks **and** selects `[N, N+1)`; drag inside the envelope seeks (tracks drag-end) and produces a range. Playhead-thumb drag scrubs without changing selection. | Simpler than splitting "playback territory" vs "editing territory"; one gesture, predictable; playhead-thumb is the escape hatch for navigate-without-select |
| Undo / redo          | Punted — per-chip removal + Discard / Save                                                                                                                                                               | Edits bar covers most "oops" cases; full undo across Saves needs pre-edit value capture or git-like history                                                  |
| Boundary semantics   | Half-open `[from, to)`, cells (not ticks)                                                                                                                                                                | Matches LeRobot's `dataset_from_index`/`dataset_to_index` and every pro video editor                                                                         |
| Edit application     | Direct staging on card change; bottom Save commits                                                                                                                                                       | Matches Figma/Photoshop; bottom Save is the single confirmation                                                                                              |
| Layout customization | Two mouse-draggable resize boundaries (vertical, horizontal); positions persist in localStorage                                                                                                          | Reuses existing `sidebar-resize-handle` pattern; everything else deferred                                                                                    |

---

## Implementation phases

### Phase A — Read-only foundation

- **A1.** Schema in `DatasetInfo`. Backend: extend `DatasetInfo` ([api/datasets.py](../api/datasets.py)) so the existing dataset-open response includes the full features dict (dtype, shape, names) — currently only feature names are returned. No new endpoint, no modal.
- **A2.** Per-frame feature values. Backend: `GET /api/datasets/{id}/episodes/{ep}/frames/{frame}/features` (skip `image`/`video`); optionally piggyback on the playback WS `seek` ([playback.py](../api/playback.py)). Frontend: schema-driven renderer registry by `(dtype, ndim)`.
- **A3.** Episode feature-series + timeline rows. Backend: `GET /api/datasets/{id}/episodes/{ep}/feature-series?features=…` → `{name: [...], ...}`; cache. Frontend: stacked rows; trim envelope tinted through all rows; outside-trim dimmed; pin/unpin per row.

### Phase B — Editing

- **B1.** Vertical-slice selection + click-to-seek-and-select. Drag inside the trim envelope → `{ episodeIndex, frameFrom, frameTo }`; vertical band across all feature rows; origin row remembered for Inspector focus; Esc clears. Click inside trim → playhead → N, selection = `[N, N+1)`. Drag inside trim → playhead tracks drag-end. Existing playhead-thumb drag remains scrub-only (selection unchanged) — the explicit navigate-without-select gesture. Click outside trim → no-op.
- **B2.** Inspector in-selection state. Scrollable card column; widgets per type (checkbox / slider+number / dropdown / text / row-of-N inputs / "Edit as JSON…" / read-only badge); `subtask_index` dropdown sourced from `meta/subtasks.parquet` with "+ new subtask…"; auto-staging with `● pending` indicator on each modified card.
- **B3.** Stage `feature_set` edits. Extend `PendingEdit` ([state.py](../state.py)). New `POST /api/edits/feature-set`; existing `DELETE /api/edits/{idx}` works. Group-by-`(feature, episode)` rendering with expand / collapse.
- **B4.** "Show pending edits" toggle. Off = current data; On = post-Save data overlaid on the timeline rows.
- **B5.** Validation + safety rails. Reuse [`validate_feature_dtype_and_shape`](../../datasets/feature_utils.py) for dtype/shape checks. Block edits on read-only features (`DEFAULT_FEATURES`, `action`, `observation.*`, image/video). Confirmation dialog for Saves touching > 10 000 frames.
- **B6.** Apply pipeline. Add `set_feature_values()` in [dataset_tools.py](../../datasets/dataset_tools.py) (see "Edit application pipeline" above). The GUI's `_apply_feature_set_edits()` in [edits.py](../api/edits.py) translates staged `PendingEdit`s into a single `set_feature_values()` call, then reloads the dataset to refresh `meta.episodes` stats.

### Phase C — Layout

- **C1.** Two mouse-draggable resize boundaries (vertical + horizontal). Reuse `sidebar-resize-handle`. Persist in localStorage.

---

## V1 scope

**In:** schema in row labels + Inspector cards · always-visible Inspector · 1D feature rows with scroll + pin/unpin · trim handles on time axis with envelope through all rows · vertical-slice selection · click-inside-trim seeks + selects `[N, N+1)`; drag = range; playhead-thumb drag scrubs without re-selecting · auto-staging Inspector cards · range value edits for `bool` / numeric scalar / numeric vector / strings · `subtask_index` dropdown sourced from `meta/subtasks.parquet` (with "+ new subtask…") · grouped expandable edits bar · "Show pending edits" toggle · two resize handles · `action` / `observation.*` / image / video / `DEFAULT_FEATURES` are read-only · **new `set_feature_values()` LeRobot API** for in-place value edits (peer to `modify_features`).

**Out (Follow-ups):** in-place segment manipulation · row context menus · multi-range selection · loop-on-selection · undo/redo across Saves · schema mutations (add/remove/rename) · derived/computed features · curve editor · **multi-dim numeric vector visualization** for `action` / `observation.*` (stacked-rows vs overlaid-curves vs collapsed-expandable — defer) · URDF / 3D trajectory views · stats viewer (P1/P2) · first-class per-episode features (format extension) · `add_frame(subtask=...)` writer · episode-list shortcuts (success column, inline `task` edit).

---

## Critical files

**LeRobot core**:

- [src/lerobot/datasets/dataset_tools.py](../../datasets/dataset_tools.py) — new `set_feature_values()` (peer to `modify_features`); reuses existing `_write_parquet`, `_recompute_episode_stats_from_data`, `_copy_episodes_metadata_and_stats`.
- [src/lerobot/datasets/utils.py](../../datasets/utils.py) — `DATA_DIR`, `DEFAULT_DATA_PATH` path templates already exist.

**GUI backend** ([api/datasets.py](../api/datasets.py), [api/playback.py](../api/playback.py), [api/edits.py](../api/edits.py), [state.py](../state.py), [api/models.py](../api/models.py)) — extend `DatasetInfo` with full features dict; new feature-values + feature-series endpoints; new `feature_set` `PendingEdit` type; `_apply_feature_set_edits()` calls `set_feature_values()`.

**GUI frontend** ([static/index.html](../static/index.html), [static/app.js](../static/app.js), [static/style.css](../static/style.css)) — Inspector, feature rows, vertical-slice selection, click-to-seek wiring, resize handles, grouped edits bar.

**Reused helpers**: [feature_utils.py](../../datasets/feature_utils.py) `validate_feature_dtype_and_shape` for staged-edit validation.

**Tests**:

- new `tests/datasets/test_set_feature_values.py` — value rewrites, stats recomputation, sidecar lookup updates (`subtasks.parquet`), in-place vs forked output, `finalize()` is called.
- extend [tests/gui/test_state.py](../../../tests/gui/test_state.py) for `feature_set` `PendingEdit` serialization.
- new `tests/gui/test_feature_endpoints.py` — schema, per-frame features, feature-series.
- new `tests/gui/test_feature_edits.py` — staging, validation, safety rails, end-to-end Save flow.

---

## Verification

`uv run pytest tests/gui -svv` after each phase. Schema validation rejects type-mismatched edits. Range semantics: staging `[120, 130)` writes 10 frames; UI shows "frames 120…129". Manual: open a dataset with `reward` → drag-select on a row → edit cards (slider, text, checkbox) → confirm chips appear and timeline preview toggles correctly → Save → reload → values persist.
