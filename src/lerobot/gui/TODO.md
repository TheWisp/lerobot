# GUI TODO

## Run Tab

- [Done] ~~Form state (FPS, profiles, episode count, etc.) lost when switching workflow tabs (e.g. teleop → policy → teleop)~~
- [High] Policy workflow task field has no auto-fill — user must manually type every time. Task info isn't stored in train_config.json and can't be reliably derived from run/dataset names
- [Done] ~~Policy workflow "Record evaluation" only supports typing a new dataset name — now has dropdown to choose existing opened datasets (with resume) or create new~~
- [Low] Text output freezes after a while — teleoperate uses ANSI cursor-up in piped stdout
- [Low] Rerun web viewer has ~200ms visual lag (Rerun 0.26 limitation)
- [Low] Replay FPS setting doesn't seem to affect playback speed — remove if not useful (needs investigation)
- [Done] ~~Run tab robot/teleop profile dropdowns don't update when profiles are added or deleted in the Robot tab~~

## Architecture

- [Mid] Cross-tab data synchronization is fragile and not scalable — current approach (refreshRunProfileSelects, refreshRunDatasetSelects, refreshExpandedSources, refreshOpenedDatasets) hardcodes specific refresh calls between tabs. Any data source update should notify all dependent UI components efficiently, e.g. via a pub/sub event bus, rather than point-to-point wiring

## Robot Tab

- [Low] UX consistency pass: ensure consistent button coloring/hierarchy across views (e.g. "Record rest position" is blue/accent while the primary "Move to rest" is grey, similar inconsistencies may exist elsewhere)
- [Low] ~1s latency when first opening the Robot tab while loading profiles

## Data Tab

- [Done] ~~Opened datasets don't refresh after recording new episodes — need to close and reopen to see changes~~
- [Done] ~~Run tab replay/record dataset options don't update when datasets are opened or closed in the Data tab~~
- [Done] ~~Run tab replay episode list doesn't update when episodes are deleted in the Data tab~~
- [Done] ~~After recording a new dataset, the record UI still shows "new dataset" form instead of switching to the now-existing dataset~~
- [Done] ~~After deleting an episode, selection moves to neighbour but playback view still shows the deleted episode~~
- [Done] ~~After trimming an episode and saving, playback still shows the old duration~~
- [Low] Cannot open a dataset in the Data tab while it's being recorded (at least for new datasets — need to verify behavior for existing ones)

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices so only one video filtering function is needed.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` (`OurStreamingVideoEncoder`, per-camera, unbounded queue, reservoir stats) vs upstream's `video_utils.py` (`StreamingVideoEncoder`, multi-camera manager, bounded queue, HW encoder support). Currently both coexist in `lerobot_dataset.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating to upstream's and removing `video_encoder.py`.

## Hardware

- [Low] Use stable Linux device paths (`/dev/serial/by-id/`, `/dev/v4l/by-id/`, `/dev/v4l/by-path/`) instead of volatile `/dev/ttyACM*` and `/dev/video*` in robot/teleop/camera profiles. Benefits: configs survive reboots and USB re-plugging. The GUI Robot tab could auto-detect `by-id` paths and offer them in port selection. For identical cameras (no unique serial), fall back to `by-path` (stable per USB port).

## Python 3.12+ Compatibility

- [Mid] Remove Python < 3.12 workarounds once we drop 3.10/3.11 support. Upstream lerobot now requires 3.12+. Our fork pins 3.10 compatibility via these changes:
  - `datasets/utils.py`: `class Backtrackable(Generic[T])` → native `class Backtrackable[T]:`
  - `motors/motors_bus.py`: `NameOrID = Union[str, int]` → native `type NameOrID = str | int`
  - `utils/io_utils.py`: module-level `T = TypeVar(...)` + regular function → native `def foo[T: Bound](...)`
  - `processor/pipeline.py`: `class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin)` → native `class DataProcessorPipeline[TInput, TOutput](HubMixin):`
  - `policies/pretrained.py`: conditional `Unpack` import from `typing_extensions` → direct `from typing import Unpack`
  - `policies/{pi0,pi0_fast,pi05,smolvla}/modeling_*.py` + `policies/factory.py`: `from typing_extensions import Unpack` → `from typing import Unpack`

## Workflow

To work on this TODO autonomously:
1. Read the TODO.md
2. Find the next incomplete task and implement it
3. Commit your changes
4. Update TODO with what you did
ONLY DO ONE TASK AT A TIME.
