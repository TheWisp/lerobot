# GUI TODO

## Run Tab

- [Done] ~~Form state (FPS, profiles, episode count, etc.) lost when switching workflow tabs (e.g. teleop → policy → teleop)~~
- [High] Policy workflow task field has no auto-fill — user must manually type every time. Task info isn't stored in train_config.json and can't be reliably derived from run/dataset names
- [Mid] Policy workflow "Record evaluation" only supports typing a new dataset name — should support choosing an existing dataset via dropdown. First check whether eval can bypass the dataset entirely (users often don't care about eval recordings). Complicated by the fact that new eval datasets aren't opened in the Data tab, and datasets can't be opened during recording, so a live dropdown may not be feasible without solving that limitation first
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

## Workflow

To work on this TODO autonomously:
1. Read the TODO.md
2. Find the next incomplete task and implement it
3. Commit your changes
4. Update TODO with what you did
ONLY DO ONE TASK AT A TIME.
