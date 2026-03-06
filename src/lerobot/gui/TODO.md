# GUI TODO

## Run Tab

- [Done] ~~Form state (FPS, profiles, episode count, etc.) lost when switching workflow tabs (e.g. teleop → policy → teleop)~~
- [High] Policy workflow task field has no auto-fill — user must manually type every time. Task info isn't stored in train_config.json and can't be reliably derived from run/dataset names
- [Low] Text output freezes after a while — teleoperate uses ANSI cursor-up in piped stdout
- [Low] Rerun web viewer has ~200ms visual lag (Rerun 0.26 limitation)
- [Low] Replay FPS setting doesn't seem to affect playback speed — remove if not useful (needs investigation)
- [Done] ~~Run tab robot/teleop profile dropdowns don't update when profiles are added or deleted in the Robot tab~~

## Robot Tab

- [Low] ~1s latency when first opening the Robot tab while loading profiles

## Data Tab

- [Mid] Opened datasets don't refresh after recording new episodes — need to close and reopen to see changes
- [Done] ~~Run tab replay/record dataset options don't update when datasets are opened or closed in the Data tab~~
- [Done] ~~Run tab replay episode list doesn't update when episodes are deleted in the Data tab~~
- [Mid] After recording a new dataset, the record UI still shows "new dataset" form instead of switching to the now-existing dataset
- [Mid] After deleting an episode, selection moves to neighbour but playback view still shows the deleted episode
- [Mid] After trimming an episode and saving, playback still shows the old duration
