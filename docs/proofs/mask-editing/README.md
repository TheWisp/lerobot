# Mask editing — captured states

Cropped to the component, one state each, from a running GUI on the
`screenshots/mask_segments_demo` dataset: 2 episodes × 90 frames, 2 cameras, cut
from real rig footage (a green ring and coloured blocks on a dark tray). Its
masks are **real SAM3 detections**, with a disabled run and an absent run cut
into them on purpose, so every lane carries all three states and every gesture
has a target.

|                                |                                                                                                                                                                                                                                                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-three-states.png`           | a mask row at rest. **Filled** is detected, **hollow** is disabled, the faint rail is absent. Not a dimmer fill or a hatch: a lane is a few pixels tall and neither survives at that size.                                                                                                                                  |
| `2-selection.png`              | a dragged range. A click seeks and leaves a one-frame selection, which is not enough to toggle.                                                                                                                                                                                                                             |
| `3-delete-affordance.png`      | reaching for a segment's trailing edge reveals its `×`. One per segment, not one per row, and only near the edge — a button over the middle of the bar sits exactly where a click means _toggle_.                                                                                                                           |
| `4-staged-toggle.png`          | after clicking a segment: it flips to the other state immediately, before any save.                                                                                                                                                                                                                                         |
| `5-pending-overlay.png`        | the same edit under **show pending edits** — an amber band on that label's lane only, since a row holds several labels.                                                                                                                                                                                                     |
| `6-inspector-dataset-tier.png` | the Inspector's **Dataset** tier, above Episode. Treatments keyed by label name, "applies to every camera", with the background as a row of its own.                                                                                                                                                                        |
| `7-overlays-no-treatments.png` | the Overlays panel with no treatment controls and no saved-label list. A treatment is dataset-wide; this panel is the live query and has no scope.                                                                                                                                                                          |
| `8-fill-gaps-dialog.png`       | the whole-dataset fill. The label set is **picked here**, not inherited from the vocabulary: that is the accumulated union of everything ever segmented anywhere, so a `blue towel` from one episode would otherwise send the job hunting across all of them. The per-label episode count is what makes the choice obvious. |
| `9-preview-seeded.png`         | the segmenter's object rows, arriving pre-filled with the vocabulary the dataset already carries rather than empty.                                                                                                                                                                                                         |
| `10-apply-control.png`         | **Apply — write masks while playing.** A checkbox, not a button: it is a mode you are in while playing. The job writes the frames and the playhead is moved to the frame count it reports, so the playhead can only ever sit on a frame whose masks are stored.                                                             |

## walkthrough.mp4

[walkthrough.mp4](walkthrough.mp4) — all 23 checklist steps driven in order
against a running GUI, 99 seconds.

**23/23 pass.** Each step asserts a RESULT read back from storage, from the
pending queue, or from the DOM — not that the gesture was dispatched. From the
last run:

| step | what was asserted, and what it saw                                                                   |
| ---- | ---------------------------------------------------------------------------------------------------- |
| 15   | ticking Apply armed the mode and wrote nothing — storage unchanged, pending still 0                  |
| 16   | playing filled the run's edit as it went: 30 → 42 → 56 camera-frames staged, in **one** pending edit |
| 17   | pausing stopped it and kept what it had staged                                                       |
| 18   | Discard threw the run away whole and touched no stored row                                           |
| 19   | Save committed it; `tray` reached the vocabulary                                                     |
| 20   | replaying never replaced what was already stored                                                     |
| 21   | 15/15 frames of the pre-existing disabled run were still muted                                       |

Step 16 measures the run edit's _contents_, not the number of pending edits: a
run is one edit by design, so a count of edits can only ever report that nothing
happened.

## What the walkthrough found

Four defects, each fixed with a regression test that fails when the bug is put back:

| defect                                                                                                                        | fix                                                                                   | test                                     |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------- |
| a run that appended a label left it invisible — no lane, no Inspector row, no fill-dialog entry — for the whole session       | the apply poll refreshes the client's schema on any terminal status                   | `tests/gui/apply_completion.test.js`     |
| two object rows with the same name stored the label twice, in a vocabulary that is positional                                 | deduped at both writers, first occurrence winning                                     | `tests/datasets/test_mask_vocabulary.py` |
| `data-label` meant the label NAME on the delete button and the lane INDEX on the segment rect                                 | both carry the name; the rect gained `data-lane`                                      | `tests/gui/feature_editing.test.js`      |
| showing the pending-edits bar shifted every timeline row up by its height, so the second of two clicks landed on the next row | the bar is overlaid instead of taking flex space; the scroll area reserves the height | `tests/gui/test_edits_bar_layout.py`     |
