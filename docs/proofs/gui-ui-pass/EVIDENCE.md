<!-- Captured evidence for one change: what was observed, not what was designed.
     Design documents live beside the code they describe. -->

# GUI UI pass

Captured on `feat/gui-ui-polish` against `thewisp/cylinder_ring_assembly`, in a
headed Chromium with classic (non-overlay) scrollbars, so the scrollbar shots
show what the operator sees. Before/after pairs come from one running server
with the served static assets swapped between `origin/main` and the branch, so
nothing but the assets differs between the two shots of a pair.

| File                                                       | What it shows                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| `1-transport-before.png` / `1-transport-after.png`         | Play against the speed picker, before and after the shared control height   |
| `2-filters-before.png` / `2-filters-after.png`             | The dataset search / sort / favourites row at body size, then at row size   |
| `3-tile-before.png` / `3-tile-after.png`                   | A camera tile's top edge: name as a header row, then as a chip in the frame |
| `4-sources-seam-before.png` / `4-sources-seam-after.png`   | The Sources/Opened boundary, before and after it became a draggable seam    |
| `5-overlays-seam-before.png` / `5-overlays-seam-after.png` | The Inspector/Overlays boundary, same                                       |
| `6-timeline-clamped.png`                                   | The camera/timeline seam dragged to its stop; the timeline is still whole   |
| `7-scrollbar-sources.png`, `7-scrollbar-timeline.png`      | The restyled scrollbar on the Sources list and on the timeline's rows       |
| `8-dialog-destructive.png`                                 | A destructive confirm: red affirmative, focus resting on Cancel             |
| `8-dialog-confirm.png`                                     | An ordinary confirm: focus resting on the affirmative                       |
| `8-dialog-prompt.png`                                      | A prompt, with the field focused and its default selected                   |
| `8-dialog-alert.png`                                       | An alert, single button                                                     |

`6-timeline-clamped.png` was taken after dragging the seam 2000 px down; the
grid stops where the timeline still fits, and the measured overhang past the
bottom of the pane that clips it was 0 px.
