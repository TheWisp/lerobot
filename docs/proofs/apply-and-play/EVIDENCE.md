<!-- Captured evidence for one change: what was observed, not what was designed.
     Design documents live beside the code they describe. -->

# Evidence: apply-and-play fills a camera that had no mask column

Both shots are the same feature rows of the same episode of the same dataset —
one that had never held a mask — cropped to the rows. The only thing between
them is one apply-and-play run over the `top` camera.

| file                                   | what it shows                                                                                                         |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `1-before-no-mask-track.png`           | the dataset's feature rows before the run. `observation.state` and `action` only — there is no `masks.top` row at all |
| `2-after-track-adopted-and-filled.png` | after the run and Save. `masks.top` exists, and its `ball` lane is filled solid across the episode                    |

Before this change the same run produced the first picture twice: it played the
whole episode, staged nothing, said "Apply complete", and returned 200 from
Save.

## What the run reported to disk

Read back with the dataset's own accessors after the server was stopped:

```
columns:  ['observation.images.top']
coverage: {'top': (12, 12), 'wrist': (0, 0)}
```

12 of 12 frames on the camera that was played and adopted; nothing on the camera
that was not selected, which is the complement — adoption reaches what the run
plays and nothing else.

## The whole run

`3-adopt-and-fill.gif` (and `3-adopt-and-fill.mp4` at full resolution) is one
recorded run over a dataset that had never held a mask, showing the entire
screen rather than a crop: the Overlays panel with SAM3 picked, the `ball`
object and the camera selection; the mask drawn on the `top` camera view as it
is produced; and the `masks.top` lane appearing in the feature rows and filling
across all 60 frames. Read back off disk afterwards as `top=60/60`.

One thing visible in it is a defect, not the feature: after the run adopts and
saves, the Overlays panel still reads "masks: feature not adopted yet". That
readout is stale and is not addressed here.

## Not captured

The consent prompt is a native browser dialog (`window.confirm`). It does not
appear in a page screenshot, and a hand-drawn imitation of it would not be
evidence, so there is no image of it here. Its text is asserted verbatim by
`tests/gui/test_apply_and_play_playwright.py`, which also covers declining it.

## How these were produced

The harness in `tests/gui/test_apply_and_play_playwright.py`, driven headless
against a throwaway dataset in a temporary directory, with
`LEROBOT_GUI_CONFIG_DIR` redirected so the run touches no real state. Only the
segmenter is faked; the drain, the write-rule filter, the staging endpoint and
the edits pipeline are the real ones.
