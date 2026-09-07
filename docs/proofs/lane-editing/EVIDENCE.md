# Lane editing — captured states

Cropped to the component, one state each, from a running GUI. Both datasets are
synthesised throwaways built for this capture — no real dataset was opened, and
nothing here touched stored user data.

- `gui-demo/flag_lanes` — 120 frames, `quality int64[1]` with flags
  `blurry` / `fumble` / `occluded`. `blurry` carries frames 0–19 and 90–109,
  `fumble` carries 40–79, `occluded` carries nothing, so one row holds a lane
  with two runs, a lane with one, and an empty lane.
- `gui-demo/mask_lanes` — 120 frames, `masks.top` with `ball` / `tray`. `ball`
  is detected over 0–39, disabled over 40–79 and absent over 80–119, so every
  mask state has a target.

## Flag lanes

|                              |                                                                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flag-1-selection.png`       | frames 10–70 dragged on the row. Nothing is hovered, so nothing is offered.                                                                                      |
| `flag-2-hover-set-run.png`   | pointing at the part of `blurry` that is set. The band is an outline over the frames going away, tagged `− blurry`, and it stops at frame 20 where the run does. |
| `flag-3-hover-clear-run.png` | pointing past that boundary, in the same selection. Filled band, `+ blurry`, frames 20–70 — the complementary span and the opposite direction.                   |
| `flag-4-empty-lane.png`      | `occluded` is carried nowhere, so its run is the whole episode and the band is the selection. The one case where the two coincide.                               |
| `flag-5-staged.png`          | after the click. The lane redraws from the merged view before any save.                                                                                          |

## Mask lanes

The same gesture and the same band on the tri-state row.

|                                    |                                                                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `mask-1-hover-detected.png`        | a detected run. It reaches training now, so the click withholds it: outline, `− ball`.                                                  |
| `mask-2-hover-disabled.png`        | a muted run. Filled band, `+ ball` — it is about to reach training again.                                                               |
| `mask-3-absent-offers-nothing.png` | an absent stretch, selected and hovered. **No band.** A mask that was never stored cannot be conjured by a click, so nothing offers to. |
| `mask-4-delete-above-band.png`     | reaching for the segment's trailing edge. The `×` draws above the band; the band is inert to the pointer.                               |
