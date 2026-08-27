# Flag feature — UI states

Captured from a real GUI (uvicorn + Chromium) against a synthetic three-episode
dataset. Frames 12–19 of episode 0 carry `fumble` on disk, so a selection
spanning 8–20 is genuinely mixed — that is the state worth photographing, and
the one a two-state checkbox cannot represent.

|                           | state                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `01-dialog-number.png`    | Add-column dialog, Number kind — dtype, shape, fill value                                                                |
| `02-dialog-flags.png`     | Add-column dialog, Flags kind — flag list only; the storage fields are gone rather than ignored                          |
| `03-none-set.png`         | selection where no frame carries a flag                                                                                  |
| `04-all-set.png`          | selection where every frame carries `fumble`                                                                             |
| `05-mixed.png`            | selection spanning both — `fumble` indeterminate, `fumble 8/12`                                                          |
| `06-after-tick.png`       | `blurry` ticked — `blurry 12/12`, `fumble` still indeterminate beside it                                                 |
| `07-after-untick.png`     | unticked again — back to exactly the pre-tick state, one edit collapsed rather than two stacked                          |
| `08-bool-parity.png`      | the flags card above a `bool[1]` card: same checkbox rendering                                                           |
| `09-row-lanes.png`        | the timeline — one lane per flag, each with a rail whether or not it fires, beside the ordinary rows it has to sit among |
| `10-rename-control.png`   | the rename affordance, revealed by hovering the flag's row and sitting outside its `<label>`                             |
| `11-rename-editor.png`    | rename opens in place, seeded with the current name, with the other names still on screen beside it                      |
| `12-rename-duplicate.png` | the server's refusal, beside the input, with the editor still open and holding what was typed                            |
| `13-after-rename.png`     | corrected to `slipped` — still `8/12` and still indeterminate, so the bit did not move                                   |
| `14-add-blank-row.png`    | adding is a blank row where the flag will be, not a dialog over the panel                                                |
| `15-add-duplicate.png`    | the same refusal on the add path                                                                                         |
| `16-rows-after-edits.png` | four flags after the edits — the row grows by a lane, not by a band                                                      |

The dialog shots are of the two kinds that differ structurally; Text differs
from Number only in which of the same rows are shown, which
`test_add_column_dialog_playwright.py` asserts by computed style rather than by
picture.

The rename shots are of the same card the tick shots use, so 05, 11 and 13 can
be compared directly: the count beside the flag reads `8/12` in all three, which
is what "renaming moves no bit" means from the outside.

Every state here is also a test in
`test_flag_vocabulary_editor_playwright.py`, which drives the same flows and
fails on the sight of a native dialog. These are the pictures; that is the
ratchet.
