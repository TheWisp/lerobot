# The overlays panel on the dataset it switched to

What the Data tab's Overlays panel shows **on the incoming dataset**, after a
dataset that already carries saved masks seeded its object rows from the stored
vocabulary. Captured 2026-09-15 on a headless Chromium at 1280×900, cropped to
the panel, from `tests/gui/test_overlays_panel_playwright.py`'s own server —
no dataset, GPU or overlay worker, and nothing of the operator's was read.

The two runs differ only in `static/overlays.js`: `456ebf50d` against the same
tree with this branch's fix.

| State                                          | Row 0 reads | The switch |
| ---------------------------------------------- | ----------- | ---------- |
| [Before](1-before-stuck-on-the-old-prompt.png) | `ring`      | threw      |
| [After](2-after-fresh-for-the-new-dataset.png) | empty       | completed  |

**Before.** The panel is on a dataset it has never seen and still shows `ring`,
the prompt belonging to the dataset just left. `snapshotConfig` threw on the way
in, so nothing restored, nothing re-rendered and the camera list was never
rebuilt. The panel looks entirely normal, which is what makes it expensive: a
running segmenter goes on addressing the previous dataset's cameras and the
screen gives no sign.

**After.** The same switch lands on the fresh, inert configuration a never-seen
dataset is supposed to get — one empty row with its placeholder and the
`name an object` hint beneath.

The captured console line from the first run:

```
TypeError: Cannot read properties of undefined (reading 'key')
    at carry (/static/overlays.js:1410)
    at snapshotConfig (/static/overlays.js:1415)
    at Panel.refreshCameras (/static/overlays.js:1453)
```

The earlier attempt at this pair screenshotted the panel back on the _seeded_
dataset and produced the same image twice — there, "stuck on the old rows" and
"correctly restored the old rows" are the same picture. The incoming dataset is
where the two states differ.
