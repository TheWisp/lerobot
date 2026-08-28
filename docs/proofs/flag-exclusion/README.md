# Excluding flagged frames — the control that starts the run

Captured from a real GUI (uvicorn + Chromium) against two synthetic datasets in
one source: one declaring `blurry`, `fumble` and `occluded`, one declaring no
flags column at all. Both states matter, and only one of them has checkboxes in
it.

|                               | state                                                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `1-no-dataset.png`            | before a dataset is chosen — the picker says what it is waiting for rather than showing an empty box                         |
| `2-dataset-without-flags.png` | a dataset with no flags column — says so, and points at where one is made, instead of rendering a control with nothing in it |
| `3-flags-offered.png`         | the declared flags, **none ticked**: the default is "train on every frame"                                                   |
| `4-one-ticked.png`            | `fumble` ticked — the only state that submits a value                                                                        |
| `5-form-in-context.png`       | the picker in the whole form, directly under the camera picker                                                               |

The last one is the placement argument: both are properties of the dataset
rather than hyperparameters, so both sit outside the advanced disclosure and
next to each other. The two are also the reason the default is worth
photographing — they look identical and mean opposite things. Cameras are an
inclusion list, so every box ticked is the default; flags are an exclusion list,
so no box ticked is. Both submit nothing in that state.

What the selection becomes, from the recipe builders themselves:

```
lerobot-train   --dataset.exclude_flags=[fumble]
hvla            --exclude-flags fumble
```

The capture script asserts the label does not contain the word "quality": these
shots were once taken before that rename and sat in the PR contradicting the
code, which a picture is uniquely good at hiding.

The same states are asserted by `flag_picker.test.js` and
`test_flag_picker_backend.py`, the latter through the HTTP route rather than the
scanner beneath it — the response model dropping the field is what these
pictures caught first.
