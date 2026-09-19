# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Both writes that change a mask row have to tell the row.

`FeatureEditing.refreshFromServer` is what makes a write the panel did not
make visible without a page reload, and it works — that is pinned behaviourally
in test_feature_panel_refresh_playwright.py. What that test cannot see is
whether anything calls it. The fix at each write site is a single line, which
is exactly the kind of line that gets dropped in a refactor and takes a
behavioural test with it only if the test drives the whole UI flow through a
segmenter and a GPU.

So the call sites are checked here instead, by reading the source. Two of them:

  * the mask save, when its job completes — the reported bug ("after saving,
    it did not refresh the feature display of the episode, and it took me a
    refresh");
  * the effects apply, because the lane names its own treatment and an apply
    changes exactly that.

A source check is a weak test in general. It is the right one here because the
failure it guards is a deletion, not a behaviour.
"""

from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"

CALL = "refreshFromServer"


@pytest.mark.parametrize(
    ("filename", "context"),
    [
        ("overlay_stream.js", "the mask save, once its job reports complete"),
        ("masks.js", "the effects apply, which changes the treatment each lane names"),
    ],
)
def test_a_write_path_refreshes_the_feature_panel(filename: str, context: str):
    source = (STATIC / filename).read_text()
    assert CALL in source, (
        f"{filename} no longer calls {CALL}, so {context} leaves the mask rows "
        "showing the pre-write state until the page is reloaded"
    )


def test_the_panel_still_offers_the_call():
    """The two call sites above are optional-chained, so a rename in the panel
    would leave them silently doing nothing rather than failing."""
    source = (STATIC / "feature_editing.js").read_text()
    assert f"function {CALL}(" in source, f"{CALL} is gone from the panel; its callers now no-op"
    assert f"        {CALL}," in source, f"{CALL} is defined but not exported, so callers no-op"
