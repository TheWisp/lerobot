# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Loading a model that is already on disk must not talk to the Hub.

`from_pretrained` re-validates every config file against huggingface.co even
when the weights are cached, and the SAM3 adapter performs six of them. Each
one is several round trips; this rig tunnels all Hub traffic, so a round trip
costs ~320 ms. Measured on a real mask save: 34 requests to huggingface.co and
14 s of a 36 s save spent before a single frame was segmented, entirely on
HEAD requests for files that had not changed. Resolving the config and
processor alone measured 11.5 s online against 2.5 s from the cache.

The cost is invisible in the way that matters — nothing fails, the save is
just slower, and it is slower by an amount that looks like segmentation being
expensive. So it is pinned here rather than left to be rediscovered:
`_from_cache_first` asks the cache and stops if the cache answers, and every
load in the adapter goes through it. The second half is a source check because
that is where the regression would actually happen — a seventh model added
later with a bare `from_pretrained` reintroduces the whole cost, and no
behavioural test would fail.
"""

import re
from pathlib import Path

import lerobot.overlays.adapters as adapters_mod
from lerobot.overlays.adapters import _from_cache_first


class _Recorder:
    """Stands in for a transformers class, recording how it was asked."""

    calls: list[bool] = []
    fail_offline = False

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append(bool(kwargs.get("local_files_only", False)))
        if cls.fail_offline and kwargs.get("local_files_only"):
            raise OSError(f"{model_id} is not in the local cache")
        return "loaded"


def test_a_cached_model_is_loaded_without_asking_the_hub():
    _Recorder.calls, _Recorder.fail_offline = [], False
    assert _from_cache_first(_Recorder, "facebook/sam3") == "loaded"
    assert _Recorder.calls == [True], (
        "the load reached the network for a model already on disk: this is the "
        "12 seconds every mask save and every preview start used to pay"
    )


def test_a_cache_miss_still_reaches_the_hub():
    """Cache-first must not mean cache-only, or a first run could never download."""
    _Recorder.calls, _Recorder.fail_offline = [], True
    assert _from_cache_first(_Recorder, "facebook/sam3") == "loaded"
    assert _Recorder.calls == [True, False]


def test_keyword_arguments_survive_both_attempts():
    """The config and dtype the adapter passes decide what is actually loaded —
    a helper that dropped them would load a differently-shaped model."""
    seen = []

    class Fussy:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("miss")
            return "loaded"

    _from_cache_first(Fussy, "facebook/sam3", config="CFG", dtype="fp16")
    assert [s.get("config") for s in seen] == ["CFG", "CFG"]
    assert [s.get("dtype") for s in seen] == ["fp16", "fp16"]


def test_no_model_load_in_the_adapter_bypasses_the_cache():
    """The check that survives someone adding a seventh model.

    A bare `X.from_pretrained(...)` anywhere in this module silently restores
    the per-load Hub round trips, and every behavioural test still passes — the
    save just gets slower again. So the module is read: the only place allowed
    to call `from_pretrained` is the helper that passes `local_files_only`.
    """
    source = Path(adapters_mod.__file__).read_text()
    helper_start = source.index("def _from_cache_first(")
    helper_end = source.index("\nclass ", helper_start)
    outside = source[:helper_start] + source[helper_end:]

    offenders = [
        line.strip()
        for line in outside.splitlines()
        if re.search(r"\.from_pretrained\s*\(", line) and "local_files_only" not in line
    ]
    assert not offenders, (
        "these loads go straight to the Hub instead of through _from_cache_first, "
        f"which costs ~2 s each on a cached model: {offenders}"
    )
