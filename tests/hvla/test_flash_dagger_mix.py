"""Tests for ThreeWayMixDataset categorical sampling."""

from __future__ import annotations

import random
from collections import Counter

from lerobot.policies.hvla.flash_dagger.mix import ThreeWayMixDataset


def _tagged_samples(tag: str, n: int) -> list[dict]:
    return [{"pool": tag, "i": i} for i in range(n)]


def test_mix_categorical_distribution_within_tolerance():
    """Empirical pool fractions should match configured mix to within ~3pp at N=10000."""
    random.seed(0)
    old = _tagged_samples("old", 50)
    flashed = [_tagged_samples("flashed", 50)]
    new = _tagged_samples("new", 50)

    mix = ThreeWayMixDataset(
        old_samples=old,
        flashed_pools=flashed,
        new_pool=new,
        old_pct=0.10,
        flashed_pct=0.25,
        length=10000,
    )

    counts = Counter()
    for i in range(len(mix)):
        s = mix[i]
        counts[s["pool"]] += 1
    total = sum(counts.values())
    fracs = {k: v / total for k, v in counts.items()}
    assert abs(fracs["old"] - 0.10) < 0.03
    assert abs(fracs["flashed"] - 0.25) < 0.03
    assert abs(fracs["new"] - 0.65) < 0.03


def test_mix_collapses_flashed_into_new_when_pool_empty():
    """First-cycle case: no flashed pool yet, flashed_pct should fold into new."""
    random.seed(1)
    old = _tagged_samples("old", 20)
    new = _tagged_samples("new", 20)

    mix = ThreeWayMixDataset(
        old_samples=old,
        flashed_pools=[],
        new_pool=new,
        old_pct=0.10,
        flashed_pct=0.25,
        length=2000,
    )

    counts = Counter()
    for i in range(len(mix)):
        counts[mix[i]["pool"]] += 1
    fracs = {k: v / len(mix) for k, v in counts.items()}
    assert "flashed" not in counts  # never sampled
    assert abs(fracs["old"] - 0.10) < 0.03
    assert abs(fracs.get("new", 0) - 0.90) < 0.03


def test_mix_skips_empty_flashed_pools():
    """Empty individual pool entries shouldn't break sampling."""
    random.seed(2)
    old = _tagged_samples("old", 30)
    flashed = [[], _tagged_samples("flashed", 30), []]
    new = _tagged_samples("new", 30)

    mix = ThreeWayMixDataset(
        old_samples=old,
        flashed_pools=flashed,
        new_pool=new,
        old_pct=0.10,
        flashed_pct=0.25,
        length=2000,
    )

    counts = Counter()
    for i in range(len(mix)):
        counts[mix[i]["pool"]] += 1
    # Should still see some flashed samples
    assert counts.get("flashed", 0) > 0


def test_mix_rejects_invalid_pcts():
    import pytest

    new = _tagged_samples("new", 10)
    with pytest.raises(AssertionError):
        ThreeWayMixDataset(
            old_samples=_tagged_samples("old", 10),
            flashed_pools=[],
            new_pool=new,
            old_pct=0.6,
            flashed_pct=0.5,
            length=100,
        )
