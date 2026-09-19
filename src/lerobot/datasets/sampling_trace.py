#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What a run actually drew, per frame, so it can be asked afterwards.

A training log says a selection was applied. It does not say which frames the
run then learned from, or how evenly -- and those are the questions asked later,
when a policy behaves oddly around a segment and nobody can tell whether it saw
that segment twice or not at all.

This records one number per dataset frame: how many times it was drawn as a
chunk start. Draws, not supervised positions, because supervision is a pure
function of the draws and the boundaries -- a start at ``s`` supervises exactly
``[s, min(episode_end(s), next_excluded_at_or_after(s)))`` -- so recording both
would put per-position work in the hot path to store something derivable.
:func:`supervised_counts` does that derivation offline.

**One counter per run, written by the main process.** Under accelerate every
rank enumerates the *whole* sampler and yields only the batches belonging to it
(``BatchSamplerShard._iter_with_no_split``), so every rank's tally is the same
full epoch rather than its own share. Per-rank files summed together therefore
reported ``world_size`` times the truth, which is why there is one.

**The counter is file-backed, not resident.** One int32 per frame is nothing at
the size of a teaching dataset and a great deal at the size of a real one: 40k
frames is 160 KB, but 1000 hours at 50 fps is 180M frames and 687 MB, which is
not something to hold for the length of a run on a machine also holding a model.
Memory-mapped, the pages are page cache -- the kernel reclaims them under
pressure and writes them back, so the cost is bounded by what is actually being
touched rather than by the dataset's length. It is also why the counts are their
own file rather than an array inside an archive: an archive would have to be
re-serialised whole at every checkpoint, which is the same problem again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DIRNAME = "sampling_trace"
COUNTS_FILE = "counts.i32"
META_FILE = "meta.npz"
COUNTS_DTYPE = np.int32


def counts_path(directory: str | Path) -> Path:
    return Path(directory) / COUNTS_FILE


def open_draw_counts(directory: str | Path, num_frames: int) -> np.memmap:
    """Create or reopen the file-backed draw counter for a run.

    Preconditions:
        Only the main process may hold one. Under accelerate every rank
        enumerates the whole sampler, so a second rank's counter would be a
        duplicate of this one rather than a complement to it -- and two ranks
        sharing one mapping would lose increments outright, since
        ``counts[i] += 1`` is a read-modify-write with no atomicity.

        ``num_frames`` is the dataset's total frame count -- the counter is
        indexed by *absolute* frame, so a run loading only some episodes still
        sizes it for all of them and stays comparable with a run that did not.

    Postconditions:
        Returns a writable ``int32`` memmap of length ``num_frames``, zeroed on
        first creation and carrying its existing counts on reopen, so a resumed
        run continues the same tally rather than starting a second one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = counts_path(directory)
    exists = path.exists() and path.stat().st_size == num_frames * COUNTS_DTYPE().itemsize
    return np.memmap(path, dtype=COUNTS_DTYPE, mode="r+" if exists else "w+", shape=(num_frames,))


def save_sampling_trace(
    directory: str | Path,
    *,
    draw_counts: np.ndarray,
    episode_from: np.ndarray,
    episode_to: np.ndarray,
    excluded_frames: np.ndarray | None = None,
    **scalars: Any,
) -> Path:
    """Flush the counter and write the metadata needed to interpret it.

    The counts themselves are not copied: they are already on disk, and at the
    sizes this is built for copying them at every checkpoint would cost more
    than the training step it interrupts.

    Postconditions:
        ``directory`` is self-describing -- nothing outside it is needed to
        derive which frames were supervised, and how often.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(draw_counts, np.memmap):
        draw_counts.flush()
    else:
        # A run that did not use the file-backed counter (tests, mostly) still
        # gets a readable trace rather than a missing one.
        np.asarray(draw_counts, dtype=COUNTS_DTYPE).tofile(counts_path(directory))
    excluded = np.asarray([] if excluded_frames is None else excluded_frames, dtype=np.int64)
    # The feature's whole promise, checked end to end at every checkpoint of
    # every real run: whatever the sampler, the trainer wiring and the resume
    # path did in between, no excluded frame was ever drawn. This is the one
    # assertion here that does not trust any of them -- it reads the tally the
    # run actually produced. It is also the only check the trainer entry points
    # get at all, since no test executes them.
    #
    # Costs one fancy-index over the excluded set, not over the dataset.
    if excluded.size:
        drawn = np.asarray(draw_counts)[excluded]
        assert not drawn.any(), (
            f"{int((drawn > 0).sum())} excluded frames were drawn as chunk starts "
            f"({int(drawn.sum())} draws) -- the run reported itself filtered and was not"
        )
    np.savez_compressed(
        directory / META_FILE,
        episode_from=np.asarray(episode_from, dtype=np.int64),
        episode_to=np.asarray(episode_to, dtype=np.int64),
        excluded_frames=excluded,
        num_frames=np.asarray(len(draw_counts), dtype=np.int64),
        **{k: np.asarray(v) for k, v in scalars.items()},
    )
    return directory


def load_sampling_trace(directory: str | Path) -> dict[str, np.ndarray]:
    """Read a trace back. ``draw_counts`` is memory-mapped read-only.

    Mapped rather than read, for the same reason it is written mapped: a caller
    asking about one episode should not pay for the whole dataset's counts.

    Raises:
        FileNotFoundError: If the counts file is absent. An empty trace read as
            all zeros would be indistinguishable from a run that drew nothing.
    """
    directory = Path(directory)
    with np.load(directory / META_FILE) as data:
        trace = {key: data[key] for key in data.files}
    num_frames = int(trace["num_frames"])
    path = counts_path(directory)
    if not path.exists():
        raise FileNotFoundError(f"no {COUNTS_FILE} in {directory}")
    trace["draw_counts"] = np.memmap(path, dtype=COUNTS_DTYPE, mode="r", shape=(num_frames,))
    return trace


def window_end(start: int, episode_to: np.ndarray, excluded_frames: np.ndarray, dataset_frames: int) -> int:
    """Where the action window of a chunk starting at ``start`` stops.

    The same rule the reader applies, restated here so a trace can be read
    without a dataset to hand: the episode end, or the first excluded frame at
    or after the start, whichever comes first.
    """
    episode = int(np.searchsorted(episode_to, start, side="right"))
    end = int(episode_to[episode]) if episode < len(episode_to) else dataset_frames
    if excluded_frames.size:
        position = int(np.searchsorted(excluded_frames, start, side="left"))
        if position < excluded_frames.size:
            end = min(end, int(excluded_frames[position]))
    return end


def supervised_counts(trace: dict[str, np.ndarray], chunk_size: int) -> np.ndarray:
    """How many times each frame was supervised, derived from the draws.

    Preconditions:
        ``trace`` is what :func:`load_sampling_trace` returns, and ``chunk_size``
        is the one the run used -- it is not recorded in the sampler, so it must
        be supplied by whoever knows the run's config.

    Postconditions:
        Zero for every excluded frame, by construction rather than by filtering:
        a window stops *at* the first excluded frame, so no draw ever covers it.
    """
    draws = trace["draw_counts"]
    excluded = trace["excluded_frames"]
    episode_to = trace["episode_to"]
    frames = int(draws.size)
    counts = np.zeros(frames, dtype=np.int64)
    for start in np.flatnonzero(draws):
        end = min(int(start) + chunk_size, window_end(int(start), episode_to, excluded, frames))
        counts[int(start) : end] += int(draws[start])
    return counts
