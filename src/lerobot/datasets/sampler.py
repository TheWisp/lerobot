#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
        draw_counts: np.ndarray | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
            draw_counts: Buffer to tally draws into, indexed by absolute frame and at
                least as long as the dataset, or None to not count at all. Pass a
                memory-mapped one for a dataset large enough that an int32 per frame
                matters -- :func:`lerobot.datasets.sampling_trace.open_draw_counts`
                makes one.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])

        # Excluding arbitrary frames cannot be expressed as episode intervals,
        # which is what this sampler stores, so the surviving starts are
        # materialised instead -- one int64 per drawable frame. That is paid
        # only when frames are actually excluded; without it the interval
        # arithmetic above is used unchanged.
        self._absolute_to_relative = absolute_to_relative_idx
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

        # How many times each *absolute* frame has been drawn as a chunk start.
        # Kept always rather than behind a flag: it is one increment per draw
        # against a video decode, and a trace nobody enabled is a trace nobody
        # has when the question comes up. Absolute so it stays comparable across
        # runs that load different episode subsets.
        #
        # Opt-in, and the buffer is the caller's to supply. Counting by default
        # would charge every sampler for a tally most of them throw away: an
        # int32 per frame is 160 KB at 40k frames and 687 MB at 1000 hours of
        # 50 fps, and a resident one is anonymous memory -- not reclaimable the
        # way the memory-mapped buffer a trainer passes is. Under accelerate
        # that would be paid once per rank for counts only the main process
        # keeps. None means this sampler does not count.
        self._dataset_frames = int(to_indices.max()) if len(to_indices) else 0
        if draw_counts is not None and len(draw_counts) < self._dataset_frames:
            raise ValueError(
                f"draw_counts has {len(draw_counts)} entries but the dataset ends at frame "
                f"{self._dataset_frames}; a short buffer would silently drop the tail's counts"
            )
        self.draw_counts = draw_counts

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _to_relative(self, absolute_idx: int) -> int:
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def _frame_index(self, position: int) -> int:
        """Position -> the index a caller consumes. Does **not** count a draw.

        Separate from :meth:`_draw` because ``indices`` walks every position for
        introspection, and counting there would record draws that never happened.
        """
        return self._to_relative(self._absolute_frame_index(position))

    def _draw(self, position: int) -> int:
        """Position -> the index a caller consumes, recording it as drawn.

        The one funnel every yielded index passes through, which is why the
        count is taken here rather than at the call sites: a second iteration
        path added later is counted without remembering to.

        **This counts enumeration, not consumption, and under accelerate those
        differ.** Sharding is applied above the sampler: every rank builds the
        same sampler, computes the same permutation -- which is what makes the
        order a pure function of (seed, epoch) and needs no shared RNG -- and
        ``BatchSamplerShard`` then keeps batch *i* on rank ``i % world_size``.
        The filter is on which batch is *yielded*; the loop walks the whole
        underlying sampler on every process. So with three ranks over a
        24-position epoch, each rank trains on 8 and passes through all 24.

        Anything hung off the sampler therefore measures the epoch once per
        rank; only something reading the dataloader's output measures a rank's
        own share. A per-rank tally of this counter is a duplicate rather than a
        complement, which is why one process keeps it -- summing them reported
        ``world_size`` times the truth.
        """
        absolute_idx = self._absolute_frame_index(position)
        if self.draw_counts is not None:
            self.draw_counts[absolute_idx] += 1
        return self._to_relative(absolute_idx)

    def _absolute_frame_index(self, position: int) -> int:
        """Position -> *absolute* frame, from the episode intervals alone.

        The seam subclasses override to change *which* starts exist without
        touching how they are permuted or resumed. Kept absolute because any
        exclusion is expressed in absolute indices and must be compared in that
        space; :meth:`_frame_index` applies the relative mapping afterwards, in
        one place.
        """
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        return int(self._starts[episode]) + position_in_episode

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._draw(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._draw(k)

    def __len__(self) -> int:
        return self._num_frames


class ExcludedStartSampler(EpisodeAwareSampler):
    """An :class:`EpisodeAwareSampler` that never draws certain frames as starts.

    An excluded flag makes some starts worthless rather than merely uninteresting:
    a flagged frame ends the action window at position zero, so a chunk
    starting there is wholly padding and contributes nothing to the loss.
    Drawing it anyway still costs a video decode, a collate and a forward pass,
    and makes the draw uniform over *frames* while the useful draws are only a
    fraction of them.

    Subclassed rather than folded into the parent because only one thing
    differs -- which starts exist. Everything that is genuinely hard here is
    inherited untouched: the per-epoch permutation that is a pure function of
    ``(seed, epoch)`` so every rank agrees without sharing an RNG, and the
    ``state_dict`` contract that makes resume sample-exact. Both fail silently
    when reimplemented slightly differently, which is a poor thing to risk for
    a filter.

    Excluding starts changes what the parent stores, though: arbitrary frames
    cannot be described by episode intervals, so the surviving starts are
    materialised here -- one int64 each. That cost is paid only by runs that
    actually exclude something.
    """

    def __init__(self, *args, excluded_frames, **kwargs):
        super().__init__(*args, **kwargs)

        excluded = np.unique(np.asarray(excluded_frames, dtype=np.int64))
        # The same mapping ``_absolute_frame_index`` performs, done once over
        # the whole range instead of once per position. A Python-level generator
        # here cost 153 s to construct on a 180M-frame dataset -- start-up time
        # that scales with the data and buys nothing, since the mapping is
        # interval arithmetic. ``test_candidates_match_the_per_position_mapping``
        # pins the two against each other.
        lengths = np.diff(self._cum_lengths, prepend=0)
        offsets = np.arange(self._num_frames, dtype=np.int64) - np.repeat(
            self._cum_lengths - lengths, lengths
        )
        candidates = np.repeat(self._starts, lengths) + offsets
        self._valid = candidates[~np.isin(candidates, excluded)]
        # The filter's postcondition, established by a different algorithm than
        # the filter: `excluded` is sorted, so membership is a searchsorted
        # probe rather than a second `isin`. Re-running `isin` here would be
        # circular -- a dtype coercion that made it match nothing would make the
        # check match nothing too, and pass.
        #
        # What this does NOT catch: an `excluded` array that is correct-looking
        # but in the wrong index space. It would remove the wrong candidates and
        # satisfy this postcondition. That is prevented upstream instead --
        # DatasetReader builds `_flagged_indices` from the dataset's own `index`
        # column, so the values are absolute frames by construction.
        if excluded.size and self._valid.size:
            probe = np.searchsorted(excluded, self._valid)
            hit = (probe < excluded.size) & (excluded[np.minimum(probe, excluded.size - 1)] == self._valid)
            assert not hit.any(), f"{int(hit.sum())} starts survived the filter that name excluded frames"
        if not self._valid.size:
            raise ValueError(
                f"Every one of the {self._num_frames} candidate start frames is excluded; "
                "there is nothing left to train on."
            )
        self._num_frames = int(self._valid.size)

    def _absolute_frame_index(self, position: int) -> int:
        return int(self._valid[position])


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}


def make_start_sampler(
    dataset_from_indices,
    dataset_to_indices,
    *,
    excluded_frames=None,
    trace_dir=None,
    **kwargs,
) -> EpisodeAwareSampler:
    """Build the sampler a training run draws its chunk starts from.

    One place both trainers call, rather than each assembling the same three
    decisions. Those decisions are: which sampler class the run needs, whether a
    draw counter is opened and where, and whether the sampler that came back
    actually excludes anything. Spread across two call sites they drifted -- and
    the wiring is the part of this feature that no test executes.

    Preconditions:
        ``excluded_frames`` are *absolute* dataset frames, or None/empty when
        the run excludes nothing. ``trace_dir`` is a directory to keep the draw
        counter in, or None for a run that should not keep one -- which is every
        rank but the main one, since under accelerate each rank enumerates the
        whole sampler and its tally would duplicate rather than complete the
        others' (see :meth:`EpisodeAwareSampler._draw`).

    Postconditions:
        Returns :class:`ExcludedStartSampler` when frames are excluded and
        :class:`EpisodeAwareSampler` when none are, so a run that excludes
        nothing keeps the parent's compact per-episode representation and pays
        for no index array. When frames *are* excluded the result offers strictly
        fewer starts than the dataset has frames -- asserted, because the failure
        it catches is a run that resolves flags, logs the count, and then trains
        on everything anyway.
    """
    excluded = None if excluded_frames is None else np.asarray(excluded_frames)
    if trace_dir is not None:
        from .sampling_trace import open_draw_counts

        # Sized for the whole dataset, not the loaded subset: the counter is
        # indexed by absolute frame so two runs over different episode
        # selections stay comparable.
        kwargs["draw_counts"] = open_draw_counts(trace_dir, int(np.asarray(dataset_to_indices)[-1]))

    if excluded is None or not excluded.size:
        return EpisodeAwareSampler(dataset_from_indices, dataset_to_indices, **kwargs)

    sampler = ExcludedStartSampler(
        dataset_from_indices, dataset_to_indices, excluded_frames=excluded, **kwargs
    )
    total = int(np.asarray(dataset_to_indices)[-1])
    assert len(sampler) < total, (
        f"{excluded.size} frames are excluded but the sampler still offers {len(sampler)} of {total} starts"
    )
    return sampler
