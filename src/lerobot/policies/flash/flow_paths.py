"""Display-only point paths composed from saved UFM image-pair flow.

Unlike TrackEverything, UFM does not produce persistent multi-frame tracks.
Advect display particles with bilinear flow inside complete recorded grid cells;
never bridge holes, use nearest-point identities, or modify the pose experiment.
"""

from __future__ import annotations

import numpy as np


class RecordedFlow:
    def __init__(self, pair):
        self.source = np.asarray(pair["source_uv"], dtype=float).reshape(-1, 2)
        self.target = np.asarray(pair["target_uv"], dtype=float).reshape(-1, 2)
        self.stage = np.asarray(pair["stage"], dtype=int)
        self.origin = self.source.min(axis=0)
        steps = []
        for axis in range(2):
            differences = np.diff(np.unique(self.source[:, axis]))
            # Coordinates were rounded to 0.001 px in the evidence file.
            steps.append(
                np.median(differences[differences < differences.min() * 1.5]) if len(differences) else 1
            )
        self.step = np.asarray(steps)
        ij = np.rint((self.source - self.origin) / self.step).astype(int)
        width, height = ij.max(axis=0) + 1
        self.grid = np.full((height, width), -1, dtype=int)
        self.grid[ij[:, 1], ij[:, 0]] = np.arange(len(ij))

    def sample(self, positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        uv = (positions - self.origin) / self.step
        # Restore exact grid nodes after evidence-coordinate rounding.
        rounded = np.rint(uv)
        uv = np.where(np.abs(uv - rounded) < 0.002, rounded, uv)
        lower = np.floor(uv).astype(int)
        fraction = uv - lower
        valid = np.ones(len(positions), dtype=bool)
        delta = np.zeros_like(positions)
        stage = np.full(len(positions), 4, dtype=int)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (
                fraction[:, 1] if dy else 1 - fraction[:, 1]
            )
            x, y = lower[:, 0] + dx, lower[:, 1] + dy
            inside = (x >= 0) & (x < self.grid.shape[1]) & (y >= 0) & (y < self.grid.shape[0])
            index = np.full(len(positions), -1, dtype=int)
            index[inside] = self.grid[y[inside], x[inside]]
            needed = weight > 0
            valid &= ~needed | (index >= 0)
            use = np.flatnonzero(needed & (index >= 0))
            delta[use] += weight[use, None] * (self.target[index[use]] - self.source[index[use]])
            stage[use] = np.minimum(stage[use], self.stage[index[use]])
        return positions + delta, stage, valid


class DenseFlow(RecordedFlow):
    """Sample UFM's full output, avoiding extra interpolation of the saved grid.

    Original sparse samples still determine seeds, coverage and conservative
    filtering labels. Only the displacement uses the full model output.
    """

    def __init__(self, pair, flow, origin, scale):
        super().__init__(pair)
        self.flow = np.asarray(flow)
        self.flow_origin = np.asarray(origin)
        self.flow_scale = scale

    def sample(self, positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        _, stage, valid = super().sample(positions)
        uv = (positions - self.flow_origin) / self.flow_scale
        lower = np.floor(uv).astype(int)
        fraction = uv - lower
        delta = np.zeros_like(positions)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (
                fraction[:, 1] if dy else 1 - fraction[:, 1]
            )
            x, y = lower[:, 0] + dx, lower[:, 1] + dy
            inside = (x >= 0) & (x < self.flow.shape[2]) & (y >= 0) & (y < self.flow.shape[1])
            valid &= (weight == 0) | inside
            rows = np.flatnonzero((weight > 0) & inside)
            delta[rows] += weight[rows, None] * self.flow[:, y[rows], x[rows]].T
        valid &= np.isfinite(delta).all(axis=1)
        return positions + delta * self.flow_scale, stage, valid


class FlowPaths:
    """Persistent display IDs, seeded from uncovered source-grid locations.

    Limits and spacing only control visualization density, never fitting.
    Keep states at each frame because the recovery method may query an older one.
    """

    def __init__(self, max_points=96, spacing_px=8):
        self.max_points = max_points
        self.spacing = spacing_px
        self.states = {}
        self.next_id = 0

    def update(self, pair, *, field=None):
        if pair is None or not pair["source_uv"]:
            return []
        field = RecordedFlow(pair) if field is None else field
        source_frame, target_frame = pair["source_frame"], pair["target_frame"]
        previous = self.states.get(source_frame, np.empty((0, 4)))  # id, hue, x, y
        targets, stages, valid = field.sample(previous[:, 2:])
        previous, targets, stages = previous[valid], targets[valid], stages[valid]
        source = previous[:, 2:]
        ids, hues = previous[:, 0].astype(int).tolist(), previous[:, 1].astype(int).tolist()

        # Reseed uncovered areas without pretending new particles are old ones.
        cells = np.floor(field.source / self.spacing).astype(int)
        _, indices = np.unique(cells, axis=0, return_index=True)
        candidates = field.source[indices]
        if len(source):
            distance = np.sum((candidates[:, None] - source[None]) ** 2, axis=2).min(axis=1)
            indices = indices[distance >= self.spacing**2]
        available = max(0, self.max_points - len(previous))
        indices = (
            indices[np.linspace(0, len(indices) - 1, min(available, len(indices)), dtype=int)]
            if len(indices)
            else indices
        )
        if len(indices):
            seed = field.source[indices]
            source = np.concatenate((source, seed))
            targets = np.concatenate((targets, field.target[indices]))
            stages = np.concatenate((stages, field.stage[indices]))
            ids.extend(range(self.next_id, self.next_id + len(seed)))
            self.next_id += len(seed)
            # Stable spatial colours, as in TrackEverything/CoTracker plots.
            span = max(float(np.ptp(field.source[:, 1])), 1)
            hues.extend(np.rint(280 * (seed[:, 1] - field.origin[1]) / span).astype(int).tolist())
        self.states[target_frame] = np.column_stack((ids, hues, targets))
        # A newly seeded source is also reusable if later recovery refers to it.
        self.states[source_frame] = np.column_stack((ids, hues, source))
        return [
            [int(i), int(h), *np.round(a, 3).tolist(), *np.round(b, 3).tolist(), int(s)]
            for i, h, a, b, s in zip(ids, hues, source, targets, stages, strict=True)
        ]
