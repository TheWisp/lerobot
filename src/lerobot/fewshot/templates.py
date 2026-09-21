# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The EXAMINE phase's product: a template bank standing in for the object model.

There is no mesh and no canonical pose. The bank is a set of observations of the
object — real views the user showed it in, plus synthetic in-plane rotations of
each — and "where is the object relative to the demo" is answered by registering
the live view against the bank and composing transforms between observations.

Rotation augmentation is what widens the basin: dense ViT features are not
rotation-equivariant, so matching degrades with in-plane angle. Rotating the
*image* and re-extracting pays that cost once, offline, per template — at match
time the nearest rotated copy is within half the rotation step, where features
still correlate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.fewshot.registration import RegistrationResult, Sim2, ransac_register


@dataclass
class Template:
    """One entry: features of one (possibly synthetically rotated) view.

    ``to_source`` maps THIS entry's coordinates back to the coordinates of the real
    captured view it derives from (identity for the unrotated original), so any
    match against a rotated copy composes back to a real observation.
    """

    view_id: str
    coords: np.ndarray  # (N, 2) in this entry's own image frame
    feats: np.ndarray  # (N, D), L2-normalised
    to_source: Sim2


@dataclass
class BankMatch:
    """Best registration of a live observation against the bank.

    ``source_to_live`` maps the matched real view's coordinates into the live
    image — rotated copies are already composed away.

    ``rotation_ambiguous`` is the bank-level trust signal, and it exists because a
    per-registration flag is not enough: measured on a rig frame, a 150-deg-rotated
    view registered against a single template with 28 inliers, a passing residual
    check, and a 168.6-deg angle error — it had locked onto the 180-flipped
    solution, confidently. Only the bank can see that failure mode: it shows up as
    a COMPETING template at a meaningfully different composed angle matching almost
    as well. Symmetric objects show up the same way, which is exactly right — for
    them the angle is unknowable and the caller should transfer translation only.
    """

    view_id: str
    source_to_live: Sim2
    result: RegistrationResult
    rotation_ambiguous: bool = True

    @property
    def trust_rotation(self) -> bool:
        return self.result.ok and self.result.n_inliers >= 12 and not self.rotation_ambiguous


class TemplateBank:
    """Examine-phase store + match. Purely in-memory for the prototype."""

    def __init__(self, extractor):
        self._extractor = extractor
        self.templates: list[Template] = []

    def add_view(
        self,
        view_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        rotations_deg: tuple[float, ...] = (0.0,),
    ) -> int:
        """Extract and store one examined view, plus rotated copies.

        Pre: image HxWx3 uint8, mask HxW bool non-empty; 0.0 should be included in
        ``rotations_deg`` (asserted) so the unrotated view is always matchable.
        Post: returns how many templates were added.
        """
        assert 0.0 in rotations_deg, "always keep the unrotated view"
        import cv2

        ys, xs = np.nonzero(mask)
        centre = (float(xs.mean()), float(ys.mean()))
        added = 0
        for deg in rotations_deg:
            if deg == 0.0:
                img_r, mask_r = image, mask
                rot = Sim2.identity()
            else:
                warp = cv2.getRotationMatrix2D(centre, deg, 1.0)
                img_r = cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]))
                mask_r = cv2.warpAffine(mask.astype(np.uint8), warp, (image.shape[1], image.shape[0])) > 0
                if not mask_r.any():
                    continue  # rotated fully out of frame — nothing to extract
                # cv2's angle is counter-clockwise in image coords with y down; build the
                # SAME transform as Sim2 so composition is exact rather than convention-lucky.
                th = -np.deg2rad(deg)
                c = np.asarray(centre)
                rmat = Sim2.from_angle(th).R
                rot = Sim2(1.0, rmat, c - rmat @ c)  # rotated = rot(source)
            coords, feats = self._extractor.extract(img_r, mask_r)
            self.templates.append(
                Template(view_id=view_id, coords=coords, feats=feats, to_source=rot.inverse())
            )
            added += 1
        assert added > 0
        return added

    def match(self, image: np.ndarray, mask: np.ndarray, **ransac_kw) -> BankMatch | None:
        """Register the live observation against every template; best inliers win.

        Post: ``None`` if nothing registers at all; otherwise ``source_to_live``
        composes the winning entry's synthetic rotation away, so callers only ever
        see transforms between REAL observations.
        """
        assert self.templates, "examine first: the bank is empty"
        live_coords, live_feats = self._extractor.extract(image, mask)
        scored: list[tuple[Template, RegistrationResult, Sim2]] = []
        for tpl in self.templates:
            res = ransac_register(tpl.coords, live_coords, tpl.feats, live_feats, **ransac_kw)
            if res.ok:
                # template_to_live ∘ (source -> template) = source -> live
                scored.append((tpl, res, res.sim2.compose(tpl.to_source.inverse())))
        if not scored:
            return None
        scored.sort(key=lambda x: -x[1].n_inliers)
        tpl, res, src_to_live = scored[0]
        # Ambiguity: a competitor whose composed angle disagrees, with enough support
        # to be a real second hypothesis. 0.45 is chosen against measured data: on rig
        # frames, wrong-angle competitors for genuinely textured objects reach at most
        # ~8% of the winner's inliers, while a symmetric object's alternate solution
        # sits near parity — the two populations are far apart, and 0.45 splits them
        # with margin on both sides. (Resampling blur costs rotated copies some
        # support, so demanding near-parity would miss real ambiguity.)
        ambiguous = any(
            r.n_inliers >= 0.45 * res.n_inliers
            and abs((c.theta - src_to_live.theta + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(20)
            for _, r, c in scored[1:]
        )
        return BankMatch(
            view_id=tpl.view_id,
            source_to_live=src_to_live,
            result=res,
            rotation_ambiguous=ambiguous,
        )
