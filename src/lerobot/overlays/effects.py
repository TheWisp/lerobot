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
"""Per-region visual treatments for data editing — the pixel transforms + composite.

Pure numpy/cv2, no dataset or model dependencies, so BOTH consumers share one
source of truth (preview == commit):
  * the live overlay worker (:mod:`lerobot.overlays.standalone`) renders the
    per-region composite on the scrubbed frame — the WYSIWYG preview;
  * the offline batch pass (:mod:`lerobot.datasets.dataset_postprocess`) runs the
    identical composite on every frame when committing to a new dataset.

Every region (each detected object, plus the background) carries one *treatment*
(``{key, params}``): Tint / Random / Blur / None. :func:`composite_regions` paints
them onto an HxWx3 uint8 RGB frame by each region's feathered mask; randomness is
drawn in :func:`sample_treatment` at the caller's cadence (once per episode for
trajectory coherence). The detection chrome (glow/label) is NOT here — it is a
display-only overlay the live worker adds and the batch pass never draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _solid(h: int, w: int, color) -> np.ndarray:
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = np.asarray(color, dtype=np.uint8)
    return out


def _noise_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Per-pixel colour static ("TV noise"), one pattern drawn per application.

    GreenAug's texture ablation (arXiv:2407.07868, Table 4) found background-texture
    ENTROPY correlates with scene generalisation: real high-entropy textures (6.81
    bits) reached 87% success vs Perlin noise (4.45 bits) at 66% and solid colours
    at 65% — "greater texture randomness leads to better performance". Per-pixel
    static is the maximum-entropy extreme of that trend (the earlier low-frequency
    block texture sat near the weak Perlin end). Known trade-off: video codecs
    low-pass or rate-inflate pure noise, so committed datasets store a softened
    version of it; a bank of random REAL texture images (GreenAug's best performer)
    is the follow-up that avoids both issues."""
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-region treatments — the unified WYSIWYG model (replaces the "protect one
# foreground, transform one background effect" special-case). Every region (each
# detected object, plus the background) carries ONE treatment; the live preview
# and the offline pass run the identical composite, so preview == commit. A
# treatment is the SAME operation pointed at a different mask — tinting an object
# and randomising the background are one mechanism.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentSpec:
    """A per-region treatment: identity + the controls the GUI renders for it."""

    key: str  # "none" | "tint" | "random" | "blur" | ...
    label: str
    controls: list[dict] = field(default_factory=list)
    randomized: bool = False  # does sample_treatment() draw anything? (per-episode)


# Order here is the GUI button order for a row's segmented control; "none" is the
# neutral/off default and is rendered last by the frontend.
TREATMENTS: list[TreatmentSpec] = [
    TreatmentSpec(
        key="tint",
        label="Tint",
        controls=[{"type": "color", "key": "color", "label": "Colour", "default": [79, 195, 247]}],
    ),
    TreatmentSpec(key="random", label="Random", randomized=True),  # per-pixel static, one draw per episode
    TreatmentSpec(
        key="blur",
        label="Blur",
        controls=[{"type": "range", "key": "strength", "label": "Blur", "min": 2, "max": 40, "default": 12}],
    ),
    TreatmentSpec(key="none", label="None"),  # neutral — region kept as-is
]

TREATMENTS_BY_KEY = {t.key: t for t in TREATMENTS}
_TINT_DEFAULT = [79, 195, 247]


def sample_treatment(key: str, params: dict, h: int, w: int, rng: np.random.Generator) -> dict:
    """Draw a treatment's per-application randomness (or ``{}`` if deterministic),
    at whatever cadence the caller chooses (once per episode for coherence)."""
    if key == "random":
        # One per-pixel static pattern per episode (see _noise_texture for the
        # entropy rationale + the real-texture-bank follow-up).
        return {"bg": _noise_texture(h, w, rng)}
    return {}


# Treatments whose output at a pixel depends ONLY on that pixel, so they can be computed
# on a crop and give bit-identical results. "blur" is deliberately absent: it reads
# neighbours, so cropping would change pixels near the region border.
_LOCAL_TREATMENTS = frozenset({"tint", "solid", "random"})


def _treat(rgb: np.ndarray, key: str, params: dict, sampled: dict) -> np.ndarray:
    """The treated pixels for one region's treatment, over the WHOLE frame (the
    caller composites them in by that region's mask). HxWx3 uint8 -> HxWx3 uint8."""
    import cv2

    h, w = rgb.shape[:2]
    if key == "tint":
        color = np.asarray(params.get("color", _TINT_DEFAULT), dtype=np.float32)
        s = float(params.get("strength", 0.55))  # blend toward colour, keeps shading
        # Round, not truncate (see composite_regions): a fractional blend strength
        # makes ~45% of values land just under an integer, biasing the tint down.
        return np.rint(np.clip(rgb.astype(np.float32) * (1.0 - s) + color * s, 0, 255)).astype(np.uint8)
    if key == "random":
        bg = sampled.get("bg")
        return bg if (bg is not None and bg.shape[:2] == (h, w)) else _solid(h, w, [0, 0, 0])
    if key == "solid":
        return _solid(h, w, params.get("color", [0, 200, 0]))
    if key == "blur":
        sigma = max(1.0, float(params.get("strength", 12)))
        k = int(sigma * 4) | 1
        return cv2.GaussianBlur(rgb, (k, k), sigma)
    raise ValueError(f"unknown treatment {key!r}")


def composite_regions(
    rgb: np.ndarray,
    regions: list[tuple[np.ndarray | None, dict]],
    sampled: list[dict],
) -> np.ndarray:
    """Composite per-region treatments onto ``rgb`` (HxWx3 uint8).

    ``regions`` is ordered ``[(alpha, treatment), …]`` — **background first, objects
    after** (painted on top) — where ``alpha`` is HxW float in [0,1] (the region's
    feathered mask) and ``treatment`` is ``{"key", "params"}``. ``sampled`` is a
    parallel list of each region's pre-drawn randomness (from :func:`sample_treatment`).

    A ``none``/empty treatment is skipped, so that region keeps its real pixels
    (all-``none`` → the original frame, unchanged). Returns a new HxWx3 uint8 frame.
    This is the single source of truth shared by the live overlay and the offline
    pass — it renders ONLY committed pixels, never the detection chrome.
    """
    import cv2

    # Nothing to paint: every region keeps its own pixels, so the answer is the
    # frame. Worth its own line because it is the recipe an operator lands on
    # first (name the objects, treat nothing yet) and because the general path
    # below would still pay a full-frame float32 round-trip to return a copy.
    if all(((tr or {}).get("key") or "none") in ("none", "") for _a, tr in regions):
        if len(regions) != len(sampled):  # the loop's strict zip, which this skips
            raise ValueError(f"{len(regions)} regions against {len(sampled)} sampled")
        return rgb.copy()

    # uint8 throughout. The float32 round-trip this replaces converted a 2.8 MB
    # frame into 11 MB, blended, and converted back — 3.46 ms against 0.45 for
    # the same blend through cv2. Measured difference from that path: at most
    # one level, on 0.0025% of pixels for a single blend and 0.19% for a
    # background plus three feathered objects, where a seam pixel is quantized
    # once per overlapping region.
    out = rgb.copy()

    for (alpha, treatment), samp in zip(regions, sampled, strict=True):
        key = (treatment or {}).get("key") or "none"
        if key in ("none", ""):
            continue  # region kept as-is (out already holds rgb there)
        # build_and_sample_regions pairs a None alpha with an untreated key and
        # nothing else; a treated region without one is a producer bug that would
        # otherwise surface as a TypeError inside boundingRect below.
        assert alpha is not None, f"treatment {key!r} carries no alpha"
        params = (treatment or {}).get("params") or {}
        # Work only where the region actually is. alpha is 0 outside its bounding box, so
        # the blend there is a no-op — but computing it full-frame made a small object cost
        # the same as a full-screen background (a single tinted object measured 19 ms on a
        # 720p frame, nearly all of it spent on pixels the alpha then discarded).
        x, y, bw, bh = cv2.boundingRect((alpha > 0).astype(np.uint8))
        if bw == 0 or bh == 0:
            continue  # empty region: nothing to composite
        sy, sx = slice(y, y + bh), slice(x, x + bw)
        if key in _LOCAL_TREATMENTS:
            # Per-pixel treatments depend on no neighbouring pixel, so cropping first is
            # EXACT. `random` carries a pre-drawn full-frame texture, cropped to match.
            samp_roi = samp or {}
            if key == "random" and isinstance(samp_roi.get("bg"), np.ndarray):
                samp_roi = {**samp_roi, "bg": samp_roi["bg"][sy, sx]}
            treated_roi = _treat(rgb[sy, sx], key, params, samp_roi)
        else:
            # Blur reads neighbours, so it must still see the whole frame; only the BLEND
            # is restricted here.
            treated_roi = _treat(rgb, key, params, samp or {})[sy, sx]
        a = np.ascontiguousarray(alpha[sy, sx], dtype=np.float32)
        # blendLinear computes (src1*w1 + src2*w2)/(w1 + w2 + 1e-5) and
        # saturate-casts; the weights sum to 1 here, so this is
        # out*(1-a) + treated*a in uint8. The epsilon pulls exact .5 results
        # just below the tie, so they round DOWN where the float path's
        # explicit +0.5 rounded up — one level, on tie pixels only, and the
        # dominant term in the measured drift below.
        out[sy, sx] = cv2.blendLinear(
            np.ascontiguousarray(out[sy, sx]), np.ascontiguousarray(treated_roi), 1.0 - a, a
        )
    return out


def build_and_sample_regions(
    masks_by_name: dict,
    obj_treatment_by_name: dict,
    background_treatment: dict | None,
    h: int,
    w: int,
    rng: np.random.Generator,
    cache: dict,
    *,
    feather: int = 5,
) -> tuple[list[tuple[np.ndarray | None, dict]], list[dict]]:
    """Build the ordered ``regions`` + parallel ``sampled`` for :func:`composite_regions`.

    ``masks_by_name`` is ``{object_name: HxW mask}`` from segmentation;
    ``obj_treatment_by_name`` maps each name to its ``{key, params}`` treatment;
    ``background_treatment`` is the background region's treatment. Regions are ordered
    **background first** (alpha = 1 − feathered union of all objects) then each object
    (its own feathered alpha), so objects paint over the background. Where object masks
    OVERLAP, the contested pixels belong to the smallest mask claiming them (the most
    specific object) — never to whichever object happens to be listed later.

    A region whose treatment is ``none`` carries ``None`` for its alpha: it keeps
    its own pixels, so :func:`composite_regions` skips it without looking, and
    computing a feathered alpha for it is pure waste.

    Randomized treatments are drawn via :func:`sample_treatment` and **memoized in
    ``cache``** keyed by region — pass a fresh dict per episode for per-episode
    coherence (or per frame for per-frame). Deterministic treatments get ``{}``.
    """
    all_masks = list(masks_by_name.values())

    # Overlap policy: a contested pixel belongs to the SMALLEST mask claiming it.
    # Deliberate deviation from the panoptic-segmentation standard (confidence-sorted
    # greedy claiming, Kirillov et al. arXiv:1801.00868; per-pixel logit argmax in
    # EfficientPS arXiv:2004.02307): our masks come from INDEPENDENT text prompts, so
    # scores are neither mutually calibrated nor fresh under tracking — and in the
    # measured failure (pick_ball: the confident arm blob swallows the whole ball on
    # every grasp frame) confidence-greedy would hand the ball's pixels to the arm.
    # Smallest-first encodes "the more specific object wins containment", which is the
    # over-segmentation case we actually observe; it is knowingly imperfect at true
    # occlusion seams (confidence is too — see Lazarow et al. CVPR 2020, who learn
    # occlusion order instead). Upgrade path if seams matter: per-pixel logit argmax
    # (needs per-concept logits through the adapter contract). Ties break by name.
    # A region whose treatment is `none` keeps its own pixels, and
    # composite_regions skips it before it ever looks at the alpha. Building
    # that alpha — a feathered full-frame float per object — was the single
    # biggest cost of the common recipe, where every object is `none` and only
    # the background is treated. Decide first, allocate second.
    def _treated(name: str) -> bool:
        return ((obj_treatment_by_name.get(name) or {}).get("key") or "none") not in ("none", "")

    treated_names = [name for name in masks_by_name if _treated(name)]

    # Arbitration exists to decide who owns a contested pixel, which only
    # matters for regions that will actually paint. With nothing to paint there
    # is nothing to arbitrate — and `.sum()` per mask is a full-frame pass.
    exclusive: dict = {}
    if treated_names:
        claimed = np.zeros((h, w), dtype=bool)
        for name, mask in sorted(masks_by_name.items(), key=lambda kv: (int(kv[1].sum()), kv[0])):
            exclusive[name] = mask & ~claimed
            claimed |= mask

    bg_treatment = background_treatment or {"key": "none"}
    bg_treated = ((bg_treatment or {}).get("key") or "none") not in ("none", "")
    regions: list[tuple[np.ndarray | None, dict]] = [
        ((1.0 - feathered_alpha(all_masks, h, w, feather)) if bg_treated else None, bg_treatment)
    ]
    ids: list[str] = ["__bg__"]
    for name in masks_by_name:
        regions.append(
            (
                feathered_alpha([exclusive[name]], h, w, feather) if _treated(name) else None,
                obj_treatment_by_name.get(name) or {"key": "none"},
            )
        )
        ids.append(name)

    sampled: list[dict] = []
    for rid, (_alpha, treatment) in zip(ids, regions, strict=True):
        key = (treatment or {}).get("key") or "none"
        spec = TREATMENTS_BY_KEY.get(key)
        if spec is not None and spec.randomized:
            if rid not in cache:
                cache[rid] = sample_treatment(key, (treatment or {}).get("params") or {}, h, w, rng)
            sampled.append(cache[rid])
        else:
            sampled.append({})
    return regions, sampled


def feathered_alpha(masks: list[np.ndarray], h: int, w: int, feather: int = 5) -> np.ndarray:
    """Union the per-object boolean masks into a soft foreground alpha. A small
    dilation + blur softens the hard SAM edge so the composited seam isn't a
    crisp cut-out (GreenAug shows imperfect masks are fine; this just avoids the
    worst artefacts). Returns HxW float in [0,1]; all-zero (no detection) means
    the whole frame is treated as background.

    Cost is proportional to the mask's bounding box, not the frame: dilation and
    the blur kernel both have finite radius ``feather``, so everything beyond the
    bbox + 2*feather margin is exactly zero — the ROI computation is
    identical to float32 rounding vs full-frame feathering (this runs per region per frame
    per camera in the live preview; full-frame passes dominated its CPU cost)."""
    import cv2

    # decode_mask returns column-major arrays (COCO RLE is column-major). OR-ing
    # them into a row-major accumulator makes numpy walk one operand with a
    # full-column stride: 1.60 ms at 720p against 0.07 when the layouts match.
    # So the accumulator adopts the operands' order, and ONE contiguous copy is
    # made afterwards for cv2, which wants row-major (dilate: 0.09 ms against
    # 0.61). Four copies at decode time were tried and cost more than they saved.
    first = next((m for m in masks if m is not None and m.shape == (h, w)), None)
    order = (
        "F" if first is not None and first.flags["F_CONTIGUOUS"] and not first.flags["C_CONTIGUOUS"] else "C"
    )
    union = np.zeros((h, w), dtype=np.uint8, order=order)
    for m in masks:
        if m is not None and m.shape == (h, w):
            union |= m.view(np.uint8) if m.dtype == np.bool_ else m.astype(np.uint8)
    if order == "F":
        union = np.ascontiguousarray(union)
    # Any nonzero means "in the region", so clamp before the feather multiplies
    # by 255: a mask using cv2's 0/255 convention would otherwise wrap (255*255
    # is 1 in uint8) and collapse the alpha to ~1/255 -- silently untreating the
    # region. The float path this replaced was saved from that by its clip.
    np.minimum(union, 1, out=union)
    if not union.any():
        return np.zeros((h, w), dtype=np.float32)
    if feather <= 0:
        return union.astype(np.float32)
    ksz = feather * 2 + 1
    x, y, bw, bh = cv2.boundingRect(union)
    # The true support is bbox + 2*feather (dilation radius + blur radius). Compute on
    # one extra `feather` of guard band and DISCARD it: the separable blur's border
    # reflection pollutes intermediates within one blur radius of the crop edge, so
    # writing the guard band back would differ from the full-frame result. Sides that
    # sit on the actual frame border keep it — there the reflection matches full-frame.
    wx0, wy0 = max(0, x - 2 * feather), max(0, y - 2 * feather)
    wx1, wy1 = min(w, x + bw + 2 * feather), min(h, y + bh + 2 * feather)
    cx0, cy0 = max(0, wx0 - feather), max(0, wy0 - feather)
    cx1, cy1 = min(w, wx1 + feather), min(h, wy1 + feather)
    roi = cv2.dilate(union[cy0:cy1, cx0:cx1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    # Feather in uint8. The float32 round-trip this replaces cost 0.658 ms
    # against 0.081 for the same blur — the alpha comes from a BINARY mask, so
    # 256 levels of feather is not a meaningful loss of resolution, and the
    # blend it feeds quantizes to uint8 immediately afterwards regardless.
    soft_roi = cv2.GaussianBlur(roi * np.uint8(255), (ksz, ksz), 0).astype(np.float32) / 255.0
    out = np.zeros((h, w), dtype=np.float32)
    out[wy0:wy1, wx0:wx1] = np.clip(soft_roi[wy0 - cy0 : wy1 - cy0, wx0 - cx0 : wx1 - cx0], 0.0, 1.0)
    return out
