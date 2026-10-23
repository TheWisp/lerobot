# Frozen internal holdout before tracker evaluation

These three YCBInEOAT sequences were unused by the six-clip development pilot. The clip for each is exactly 31 frames centered on the sequence midpoint (`first = floor(frame_count / 2) - 15`). No tracking result was inspected when selecting the intervals. This is an **internal sequence holdout from the same public dataset**, not an independent-source or blind leaderboard test.

The reference builder uses 64 spatially spread, depth-backed pixels inside the source object's first-frame ground-truth mask. It transports each source-depth point through the dataset's annotated object poses. A reference point is eligible only where the later ground-truth mask and measured depth support it within the declared 10 mm visibility threshold. Missing predictions remain failures. The pose and masks are never passed to the point model after seeding; the mask used to select the initial pixels is an experimental oracle, not a live segmentation result.

| Sequence                      | Source frames | Query file SHA-256                                                 | First source RGB SHA-256                                           |
| ----------------------------- | ------------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `bleach_hard_00_03_chaitanya` |       205–235 | `ad14da0beb1ecd55eaf9bcf5a821d5054893d05483517edab59ee6dd1ef309e8` | `f28e293bf9b157fdc7489ab4eba572a0941b5c9c17a10ba350d1c4904a648a1a` |
| `cracker_box_yalehand0`       |       648–678 | `3775f4d8c203ddacac7db4cb892b990abb3491931013558f5377e68a7767ff2e` | `db7dbcf739e07e6ed800858f7b81863bc969ea626654181cd26c25096357abdd` |
| `sugar_box_yalehand0`         |       486–516 | `efb9e695e8fbd27e3072c1e38c7a183c645026ac2a748f3bbffcad5eb63cfa3b` | `9fe79f900d98fa1882c9a09ca5913edef414731e1f7289cb7bf3f4efe612330a` |

Run all available raw point adapters once with their development-pilot settings, then report the three clips separately with no setting changes. CoTracker3, TrackEverything and TAPIP3D are full-clip methods and must not be ranked as causal 30 Hz alternatives. The reference's depth disagreement and eligible fraction must accompany any accuracy summary. No candidate is selected from three correlated sequences.
