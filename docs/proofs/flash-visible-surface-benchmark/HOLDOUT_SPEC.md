# Frozen internal holdout before tracker evaluation

These three YCBInEOAT sequences were unused by the six-clip development pilot. The clip for each is exactly 31 frames centered on the sequence midpoint (`first = floor(frame_count / 2) - 15`). No tracking result was inspected when selecting the intervals. This is an **internal sequence holdout from the same public dataset**, not an independent-source or blind leaderboard test.

The reference builder uses 64 spatially spread, depth-backed pixels inside the source object's first-frame ground-truth mask. It transports each source-depth point through the dataset's annotated object poses. A reference point is eligible only where the later ground-truth mask and measured depth support it within the declared 10 mm visibility threshold. Missing predictions remain failures. The pose and masks are never passed to the point model after seeding; the mask used to select the initial pixels is an experimental oracle, not a live segmentation result.

| Sequence                      | Source frames | Query file SHA-256                                                 | First source RGB SHA-256                                           |
| ----------------------------- | ------------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `bleach_hard_00_03_chaitanya` |       205–235 | `ad14da0beb1ecd55eaf9bcf5a821d5054893d05483517edab59ee6dd1ef309e8` | `f28e293bf9b157fdc7489ab4eba572a0941b5c9c17a10ba350d1c4904a648a1a` |
| `cracker_box_yalehand0`       |       648–678 | `3775f4d8c203ddacac7db4cb892b990abb3491931013558f5377e68a7767ff2e` | `db7dbcf739e07e6ed800858f7b81863bc969ea626654181cd26c25096357abdd` |
| `sugar_box_yalehand0`         |       486–516 | `efb9e695e8fbd27e3072c1e38c7a183c645026ac2a748f3bbffcad5eb63cfa3b` | `9fe79f900d98fa1882c9a09ca5913edef414731e1f7289cb7bf3f4efe612330a` |

Run all available raw point adapters once with their development-pilot settings, then report the three clips separately with no setting changes. CoTracker3, TrackEverything and TAPIP3D are full-clip methods and must not be ranked as causal 30 Hz alternatives. The reference's depth disagreement and eligible fraction must accompany any accuracy summary. No candidate is selected from three correlated sequences.

## Frozen motion-stratified companion

The midpoint rule can select nearly static portions of a sequence. To exercise visible-surface tracking under substantial motion, a second set is frozen **before scoring its tracker outputs**. For each same unused sequence, evaluate every 31-frame window whose every frame has at least 1,000 pixels inside both the annotated object mask and valid depth. Choose the window with the largest maximum translation of the annotated object origin from its first frame; ties use the earliest start. The 1,000-pixel condition keeps a moving object visibly present throughout. The translation is a selection variable only; the full pose annotation still defines the labels. No model score or prediction affects selection. This is motion-stratified test design within the same dataset, and its scores remain separate from the midpoint holdout.

| Sequence                      | Source frames | Max translation | Query file SHA-256                                                 | First source RGB SHA-256                                           |
| ----------------------------- | ------------: | --------------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `bleach_hard_00_03_chaitanya` |         38–68 |          423 mm | `ec4e8a0a7031a6ed371349344e83950053e03f4f9306442983375a6b4f592f7d` | `a08b55d99833aee09f5f7ea363120322ec22faae6ddc0d34c594c034f26fb107` |
| `cracker_box_yalehand0`       |     1253–1283 |          799 mm | `42e6ae93c9143ee10e6e627e0d9a3df4c890fd9a2c02e10540c8fb08fa30ced5` | `c7af85a468fb1ea0c578fe42bb7929e8ed34686c45f0bf15aa61a3683f431783` |
| `sugar_box_yalehand0`         |       619–649 |          172 mm | `a5db7b7e2059d30ed0aa9b1b140dc5e147a8e637705a9db9c5c516c39b384b2f` | `79c5ff679eaf0de1ef7c2a1e60e5765f212c012c45807ae122b2ff0ff7676fba` |
