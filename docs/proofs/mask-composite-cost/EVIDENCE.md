<!-- Captured evidence for one change: what was observed, not what was designed.
     Design documents live beside the code they describe. -->

# Evidence: what this change is worth

Dataset `thewisp/instr2_104636` — 4 cameras, 720×1280, one tinted object,
background `random`. Masks cover 7.78% of the frame. Every number below is the
median of independent trials in separate processes; the raw output is next to
this file.

## End-to-end, one frame served

`end-to-end-trials.txt` — five trials per branch, each the median of 40 frames.
Decode is one call for all four cameras; compositing and encoding are per
camera.

| stage                                        | without  | with        |
| -------------------------------------------- | -------- | ----------- |
| decode, all 4 cameras                        | 13.07 ms | 13.46 ms    |
| composite, per camera, as shipped (no cache) | 7.71 ms  | 6.20 ms     |
| composite, per camera, cache passed          | 5.29 ms  | **4.41 ms** |
| JPEG encode, per camera                      | 4.93 ms  | 4.98 ms     |

Shipped today the playback path passes no cache, so the comparison that matters
is `no cache, without` against `cache, with`:

|               | per frame            | fps             |
| ------------- | -------------------- | --------------- |
| 1 camera      | 25.71 → 22.85 ms     | 38.9 → 43.8     |
| **4 cameras** | **63.63 → 51.02 ms** | **15.7 → 19.6** |

**19.8% less work per frame on the four-camera case.**

## Compositing alone

`composite-trials.txt` — seven trials per branch. Isolates the two halves:

|                    | median      | range       |
| ------------------ | ----------- | ----------- |
| shipped (no cache) | 9.03 ms     | 8.76 – 9.30 |
| cache only         | 6.99 ms     | 6.77 – 7.12 |
| code only          | 8.09 ms     | 7.98 – 9.69 |
| both               | **5.76 ms** | 5.56 – 7.31 |

The distributions of "shipped" and "both" do not overlap: the slowest of the
seven improved trials still beats the fastest baseline trial.

## At the real endpoint

Requesting composited frames straight from `/api/datasets/.../frame/N?masks=composited`,
40 fresh frames per branch, one server at a time. The server's own log confirms
what ran: `masks=composited cams=4 composited=4`.

|                  | median    | fastest  |
| ---------------- | --------- | -------- |
| baseline         | 136.77 ms | 74.14 ms |
| with this change | 133.94 ms | 69.74 ms |

**2.8 ms of ~135 ms — about 2%.** Frame-to-frame variance is 70–192 ms, so the
saving is inside the noise: playback stays at roughly 7.5 fps either way.

## Why the function is 36% faster but the frame is 2% faster

Stages of one composited frame, measured with the server's own helpers:

| stage                              | ms               |
| ---------------------------------- | ---------------- |
| decode, all 4 cameras              | 15.53            |
| composite, per camera              | 4.36 (×4 = 17.4) |
| `encode_frame_to_jpeg`, per camera | 4.11 (×4 = 16.4) |

Compositing is ~17 ms of the frame. Cutting it by 36% removes ~6 ms, which is
what the endpoint shows once noise is accounted for.

**These stages sum to ~49 ms, and the endpoint takes ~135 ms.** The missing
~85 ms per frame is not in decode, compositing or JPEG encoding, and nothing
here explains it. That gap — not compositing — is where playback speed is, and
it has not been investigated.
