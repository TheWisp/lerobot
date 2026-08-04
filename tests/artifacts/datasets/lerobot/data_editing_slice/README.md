# `data_editing_slice` — a committed, photorealistic fixture

A 2-episode x 12-frame slice of the public
[`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
dataset (Apache-2.0, same project as this repo), stored video-backed exactly as a
real recording would be.

It exists so `tests/gui/test_process_realistic_slice.py` can run the data-editing
job over **real imagery with the real SAM3 model** without a network fetch, an HF
cache dependency, or re-encoding the video on every run. Committing it also makes
the run deterministic: the same pixels every time, so the measured "fraction of
the frame preserved" is a stable baseline rather than something that drifts with a
re-encode.

Regenerate (only if the fixture must change) by slicing the public source with
`LeRobotDataset.create(..., use_videos=True)` and copying the first 12 frames of
episodes 0 and 1 for `observation.images.top`.
