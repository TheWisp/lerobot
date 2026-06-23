# SAM3-Video live streaming: unbounded GPU memory + how to track one object — expert request

## The ask

We run `Sam3VideoModel` (transformers) in an **indefinite live loop** (robot camera, one
frame at a time) to track a single known object. Two coupled problems we think stem from
**using the API wrong**, not from the model:

1. The streaming session's GPU memory grows **without bound** (~6.8 MB/frame, linear, no
   plateau) → OOM in minutes.
2. Every workaround we tried to bound it, or to keep a single stable track, is a hack that
   breaks tracking.

We want the **supported** way to do bounded, indefinite, single-object live tracking.

## Setup

- transformers **5.3.0**, model **`facebook/sam3`** (SAM 3.0), torch 2.10, CUDA, RTX 5090 (32 GB).
- Source: `Sam3VideoModel` / `Sam3VideoProcessor` (`transformers/models/sam3_video/`).
- Use case: top-down robot camera, 30 fps; our inference runs ~10–15 fps, so the model sees
  every 2nd–3rd frame. Goal: track **one** object (a green ring) persistently through
  gripper occlusion, to drive a downstream 6-DoF pose overlay.

### Exactly what we run (the streaming loop)

```python
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.float16).to("cuda").eval()

session = processor.init_video_session(inference_device="cuda", dtype=torch.float16)
processor.add_text_prompt(session, ["green ring"])

# then, forever, one live frame at a time:
inp = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
pv = inp["pixel_values"][0].to("cuda", torch.float16)
with torch.inference_mode():
    out = model(inference_session=session, frame=pv)
res = processor.postprocess_outputs(session, out, original_sizes=[[h, w]])
# res: object_ids, scores, boxes, masks, prompt_to_obj_ids
```

## What works

On **short** clips (a few seconds) this tracks beautifully — temporally consistent masks.
The problems only appear over a **long** stream.

## Problem 1 — unbounded GPU memory

The session keeps per-frame state forever (`session.output_dict_per_obj[obj]["non_cond_frame_outputs"][frame_idx]`
and `session.processed_frames`). Measured `torch.cuda.memory_allocated()` over a looped clip:

| frames (stride 1) | GPU, `inference_state_device=cuda` | GPU, `inference_state_device="cpu"` |
| ----------------- | ---------------------------------- | ----------------------------------- |
| 150               | —                                  | 2.81 GB                             |
| 385               | 1.85 → **5.22 GB** (+3.37)         | —                                   |
| 450               | —                                  | 4.85 GB                             |
| 750               | —                                  | 6.89 GB                             |
| 1050              | —                                  | **8.92 GB**                         |

→ **Linear ~6.8 MB/frame, no plateau, with or without CPU state offload.** `init_video_session`
exposes `inference_state_device`, `processing_device`, `video_storage_device`,
`max_vision_features_cache_size` (default 1) — we tried `inference_state_device="cpu"`; it
shifts ~20% off-GPU but the GPU curve is still linear.

## Problem 2 — our memory workaround breaks tracking

To bound memory we deleted old entries from `non_cond_frame_outputs` / `processed_frames`,
keeping only the last N (cond/prompt frames preserved). **This breaks the detector→track
association**: detections stop matching the existing tracklet and get re-added as _new_
objects. Measured on the same motion clip at 30 fps:

|                            | distinct instance IDs over 13 s | GPU growth            |
| -------------------------- | ------------------------------- | --------------------- |
| **prune (keep 32 frames)** | **4** (1→2→3→4, accumulating)   | +0.3 GB (bounded ✓)   |
| **no prune**               | **1** (stable)                  | +3.4 GB (unbounded ✗) |

Live, with the prune, the track also goes **stale after ~10 s and never recovers** (the real
object becomes a phantom ID, then the keep-alive logic drops it). So: pruning bounds memory
but destroys tracking; not pruning tracks correctly but OOMs.

(We also tried an output-side "lock to one instance per concept" filter on top of the prune.
It hid the extra IDs for a while — 0 ID-switches offline — but live it still went stale,
because it masks the symptom while the prune keeps corrupting the underlying track.)

## What we found but couldn't make work

`Sam3VideoConfig` / `Sam3VideoModel` expose, among others: `score_threshold_detection=0.5`,
`new_det_thresh`, `det_nms_thresh=0.1`, `max_num_objects`, `init/max/min_trk_keep_alive=30/30/-1`,
`suppress_overlapping_based_on_recent_occlusion_threshold=0.7`, `recondition_every_nth_frame`.
Setting `max_num_objects=1` does cap instances, but it's a coarse global cap and doesn't
touch the memory growth (which is per-frame, independent of object count).

## Current state of our code

Reverted to the **honest unbounded baseline** (no prune, no lock): tracks correctly, OOMs
over time. We'd rather run it correctly than ship the hack.

## Questions for an expert

1. **What is the supported way to run `Sam3VideoModel` in an indefinite live loop without
   unbounded GPU growth?** Is there an official rolling-window / eviction for the session's
   per-frame memory, or a config that caps it? How should `inference_state_device` /
   `processing_device` / `video_storage_device` / `max_vision_features_cache_size` be set
   for live streaming (vs the finite-video default)?
2. **Is manual deletion of old `non_cond_frame_outputs` simply the wrong approach?** Is there
   a method that evicts old frames while preserving the detector→track association state?
3. **For persistent single-object tracking, is `Sam3VideoModel` (concept/text) the right tool
   at all, or should we use `Sam3TrackerVideoModel` (geometric, seed-once-and-propagate)?**
   The concept model keeps detecting new matches over time — is that expected, and is the
   geometric tracker the intended path for "lock onto one object and track through occlusion"?
4. **Are `score_threshold_detection` / `new_det_thresh` / `max_num_objects` / `trk_keep_alive`
   the intended knobs to stop instance proliferation in streaming?** Recommended values for a
   single known object?
5. **Is there a reference example of indefinite live (webcam/robot) streaming with
   `sam3_video`** that handles unbounded duration?

## Minimal repro

Run the streaming loop above on any video looped a few times; log
`torch.cuda.memory_allocated()` every ~150 frames → linear growth. Add the prune (delete
`non_cond_frame_outputs` beyond the last 32) → memory flattens but `len(res["object_ids"])`
climbs frame over frame on a moving scene.
