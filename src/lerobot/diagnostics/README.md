# SmolVLA replay input capture

Enable `--replay_capture=true` on `lerobot-record`, or select **Save replay inputs (SmolVLA)** in the Policy GUI. Default is off. The first version supports a local SmolVLA checkpoint, standard action queues, no RTC, no compiled sampler, and interpolation multiplier 1. Validation runs before connecting the robot. No training image rebuild or new dependency is required.

## Stored data

Each dataset has an optional `replay_capture/<session UUID>/` directory. Ordinary dataset/video files are unchanged.

- `session.json`: format version, checkpoint path and weight SHA256, effective policy/record configuration, camera order, code identity and numeric runtime settings.
- `checkpoint_config/`: configuration and processor statistics copied once; model weights are referenced, not duplicated. Retain the checkpoint for replay.
- `attempt_NNNNNN/attempt.json`: episode index, saved/discarded/aborted disposition, completeness and counters. Attempt IDs never repeat within a session, including Re-record.
- `attempt_NNNNNN/index.jsonl`: prediction-to-input-frame mapping and measured snapshot cost.
- `attempt_NNNNNN/prediction_NNNNNN.pt`: ordinary tensors and primitive metadata, readable with `torch.load(..., weights_only=True)`.

Prediction tensors are **exact sampler inputs**, after image resize/padding and normalization: `images`, `img_masks`, `lang_tokens`, `lang_masks`, `state`, and the **actual initial `noise`** used by that call. `actions` is the full normalized, padded action chunk before unpadding, postprocessing, robot limits or truncation on early Stop. `capture.frame_index` is the episode-local input frame. Execution of an already queued action does not produce another capture. Final recording may use only part of the final predicted chunk; `recorded_frames` bounds the retained episode.

Images follow `session.json:image_keys`. Original feature dimensions and resize settings are in `policy_config`. SmolVLA pads images on the **left/top**; remove that padding when mapping model-resolution visualization back to original camera frames. The capture contains no rendered heatmaps, gradients, KV cache, or all-layer attention matrices. Those can be computed later using the identified checkpoint and numerical settings.

## Recording lifecycle

A bounded 256 MiB CPU snapshot budget includes queued and in-flight tensor data. GPU-to-CPU copies are synchronous and owned; disk serialization runs on a dedicated thread. Each prediction file is atomically renamed after serialization. There is no per-frame model weight copy or extra sampler/gradient pass.

Episode save and Re-record enqueue disposition markers in order. Discarded attempts remain separately marked for recovery and **must be excluded from ordinary replay**. They never become the next episode's captures. Session UUIDs also separate resumed recording sessions.

Only attempts with `status == complete` and `disposition == saved` belong in normal replay. `recording`, `aborted`, `discarded` or `incomplete` attempts are not silently treated as valid. An interrupted process may leave recoverable complete prediction files inside an unfinished attempt. Background write failures or a full buffer are reported; they never make inference wait for disk capacity. Missing records make an attempt incomplete.

Capture draining occurs after robot disconnect in normal shutdown, with a five-second bound. A timeout is explicitly logged and the session is not marked successfully closed. This preserves control shutdown responsiveness; it cannot guarantee data retention on a failed disk or forced process termination. Filesystem/OS caching remains subject to normal power-loss behavior.

## Offline reconstruction

Load the matching policy from the stored local checkpoint. Move tensor inputs to its device, reproduce runtime precision settings, then call the **model** sampler without the stateful policy action queue:

```python
payload = torch.load(prediction_path, weights_only=True)
# Recursively move tensors in lists/dicts to the model device first.
inputs = {k: v for k, v in payload.items() if k not in ("actions", "capture")}
with torch.inference_mode():
    replayed = policy.model.sample_actions(**inputs)
```

Compare `replayed` with the stored `actions` before adding visualization. Cross-version/device numerical identity is not promised. For image gradients, create fresh image leaves with `requires_grad_(True)` outside inference mode; keep model weights frozen and use a separate offline pass. Never feed replayed actions to a robot from a visualization tool.
