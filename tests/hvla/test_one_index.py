"""The one-index-per-chunk invariant: shift travels with its chunk everywhere.

Two regressions pinned here:
  - reset_episode_state must clear the stitch shift with the rest of the chunk
    state, so no stale shift can survive into the next episode.
  - the executor must be able to read (chunk, shift) in one lock acquisition;
    reading them separately can pair the previous chunk with the next chunk's
    shift for one control step.
"""

import numpy as np
import torch

from .test_inference_thread import _SO107_JOINTS, MockRTCPolicy, MockSharedCache
from lerobot.policies.hvla.s1_inference import InferenceThread


def _thread(**kw):
    d = {
        "policy": MockRTCPolicy(),
        "preprocessor": lambda b: b,
        "postprocessor": lambda a: a,
        "shared_cache": MockSharedCache(),
        "s2_latent_key": "observation.s2_latent",  # gitleaks:allow
        "s1_image_keys": ["observation.images.front"],
        "joint_names": list(_SO107_JOINTS),
        "device": torch.device("cpu"),
        "resize_to": None,
        "fps": 30,
    }
    d.update(kw)
    return InferenceThread(**d)


def test_reset_clears_the_stitch_shift():
    t = _thread(rtc_stitch_search=12)
    with t._chunk_lock:
        t._chunk_stitch_offset = 7  # as if a stitched chunk had been published
    t.reset_episode_state()
    assert t.stitch_offset == 0, "a stale shift survived the episode reset"


def test_chunk_and_shift_are_read_in_one_acquisition():
    t = _thread(rtc_stitch_search=12)
    chunk = np.zeros((50, 14), dtype=np.float32)
    with t._chunk_lock:
        t._chunk_data = chunk
        t._chunk_t_origin = 123.0
        t._chunk_t_obs = 123.0
        t._chunk_stitch_offset = 5
    c, origin, t_obs, shift = t.get_chunk_with_offset()
    assert c is chunk
    assert origin == 123.0 and t_obs == 123.0
    assert shift == 5, "the shift returned must be the one published with this chunk"


def test_combined_accessor_defaults_match_disabled_state():
    t = _thread()  # no stitching configured
    c, _o, _t, shift = t.get_chunk_with_offset()
    assert c is None
    assert shift == 0
