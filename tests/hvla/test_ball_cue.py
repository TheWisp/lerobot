"""The ball cue, and the promise that it changes nothing when it is off.

The control arm of the experiment is a checkpoint trained BEFORE these flags
existed. That comparison is only valid if a model built with both flags off is
identical to one built without the feature at all -- an unconditional
nn.Linear would silently consume RNG draws and shift every subsequent
initialisation, invalidating the control without failing anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.ball_cue import (
    NOT_VISIBLE,
    ball_cue,
    render_ball_view,
)


def _config(**overrides):
    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config

    base = {
        "action_dim": 4,
        "action_feature_names": [f"j{i}.pos" for i in range(4)],
        "robot_state_feature": True,
        "state_dim": 4,
        "state_feature_names": [f"j{i}.pos" for i in range(4)],
        "image_features": {},
        "image_resize_shape": (224, 224),
        "use_dino_backbone": False,
        "hidden_dim": 32,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 2,
        "dim_feedforward": 32,
        "chunk_size": 2,
    }
    base.update(overrides)
    return FlowMatchingS1Config(**base)


def _params(cfg):
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    torch.manual_seed(0)
    model = FlowMatchingS1Policy(cfg)
    return {n: tuple(p.shape) for n, p in model.named_parameters()}


def test_flags_off_leave_the_model_bit_for_bit_the_same():
    """The control arm depends on this."""
    off = _params(_config())
    explicit_off = _params(_config(ball_token=False, ball_view=False))
    assert off == explicit_off
    assert not any("ball" in n for n in off), "a disabled feature must build no parameters"


def test_enabling_the_token_adds_exactly_one_projection():
    on = _params(_config(ball_token=True))
    added = {n: s for n, s in on.items() if "ball" in n}
    assert added == {"model.ball_proj.weight": (32, 3), "model.ball_proj.bias": (32,)}, added


def test_a_ball_checkpoint_refuses_a_batch_without_the_cue():
    """Silently dropping the cue would train or infer on a different model than
    the one the checkpoint describes; the failure has to be loud."""
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    torch.manual_seed(0)
    model = FlowMatchingS1Policy(_config(ball_token=True))
    batch = {
        "observation.state": torch.zeros(1, 4),
        "action": torch.zeros(1, 2, 4),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    with pytest.raises(KeyError, match="observation.ball"):
        model(batch)


def test_cue_is_normalised_and_resolution_independent():
    small = np.zeros((100, 200), bool)
    small[40:60, 90:110] = True
    big = np.zeros((400, 800), bool)
    big[160:240, 360:440] = True
    xs, ys, vs = ball_cue(small)
    xb, yb, vb = ball_cue(big)
    assert vs == vb == 1.0
    # Not exact, and the bound is not arbitrary: a discrete region's centroid
    # sits half a pixel off centre, which in normalised units is 1/(2*size) and
    # therefore shrinks as resolution grows. Anything larger than that would be
    # a real resolution dependence.
    assert abs(xs - xb) < 0.5 / small.shape[1], "x must not depend on resolution beyond half a pixel"
    assert abs(ys - yb) < 0.5 / small.shape[0], "y must not depend on resolution beyond half a pixel"


def test_a_miss_is_reported_not_guessed():
    assert ball_cue(None) == NOT_VISIBLE
    assert ball_cue(np.zeros((10, 10), bool)) == NOT_VISIBLE
    assert NOT_VISIBLE[2] == 0.0, "visible flag must be the thing a model branches on"
    assert not (0.0 <= NOT_VISIBLE[0] <= 1.0), "sentinel must be outside the valid range"


def test_rendered_view_keeps_only_the_mask():
    frame = np.random.default_rng(0).integers(1, 255, (60, 80, 3), dtype=np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[10:20, 30:40] = True
    view = render_ball_view(frame, mask)
    assert (view[mask] == frame[mask]).all()
    assert (view[~mask] == 0).all()
    assert not render_ball_view(frame, None).any(), "a miss renders black, the in-distribution case"


def test_a_mask_of_the_wrong_size_is_refused():
    frame = np.zeros((60, 80, 3), np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        render_ball_view(frame, np.ones((30, 40), bool))


def test_inference_plumbing_carries_the_cue_end_to_end():
    """The inference path, with fake data: processor output -> batch -> model.

    No SAM3, no checkpoint, no rig. This proves the plumbing -- that an
    observation carrying the cue produces a batch the model accepts, and that
    a rendered view is not mistaken for a missing camera. It says nothing
    about whether the policy is any good; that needs training.
    """
    import torch

    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY
    from lerobot.policies.hvla.s1.flow_matching.model import S2_LATENT_KEY, FlowMatchingS1Policy
    from lerobot.policies.hvla.s1_process import obs_to_s1_batch

    # what BallCueProcessor writes into an observation
    robot_obs = {
        "top_l": np.zeros((48, 64, 3), np.uint8),
        "ball_view": np.zeros((48, 64, 3), np.uint8),
        "ball.x": 0.25,
        "ball.y": 0.75,
        "ball.visible": 1.0,
        **{f"j{i}.pos": 0.1 * i for i in range(4)},
    }
    batch = obs_to_s1_batch(
        robot_obs,
        s1_image_keys=["observation.images.top_l", BALL_VIEW_KEY],
        shared_cache=None,
        s2_latent_key=S2_LATENT_KEY,
        device=torch.device("cpu"),
        joint_names=[f"j{i}.pos" for i in range(4)],
        resize_to=(28, 28),
        state_feature_names=[f"j{i}.pos" for i in range(4)],
        ball_token=True,
    )
    assert "observation.ball" in batch, "the cue must reach the batch"
    assert batch["observation.ball"].tolist() == [[0.25, 0.75, 1.0]]
    assert BALL_VIEW_KEY in batch, "the rendered view must be carried like a camera"

    torch.manual_seed(0)
    model = FlowMatchingS1Policy(_config(ball_token=True))
    model_batch = {
        "observation.state": torch.zeros(1, 4),
        "observation.ball": batch["observation.ball"],
        "action": torch.zeros(1, 2, 4),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    out = model(model_batch)  # must not raise
    assert out is not None


def test_the_view_arm_reports_where_the_ball_actually_is(caplog):
    """The rendered arm adds no parameters, so the log is the only evidence.

    A view that arrives uniformly black — the failure this guards — is
    indistinguishable from a working one in the loss curve.
    """
    import logging

    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    spec = {"shape": (3, 224, 224)}
    cfg = _config(ball_view=True, image_features={"observation.images.top_l": spec, BALL_VIEW_KEY: spec})
    model = FlowMatchingS1Policy(cfg)
    assert model.model._ball_view_idx == 1, "the view must be found at its position in the camera order"

    # A ball rendered a quarter across and three quarters down, on black.
    view = torch.zeros(2, 3, 224, 224)
    view[:, :, 160:180, 46:66] = 0.9
    images = [torch.zeros(2, 3, 224, 224), view]

    with caplog.at_level(logging.INFO):
        model.model._report_ball_view(images)
    line = next(r.getMessage() for r in caplog.records if "Ball view reaching the model" in r.getMessage())

    assert "2/2 rendered" in line
    x_lo = float(line.split("x [")[1].split(",")[0])
    y_lo = float(line.split("y [")[1].split(",")[0])
    assert abs(x_lo - 0.25) < 0.02, f"reported x should track the rendered ball: {line}"
    assert abs(y_lo - 0.76) < 0.02, f"reported y should track the rendered ball: {line}"


def test_an_all_black_view_is_reported_as_nothing_rendered(caplog):
    import logging

    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    spec = {"shape": (3, 224, 224)}
    cfg = _config(ball_view=True, image_features={"observation.images.top_l": spec, BALL_VIEW_KEY: spec})
    model = FlowMatchingS1Policy(cfg)

    with caplog.at_level(logging.INFO):
        model.model._report_ball_view([torch.zeros(2, 3, 224, 224), torch.zeros(2, 3, 224, 224)])
    line = next(r.getMessage() for r in caplog.records if "Ball view reaching the model" in r.getMessage())
    assert "0/2 rendered" in line, f"an empty view must read as empty, not as a working one: {line}"


def test_the_view_arm_refuses_a_config_that_never_delivers_it():
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    cfg = _config(ball_view=True, image_features={"observation.images.top_l": {"shape": (3, 224, 224)}})
    with pytest.raises(ValueError, match="not among the image features"):
        FlowMatchingS1Policy(cfg)


def _warmup_config(**overrides):
    """A checkpoint-shaped config for the warmup path."""
    spec = {"dtype": "video", "shape": [3, 224, 224], "names": ["channels", "height", "width"]}
    base = {"image_features": {"observation.images.top_l": spec}, "use_dino_backbone": False}
    base.update(overrides)
    return _config(**base)


@pytest.mark.parametrize(
    ("flags", "label"),
    [
        ({"ball_token": True, "ball_source": "observation.images.top_l"}, "token"),
        ({"ball_view": True, "ball_source": "observation.images.top_l"}, "view"),
    ],
)
def test_warmup_builds_a_batch_the_checkpoint_accepts(flags, label):
    """The warmup runs before any observation, so no processor has run yet.

    It hand-builds a batch that has to satisfy the same model the control loop
    will. When it did not, a ball checkpoint died at startup on its own guard --
    after loading, compiling and warming up, so the failure looked like a model
    problem rather than a missing key.
    """
    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
    from lerobot.policies.hvla.s1_process import _warmup_s1

    spec = {"dtype": "video", "shape": [3, 224, 224], "names": ["channels", "height", "width"]}
    features = {"observation.images.top_l": spec}
    if flags.get("ball_view"):
        features[BALL_VIEW_KEY] = spec
    cfg = _warmup_config(image_features=features, **flags)
    policy = FlowMatchingS1Policy(cfg)

    # No exception is the assertion: this is the exact startup path that failed.
    _warmup_s1(
        policy,
        lambda b: b,
        list(cfg.image_features),
        torch.device("cpu"),
        (224, 224),
        state_dim=cfg.state_dim,
        use_s2=False,
    )


@pytest.mark.parametrize(
    ("flags", "label"),
    [({"ball_token": True}, "token"), ({"ball_view": True}, "view")],
)
def test_a_saved_checkpoint_starts_up_and_acts_on_a_segmented_frame(flags, label, tmp_path, monkeypatch):
    """Save, load, warm up, segment, act -- the sequence the rig runs.

    Three defects reached the rig because each earlier test hand-wrote the
    inputs to the one function it was checking, so every boundary between
    those functions was unguarded: the checkpoint writer dropped the flags,
    the warmup built a batch the model refused, and the processor was not
    installed. This walks the real artifacts in the real order instead, with
    only the segmenter replaced -- SAM3 is the one part that needs a GPU and a
    model, and what it returns is a mask either way.
    """
    import json

    from safetensors.torch import load_file, save_file

    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY
    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
    from lerobot.policies.hvla.s1.flow_matching.model import S2_LATENT_KEY, FlowMatchingS1Policy
    from lerobot.policies.hvla.s1.flow_matching.train import checkpoint_config_dict
    from lerobot.policies.hvla.s1_process import _warmup_s1, obs_to_s1_batch

    source = "observation.images.top_l"
    spec = {"dtype": "video", "shape": [3, 224, 224], "names": ["channels", "height", "width"]}
    features = {source: spec}
    if flags.get("ball_view"):
        features[BALL_VIEW_KEY] = spec
    trained = _warmup_config(image_features=features, ball_source=source, **flags)

    # 1. write the checkpoint through the real writer
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config_dict(trained)))
    save_file(FlowMatchingS1Policy(trained).state_dict(), str(tmp_path / "model.safetensors"))

    # 2. read it back through the real reader; the weights must all land
    loaded = FlowMatchingS1Config.from_checkpoint_dict(json.loads((tmp_path / "config.json").read_text()))
    policy = FlowMatchingS1Policy(loaded)
    missing, unexpected = policy.load_state_dict(load_file(str(tmp_path / "model.safetensors")), strict=False)
    assert not missing and not unexpected, (list(missing), list(unexpected))
    assert loaded.ball_token == trained.ball_token and loaded.ball_view == trained.ball_view

    # 3. the startup warmup, before any observation exists
    _warmup_s1(
        policy,
        lambda b: b,
        list(loaded.image_features),
        torch.device("cpu"),
        (224, 224),
        state_dim=loaded.state_dim,
        use_s2=False,
    )

    # 4. a frame through the REAL processor, with only the segmenter replaced
    ball = np.zeros((48, 64), bool)
    ball[30:38, 12:20] = True

    class _StubAdapter:
        def set_control(self, _):
            pass

        def set_camera(self, _):
            pass

        def segment(self, _frame):
            return {"yellow ball": ball}

    monkeypatch.setattr("lerobot.overlays.adapters.build_adapter", lambda *a, **k: _StubAdapter())
    from lerobot.policies.hvla.s1.flow_matching.ball_processor import BallCueProcessor

    processor = BallCueProcessor(loaded.ball_source, want_view=loaded.ball_view, device="cpu")
    observation = {
        "top_l": np.full((48, 64, 3), 200, np.uint8),
        **{f"j{i}.pos": 0.1 * i for i in range(4)},
    }
    # Dispatched exactly as run_s1 does. Calling the processor any other way is
    # how a step that the loop could not call shipped and reached the rig.
    for step in [processor]:
        observation = step.observation(observation)

    # 5. the processor's own output into the batch, and the batch into the model
    batch = obs_to_s1_batch(
        observation,
        s1_image_keys=list(loaded.image_features),
        shared_cache=None,
        s2_latent_key=S2_LATENT_KEY,
        device=torch.device("cpu"),
        joint_names=[f"j{i}.pos" for i in range(4)],
        resize_to=(224, 224),
        state_feature_names=[f"j{i}.pos" for i in range(4)],
        ball_token=loaded.ball_token,
    )
    actions = policy.predict_action_chunk(batch)
    assert actions.shape[0] == 1 and actions.shape[-1] == loaded.action_dim, actions.shape

    # the cue that arrived is the stub's blob, not a default
    if loaded.ball_token:
        x, y, vis = batch["observation.ball"][0].tolist()
        assert vis == 1.0 and abs(x - 0.25) < 0.02 and abs(y - 0.70) < 0.02, (x, y, vis)
    if loaded.ball_view:
        assert batch[BALL_VIEW_KEY].max() > 0, "the rendered view reached the batch all black"


class _StubAdapter:
    """SAM3 stands in for a GPU and a model; what it returns is a mask either way."""

    mask = None

    def set_control(self, _):
        pass

    def set_camera(self, _):
        pass

    def segment(self, _frame):
        return {"yellow ball": _StubAdapter.mask}


def test_a_ball_checkpoint_gets_a_segmenter_and_a_plain_one_does_not(monkeypatch):
    """The installation decision, which nothing could reach while it sat in run_s1.

    A ball checkpoint that starts without its segmenter produces the sentinel
    on every frame -- a policy that reads as merely bad rather than unwired.
    """
    from lerobot.policies.hvla.s1.flow_matching.ball_processor import BallCueProcessor
    from lerobot.policies.hvla.s1_process import ball_processor_for

    monkeypatch.setattr("lerobot.overlays.adapters.build_adapter", lambda *a, **k: _StubAdapter())

    assert ball_processor_for(_config(), "cpu") is None, "a plain checkpoint must not segment"

    for flags in ({"ball_token": True}, {"ball_view": True}, {"ball_token": True, "ball_view": True}):
        step = ball_processor_for(_config(ball_source="observation.images.top_l", **flags), "cpu")
        assert isinstance(step, BallCueProcessor), flags
        assert step.want_view == flags.get("ball_view", False), flags


def test_a_cue_checkpoint_without_a_source_is_refused():
    from lerobot.policies.hvla.s1_process import ball_processor_for

    with pytest.raises(ValueError, match="no ball_source"):
        ball_processor_for(_config(ball_token=True), "cpu")


def test_the_cue_step_exposes_what_the_loop_actually_calls(monkeypatch):
    """Read the required method off run_s1's source, not off the class.

    The step shipped implementing ``__call__`` while the loop dispatches
    ``step.observation(obs)``, so it raised AttributeError on the rig after
    everything else had been verified. Every test that exercised the processor
    called it the way it was written rather than the way it is used, which
    could never have caught that -- so this derives the name from the caller.
    """
    import inspect
    import re

    from lerobot.policies.hvla.s1_process import ball_processor_for, run_s1

    called = set(re.findall(r"\bstep\.(\w+)\(", inspect.getsource(run_s1)))
    assert called, "no step dispatch found in run_s1; this test's premise is stale"

    monkeypatch.setattr("lerobot.overlays.adapters.build_adapter", lambda *a, **k: _StubAdapter())
    step = ball_processor_for(_config(ball_token=True, ball_source="observation.images.top_l"), "cpu")
    missing = [name for name in called if not callable(getattr(step, name, None))]
    assert not missing, f"the loop calls step.{missing} and the cue step does not provide it"


def test_recording_ignores_a_derived_view_the_robot_never_produced(tmp_path):
    """A cue view in the observation must not enter the eval recording.

    The recording dataset declares the ROBOT's cameras. add_frame rejects a
    frame carrying a feature the schema lacks, so the rendered view -- injected
    into every observation by the cue processor -- crashes the recording on its
    first frame. It is also derived from a camera that IS recorded, and keeping
    it would give the image arm a different eval schema from the token arm and
    the baseline.
    """
    from lerobot.policies.hvla.s1_process import _add_frame_to_dataset

    declared = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.images.top_l": {"dtype": "video", "shape": (48, 64, 3)},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }

    class _Meta:
        features = declared

    class _Dataset:
        meta = _Meta()

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            extra = set(frame) - set(declared) - {"task"}
            if extra:  # what LeRobotDataset.validate_frame does
                raise ValueError(f"Extra features: {sorted(extra)}")
            self.frames.append(frame)

    ds = _Dataset()
    obs = {
        "top_l": np.zeros((48, 64, 3), np.uint8),
        "ball_view": np.zeros((48, 64, 3), np.uint8),  # the processor's derived image
        "a": 0.0,
        "b": 1.0,
    }
    _add_frame_to_dataset(ds, obs, np.zeros(2, np.float32), ["a", "b"], "task")

    assert len(ds.frames) == 1
    assert "observation.images.top_l" in ds.frames[0], "the real camera must still be recorded"
    assert "observation.images.ball_view" not in ds.frames[0]


def test_the_control_loop_reads_the_robot_in_exactly_one_place():
    """Every published observation must have been through the processors.

    The steps ADD policy inputs, so an observation that skips them is missing
    inputs the checkpoint declares. run_s1 had three publish sites; two applied
    the steps and one published the raw initial observation, which reached the
    inference thread as a KeyError naming a camera the robot never had. Two
    correct sites are what made the third invisible.

    Pinning the count rather than the shape: a fourth site added later cannot
    quietly diverge, because adding one means either routing through observe()
    or failing here.
    """
    import inspect

    from lerobot.policies.hvla.s1_process import run_s1

    src = inspect.getsource(run_s1)
    reads = src.count("robot.get_observation()")
    assert reads == 1, (
        f"run_s1 reads the robot in {reads} places; every read must go through observe(), "
        "which applies the observation processors"
    )

    body = src[src.index("def observe(") :].split("return obs")[0]
    assert "step.observation(obs)" in body, "the single read does not apply the processors"


def test_the_segmenter_runs_on_the_inference_thread_not_the_control_loop():
    """SAM3 must not sit in the observation path.

    The cue is a policy input, consumed once per inference (~1.6 s), not once
    per control step. Segmenting it inline took the observation step from
    2.2 ms to 37 ms against a 33 ms budget at 30 Hz, so the loop overran on
    every iteration; and because the executed chunk index is derived from wall
    clock, an overrunning loop skips chunk entries and the arm's motion is
    amplified and made irregular. The cost is invisible in any unit test, so
    the placement is pinned here instead.
    """
    import inspect

    from lerobot.policies.hvla.s1_inference import InferenceThread
    from lerobot.policies.hvla.s1_process import run_s1

    src = inspect.getsource(run_s1)
    assert "ball_processor_for(" in src, "run_s1 no longer builds the cue step; this test is stale"
    assert "obs_processor_steps.append(_ball_step)" not in src, (
        "the segmenter is back in the observation path, which runs it at the control rate"
    )
    assert "ball_step=_ball_step" in src, "the cue step is built but never handed to the inference thread"

    loop = inspect.getsource(InferenceThread._loop)
    assert "_ball_step.observation(obs)" in loop, (
        "the inference thread does not apply the cue, so a ball checkpoint would get no cue at all"
    )
    assert loop.index("_ball_step.observation(obs)") < loop.index("obs_to_s1_batch("), (
        "the cue must be applied before the batch is built from the observation"
    )
    assert "ball_step" in inspect.signature(InferenceThread.__init__).parameters


def _aux_config(**overrides):
    from lerobot.policies.hvla.s1.flow_matching.ball_cue import BALL_VIEW_KEY  # noqa: F401

    spec = {"dtype": "video", "shape": [3, 224, 224], "names": ["channels", "height", "width"]}
    base = {
        "image_features": {"observation.images.top_l": spec, "observation.images.wrist": spec},
        "use_dino_backbone": False,
        "ball_aux": True,
        "ball_source": "observation.images.top_l",
    }
    base.update(overrides)
    return _config(**base)


def test_the_auxiliary_head_reads_the_source_camera_and_scores_its_own_patches():
    """It must read the cue camera's block, not another camera's or the state.

    The arm moves toward the ball, so the state token predicts ball position; a
    head that could see it would satisfy the loss through that shortcut and
    leave the visual features -- the thing this exists to change -- untouched.
    """
    from lerobot.policies.hvla.s1.flow_matching.model import OBS_BALL, FlowMatchingS1Policy

    cfg = _aux_config()
    model = FlowMatchingS1Policy(cfg).model
    assert model.ball_aux_head is not None
    assert model.ball_aux_head.in_features == cfg.hidden_dim
    assert model.ball_aux_head.out_features == 1, "one score per patch, not a coordinate regressor"

    n_patch, b = 256, 4
    model._ctx_layout = {"n_cams": 2, "patches_per_cam": n_patch}
    ctx = torch.zeros(b, 2 * n_patch + 1, cfg.hidden_dim)
    # put a distinctive signal in the SECOND camera's block; the head must ignore it
    ctx[:, n_patch : 2 * n_patch] = 5.0
    batch = {OBS_BALL: torch.tensor([[0.25, 0.75, 1.0]] * b)}
    loss_a, stats = model.ball_aux_loss(ctx, batch)

    ctx2 = ctx.clone()
    ctx2[:, n_patch : 2 * n_patch] = -5.0  # change only the other camera
    loss_b, _ = model.ball_aux_loss(ctx2, batch)
    assert torch.allclose(loss_a, loss_b), "the head is reading a camera it should not"
    assert stats["ball_aux_seen"] == b


def test_a_frame_with_no_detection_is_excluded_rather_than_regressed():
    """Regressing a miss to the sentinel teaches the head to point at a corner."""
    from lerobot.policies.hvla.s1.flow_matching.model import OBS_BALL, FlowMatchingS1Policy

    cfg = _aux_config()
    model = FlowMatchingS1Policy(cfg).model
    n_patch = 256
    model._ctx_layout = {"n_cams": 2, "patches_per_cam": n_patch}
    ctx = torch.randn(3, 2 * n_patch + 1, cfg.hidden_dim)
    seen_only = torch.tensor([[0.4, 0.6, 1.0], [0.4, 0.6, 1.0], [0.4, 0.6, 1.0]])
    with_miss = torch.tensor([[0.4, 0.6, 1.0], [0.4, 0.6, 1.0], [-1.0, -1.0, 0.0]])

    torch.manual_seed(0)
    a, sa = model.ball_aux_loss(ctx, {OBS_BALL: seen_only})
    b, sb = model.ball_aux_loss(ctx[:2], {OBS_BALL: seen_only[:2]})
    c, sc = model.ball_aux_loss(ctx, {OBS_BALL: with_miss})
    assert sc["ball_aux_seen"] == 2 and sa["ball_aux_seen"] == 3
    assert torch.allclose(c, b), "the miss changed the loss, so it was not excluded"

    all_miss = torch.tensor([[-1.0, -1.0, 0.0]] * 3)
    z, sz = model.ball_aux_loss(ctx, {OBS_BALL: all_miss})
    assert float(z) == 0.0 and sz["ball_aux_seen"] == 0


def test_the_auxiliary_target_is_required_and_says_so():
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

    model = FlowMatchingS1Policy(_aux_config()).model
    model._ctx_layout = {"n_cams": 2, "patches_per_cam": 256}
    with pytest.raises(KeyError, match="as a TARGET"):
        model.ball_aux_loss(torch.zeros(2, 513, model.config.hidden_dim), {})


def test_cue_dropout_replaces_the_right_share_and_only_while_training():
    """Inference must always see the real cue; only training drops it.

    The sentinel is what the policy sees on a genuine miss, so dropping to it
    stays in distribution -- but dropping at inference would throw away the cue
    the run is meant to test.
    """
    from lerobot.policies.hvla.s1.flow_matching.ball_cue import NOT_VISIBLE
    from lerobot.policies.hvla.s1.flow_matching.model import OBS_BALL, FlowMatchingS1Policy

    cfg = _config(ball_token=True, ball_source="observation.images.top_l", ball_token_dropout=0.5)
    model = FlowMatchingS1Policy(cfg).model
    cue = torch.tensor([[0.4, 0.6, 1.0]] * 400)
    batch = {OBS_BALL: cue, "observation.state": torch.zeros(400, cfg.state_dim)}

    torch.manual_seed(0)
    model.train()
    model.encode_observations(dict(batch))
    rate = model._dropped / model._drop_seen
    assert 0.42 < rate < 0.58, f"realised dropout {rate:.3f} is not the requested 0.5"

    model.eval()
    before = model._dropped
    model.encode_observations(dict(batch))
    assert model._dropped == before, "the cue was dropped at inference"

    # and the caller's tensor is never mutated in place
    assert torch.equal(cue, torch.tensor([[0.4, 0.6, 1.0]] * 400))

    off = FlowMatchingS1Policy(_config(ball_token=True, ball_source="x")).model
    off.train()
    off.encode_observations({OBS_BALL: cue.clone(), "observation.state": torch.zeros(400, cfg.state_dim)})
    assert off._drop_seen == 0, "dropout ran with the feature off"
    assert NOT_VISIBLE == (-1.0, -1.0, 0.0)
