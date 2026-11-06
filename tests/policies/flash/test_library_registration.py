"""Exercise optional fitters against known rigid motion and real geometry guards."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_library_registration import LibraryConfig, LibraryTrial, measured_colors
from benchmarks.flash_partial_surface_control import box_observation, perturbed_pose, pose_error
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("backend", ["small_gicp", "open3d_plane", "open3d_color"])
def test_configured_library_handles_measured_motion_and_preserves_depth_gate(backend):
    from dataclasses import replace

    from lerobot.policies.flash.surface_registration import LibraryConfig

    pytest.importorskip("small_gicp" if backend == "small_gicp" else "open3d")
    source, current, k, truth = textured_box_pair()
    shape = replace(ShapeConfig(), registration=LibraryConfig(backend))
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, shape, reference_observation=source)
    pose, detail = fit.correct(perturbed_pose(truth))
    assert pose is not None
    assert detail["library"] == backend
    assert pose_error(fit.reference, pose, truth)["median_point_error_mm"] < 8
    wrong = truth.copy()
    wrong[2, 3] -= 0.05
    assert not fit.check_pose(wrong)["accepted"]


def test_registration_config_roundtrip_and_validation():
    from dataclasses import asdict

    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    data = asdict(VisionPipelineConfig())
    data["shape"]["registration"]["backend"] = "open3d_color"
    data["shape"]["registration"]["voxel_sizes_m"] = [0.004, 0.002]
    restored = VisionPipelineConfig.from_dict(data)
    assert restored.shape.registration.backend == "open3d_color"
    assert restored.shape.registration.voxel_sizes_m == (0.004, 0.002)
    for change in (
        {"backend": "typo"},
        {"threads": 1.5},
        {"appearance_weight": float("nan")},
        {"voxel_sizes_m": [0.0]},
        {"iterations_per_level": True},
    ):
        with pytest.raises(ValueError):
            LibraryConfig(**change)


def test_native_iterates_share_queries_without_changing_scores_or_geometry():
    from dataclasses import replace
    from types import SimpleNamespace

    pytest.importorskip("small_gicp")
    source, current, k, truth = textured_box_pair()
    config = replace(ShapeConfig(), registration=LibraryConfig("small_gicp", threads=1))
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(reference, current, k, config, reference_observation=source)
    scalar = WholeSurfaceFit(reference, current, k, config, reference_observation=source)
    queries = []
    for index, view in enumerate([fit, *(view for _, view in fit.reverse_fits)]):
        original = view.target_tree.query

        def query(points, *, original=original, index=index, **kwargs):
            queries.append((index, len(points)))
            return original(points, **kwargs)

        view.target_tree = SimpleNamespace(query=query)
    _, _, iterates = fit._solve(perturbed_pose(truth), 40)
    assert len(iterates) > 2
    # Through the real native solver and production scorer: one query batch per
    # measured view, rather than one dispatch per returned pose.
    assert queries == [
        (index, len(iterates) * len(view.reference_queries))
        for index, view in enumerate([fit, *(view for _, view in fit.reverse_fits)])
    ]
    for score, pose in iterates:
        assert score == scalar.score(pose)
        assert fit.check_pose(pose) == scalar.check_pose(pose)
    assert any(fit.check_pose(pose)["accepted"] for _, pose in iterates)


@pytest.mark.parametrize("backend", ["small_gicp", "open3d_plane"])
def test_hidden_back_surface_is_not_pulled_onto_visible_front(backend):
    from tests.policies.flash.test_surface_publication import frame

    pytest.importorskip("small_gicp" if backend == "small_gicp" else "open3d")
    prepared = frame(0)
    current = prepared["current"]
    front = current["obs"]["points"]
    back = front.copy()
    back[:, 2] += 0.012
    # The stored object has two faces, but depth measures only the front.
    # Matching both faces to that one observation would move it forward 6 mm.
    fit = WholeSurfaceFit(np.concatenate([front, back]), current, prepared["intrinsics"], ShapeConfig())
    initial = np.eye(4)
    initial[2, 3] = 0.002
    pose, _ = LibraryTrial(LibraryConfig(backend)).refine(fit, initial)
    assert abs(pose[2, 3]) < 0.001
    assert fit.check_pose(pose)["accepted"]


@pytest.mark.parametrize("backend", ["small_gicp", "open3d_plane"])
def test_library_direction_and_translation_on_asymmetric_surface(backend):
    pytest.importorskip("small_gicp" if backend == "small_gicp" else "open3d")
    # Independently generated bumpy surface, with known metre-scale translation.
    x, y = np.meshgrid(np.linspace(-0.06, 0.06, 41), np.linspace(-0.04, 0.04, 31))
    z = 0.7 + 0.012 * np.sin(x * 70) + 0.008 * np.cos(y * 90) + x * y
    source = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [4, -3, 7], degrees=True).as_matrix()
    truth[:3, 3] = [0.014, -0.008, 0.007]
    target = move(source, truth)
    initial = truth.copy()
    initial[:3, 3] += [0.004, -0.003, 0.003]
    from types import SimpleNamespace

    fitter = SimpleNamespace(reference=source, target=target, reference_queries=source, target_queries=target)
    fitter.score = lambda pose: float(np.mean((move(source, pose) - target) ** 2))
    trial = LibraryTrial(LibraryConfig(backend))
    result, _ = trial.refine(fitter, initial)
    assert np.linalg.norm(result[:3, 3] - truth[:3, 3]) < 0.002
    assert Rotation.from_matrix(result[:3, :3] @ truth[:3, :3].T).magnitude() < np.deg2rad(1)
    assert np.mean(np.linalg.norm(move(source, result) - target, axis=1)) < 0.001
    assert fitter.score(result) < fitter.score(initial) / 4


@pytest.mark.parametrize("backend", ["small_gicp", "open3d_plane"])
def test_replacement_preserves_full_validation_and_restores_original(backend):
    pytest.importorskip("small_gicp" if backend == "small_gicp" else "open3d")
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(20)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)
    original = WholeSurfaceFit._solve
    mask, depth = current["mask"].copy(), current["depth"].copy()
    with LibraryTrial(LibraryConfig(backend)) as trial:
        pose, detail = fit.correct(perturbed_pose(truth))
    assert WholeSurfaceFit._solve is original
    assert trial.calls
    assert detail["cost"] == fit.score(pose)
    assert detail["fit_reference_points"] == len(fit.reference_queries)
    np.testing.assert_array_equal(current["mask"], mask)
    np.testing.assert_array_equal(current["depth"], depth)
    # A solver returning a plausible transform is not automatic acceptance.
    wrong = truth.copy()
    wrong[0, 3] += 0.04
    assert not check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())["accepted"]
    assert check_surface_contradiction(fit, truth, PoseUncertaintyConfig())["accepted"]
    assert pose_error(reference, pose, truth)["median_point_error_mm"] < 8
    with pytest.raises(RuntimeError, match="interrupted"):
        with LibraryTrial(LibraryConfig(backend)):
            raise RuntimeError("interrupted")
    assert WholeSurfaceFit._solve is original


def test_color_is_not_borrowed_from_occluding_foreground():
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, 2.0]])
    k = np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((6, 6, 3), np.uint8)
    rgb[2, 2] = [255, 0, 0]
    obs = {"rgb": rgb, "depth": np.ones((6, 6)), "mask": np.ones((6, 6), bool)}
    colors, valid = measured_colors(points, obs, k, np.eye(4), 0.003)
    assert valid.tolist() == [True, True, False]
    np.testing.assert_array_equal(colors[0], [1, 0, 0])


def textured_box_pair():
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(20)
    for obs, pose in ((source, np.eye(4)), (current, truth)):
        y, x = np.indices(obs["depth"].shape)
        z = obs["depth"]
        points = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), -1)
        fixed = move(points.reshape(-1, 3), np.linalg.inv(pose))
        obs["rgb"] = ((np.sin(fixed * 80) + 1) * 127).astype(np.uint8).reshape(*z.shape, 3)
    return source, current, k, truth


def test_colored_solver_uses_observed_rgb_on_known_partial_motion():
    pytest.importorskip("open3d")
    source, current, k, truth = textured_box_pair()
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)
    with LibraryTrial(LibraryConfig("open3d_color")) as trial:
        pose, _ = fit.correct(perturbed_pose(truth))
    assert trial.calls[0]["inliers"] > 50
    assert pose_error(reference, pose, truth)["median_point_error_mm"] < 8


def test_colored_replay_session_supplies_rgb_without_changing_geometry_views():
    pytest.importorskip("open3d")
    from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession

    source, current, k, truth = textured_box_pair()
    session = GrowingSurfaceSession(ShapeConfig())
    original = GrowingSurfaceSession.update
    assert not session.config.partial_surface_matching
    with LibraryTrial(LibraryConfig("open3d_color")) as trial:
        for i, obs in enumerate((source, current)):
            session.update(
                {
                    "frame_id": i,
                    "status": None,
                    "current": obs,
                    "intrinsics": k,
                    "step": truth if i else None,
                    "match_source_frame": 0 if i else None,
                },
                float(i),
            )
        assert session._fit_views() == ()
        assert trial.calls and all(row["levels"] > 0 for row in trial.calls)
        assert max(row["inliers"] for row in trial.calls) > 50
    assert GrowingSurfaceSession.update is original


def test_colored_fitter_fails_loudly_without_reference_rgb():
    pytest.importorskip("open3d")
    source, current, k, _ = textured_box_pair()
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig())
    with pytest.raises(ValueError, match="measured reference RGB-D"):
        LibraryTrial(LibraryConfig("open3d_color")).refine(fit, np.eye(4))


def test_prepared_clouds_are_not_reported_as_a_successful_library_solve(monkeypatch):
    pytest.importorskip("open3d")
    source, current, k, truth = textured_box_pair()
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    trial = LibraryTrial(LibraryConfig("open3d_color"))

    def no_correspondences(*args):
        raise RuntimeError("No correspondences found between source and target pointcloud.")

    monkeypatch.setattr(trial.library.pipelines.registration, "registration_colored_icp", no_correspondences)
    pose, details = trial.refine(fit, truth)
    np.testing.assert_array_equal(pose, truth)
    assert details["library_levels"] > 0
    assert details["library_attempted_levels"] == 1
    assert details["library_solved_levels"] == 0
    assert details["library_failure"] == "no_correspondences"
    assert trial.summary()["refinements_with_library_solve"] == 0


def test_nonfinite_library_warm_start_is_not_sent_into_next_solver(monkeypatch):
    pytest.importorskip("open3d")
    from types import SimpleNamespace

    source, current, k, truth = textured_box_pair()
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    trial = LibraryTrial(LibraryConfig("open3d_color"))
    reg = trial.library.pipelines.registration
    monkeypatch.setattr(
        reg, "registration_icp", lambda *args: SimpleNamespace(transformation=np.full((4, 4), np.nan))
    )

    def forbidden(*args):
        raise AssertionError("The next solver must not consume a nonfinite transform")

    monkeypatch.setattr(reg, "registration_colored_icp", forbidden)
    pose, detail = trial.refine(fit, truth)
    np.testing.assert_array_equal(pose, truth)
    assert detail["library_failure"] == "nonfinite_transform"
    assert detail["library_solved_levels"] == 0
    assert detail["evaluations"] > 0
