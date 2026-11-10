"""The frame-processing owner shared by recorded experiments and live RGB-D.

Only adapters acquire frames/masks or persist evidence. Matching, group evidence,
pose validation, Kalman updates and surface growth execute here for both inputs.
No camera hardware, dataset loader, labels, robot control or output files.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from .experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from .flow_paths import DenseFlow, FlowPaths
from .motion_groups import SceneMotionGroups
from .object_group_association import ObjectGroupAssociation, ObjectMotionSource
from .scene_groups import ScenePointTracker
from .shared_image_tracking import SharedFrameTracker
from .surface_correspondence import crop_pair, measured_pairs, robust_fit, ufm_pairs
from .surface_observation import move, prepare_surface_frame
from .surface_tracking import SurfaceTrackingSession


class MotionChain:
    """A failed link cannot silently become a stationary measurement."""

    def __init__(self):
        self.pose = None
        self.broken = False

    def update(self, prepared):
        if prepared["status"] is not None:
            # prepare() keeps the last usable image in this case. A future
            # correspondence can still bridge to it, but this frame is unknown.
            return None
        if self.broken:
            return None
        if self.pose is None:
            self.pose = np.eye(4)
        elif prepared["step"] is None:
            # prepare() advanced its image even though this pair did not fit.
            self.broken = True
            return None
        else:
            self.pose = prepared["step"] @ self.pose
        return self.pose.copy()


def scene_group_record(snapshot, scene, frame):
    """All measured scene points, preserving the identities used by the wireframe."""
    visible = {int(point_id): index for index, point_id in enumerate(scene.ids)}
    camera_from_base = np.linalg.inv(frame.base_from_camera)
    record = []
    for group in snapshot.groups.values():
        measured = [int(point_id) for point_id in group.ids if int(point_id) in visible]
        indices = [visible[point_id] for point_id in measured]
        hidden = group.base_xyz[~group.measured]
        predicted_xyz = move(hidden, camera_from_base)
        z = predicted_xyz[:, 2] if len(predicted_xyz) else np.empty(0)
        valid = z > 0
        predicted_xyz = predicted_xyz[valid]
        z = z[valid]
        pixels = (
            np.column_stack(
                (
                    frame.intrinsics[0, 0] * predicted_xyz[:, 0] / z + frame.intrinsics[0, 2],
                    frame.intrinsics[1, 1] * predicted_xyz[:, 1] / z + frame.intrinsics[1, 2],
                )
            )
            if len(z)
            else np.empty((0, 2))
        )
        record.append(
            {
                "id": group.id,
                "confirmed": group.confirmed,
                "supported": group.supported,
                "member_count": len(group.ids),
                "measured_count": int(group.measured.sum()),
                "residual_mm": 1000 * group.residual_m if np.isfinite(group.residual_m) else None,
                "ids": measured,
                "pixels": np.rint(scene.xy[indices]).astype(int).tolist(),
                "camera_xyz": np.round(move(scene.base_xyz[indices], camera_from_base), 4).tolist(),
                "predicted_pixels": np.rint(pixels).astype(int).tolist(),
                "predicted_camera_xyz": np.round(predicted_xyz, 4).tolist(),
            }
        )
    return record


class RetainedMotionReference:
    """Experiment: retry from the last connected image after a missed match.

    This is a recovery hypothesis, not a pose acceptance rule. Long-gap UFM
    correspondences can still be wrong; the labelled cases measure that risk.
    """

    def __init__(self, matcher, config, *, record_correspondences=False, shared_matching=None):
        self.matcher, self.config = matcher, config
        self.shared_matching = shared_matching
        self.record_correspondences = record_correspondences
        self.evidence = None
        self.paths = FlowPaths()
        self.reference = None
        self.reference_frame = None
        self.pose = np.eye(4)

    def update(self, prepared, previous):
        self.evidence = None
        diagnostics = {"reference_frame": self.reference_frame, "extra_match_ms": 0.0}
        if prepared["status"] is not None:
            return None, diagnostics
        current = prepared["current"]
        if self.reference is not None:
            step = prepared["step"]
            evidence = prepared.get("correspondences")
            if self.reference is not previous:
                started = time.perf_counter()
                a, b, crop = crop_pair(self.reference, current, self.config)
                evidence = {} if self.record_correspondences else None
                kwargs = {"evidence": evidence} if evidence is not None else {}
                if self.shared_matching is not None:
                    source, target = self.shared_matching.pairs(
                        self.reference, current, prepared["intrinsics"], evidence=evidence
                    )
                else:
                    source, target = ufm_pairs(
                        self.matcher,
                        a,
                        b,
                        crop,
                        self.reference,
                        current,
                        prepared["intrinsics"],
                        self.config,
                        **kwargs,
                    )
                step, inliers, residual = robust_fit(source, target, self.config)
                diagnostics.update(
                    extra_match_ms=(time.perf_counter() - started) * 1000,
                    matches=len(source),
                    inliers=inliers,
                    residual_mm=float(residual * 1000) if np.isfinite(residual) else None,
                )
            if evidence is not None:
                self.evidence = match_record(
                    evidence,
                    self.reference,
                    current,
                    prepared["intrinsics"],
                    step,
                    self.config,
                    paths=self.paths,
                )
            if step is None:
                return None, diagnostics
            self.pose = step @ self.pose
        self.reference = current
        self.reference_frame = prepared["frame_id"]
        return self.pose.copy(), diagnostics


def match_record(evidence, source, target, intrinsics, step, config, *, paths=None):
    """Save measured matches and fit agreement; never synthesize matches from a pose."""
    stages = evidence["stage"].copy()
    rows = np.flatnonzero(stages == 3)
    if step is not None and len(rows):
        a, b = measured_pairs(
            source, target, evidence["source_uv"][rows], evidence["target_uv"][rows], intrinsics
        )
        if len(a) != len(rows):
            raise ValueError("captured correspondence filter disagrees with the estimator")
        errors = np.linalg.norm(a @ step[:3, :3].T + step[:3, 3] - b, axis=1)
        stages[rows[errors < config.ransac_distance_m]] = 4
    record = {
        "source_frame": source["frame_id"],
        "target_frame": target["frame_id"],
        "source_uv": np.round(evidence["source_uv"], 3).tolist(),
        "target_uv": np.round(evidence["target_uv"], 3).tolist(),
        "stage": stages.tolist(),
        "fit_available": step is not None,
    }
    if "track_ids" in evidence:
        record["backend"] = evidence["backend"]
        record["track_ids"] = evidence["track_ids"].tolist()
        record["paths"] = [
            [int(i), int((i * 137) % 280), *np.round(a, 3).tolist(), *np.round(b, 3).tolist(), int(s)]
            for i, a, b, s in zip(
                evidence["track_ids"], evidence["source_uv"], evidence["target_uv"], stages, strict=True
            )
        ]
        record["path_sampling"] = "shared_point_identities"
    elif paths is not None and "flow" in evidence:
        crop = evidence["crop"]
        field = (
            DenseFlow(record, evidence["flow"], crop[:2], crop[2] / config.input_size_px)
            if len(stages)
            else None
        )
        record["paths"] = paths.update(record, field=field)
        record["path_sampling"] = "full_ufm_output"
    return record


class SurfaceFrameSession:
    def __init__(
        self,
        roles,
        model,
        config,
        kernels=None,
        search_pool=None,
        *,
        retain_motion_reference=False,
        record_correspondences=False,
        surface_experiment=None,
        group_experiment=None,
        shared_matching_config=None,
        surface_methods=None,
        single_method=None,
        parallel_objects=True,
    ):
        if single_method is not None:
            if surface_methods is not None or retain_motion_reference:
                raise ValueError("single_method cannot also select comparison methods or a retry baseline")
            surface_methods = [single_method]
        shared = SharedFrameTracker(shared_matching_config, model) if shared_matching_config else None
        sessions = {
            role: SurfaceTrackingSession(
                model,
                observation_config=config.observation,
                shape_config=config.shape,
                motion_config=config.motion,
                fit_kernels=kernels,
                search_pool=search_pool,
                record_correspondences=record_correspondences,
                shared_matching=shared,
            )
            for role in roles
        }
        chains = {role: MotionChain() for role in sessions}
        paths = {role: FlowPaths() for role in sessions}
        retained = {
            role: RetainedMotionReference(
                model, config.motion, record_correspondences=record_correspondences, shared_matching=shared
            )
            for role in sessions
        }
        if group_experiment is not None and surface_experiment is None:
            raise ValueError("group variants require the surface experiment")
        variants = [("conditioned_fixed", False), ("conditioned_growing", True)]
        if group_experiment is not None:
            variants += [("group_fixed", False), ("group_growing", True)]
        if surface_methods is not None:
            if (
                not surface_experiment
                or not surface_methods
                or set(surface_methods) - {n for n, _ in variants}
            ):
                raise ValueError("surface_methods must select configured surface experiments")
            variants = [(name, grow) for name, grow in variants if name in surface_methods]
        candidates = (
            {
                name: {
                    role: GrowingSurfaceSession(
                        config.shape,
                        GrowingSurfaceConfig(
                            **{
                                **asdict(surface_experiment),
                                "grow": grow,
                                "retain_unobserved_pose": name.startswith("group_"),
                            }
                        ),
                        fit_kernels=kernels,
                        search_pool=search_pool,
                    )
                    for role in sessions
                }
                for name, grow in variants
            }
            if surface_experiment is not None
            else {}
        )
        scene_tracker = ScenePointTracker() if group_experiment is not None and shared is None else None
        scene_groups = SceneMotionGroups(group_experiment) if group_experiment is not None else None
        sources: dict[str, dict[str, ObjectMotionSource]] = {
            name: {
                role: ObjectGroupAssociation(
                    scene_groups.config,
                    point_evidence_weight=surface_experiment.group_point_evidence_weight,
                    min_association_points=surface_experiment.group_association_min_points,
                )
                for role in sessions
            }
            for name in candidates
            if name.startswith("group_")
        }
        self.scene_groups = scene_groups
        self.shared = shared
        self.sessions = sessions
        self.chains = chains
        self.paths = paths
        self.retained = retained
        self.candidates = candidates
        self.scene_tracker = scene_tracker
        self.motion_sources = sources
        self.config, self.model = config, model
        self.single_method = single_method
        self.retain_motion_reference = retain_motion_reference
        self.references, self.reference_frames = {}, {}
        self.last_frame_id = None
        self.object_pool = (
            ThreadPoolExecutor(max_workers=min(config.surface_object_workers, len(sessions)))
            if parallel_objects
            and shared is not None
            and single_method is not None
            and config.parallel_surface_objects
            and len(sessions) > 1
            else None
        )

    def step(self, frame, masks, *, mask_ms=0.0, archive_callback=None):
        if self.last_frame_id is not None and frame.frame_id <= self.last_frame_id:
            raise ValueError("Frame IDs must increase; create a new session to reseed")
        for role in self.sessions:
            if role not in masks or masks[role].shape != frame.depth.shape:
                raise ValueError(f"Missing or unaligned mask for {role}")
        self.last_frame_id = frame.frame_id
        frame_started = time.perf_counter() - mask_ms / 1000
        excluded_s = 0.0
        shared = self.shared
        sessions = self.sessions
        chains = self.chains
        paths = self.paths
        retained = self.retained
        candidates = self.candidates
        scene_tracker = self.scene_tracker
        sources = self.motion_sources
        config = self.config
        model = self.model
        single_method = self.single_method
        retain_motion_reference = self.retain_motion_reference
        references = self.references
        reference_frames = self.reference_frames
        object_pool = self.object_pool
        row = {"frame": frame.frame_id, "time_s": frame.source_time_s, "mask_ms": mask_ms, "objects": {}}
        if shared is not None:
            scene = shared.update(frame, masks)
            row["shared_matching"] = dict(shared.diagnostics)
            row["scene_point_tracking_ms"] = shared.diagnostics["total_ms"]
        if scene_tracker is not None:
            begin = time.perf_counter()
            scene = scene_tracker.update(frame)
            row["scene_point_tracking_ms"] = (time.perf_counter() - begin) * 1000
        if self.scene_groups is not None:
            begin = time.perf_counter()
            group_frame = self.scene_groups.update(scene)
            row["group_update_ms"] = (time.perf_counter() - begin) * 1000
            row["scene_groups"] = {
                "groups": len(group_frame.groups),
                "points": group_frame.point_count,
                "supported": sum(g.supported for g in group_frame.groups.values()),
                "confirmed": sum(g.confirmed for g in group_frame.groups.values()),
            }
            row["scene_group_tracks"] = scene_group_record(group_frame, scene, frame)
        depth_m = frame.depth.astype(np.float32) * frame.depth_m_per_unit
        frame_filters = prepare_surface_frame(depth_m, asdict(config.observation))
        match_row = {"frame": frame.frame_id, "objects": {}}

        def process_role(
            role_session,
            frame=frame,
            masks=masks,
            scene=scene if shared is not None or scene_tracker is not None else None,
            depth_m=depth_m,
            frame_filters=frame_filters,
        ):
            role, session = role_session
            new_reference = new_reference_frame = None
            excluded_role_s = 0.0
            begin = time.perf_counter()
            previous = session.previous
            prepared = session.prepare(
                frame.rgb,
                depth_m,
                masks[role],
                frame.intrinsics,
                frame.frame_id,
                frame_filters=frame_filters,
            )
            prepare_ms = (time.perf_counter() - begin) * 1000
            pair = None
            if prepared.get("correspondences") is not None:
                pair = match_record(
                    prepared["correspondences"],
                    previous,
                    prepared["current"],
                    frame.intrinsics,
                    prepared["step"],
                    config.motion,
                    paths=paths[role],
                )
            match_object = {"pair": pair}
            if role not in references and prepared["status"] is None:
                new_reference = prepared["current"]["obs"]["points"].copy()
                new_reference_frame = frame.frame_id
            object_row = {
                "prepare_ms": prepare_ms,
            }
            if single_method is None:
                chain = chains[role].update(prepared)
                begin = time.perf_counter()
                fitted = session.finish(prepared)
                finish_ms = (time.perf_counter() - begin) * 1000
                object_row.update(
                    ufm_chain=chain.tolist() if chain is not None else None,
                    surface_corrected=fitted.get("camera_from_reference"),
                    status=fitted["status"],
                    motion_chain_broken=chains[role].broken,
                    finish_ms=finish_ms,
                    diagnostics=fitted["diagnostics"],
                )
            if retain_motion_reference:
                retry_pose, retry_diagnostics = retained[role].update(prepared, previous)
                match_object["retry"] = retained[role].evidence
                object_row.update(
                    ufm_retained_reference=retry_pose.tolist() if retry_pose is not None else None,
                    retained_reference_diagnostics=retry_diagnostics,
                )
            for name, tracks in candidates.items():
                begin = time.perf_counter()
                track = tracks[role]
                candidate_input = prepared
                # Match against this track's last accepted RGB-D observation.
                # A rejected/predicted frame cannot replace that reference.
                if (
                    prepared["status"] is None
                    and track.accepted_observation is not None
                    and prepared.get("match_source_frame") != track.accepted_frame
                ):
                    candidate_input = dict(prepared)
                    old = track.accepted_observation
                    current = prepared["current"]
                    if shared is not None:
                        source, target = shared.pairs(old, current, prepared["intrinsics"])
                    else:
                        a, b, crop = crop_pair(old, current, config.motion)
                        source, target = ufm_pairs(
                            model, a, b, crop, old, current, prepared["intrinsics"], config.motion
                        )
                    step, _, _ = robust_fit(source, target, config.motion)
                    candidate_input.update(
                        step=step, point_pairs=(source, target), match_source_frame=track.accepted_frame
                    )
                geometry = track.prepare_geometry(candidate_input)
                hint = None
                if name in sources:
                    hint, group_details = sources[name][role].query(
                        group_frame,
                        frame,
                        scene,
                        masks[role],
                        track.reference_frame,
                        object_usable=prepared["current"]["usable"] if shared else None,
                    )
                result = track.update(
                    candidate_input,
                    frame.source_time_s,
                    hint=hint,
                    base_from_camera=frame.base_from_camera,
                    geometry=geometry,
                )
                details = result["diagnostics"]
                if name in sources:
                    sources[name][role].record_observation(
                        frame,
                        scene,
                        masks[role],
                        result,
                        track.reference,
                        **({"object_usable": prepared["current"]["usable"]} if shared else {}),
                    )
                    details["group"] = group_details
                object_row.update(
                    {
                        name: result["camera_from_reference"],
                        name + "_ms": (time.perf_counter() - begin) * 1000,
                        name + "_status": result["status"],
                        name + "_details": details,
                    }
                )
                if archive_callback is None:
                    continue
                archive_started = time.perf_counter()
                archive_callback(name, role, track, details)
                excluded_role_s += time.perf_counter() - archive_started
            return role, object_row, match_object, new_reference, new_reference_frame, excluded_role_s

        role_results = (
            object_pool.map(process_role, sessions.items())
            if object_pool is not None
            else map(process_role, sessions.items())
        )
        for (
            role,
            object_row,
            match_object,
            new_reference,
            new_reference_frame,
            excluded_role_s,
        ) in role_results:
            row["objects"][role] = object_row
            match_row["objects"][role] = match_object
            if new_reference is not None:
                references[role] = new_reference
                reference_frames[role] = new_reference_frame
            excluded_s += excluded_role_s

        row["processing_ms"] = (time.perf_counter() - frame_started - excluded_s) * 1000
        return row, match_row

    def close(self):
        if self.object_pool is not None:
            self.object_pool.shutdown(wait=True)
            self.object_pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
