"""One rigid fitting loop with visibility-tested correspondences.

The loop proposes poses; the shared matcher separately checks and ranks them.
Point transforms, neighbour searches and the weighted rigid solve are array or
compiled operations. No geometry rejection rule is implemented here.
"""

import numpy as np

from .surface_observation import move


def measured_pairs(view, predicted):
    weight = np.maximum(view._projection_evidence(predicted)[0], view._scene_depth_evidence(predicted)[2])
    distance, indices = view.target_tree.query(predicted)
    weight[distance > view.config.max_surface_distance_m] = 0
    return view.target[indices], weight / (len(predicted) * view.config.depth_tolerance_m**2)


def rigid_steps(problem, initial, max_iterations):
    """Refresh correspondences and make one weighted rigid update per iteration."""
    geometry = getattr(problem, "geometry", problem)
    pose = initial.copy()
    for _ in range(max_iterations):
        predicted = move(geometry.reference, pose)
        target, weight = measured_pairs(geometry, predicted)
        sources, targets, weights = [predicted], [target], [weight]
        reverse = []
        for camera, view in geometry.reverse_fits:
            old = move(view.reference, camera @ np.linalg.inv(pose))
            old_target, old_weight = measured_pairs(view, old)
            reverse.append((move(old_target, pose @ np.linalg.inv(camera)), view.reference, old_weight))
        if reverse:
            # Match the objective's maximum over measured views: seeing a point
            # in several old frames must not multiply that point's influence.
            errors = np.asarray([np.sum((a - b) ** 2, axis=1) * w for a, b, w in reverse])
            chosen, indices = errors.argmax(0), np.arange(errors.shape[1])
            sources.append(np.asarray([a for a, _, _ in reverse])[chosen, indices])
            targets.append(reverse[0][1])
            weights.append(np.asarray([w for _, _, w in reverse])[chosen, indices])
        else:
            # The existing objective also rewards covering the current surface
            # when no earlier depth views are available. Omitting this direction
            # lets the solver stop with visible geometry left unmatched.
            query = (geometry.target_queries - pose[:3, 3]) @ pose[:3, :3]
            distance, indices = geometry.reference_tree.query(query)
            sources.append(move(geometry.reference[indices], pose))
            targets.append(geometry.target_queries)
            weights.append(
                (distance <= geometry.config.max_surface_distance_m).astype(float)
                / (len(query) * geometry.config.depth_tolerance_m**2)
            )
        if problem.has_image_evidence:
            evidence = [
                (problem.image_reference, problem.image_current, problem.image_scale_m, problem.image_weight),
                *((p.reference, p.current, p.scale_m, p.weight) for p in problem.point_evidence),
            ]
            for reference, current, scale, strength in evidence:
                if not len(reference) or not strength:
                    continue
                predicted = move(reference, pose)
                weight = strength / (len(predicted) * scale**2)
                weight = weight / np.maximum(1, np.linalg.norm(predicted - current, axis=1) / scale)
                sources.append(predicted)
                targets.append(current)
                weights.append(weight)
        source, target, weight = map(np.concatenate, (sources, targets, weights))
        active = weight > 0
        if np.count_nonzero(active) < 3:
            return
        source, target, weight = source[active], target[active], weight[active]
        weight /= weight.sum()
        x, y = np.sum(source * weight[:, None], axis=0), np.sum(target * weight[:, None], axis=0)
        cross = ((source - x) * weight[:, None]).T @ (target - y)
        if np.linalg.matrix_rank(cross) < 2:
            return
        u, _, vh = np.linalg.svd(cross)
        rotation = (u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh).T
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = rotation, y - rotation @ x
        pose = step @ pose
        yield pose.copy()
        if np.linalg.norm(step - np.eye(4)) < problem.config.solver_tolerance:
            return


def solve(problem, initial, max_iterations):
    poses = [initial, *rigid_steps(problem, initial, max_iterations)]
    problem.batch_residuals(poses)
    converged = len(poses) > 1 and bool(
        np.linalg.norm(poses[-1] @ np.linalg.inv(poses[-2]) - np.eye(4)) < problem.config.solver_tolerance
    )
    return (
        poses[-1],
        {
            "evaluations": len(poses) - 1,
            "library": "visible_icp",
            "converged": converged,
        },
        [(problem.score(pose), pose) for pose in poses],
    )
