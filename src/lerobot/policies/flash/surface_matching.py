"""One bounded matching transaction, with shared preparation and evaluation.

Python schedules a small set of starts. Point transforms, nearest-neighbour
queries, depth checks, SVD and numerical optimization are vectorized or compiled.
Geometry checks govern every returned candidate, independently of its score.
Texture and outline coverage are ranking preferences, never permission to
violate measured geometry. A candidate's score cannot turn rejection into
acceptance; missing/occluded depth is unknown, not measured empty space.
"""

from __future__ import annotations

from typing import Protocol


class PoseMatcher(Protocol):
    def match(
        self,
        problem,
        proposal,
        *,
        geometry=None,
        previous=None,
        check=None,
        extra=(),
        coarse=True,
        recovery_starts=0,
        evaluation_budget=None,
    ) -> tuple[list[dict], dict]: ...


class SharedGeometryMatcher:
    def match(
        self,
        problem,
        proposal,
        *,
        geometry=None,
        previous=None,
        check=None,
        extra=(),
        coarse=True,
        recovery_starts=0,
        evaluation_budget=None,
    ):
        """Fit independent starts; use geometry to validate and rank every result.

        Image/group motion supplies a start. The previous placement supplies
        another. Neither fit is fed into a second fitting branch. Coarse starts
        are tried only when promising, with extra orientations only on failure.
        Solver iterates remain proposals until they pass the geometry checks.
        """
        from .surface_grounding import pca_candidates

        geometry = problem if geometry is None else geometry
        check = geometry.check_pose if check is None else check
        starts = 1 + (previous is not None) + coarse + recovery_starts
        single_loop = problem.config.registration.backend == "visible_icp"
        budget = (
            evaluation_budget if evaluation_budget is not None else problem.config.max_evaluations * starts
        )
        candidates, checks, scores, seen = [], {}, {}, set()
        used, numerical_solves = 0, 0

        def score(pose):
            key = pose.tobytes()
            if key not in scores:
                scores[key] = geometry.score(pose)
            return scores[key]

        def checked(pose):
            key = pose.tobytes()
            if key not in checks:
                checks[key] = check(pose)
            return checks[key]

        def add(pose, name, detail=None):
            candidates.append(
                {
                    "pose": pose.copy(),
                    "name": name,
                    "check": checked(pose),
                    "cost": score(pose),
                    "detail": detail or {},
                }
            )

        def score_all(poses):
            missing = {p.tobytes(): p for p in poses if p.tobytes() not in scores}
            # One batched spatial query instead of one thread dispatch per pose.
            # score() then reads these exact cached residuals and adds RGB cost.
            if missing:
                geometry.batch_residuals(list(missing.values()))
                scores.update((key, geometry.score(pose)) for key, pose in missing.items())

        def best_valid(poses):
            score_all(poses)
            return next((p for p in sorted(poses, key=score) if checked(p)["accepted"]), None)

        def run(initial, name, objective=None):
            nonlocal used, numerical_solves
            objective = geometry if objective is None else objective
            key = id(objective), initial.tobytes()
            if key in seen or used >= budget:
                return
            seen.add(key)
            numerical_solves += 1
            final, detail, iterates = objective._solve(
                initial, min(problem.config.max_evaluations, budget - used)
            )
            used += detail["evaluations"]
            # The solver already scored these poses against this same objective.
            # Reuse those values rather than projecting every surface twice.
            if objective is geometry:
                scores.update((pose.tobytes(), cost) for cost, pose in iterates)
            best = best_valid([pose for _, pose in iterates])
            detail["retained_admissible_iterate"] = best is not None
            add(final if best is None else best, name, detail)

        run(
            proposal,
            "surface_from_motion" if geometry is not problem else "local",
            problem if problem.has_image_evidence and single_loop else geometry,
        )
        if previous is not None:
            add(previous, "previous")
            run(previous, "surface_from_previous")
        for name, pose in extra:
            add(pose, name)
        if coarse and used < budget:
            seeds = pca_candidates(problem.reference, problem.observation["obs"]["points"])
            score_all(seeds)
            seed = min(seeds, key=score)
            valid = [c for c in candidates if c["check"]["accepted"]]
            if not valid or score(seed) < min(c["cost"] for c in valid):
                run(seed, "local_coarse")
        if recovery_starts and not any(c["check"]["accepted"] for c in candidates):
            seeds = pca_candidates(
                problem.reference, problem.observation["obs"]["points"], all_axis_orders=True
            )
            score_all(seeds)
            for i, seed in enumerate(sorted(seeds, key=score)[:recovery_starts]):
                run(seed, f"recovery_{i}")
        return candidates, {
            "evaluations": used,
            "evaluation_budget": budget,
            "validation_count": len(checks),
            "numerical_solves": numerical_solves,
            "distinct_scores": len(scores),
        }


DEFAULT_MATCHER: PoseMatcher = SharedGeometryMatcher()
