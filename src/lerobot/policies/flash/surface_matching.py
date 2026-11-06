"""One bounded matching transaction, with shared preparation and evaluation.

Python schedules a small set of starts. Point transforms, nearest-neighbour
queries, depth checks, SVD and numerical optimization are vectorized or compiled.
Geometry checks govern every returned candidate, independently of its score.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


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
        """One owner for all starts, validity checks, and retained alternatives.

        Numerical fitting and correspondence updates only propose poses. Every
        candidate uses the same cached check. A lower-cost invalid candidate can
        never displace a valid one. Budgets count solver evaluations (excluding
        finite-difference probes) and correspondence updates, as in max_nfev.
        """
        from .surface_grounding import pca_candidates
        from .surface_observation import move

        geometry = problem if geometry is None else geometry
        check = geometry.check_pose if check is None else check
        branches = 1 if geometry is problem else 3
        # Preserve the previous maximum work, but account for it in one place:
        # each branch had two starts, each with a solve and a corrective pass.
        budget = (
            evaluation_budget
            if evaluation_budget is not None
            else (problem.config.max_evaluations * (branches * (4 if coarse else 2) + 2 * recovery_starts))
        )
        remaining = budget
        candidates, checks, scores, solves, corrections = [], {}, {}, {}, {}
        native = problem.config.registration.backend != "scipy"
        numerical_solves = 0

        def score(objective, pose):
            key = id(objective), pose.tobytes()
            if key not in scores:
                scores[key] = objective.score(pose)
            return scores[key]

        def checked(pose):
            key = pose.tobytes()
            if key not in checks:
                checks[key] = check(pose)
            return checks[key]

        def add(pose, name, detail=None):
            record = {
                "pose": pose.copy(),
                "name": name,
                "check": checked(pose),
                "cost": score(geometry, pose),
                "detail": detail or {},
            }
            candidates.append(record)
            return record

        def run(objective, initial, name):
            nonlocal remaining, numerical_solves
            if remaining <= 0:
                return None
            # Native registration proposes the same iterates regardless of the
            # image residual. SciPy also shares a solve when that residual is empty.
            owner = geometry if native or not objective.has_image_evidence else objective
            key = id(owner), initial.tobytes()
            if key not in solves:
                final, detail, iterates = objective._solve(
                    initial, min(problem.config.max_evaluations, remaining)
                )
                remaining -= detail["evaluations"]
                numerical_solves += 1
                solves[key] = final, dict(detail), [p for _, p in iterates]
            else:
                final, detail, poses = solves[key]
                detail = {**detail, "evaluations": 0, "reused_solve": True}
                iterates = [(score(objective, p), p) for p in poses]
            # Evaluate promising iterates first; numerical derivative probes
            # share the same cache as all other candidates.
            valid = next((p for _, p in sorted(iterates, key=lambda x: x[0]) if checked(p)["accepted"]), None)
            correction_count = 0
            # A solver can slide off a flat surface and leave only its input
            # admissible. Refine that input rather than treating it as a
            # successful solve or correcting the diverged endpoint.
            unchanged = valid is not None and np.array_equal(valid, initial)
            if remaining > 0 and (
                valid is None or (unchanged and score(objective, final) > score(objective, initial))
            ):
                correction_start = initial if unchanged else final
                key = correction_start.tobytes()
                if key not in corrections:
                    corrected = list(
                        objective._refine_visible_correspondences(
                            correction_start, min(problem.config.max_evaluations, remaining)
                        )
                    )
                    corrections[key] = corrected
                    correction_count = len(corrected)
                    remaining -= correction_count
                else:
                    corrected = corrections[key]
                costs = [float(r @ r) for r in objective.batch_residuals(corrected)]
                corrected_valid = next(
                    (
                        p
                        for _, p in sorted(zip(costs, corrected, strict=True), key=lambda x: x[0])
                        if checked(p)["accepted"]
                    ),
                    None,
                )
                if corrected_valid is not None and (
                    valid is None or score(objective, corrected_valid) < score(objective, valid)
                ):
                    valid = corrected_valid
            detail.update(
                visible_correction_iterations=correction_count, retained_admissible_iterate=valid is not None
            )
            result = add(final if valid is None else valid, name, detail)
            return result if result["check"]["accepted"] else None

        coarse_seeds = None

        def branch(objective, initial, name):
            nonlocal coarse_seeds
            local = run(objective, initial, name)
            if not coarse or remaining <= 0:
                return local
            predicted = move(problem.reference, initial)
            if coarse_seeds is None:
                coarse_seeds = pca_candidates(problem.reference, problem.observation["obs"]["points"])
            seed = min(
                coarse_seeds,
                key=(
                    (lambda pose: score(objective, pose))
                    if problem.config.registration.backend != "scipy"
                    else lambda p: np.mean(np.sum((move(problem.reference, p) - predicted) ** 2, axis=1))
                ),
            )
            if local is not None and score(objective, seed) >= score(objective, local["pose"]):
                return local
            alternative = run(objective, seed, name + "_coarse")
            valid = [r for r in (local, alternative) if r is not None]
            return min(valid, key=lambda r: score(objective, r["pose"])) if valid else None

        joint = branch(problem, proposal, "joint_fit" if geometry is not problem else "local")
        if geometry is not problem:
            add(proposal, "proposal")
            branch(geometry, joint["pose"] if joint is not None else proposal, "surface_from_joint")
            branch(geometry, previous, "surface_from_previous")
            add(previous, "previous")
        for name, pose in extra:
            add(pose, name)
        if not any(c["check"]["accepted"] for c in candidates) and recovery_starts:
            seeds = pca_candidates(
                problem.reference, problem.observation["obs"]["points"], all_axis_orders=True
            )
            for i, seed in enumerate(sorted(seeds, key=lambda pose: score(geometry, pose))[:recovery_starts]):
                run(geometry, seed, f"recovery_{i}")
        return candidates, {
            "evaluations": budget - remaining,
            "evaluation_budget": budget,
            "validation_count": len(checks),
            "numerical_solves": numerical_solves,
            "distinct_scores": len(scores),
        }


DEFAULT_MATCHER: PoseMatcher = SharedGeometryMatcher()
