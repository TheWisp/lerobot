# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The closed loop: image-plane difference in, joint deltas out.

Invariant 1 in one function. The error is never "where is the object" — it is the
gap between where the held end SHOULD be (the taught configuration, transported by
the target team's own live fit) and where it IS (the same taught points, transported
by the held team's live fit). Both ends are measured through one lens on one frame,
so hand-eye calibration error, FK error and grasp offset all appear in both terms
and cancel in the difference.

**Deviation from the spec, deliberate.** §3.4 asks for damped-least-squares
differential IK, which normally means the FK Jacobian. This rig's encoders are good
but its kinematic MODEL is not (see ``lerobot/fewshot/README.md``), and an FK
Jacobian would inject exactly the error the difference above just cancelled — plus
it needs hand-eye calibration and a depth estimate per point, and §6 rules depth out
of v0. So the Jacobian mapping joint deltas to image motion is measured instead:
bootstrapped by a short probe at stage start, then Broyden-updated from the motion
the loop is already producing. DLS is unchanged — it is the inverse, and an
empirically estimated Jacobian is precisely the ill-conditioned case damping exists
for. The cost is a probe per stage; the gain is that no calibration exists to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.showservo.grouping import TeamFit
from lerobot.showservo.pose import RigidFit, rotation_vector


@dataclass
class ServoError:
    """The difference to drive to zero, plus whether it may be believed at all."""

    ok: bool
    e_uv: np.ndarray = field(default_factory=lambda: np.zeros(2))
    per_point: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    reason: str = ""

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.e_uv))


def servo_error(taught_held_uv: np.ndarray, target_fit: TeamFit, held_fit: TeamFit) -> ServoError:
    """Where the held end should be, minus where it is. Post: ``ok=False`` abstains.

    Pre: ``taught_held_uv`` is (N, 2) in taught-image pixels — the stage's
    ``goal_relation.held_uv``. Both fits map TAUGHT coordinates to the live frame.

    GEOMETRIC PRECONDITION, and the sharpest limit in v0. Transporting the goal
    through the TARGET team's fit is exact only when (a) the camera's optical axis is
    roughly perpendicular to the plane the object moves in, and (b) the held team's
    features are COPLANAR with the target team's. Break (b) and the goal is wrong by

        |d| * (z_target - z_held) / (H - z_target)

    for object displacement ``d`` and camera height ``H`` — 16 mm for a gripper
    feature 8 cm high after an 11 cm move, which is a missed grasp reported as a clean
    convergence. **A closed loop does not repair this**: it drives the error it is
    given to zero, so a wrong setpoint means confident convergence to the wrong place.
    Track the FINGERTIPS, which sit on the object's own surface at the grasp instant.
    ``tests/showservo/test_goal_transport_geometry.py`` measures both error sources
    and shows coplanarity is worth ~50x the mounting angle.

    Post: ``e_uv`` is the mean image-plane error in pixels; a failed fit on either end
    yields ``ok=False`` with a reason, and never a zero error, because "I cannot see
    it" and "it is already there" must not produce the same command (invariant 5).
    """
    taught = np.asarray(taught_held_uv, dtype=np.float64).reshape(-1, 2)
    assert len(taught) >= 1, "no held points: nothing to servo"
    if not target_fit.ok:
        return ServoError(ok=False, reason="target fit failed")
    if not held_fit.ok:
        return ServoError(ok=False, reason="held fit failed")

    desired = target_fit.sim2.apply(taught)
    current = held_fit.sim2.apply(taught)
    per_point = desired - current
    return ServoError(ok=True, e_uv=per_point.mean(axis=0), per_point=per_point)


@dataclass
class ServoError3D:
    """The 6-DoF difference to drive to zero. Units: metres and radians."""

    ok: bool
    e_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    e_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reason: str = ""

    @property
    def twist(self) -> np.ndarray:
        """Post: (6,) — translation then rotation, the vector the servo drives."""
        return np.concatenate([self.e_t, self.e_r])

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.e_t))

    @property
    def angle(self) -> float:
        return float(np.linalg.norm(self.e_r))


def servo_error_3d(
    taught_held_xyz: np.ndarray,
    target_fit: RigidFit,
    held_fit: RigidFit,
    *,
    check_scale: bool = True,
) -> ServoError3D:
    """The PRIMARY error: where the held end should be in 3D, minus where it is.

    Pre: ``taught_held_xyz`` is (N, 3) camera-frame points from the demo keyframe —
    the held end's tracked features, lifted by depth. Both fits map TAUGHT 3D
    coordinates to the live frame and come from :func:`~lerobot.showservo.pose
    .ransac_fit_rigid`.

    Post: ``e_t`` is the translation error measured AT THE HELD CENTROID, and ``e_r``
    the rotation error. Taking translation at the centroid rather than at the camera
    origin is what keeps a pure rotation from reporting a huge spurious translation.

    This supersedes the image-plane :func:`servo_error`. Because both ends are lifted
    to 3D and compared as rigid motions, the camera may sit at any angle and height,
    and the held end may sit at any depth relative to the target — the coplanarity and
    perpendicularity preconditions of the 2D path simply do not arise. What is
    preserved is the thing that mattered: both ends are still measured through one
    sensor in one frame, so the camera's own pose never enters the difference.
    """
    taught = np.asarray(taught_held_xyz, dtype=np.float64).reshape(-1, 3)
    assert len(taught) >= 1, "no held points: nothing to servo"
    if not target_fit.ok:
        return ServoError3D(ok=False, reason="target fit failed")
    if not held_fit.ok:
        return ServoError3D(ok=False, reason="held fit failed")
    if check_scale:
        # A rigid object cannot change size. If a fit says otherwise the depth is
        # lying, and every metre below is derived from that lie.
        if not target_fit.scale_is_plausible():
            return ServoError3D(ok=False, reason=f"target depth implausible (scale {target_fit.scale:.3f})")
        if not held_fit.scale_is_plausible():
            return ServoError3D(ok=False, reason=f"held depth implausible (scale {held_fit.scale:.3f})")

    desired = target_fit.transform.apply(taught)
    current = held_fit.transform.apply(taught)
    e_t = (desired - current).mean(axis=0)
    e_r = rotation_vector(target_fit.transform.rot @ held_fit.transform.rot.T)
    return ServoError3D(ok=True, e_t=e_t, e_r=e_r)


class PIController:
    """``u = clip(Kp·e + Ki·∫e, v_max)`` with a backlash deadband on the output.

    The deadband belongs on the OUTPUT, not the error: a command smaller than the
    backlash does not move the joint, it winds up the gear train, and issuing it
    teaches the loop that its command had no effect. Suppressing those commands lets
    the integral term accumulate until it can break through in one real move — which
    is why ``integral_limit`` must stay comfortably above ``deadband``, and why the
    constructor refuses a configuration where it is not.
    """

    def __init__(
        self,
        kp: float = 2.5,
        ki: float = 0.0,
        *,
        v_max: float = 0.05,
        deadband: float = 0.0,
        integral_limit: float = 1.0,
    ):
        assert kp > 0 and ki >= 0 and v_max > 0 and deadband >= 0
        assert integral_limit > 0
        if ki > 0 and deadband > 0:
            assert ki * integral_limit > deadband, (
                "integral authority is below the deadband: the loop can never break "
                "through backlash — raise integral_limit or ki, or lower deadband"
            )
        self.kp, self.ki = float(kp), float(ki)
        self.v_max, self.deadband = float(v_max), float(deadband)
        self.integral_limit = float(integral_limit)
        self._integral = np.zeros(0)

    def reset(self) -> None:
        """Pre: called on every re-bind and every retry — a retry starts from a known
        backed-off pose, so integral state earned against the old pose is a lie."""
        self._integral = np.zeros(0)

    def step(self, e: np.ndarray, dt: float) -> np.ndarray:
        """Post: (m,) command, zeroed where it could not overcome backlash."""
        e = np.asarray(e, dtype=np.float64).ravel()
        assert dt > 0, "dt must be positive"
        if self._integral.shape != e.shape:
            self._integral = np.zeros_like(e)

        if self.ki > 0:
            self._integral = np.clip(self._integral + e * dt, -self.integral_limit, self.integral_limit)
        u = self.kp * e + self.ki * self._integral

        norm = float(np.linalg.norm(u))
        if norm > self.v_max:
            u = u * (self.v_max / norm)
        if norm < self.deadband:
            return np.zeros_like(u)
        return u


class JacobianEstimator:
    """Measured mapping joint delta -> image motion, inverted by damped least squares.

    Pre: joint deltas are in the units the bus takes (degrees here) and image motion
    in pixels, consistently for the life of an instance.
    """

    def __init__(self, n_joints: int, m: int = 2, *, damping: float = 1e-2):
        assert n_joints >= 1 and m >= 1 and damping > 0
        self.n_joints, self.m = int(n_joints), int(m)
        self.damping = float(damping)
        self._j: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._j is not None

    @property
    def matrix(self) -> np.ndarray:
        assert self._j is not None, "Jacobian used before the probe seeded it"
        return self._j.copy()

    def seed_from_probe(self, dq: np.ndarray, de: np.ndarray) -> np.ndarray:
        """Least-squares fit from the bootstrap probe. Post: ``ready`` is True.

        Pre: ``dq`` is (K, n_joints) commanded deltas and ``de`` is (K, m) the image
        motion each produced, K >= 1. The probe must actually excite the joints it
        wants columns for: a joint never moved gets a zero column, and DLS will then
        correctly refuse to use it rather than inventing a gain for it.
        """
        dq = np.asarray(dq, dtype=np.float64).reshape(-1, self.n_joints)
        de = np.asarray(de, dtype=np.float64).reshape(-1, self.m)
        assert len(dq) == len(de) and len(dq) >= 1, "probe needs matched dq/de pairs"
        self._j = de.T @ np.linalg.pinv(dq.T)
        assert self._j.shape == (self.m, self.n_joints)
        return self.matrix

    def update(self, dq: np.ndarray, de: np.ndarray, *, min_step: float = 1e-6) -> None:
        """Broyden rank-1 update from one executed step. Pre: ``ready``.

        Steps below ``min_step`` are ignored: dividing by a near-zero ``dq·dq`` turns
        sensor noise into an unbounded Jacobian correction, which is the classic way
        uncalibrated visual servoing destroys itself.
        """
        assert self._j is not None, "update() before seed_from_probe()"
        dq = np.asarray(dq, dtype=np.float64).ravel()
        de = np.asarray(de, dtype=np.float64).ravel()
        assert dq.size == self.n_joints and de.size == self.m
        denom = float(dq @ dq)
        if denom < min_step:
            return
        self._j = self._j + np.outer(de - self._j @ dq, dq) / denom

    def solve(self, e: np.ndarray, *, damping: float | None = None) -> np.ndarray:
        """Damped least squares ``dq = Jᵀ(JJᵀ + λ²I)⁻¹ e``. Post: (n_joints,).

        Damping is what makes this safe near singularities: undamped, a Jacobian that
        has lost rank (the probe under-excited a joint, or the arm is stretched out)
        commands an unbounded joint move for a small image error.
        """
        assert self._j is not None, "solve() before seed_from_probe()"
        e = np.asarray(e, dtype=np.float64).ravel()
        assert e.size == self.m
        lam = self.damping if damping is None else float(damping)
        j = self._j
        return j.T @ np.linalg.solve(j @ j.T + (lam**2) * np.eye(self.m), e)


class ConvergenceCertificate:
    """The error-decreasing check of invariant 5.

    A servo that is not reducing its error is not servoing — it is grinding, and on a
    position-controlled arm with no force sensing that is how things get bent. This
    watches ‖e‖ over a window and reports failure to progress so the monitor can drop
    to the retry ladder instead of running out the clock.
    """

    def __init__(self, *, window: int = 30, min_improvement: float = 0.05):
        assert window >= 2 and 0.0 < min_improvement < 1.0
        self._window = window
        self._min_improvement = float(min_improvement)
        self._history: list[float] = []

    def reset(self) -> None:
        self._history.clear()

    def update(self, err_norm: float) -> None:
        self._history.append(float(err_norm))

    @property
    def progressing(self) -> bool:
        """True while the window is still filling — absence of evidence is not failure."""
        if len(self._history) < self._window:
            return True
        first = self._history[-self._window]
        last = self._history[-1]
        if first <= 1e-9:
            return True
        return (first - last) / first >= self._min_improvement

    @property
    def best(self) -> float:
        return min(self._history) if self._history else float("inf")

    @property
    def energy(self) -> float:
        """Σ‖e‖ over the attempt — the logged scalar that separates a clean convergence
        from one that thrashed its way to the same place (§3.5)."""
        return float(sum(self._history))
