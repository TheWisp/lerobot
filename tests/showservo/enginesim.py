# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A small synthetic scene for the act engine's tests: a camera over a table, box-shaped objects, depth frames and
2D tracks rendered from them, an arm whose joints are its fingertip's pose, and a motor loop that plays a playback
queue the way the jog does.

The camera is the rig's (intrinsics and pose from its act records of 2026-10-13: 848 x 480, fx 604, 0.40 m above the
table and 20 degrees off vertical), and objects sit where its axis meets the table (``WORK``), so millimetres and
pixels relate as they do there: a 50 mm vertical lift moves about 25 px in the image. Depth is rendered exactly (the table by ray and plane, objects by a
z-buffer of dense surface points), so a test that fails here fails on geometry, not on noise.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.showservo.engine.scene import Camera, Depth, Tracks, View

W, H = 848, 480
K = np.array([[604.2, 0.0, 419.2], [0.0, 604.2, 250.6], [0.0, 0.0, 1.0]])
T_BC = np.array(
    [
        [-1.0, -0.024, -0.017, -0.178],
        [-0.016, 0.937, -0.350, -0.082],
        [0.025, -0.349, -0.937, 0.401],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_BC[:3, :3] = (
    np.linalg.svd(T_BC[:3, :3])[0] @ np.linalg.svd(T_BC[:3, :3])[2]
)  # the rounded rotation, made exact
WORK = T_BC[:3, 3] + T_BC[:3, 2] * (-T_BC[2, 3] / T_BC[2, 2])  # where the camera's axis meets the table


def camera(work=None) -> Camera:
    """The rig's camera; with ``work``, moved so its axis meets the table at ``work`` as it meets it at ``WORK``."""
    t_bc = T_BC.copy()
    if work is not None:
        t_bc[:3, 3] += np.asarray(work, dtype=float) - WORK
    return Camera(K=K.copy(), t_bc=t_bc, size=(H, W))


def pose(xyz=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    out[:3, 3] = xyz
    return out


def box_points(size_m, centre, step_m=0.001) -> np.ndarray:
    """Points on the surface of an axis-aligned box of ``size_m`` about ``centre``, base frame."""
    sx, sy, sz = (float(s) for s in size_m)
    gx, gy, gz = (np.arange(-s / 2, s / 2 + 1e-9, step_m) for s in (sx, sy, sz))
    faces = []
    for a, b, axis, fixed in ((gx, gy, 2, sz / 2), (gx, gz, 1, sy / 2), (gy, gz, 0, sx / 2)):
        u, v = np.meshgrid(a, b, indexing="ij")
        for sign in (-1.0, 1.0):
            p = np.zeros((u.size, 3))
            free = [k for k in range(3) if k != axis]
            p[:, free[0]], p[:, free[1]] = u.ravel(), v.ravel()
            p[:, axis] = sign * fixed
            faces.append(p)
    return np.concatenate(faces) + np.asarray(centre, dtype=float)


@dataclasses.dataclass
class Body:
    """An object: its surface points at its demo pose (base frame) and its motion since then."""

    points: np.ndarray
    motion: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(4))

    def now(self) -> np.ndarray:
        return self.points @ self.motion[:3, :3].T + self.motion[:3, 3]

    @property
    def centre(self) -> np.ndarray:
        return self.points.mean(axis=0)


def render_depth(cam: Camera, bodies: list[np.ndarray], t: float = 0.0, table_z: float = 0.0) -> Depth:
    """Depth (metres, camera z) of the table at ``table_z`` and every body's points in front of it."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    r_bc, origin = cam.t_bc[:3, :3], cam.t_bc[:3, 3]
    dirs = rays @ r_bc.T  # base frame, per unit of camera z
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (table_z - origin[2]) / dirs[..., 2]
    depth = np.where(z > 0, z, 0.0)
    nearest = np.full((H, W), np.inf)
    for pts in bodies:
        uv, zc = cam.project(pts)
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (zc > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        for du in (0, 1):  # each point covers 2 x 2 pixels, so a millimetre's spacing leaves no holes
            for dv in (0, 1):
                np.minimum.at(
                    nearest, (np.clip(vi[ok] + dv, 0, H - 1), np.clip(ui[ok] + du, 0, W - 1)), zc[ok]
                )
    depth = np.where(np.isfinite(nearest) & ((depth == 0) | (nearest < depth)), nearest, depth)
    return Depth(t=t, metres=depth)


def track_points(body: Body, n: int = 24, seed: int = 0) -> np.ndarray:
    """Indices of ``n`` of the body's surface points that its tracks follow."""
    return np.random.default_rng(seed).choice(len(body.points), size=n, replace=False)


def render_tracks(
    cam: Camera, body: Body, rows: np.ndarray, t: float, hidden: np.ndarray | None = None
) -> Tracks:
    """The body's tracked points projected now; ``hidden`` (per row) marks those the gripper covers."""
    uv, _ = cam.project(body.now()[rows])
    visible = np.ones(len(rows), dtype=bool) if hidden is None else ~np.asarray(hidden, dtype=bool)
    return Tracks(t=t, ids=np.arange(len(rows)), uv=uv, visible=visible)


# ── an arm, a motor loop and a world for the engine's tests ──

DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # the fingertip pointing straight down


class FakeKin:
    """Joints that are the fingertip's pose: x, y, z at ``PER_M`` degrees a metre (so the joint bound means about
    the rig's speeds), then its turn from pointing down as a rotation vector in degrees; the gripper last."""

    PER_M = 500.0
    GRIPPER = 6

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        out = np.eye(4)
        out[:3, :3] = DOWN @ Rotation.from_rotvec(np.radians(q[3:6])).as_matrix()
        out[:3, 3] = q[:3] / self.PER_M
        return out

    def inverse_kinematics(self, seed, target):
        q = np.asarray(seed, dtype=float).copy()
        q[:3] = np.asarray(target)[:3, 3] * self.PER_M
        q[3:6] = np.degrees(Rotation.from_matrix(DOWN.T @ np.asarray(target)[:3, :3]).as_rotvec())
        return q


class Motor:
    """The motor loop as the jog runs a live playback: each tick it sends the newest due command, never skipping one
    where the gripper starts, stops or turns back, and moves each arm joint at most ``step_deg`` from the command
    before. The arm follows its command a share ``follow`` of the way each tick."""

    def __init__(self, q0, hz=30.0, step_deg=3.0, follow=0.5, gripper=FakeKin.GRIPPER):
        self.samples, self.due = [np.asarray(q0, dtype=float).copy()], [0.0]
        self.sent_i, self.t0, self.now = -1, 0.0, 0.0
        self.q_cmd = np.asarray(q0, dtype=float).copy()
        self.q_obs = self.q_cmd.copy()
        self.hz, self.step_deg, self.follow, self.gi = hz, step_deg, follow, gripper
        self.log = []  # (t, sent index, q_cmd, q_obs)
        self.clamped = 0.0  # the most a command was held back by the per-tick bound

    def start(self, now):
        self.t0 = self.now = now

    def sent(self):
        return self.sent_i

    def clock(self):
        return self.now - self.t0

    def append(self, expected_tail, samples, due):
        if expected_tail != len(self.samples) - 1:
            return False
        assert all(b >= a for a, b in zip([self.due[-1], *due[:-1]], due, strict=True)), (
            "due times go forward"
        )
        self.samples.extend(np.asarray(s, dtype=float).copy() for s in samples)
        self.due.extend(due)
        return True

    def cancel(self, expected_sent):
        if expected_sent != self.sent_i:
            return False
        del self.samples[expected_sent + 1 :]
        del self.due[expected_sent + 1 :]
        return True

    def _turns(self, k):
        g = self.gi
        prev = self.samples[k - 1][g] if k > 0 else self.samples[k][g]
        if k + 1 >= len(self.samples):
            return True
        way = [
            np.sign(round(b - a, 6))
            for a, b in ((prev, self.samples[k][g]), (self.samples[k][g], self.samples[k + 1][g]))
        ]
        return way[0] != way[1]

    def tick(self, now):
        self.now = now
        rel = now - self.t0
        k = self.sent_i + 1
        if k < len(self.samples) and self.due[k] <= rel + 1e-9:
            while k + 1 < len(self.samples) and self.due[k + 1] <= rel + 1e-9 and not self._turns(k):
                k += 1
            self.sent_i = k
        if self.sent_i >= 0:
            target = self.samples[self.sent_i]
            arm = np.arange(len(target)) != self.gi
            step = np.clip(target - self.q_cmd, -self.step_deg, self.step_deg)
            new = np.where(arm, self.q_cmd + step, target)
            self.clamped = max(self.clamped, float(np.max(np.abs(target - new)[arm])))
            self.q_cmd = new
        self.q_obs = self.q_obs + (self.q_cmd - self.q_obs) * self.follow
        self.log.append((now, self.sent_i, self.q_cmd.copy(), self.q_obs.copy()))


class World:
    """Objects on a table, the gripper's contact, and what the camera reports, 9.5 frames a second as on the rig.

    The gripper holds an object when it closes past ``close_at`` with the fingertip within ``reach_m`` of the
    object's centre, unless ``misses`` says to miss that many grasps; it lets go when it opens again, and the object
    falls straight down onto whatever is under its centre. Views are reported only of objects not held (the rig's
    pose rule rejects a held object's), with ``view_noise_m`` of noise; tracked points are hidden where something
    nearer covers them; depth is exact to the millimetre, as the rig's depth images are."""

    def __init__(
        self, cam, bodies, kin, gripper=FakeKin.GRIPPER, close_at=95.0, reach_m=0.015, seed=0, table_z=0.0
    ):
        self.cam, self.bodies, self.kin, self.gi = cam, bodies, kin, gripper
        self.table_z = table_z
        self.close_at, self.reach_m = close_at, reach_m
        self.rows = {n: track_points(b, n=30, seed=k) for k, (n, b) in enumerate(bodies.items())}
        self.held, self.hold = None, None
        self.misses = 0
        self.view_noise_m, self.track_noise_px = 0.0, 0.0
        self.hidden: set[str] = set()
        self.rng = np.random.default_rng(seed)
        self.last_grip = None
        self.events = []  # (t, what)
        self.frame_k = -1
        self.at = []  # scheduled changes: (t, fn)

    def gripper_points(self, tip):
        """The fingers, as the depth camera sees them: a 30 x 30 x 50 mm block above the fingertip."""
        c = (tip @ np.array([0.0, 0.0, -0.03, 1.0]))[:3]
        return box_points((0.03, 0.03, 0.05), c, step_m=0.002)

    def _support_z(self, name, xy):
        top = self.table_z
        for other, b in self.bodies.items():
            if other == name:
                continue
            pts = b.now()
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            if lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]:
                top = max(top, hi[2])
        return top

    def drop(self, name, t):
        b = self.bodies[name]
        pts = b.now()
        fall = pts[:, 2].min() - self._support_z(name, pts.mean(axis=0)[:2])
        m = b.motion.copy()
        m[2, 3] -= fall
        b.motion = m
        self.events.append((t, f"{name} fell {fall * 1000:.0f} mm"))

    def physics(self, t, q_obs):
        for when, fn in list(self.at):
            if t >= when:
                fn(self, t)
                self.at.remove((when, fn))
        tip = self.kin.forward_kinematics(q_obs)
        g = float(q_obs[self.gi])
        closing = self.last_grip is not None and g > self.last_grip
        if self.held is not None:
            self.bodies[self.held].motion = tip @ self.hold
            if g < self.close_at - 2.0:
                name, self.held = self.held, None
                self.events.append((t, f"let go of {name}"))
                self.drop(name, t)
        elif closing and self.last_grip < self.close_at <= g:
            near = min(
                self.bodies, key=lambda n: np.linalg.norm(self.bodies[n].now().mean(axis=0) - tip[:3, 3])
            )
            if np.linalg.norm(self.bodies[near].now().mean(axis=0) - tip[:3, 3]) <= self.reach_m:
                if self.misses > 0:
                    self.misses -= 1
                    self.events.append((t, f"missed {near}"))
                else:
                    self.held, self.hold = near, np.linalg.inv(tip) @ self.bodies[near].motion
                    self.events.append((t, f"holding {near}"))
        self.last_grip = g

    def perceive(self, t, scene, tip):
        """Report a camera frame at ``t`` when one is due (9.5 a second)."""
        k = int(np.floor(t * 9.5 + 1e-9))
        if k == self.frame_k:
            return False
        self.frame_k = k
        depth = render_depth(
            self.cam,
            [b.now() for b in self.bodies.values()] + [self.gripper_points(tip)],
            t=t,
            table_z=self.table_z,
        )
        depth = Depth(t=t, metres=np.round(depth.metres, 3))
        scene.add_depth(depth)
        for name, b in self.bodies.items():
            tr = render_tracks(self.cam, b, self.rows[name], t)
            uv, z = self.cam.project(b.now()[self.rows[name]])
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
            covered = depth.metres[vi, ui] < z - 0.004
            uvn = (
                tr.uv + self.rng.normal(0.0, self.track_noise_px, tr.uv.shape)
                if self.track_noise_px
                else tr.uv
            )
            visible = ~covered & (name not in self.hidden)
            scene.add_tracks(name, Tracks(t=t, ids=tr.ids, uv=uvn, visible=visible))
            if name != self.held:
                m = b.motion.copy()
                if self.view_noise_m:
                    m[:3, 3] += self.rng.normal(0.0, self.view_noise_m, 3)
                scene.add_view(name, View(t=t, motion=m))
        return True


def pick_place_demo(
    fps=30.0, dense=False, work=None, rot=None, kin=None, seed_q=None, gripper=FakeKin.GRIPPER
):
    """A demo of picking a 40 x 20 x 12 mm block off the table and setting it on a 40 x 40 x 20 mm block, made of
    timed waypoints (straight at a steady speed between them), with its key frames as ``core.key_frames`` makes them.
    The arm stands while the gripper closes and opens, as a teleoperator does. ``dense`` marks a key frame on every
    demo frame of the grasp. ``work`` (where the block lies, on the table), ``rot`` (the fingertip's turn
    throughout), ``kin`` and ``seed_q`` make the same demo for another arm: its joints are solved along the path from
    ``seed_q``."""
    from lerobot.showservo.engine.timeline import BASE, Demo

    work = WORK if work is None else np.asarray(work, dtype=float)
    dowel_c = work + np.array([0.0, 0.0, 0.006])
    ring_c = work + np.array([0.08, 0.05, 0.010])
    bodies = {
        "dowel": Body(box_points((0.040, 0.020, 0.012), dowel_c)),
        "ring": Body(box_points((0.04, 0.04, 0.02), ring_c)),
    }
    on_ring = ring_c + np.array([0.0, 0.0, 0.010 + 0.006])  # the block's centre, lying on top of the ring
    turn = DOWN if rot is None else np.asarray(rot, dtype=float)

    def tip(p):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = turn, p
        return out

    way = [  # (t, fingertip centre, gripper command)
        (0.0, dowel_c + (-0.05, -0.06, 0.12), 20.0),
        (1.5, dowel_c + (0.0, 0.0, 0.05), 20.0),
        (2.3, dowel_c, 20.0),
        (2.8, dowel_c, 100.0),
        (3.0, dowel_c, 100.0),
        (3.6, dowel_c + (0.0, 0.0, 0.06), 100.0),
        (5.2, on_ring + (0.0, 0.0, 0.05), 100.0),
        (6.0, on_ring, 100.0),
        (6.1, on_ring, 100.0),
        (6.5, on_ring, 20.0),
        (6.8, on_ring, 20.0),
        (7.6, on_ring + (0.0, 0.0, 0.06), 20.0),
    ]
    n = int(np.floor(7.6 * fps)) + 1
    t = np.arange(n) / fps
    wt = np.array([w[0] for w in way])
    xyz = np.stack([np.interp(t, wt, [w[1][k] for w in way]) for k in range(3)], axis=1)
    grip = np.interp(t, wt, [w[2] for w in way])
    grip_obs = np.where((t >= 2.3) & (t <= 6.5), np.minimum(grip, 97.9), grip)  # the block stops the fingers
    tips = np.stack([tip(p) for p in xyz])
    if kin is None:
        kin = FakeKin()
        q = np.stack([kin.inverse_kinematics(np.zeros(7), p) for p in tips])
    else:  # a real arm records its fingertip from its joints: solve each pose to the act's tolerance, then read it back
        from lerobot.showservo import motion as core

        n_q = len(seed_q)
        solved = core.solve_plan_joints(
            kin, tips, grip, np.zeros((len(tips), n_q)), np.asarray(seed_q, dtype=float), gripper
        )
        q = solved["q"]
        tips = np.stack([np.asarray(kin.forward_kinematics(qi), dtype=float) for qi in q])
        # The objects lie where this arm's fingertip actually closed and opened, as a recorded demo's do.
        at = lambda when: tips[int(np.argmin(np.abs(t - when)))][:3, 3]  # noqa: E731
        bodies["dowel"] = Body(box_points((0.040, 0.020, 0.012), at(2.8)))
        bodies["ring"] = Body(box_points((0.04, 0.04, 0.02), at(6.3) - np.array([0.0, 0.0, 0.016])))
    q[:, gripper] = grip
    demo = Demo(t=t, tips=tips, grip=grip, grip_obs=grip_obs, q=q)
    kf = [
        {
            "t": 1.5,
            "frame": "dowel",
            "move": "line",
            "grip": "first",
            "constraint": "pose",
            "label": "pre-grasp 1",
        },
        {
            "t": 3.0,
            "frame": "dowel",
            "move": "follow",
            "grip": "first",
            "constraint": "pose",
            "label": "grasp",
        },
        {
            "t": 3.6,
            "frame": BASE,
            "move": "lift",
            "grip": "first",
            "up_mm": 60.0,
            "constraint": "best_effort",
            "label": "lift 1",
        },
        {
            "t": 5.2,
            "frame": "ring",
            "move": "line",
            "grip": "first",
            "constraint": "pose",
            "label": "pre-place 1",
        },
        {
            "t": 6.8,
            "frame": "ring",
            "move": "follow",
            "grip": "first",
            "constraint": "pose",
            "label": "place",
        },
        {
            "t": 7.6,
            "frame": "ring",
            "move": "line",
            "grip": "first",
            "constraint": "pose",
            "label": "key frame 1",
        },
    ]
    if dense:
        extra = [
            {
                "t": float(x),
                "frame": "dowel",
                "move": "follow",
                "grip": "first",
                "constraint": "pose",
                "label": f"dense {k}",
            }
            for k, x in enumerate(t[(t > 1.5 + 1e-9) & (t < 3.0 - 1e-9)])
        ]
        kf = sorted(kf + extra, key=lambda k: k["t"])
    return demo, kf, bodies


def visible_surface(cam, body, table_z=0.0):
    """What the act knows of an object's shape: the points its camera saw of it, at its demo pose."""
    uv, z = cam.project(body.points)
    seen = render_depth(cam, [body.points], table_z=table_z)
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    return body.points[np.abs(seen.metres[vi, ui] - z) < 0.002]


@dataclasses.dataclass
class Sim:
    engine: object
    world: World
    motor: Motor
    scene: object
    started: float


def run_act(
    demo,
    key_frames,
    bodies,
    speed=1.0,
    perturb=None,
    setup=None,
    until_s=40.0,
    step_deg=3.0,
    settings=None,
    kin=None,
    gripper=FakeKin.GRIPPER,
    seed_q=None,
    work=None,
):
    """One act in the simulated world: the arm starts where the demo did, the camera has seen the objects at rest for
    a second, and the engine steps once per 30 Hz tick before the motor loop does."""
    from lerobot.showservo.engine.engine import Arm, Engine
    from lerobot.showservo.engine.objects import Objects
    from lerobot.showservo.engine.scene import SceneLog
    from lerobot.showservo.engine.timeline import Timeline

    cam = camera(work)
    kin = FakeKin() if kin is None else kin
    scene = SceneLog(cam, keep_s=120.0)
    world = World(cam, bodies, kin, gripper=gripper, table_z=0.0 if work is None else float(work[2]))
    if setup is not None:
        setup(world)
    q0 = (
        np.asarray(demo.q[0], dtype=float).copy()
        if seed_q is not None
        else kin.inverse_kinematics(np.zeros(7), demo.tips[0])
    )
    q0[gripper] = demo.grip[0]
    t, dt = 0.0, 1.0 / 30.0
    for _ in range(30):
        world.perceive(t, scene, kin.forward_kinematics(q0))
        scene.add_tip(t, kin.forward_kinematics(q0))
        t += dt
    motor = Motor(q0, step_deg=step_deg, gripper=gripper)
    arm = Arm(kin, gripper=gripper, step_deg=step_deg, hz=30.0, walk_m_s=0.04, walk_rad_s=np.radians(30.0))
    objects = Objects(
        scene,
        {n: b.motion.copy() for n, b in bodies.items()},
        {n: visible_surface(cam, Body(b.points), world.table_z) for n, b in bodies.items()},
        arm.walk_m_s,
        arm.walk_rad_s,
    )
    engine = Engine(
        Timeline(demo, key_frames),
        arm,
        scene,
        motor,
        objects,
        speed=speed,
        perturb=perturb,
        settings=settings,
    )
    motor.start(t)
    engine.start(q0, t)
    started = t
    while not engine.done and t < started + until_s:
        t += dt
        tip = kin.forward_kinematics(motor.q_obs)
        world.perceive(t, scene, tip)
        scene.add_tip(t, tip)
        engine.step(t)
        motor.tick(t)
        world.physics(t, motor.q_obs)
    return Sim(engine, world, motor, scene, started)
