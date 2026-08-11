# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Show-and-Servo: few-demo manipulation by closed-loop visual servoing.

PROTOTYPE (v0). A demonstration is mined for BOUNDARY CONDITIONS — keyframe relations
between tracked visual features — and its motion is discarded. Motion is regenerated
at runtime by a visual servo that measures both ends of the relation in one camera,
so calibration, FK and grasp-offset errors cancel in the difference. Every stage
emits a certificate or abstains; failures are detected and retried, never absorbed.

See ``README.md`` in this package for the design invariants, what is proven versus
merely measured, and what still needs the rig.
"""

from lerobot.showservo.binder import BindGate, BindResult, DinoBinder, SiftBinder, sift_keypoints
from lerobot.showservo.card import (
    Budget,
    Card,
    Chapter,
    GoalRelation,
    Keypoint,
    Termination,
)
from lerobot.showservo.grouping import (
    AttachmentEvent,
    AttachmentMonitor,
    TeamFit,
    evict,
    fit_team,
)
from lerobot.showservo.monitor import (
    AttemptLog,
    ChapterMonitor,
    Decision,
    Event,
    Rung,
    State,
)
from lerobot.showservo.servo import (
    ConvergenceCertificate,
    JacobianEstimator,
    PIController,
    ServoError,
    servo_error,
)
from lerobot.showservo.tracker import KLTTracker, TrackState, shi_tomasi_points

__all__ = [
    "AttachmentEvent",
    "AttachmentMonitor",
    "AttemptLog",
    "BindGate",
    "BindResult",
    "Budget",
    "Card",
    "Chapter",
    "ChapterMonitor",
    "ConvergenceCertificate",
    "Decision",
    "DinoBinder",
    "Event",
    "GoalRelation",
    "JacobianEstimator",
    "KLTTracker",
    "Keypoint",
    "PIController",
    "Rung",
    "ServoError",
    "SiftBinder",
    "State",
    "TeamFit",
    "Termination",
    "TrackState",
    "evict",
    "fit_team",
    "servo_error",
    "shi_tomasi_points",
    "sift_keypoints",
]
