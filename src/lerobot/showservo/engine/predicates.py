# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""How a check answers from the world: an object's measured pose against the two outcomes' predictions of it
(act_execution.md, "What a check answers, and how close counts").

Every answer is success (YES), failure (NO) or can't tell (UNKNOWN), with the evidence. No threshold here is a number
fitted to a rig: each comes from the noise measured on the object in the same act. One count is a choice, named where
it is used: three values for a median.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np


class Answer(enum.Enum):
    YES = "success"
    NO = "failure"
    UNKNOWN = "can't tell"


@dataclasses.dataclass(frozen=True)
class Verdict:
    answer: Answer
    why: str
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def yes(self) -> bool:
        return self.answer is Answer.YES

    @property
    def no(self) -> bool:
        return self.answer is Answer.NO


def _unknown(why: str, **evidence: Any) -> Verdict:
    return Verdict(Answer.UNKNOWN, why, evidence)


MEDIAN_MIN = 3  # a choice: the fewest values whose median ignores one wrong one


def decide(obs: np.ndarray, new: np.ndarray, old: np.ndarray, noise: float, unit: str) -> Verdict:
    """Which of two predictions an observation is: ``new`` (success) or ``old`` (failure), in the same units as the
    ``noise`` measured on what is observed. Undecided until the predictions are more than twice the noise apart;
    then success within half their distance less the noise of ``new``, failure within that of ``old`` or beyond half
    their distance plus the noise from both (something else moved it)."""
    obs, new, old = (np.asarray(x, dtype=float) for x in (obs, new, old))
    apart = float(np.linalg.norm(new - old))
    to_new, to_old = float(np.linalg.norm(obs - new)), float(np.linalg.norm(obs - old))
    ev = {"apart": apart, "to_new": to_new, "to_old": to_old, "noise": noise, "unit": unit}
    if apart <= 2.0 * noise:
        return _unknown(
            f"the two outcomes are {apart:.1f} {unit} apart, within the noise ({noise:.1f})", **ev
        )
    near = apart / 2.0 - noise
    if to_new <= near:
        return Verdict(
            Answer.YES, f"as expected: {to_new:.1f} {unit} from it, {apart:.1f} from the other", ev
        )
    if to_old <= near:
        return Verdict(
            Answer.NO, f"the other outcome: {to_old:.1f} {unit} from it, {apart:.1f} from the expected", ev
        )
    if to_new > apart / 2.0 + noise and to_old > apart / 2.0 + noise:
        return Verdict(
            Answer.NO, f"neither: {to_new:.1f} {unit} from the expected, {to_old:.1f} from the other", ev
        )
    return _unknown(
        f"between the two: {to_new:.1f} {unit} from the expected, {to_old:.1f} from the other", **ev
    )
