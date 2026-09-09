"""The ladder's rule, driven directly.

For a week this rule could only be exercised by running a browser against an
emulated link, and it collected the kind of defect that survives an end-to-end
test: a step judged on the round trip rather than the transfer, a single
window's evidence moving the rung and the next window moving it back, a
counter that was never declared. None of them is visible from "playback did
not hold"; all of them are obvious when the decision is a function you can
call with a state and an expectation.

`window_player.js` exports that function. These tests run it in node, with no
server, no video and no link, and enumerate what it must decide. The
emulated-link profiles in test_window_adaptation.py remain the integration
check: the two answer different questions, and the rule's own arithmetic
belongs here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PLAYER = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static" / "window_player.js"

#: The ladder the server serves, and the cost of a rung per second of media at
#: its nominal cap, which is what the page starts from before it has measured.
KBPS = {"160": 150, "320": 300, "640": 800, "1280": 1500}
RUNGS = ["160", "320", "640", "1280"]
LENGTHS = [0.5, 1.0, 2.0, 4.0]


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node


def run_rule(calls: list[dict]) -> list:
    """Run a list of ``{fn, state}`` calls against the exported rule.

    ``state.cost`` is a mapping rung -> bytes per second of media (or a single
    number for ``nextLength``); it becomes the function the rule expects. A
    call that raises comes back as ``{"violation": message}``.
    """
    script = """
      const fs = require('fs');
      const src = fs.readFileSync(process.argv[1], 'utf8');
      const win = {};
      new Function('window', 'performance', 'localStorage', src)(
        win, { now: () => 0 }, { getItem: () => null, setItem: () => {} },
      );
      const R = win.WindowPlayer;
      const out = [];
      for (const call of JSON.parse(process.argv[2])) {
        const s = Object.assign({}, call.state);
        if (call.fn === 'nextRung') { const c = s.cost; s.cost = (r) => c[r]; }
        if (call.fn === 'fetchPlan') { const k = call.allowedIndex; s.allowedIndex = () => k; }
        try {
          out.push(R[call.fn](s));
        } catch (e) {
          out.push({ violation: e.message, kind: e.constructor.name });
        }
      }
      process.stdout.write(JSON.stringify(out));
    """
    proc = subprocess.run(
        [_node(), "-e", script, str(PLAYER), json.dumps(calls)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def rung_state(**over) -> dict:
    """A state at the bottom of the ladder on a link that carries everything."""
    s = {
        "rungs": RUNGS,
        "kbps": KBPS,
        "rung": "160",
        "buffered": 4.0,
        "rate": 1,
        "linkMbps": 10.0,
        "cost": {r: KBPS[r] * 1000 / 8 for r in RUNGS},  # bytes per second of media
        "slow": False,
        "upVotes": 0,
    }
    s.update(over)
    return s


def length_state(**over) -> dict:
    s = {
        "lengths": LENGTHS,
        "buffered": 4.0,
        "rate": 1,
        "linkMbps": None,
        "cost": None,
        "inflight": 0,
        "rttMs": 250,
        "buildFixed": 60,
        "buildPerSec": 90,
    }
    s.update(over)
    return s


def rungs(*states) -> list[dict]:
    return run_rule([{"fn": "nextRung", "state": s} for s in states])


def lengths(*states) -> list[int]:
    return run_rule([{"fn": "nextLength", "state": s} for s in states])


# ── stepping up ─────────────────────────────────────────────────────────────


def test_a_step_up_needs_two_windows_of_evidence():
    """One window's evidence moved the rung and the next moved it back, twice a
    second, on the reviewer's link."""
    first, second = rungs(rung_state(upVotes=0), rung_state(upVotes=1))
    assert first == {"rung": "160", "upVotes": 1, "reason": "hold"}
    assert second == {"rung": "320", "upVotes": 0, "reason": "up"}


def test_a_step_up_is_one_rung_even_when_the_link_carries_the_top():
    """A link with room for 1280 still climbs 160 -> 320: the cost of a rung is
    measured, not assumed, so each step must be paid for by a window at it."""
    out = rungs(rung_state(upVotes=1, linkMbps=1000.0))[0]
    assert out["rung"] == "320"


def test_the_ladder_is_ordered_by_quality_not_by_cost():
    """`full` is the archive's own samples: on a well-encoded archive it can
    cost less per second than a rung transcoded in a hundred milliseconds. An
    invariant that the ladder rises in cost fires on exactly those datasets,
    which is how this one was found."""
    rungs_with_full = [*RUNGS, "full"]
    kbps = {**KBPS, "full": 5}  # a tiny synthetic archive, cheaper than 1280
    out = rungs(
        rung_state(
            rungs=rungs_with_full,
            kbps=kbps,
            rung="1280",
            upVotes=1,
            linkMbps=10.0,
            cost={**{r: KBPS[r] * 1000 / 8 for r in RUNGS}, "full": 5 * 1000 / 8},
        )
    )[0]
    assert out == {"rung": "full", "upVotes": 0, "reason": "up"}, out


def test_the_top_of_the_ladder_holds():
    out = rungs(rung_state(rung="1280", upVotes=1, linkMbps=1000.0))[0]
    assert out == {"rung": "1280", "upVotes": 0, "reason": "hold"}


def test_a_thin_buffer_withholds_the_step_up_and_forgets_the_vote():
    """Slack in the buffer is what pays for a bigger window; without it the
    evidence does not accumulate."""
    out = rungs(rung_state(upVotes=1, buffered=0.8))[0]
    assert out == {"rung": "160", "upVotes": 0, "reason": "hold"}


def test_without_a_measured_rate_a_full_buffer_stands_in():
    """At the lowest rung no window is large enough to measure a rate. A buffer
    at three quarters of its target is then the evidence: the link outpaces
    this rung by some unknown amount, which is worth one step."""
    voted, thin = rungs(
        rung_state(linkMbps=None, upVotes=1, buffered=3.5),
        rung_state(linkMbps=None, upVotes=1, buffered=1.5),
    )
    assert voted["rung"] == "320"
    assert thin["rung"] == "160", "a buffer below the target is not evidence of headroom"


# ── stepping down ───────────────────────────────────────────────────────────


def test_a_rate_that_no_longer_carries_the_rung_steps_down_at_once():
    out = rungs(rung_state(rung="1280", linkMbps=0.5))[0]
    assert out == {"rung": "640", "upVotes": 0, "reason": "down:rate"}


def test_a_thin_buffer_and_a_slow_transfer_step_down():
    out = rungs(rung_state(rung="640", buffered=0.3, slow=True, linkMbps=None))[0]
    assert out == {"rung": "320", "upVotes": 0, "reason": "down:thin"}


def test_a_slow_transfer_with_a_healthy_buffer_holds():
    """A window slower than it plays while four seconds are buffered is the
    fetcher running ahead, not the link failing."""
    out = rungs(rung_state(rung="640", buffered=4.0, slow=True, linkMbps=None))[0]
    assert out["reason"] != "down:thin"


def test_the_bottom_of_the_ladder_holds_however_bad_the_link():
    out = rungs(rung_state(rung="160", buffered=0.0, slow=True, linkMbps=0.01))[0]
    assert out == {"rung": "160", "upVotes": 0, "reason": "hold"}


def test_the_margin_is_what_decides_a_borderline_link():
    """A rung that exactly fits the measured rate is not carried: the margin
    exists so a link at the edge does not spend the buffer proving it."""
    exact = KBPS["320"] / 1000  # Mbit/s, the nominal cost of 320
    fits, doesnt = rungs(
        rung_state(rung="320", linkMbps=exact * 1.25),
        rung_state(rung="320", linkMbps=exact * 1.15),
    )
    assert fits["reason"] == "hold"
    assert doesnt == {"rung": "160", "upVotes": 0, "reason": "down:rate"}


def test_the_playback_rate_scales_what_a_rung_costs():
    """At 2x the page needs twice the bytes per second of wall time, so a link
    that carries a rung at 1x may not carry it at 2x."""
    at_1x, at_2x = rungs(
        rung_state(rung="640", linkMbps=1.2, rate=1),
        rung_state(rung="640", linkMbps=1.2, rate=2, buffered=8.0),
    )
    assert at_1x["reason"] == "hold"
    assert at_2x == {"rung": "320", "upVotes": 0, "reason": "down:rate"}


# ── the rule does not hunt on a link that does not move ─────────────────────


def _settle(link: float, start: str, rounds: int = 20) -> list[str]:
    """Feed the rule its own choice back, on a link that never changes."""
    state = rung_state(rung=start, linkMbps=link, buffered=4.0)
    chosen = []
    for _ in range(rounds):
        out = rungs(state)[0]
        chosen.append(out["rung"])
        state = rung_state(rung=out["rung"], linkMbps=link, buffered=4.0, upVotes=out["upVotes"])
    return chosen


def _highest_that_fits(link: float) -> str:
    """The rung the rule should settle on: the highest whose cost, at the
    margin, the link carries. Derived from the same numbers the rule reads, so
    this stays true if the ladder or the margin changes."""
    fits = [r for r in RUNGS if (KBPS[r] / 1000) * 1.2 <= link]
    return fits[-1] if fits else RUNGS[0]


@pytest.mark.parametrize("link", [0.2, 0.45, 1.0, 3.0])
def test_a_stationary_link_settles_and_stops_moving(link):
    """The property the reviewer's session violated: on a link that does not
    change, the rung must converge -- to the highest rung that link carries --
    and then stay there. His log showed 24 changes across 99 windows."""
    chosen = _settle(link, "160")
    want = _highest_that_fits(link)
    assert chosen[-6:] == [want] * 6, (link, chosen)
    changes = sum(1 for a, b in zip(chosen, chosen[1:], strict=False) if a != b)
    assert changes <= RUNGS.index(want) + 1, f"the rung moved {changes} times on a fixed link: {chosen}"


@pytest.mark.parametrize("link", [0.2, 0.45, 1.0])
def test_it_settles_on_the_same_rung_coming_down(link):
    """Starting above the link's capacity must reach the same answer as
    starting below it, or the rung depends on where the session began."""
    assert _settle(link, "1280")[-3:] == [_highest_that_fits(link)] * 3


# ── the invariants, at runtime ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("over", "says"),
    [
        ({"rung": "999"}, "not a rung"),
        ({"rungs": []}, "ladder is empty"),
        ({"rungs": ["160", "320", "320"]}, "repeats a rung"),
        ({"kbps": {"160": 0, "320": 300, "640": 800, "1280": 1500}}, "no cost"),
        ({"rate": 0}, "playback rate"),
    ],
)
def test_a_broken_input_is_a_reported_violation(over, says):
    """These are programming errors, not link conditions: they are raised, and
    the player records them where every test can see them."""
    out = rungs(rung_state(**over))[0]
    assert out.get("kind") == "RuleViolation", out
    assert says in out["violation"], out


def test_a_length_below_the_buffer_floor_is_always_the_shortest():
    assert lengths(length_state(buffered=0.1))[0] == 0


def test_the_length_grows_with_the_buffer():
    got = lengths(*[length_state(buffered=b) for b in (0.3, 0.9, 2.0, 4.0)])
    assert got == [0, 1, 2, 3], got


def test_the_length_is_bounded_by_what_can_arrive_in_time():
    """Four seconds buffered allows a four-second window only if it can arrive
    before those four seconds run out."""
    cost = 800 * 1000 / 8  # bytes per second of media at the 640 rung
    fast, slow = lengths(
        length_state(buffered=4.0, cost=cost, linkMbps=10.0),
        length_state(buffered=4.0, cost=cost, linkMbps=0.9),
    )
    assert fast == 3
    assert slow < 3, "a window that cannot arrive in time was chosen anyway"


def test_two_requests_in_flight_share_the_link():
    cost = 800 * 1000 / 8
    alone, shared = lengths(
        length_state(buffered=4.0, cost=cost, linkMbps=2.0, inflight=0),
        length_state(buffered=4.0, cost=cost, linkMbps=2.0, inflight=1),
    )
    assert shared <= alone, (alone, shared)


def test_a_length_state_that_cannot_be_satisfied_is_a_violation():
    out = run_rule([{"fn": "nextLength", "state": length_state(lengths=[], buffered=1.0)}])[0]
    assert out.get("kind") == "RuleViolation" and "no window lengths" in out["violation"], out


# ── the buffer walk ─────────────────────────────────────────────────────────


def coverage(**over) -> int:
    s = {"held": [], "clock": 0, "rangeStart": 0, "rangeEnd": 40}
    s.update(over)
    return run_rule([{"fn": "coverageFrom", "state": s}])[0]


def whole(start: int, end: int) -> dict:
    """A window whose frames are all decoded."""
    return {"start": start, "end": end, "readyTo": end}


def test_coverage_counts_contiguous_ready_media_from_the_clock():
    assert coverage(held=[whole(0, 40)], clock=10) == 40 - 10 + 10, "the walk must wrap"
    assert coverage(held=[whole(0, 20)], clock=5) == 15
    assert coverage(held=[whole(0, 20), whole(20, 40)], clock=30) == 40


def test_coverage_stops_at_the_first_gap():
    assert coverage(held=[whole(0, 10), whole(20, 30)], clock=0) == 10


def test_coverage_stops_inside_a_window_still_decoding():
    assert coverage(held=[{"start": 0, "end": 20, "readyTo": 7}], clock=0) == 7
    assert coverage(held=[{"start": 0, "end": 20, "readyTo": 7}], clock=8) == 0


def test_a_fully_buffered_episode_is_counted_once():
    """The invariant that caught this: walking past the wrap counted the window
    the clock sits in a second time, so a 40-frame episode reported 70 frames
    buffered and the rule read slack that was not there."""
    for clock in (0, 1, 13, 39):
        assert coverage(held=[whole(0, 40)], clock=clock) == 40, clock


def test_a_clock_outside_the_range_is_read_as_the_range_start():
    """A trim dragged while paused leaves the clock outside the range until
    play moves it in. That is a state, not an error: the walk starts where
    playback will, so the buffer is measured for the range that is about to be
    played rather than reported as empty."""
    held = [{"start": 10, "end": 20, "readyTo": 20}]
    assert coverage(held=held, clock=0, rangeStart=10, rangeEnd=20) == 10
    out = run_rule(
        [{"fn": "coverageFrom", "state": {"held": [], "clock": 0, "rangeStart": 20, "rangeEnd": 20}}]
    )[0]
    assert out.get("kind") == "RuleViolation" and "empty range" in out["violation"], out


# ── the fetch plan ──────────────────────────────────────────────────────────


def plan(**over) -> list[dict]:
    s = {
        "held": [],
        "inflight": [],
        "clock": 0,
        "rangeStart": 0,
        "rangeEnd": 400,
        "fps": 10,
        "targetFrames": 40,
        "maxInflight": 2,
        "lengths": [0.5, 1.0, 2.0, 4.0],
        "allowedIndex": 0,
    }
    s.update(over)
    index = s.pop("allowedIndex")
    return run_rule([{"fn": "fetchPlan", "state": s, "allowedIndex": index}])[0]


def test_the_plan_asks_for_the_grid_cell_at_each_gap():
    assert plan(clock=7) == [{"start": 5, "len": 0.5}, {"start": 10, "len": 0.5}]


def test_the_plan_skips_what_is_held_or_already_in_flight():
    """The walk steps over the held window and the standing request and asks
    for the gap after them -- one window, because one request already stands
    and two is the limit."""
    got = plan(clock=0, held=[{"start": 0, "end": 20}], inflight=[{"start": 20, "end": 25}])
    assert got == [{"start": 25, "len": 0.5}], got


def test_the_plan_never_exceeds_the_requests_in_flight():
    assert plan(inflight=[{"start": 100, "end": 105}, {"start": 105, "end": 110}]) == []
    assert len(plan(inflight=[{"start": 100, "end": 105}])) == 1


def test_the_plan_stops_at_the_target():
    """Five half-second windows cover the target of four seconds, but only two
    may stand at once."""
    assert len(plan(targetFrames=40, maxInflight=9)) == 8


def test_a_longer_window_still_starts_on_the_shortest_grid_cell():
    """The window covering the gap starts at the gap's own half-second cell and
    takes the longest length that cell also sits on. Starting at the chosen
    length's cell instead would begin before media already held; the fetch is
    then dropped as a duplicate and the buffer stops growing."""
    deep = {"allowedIndex": 3, "maxInflight": 1, "targetFrames": 200}
    assert plan(clock=0, held=[{"start": 0, "end": 5}], **deep) == [{"start": 5, "len": 0.5}]
    assert plan(clock=0, held=[{"start": 0, "end": 20}], **deep) == [{"start": 20, "len": 2.0}]
    assert plan(clock=0, held=[{"start": 0, "end": 40}], **deep) == [{"start": 40, "len": 4.0}]


def test_the_plan_wraps_and_stops_at_the_clock():
    """Near the end of a short range the walk continues from its start, and
    stops when it comes back to the clock rather than looping forever."""
    got = plan(clock=35, rangeEnd=40, maxInflight=9, targetFrames=400)
    assert [w["start"] for w in got] == [35, 0, 5, 10, 15, 20, 25, 30], got


# ── eviction ────────────────────────────────────────────────────────────────


def drop(**over) -> list[int]:
    s = {"held": [], "clock": 0, "rangeStart": 0, "rangeEnd": 400, "keepBehind": 20, "keepAhead": 60}
    s.update(over)
    return run_rule([{"fn": "windowsToDrop", "state": s}])[0]


def test_eviction_keeps_what_is_just_behind_and_within_reach_ahead():
    held = [
        {"start": 0, "end": 10},
        {"start": 90, "end": 100},
        {"start": 100, "end": 110},
        {"start": 150, "end": 160},
    ]
    assert drop(held=held, clock=105) == [0], drop(held=held, clock=105)


def test_eviction_never_drops_the_window_the_clock_is_inside():
    held = [{"start": 100, "end": 110}]
    assert drop(held=held, clock=105, keepBehind=0, keepAhead=0) == []


def test_eviction_measures_around_the_wrap():
    """A window at the start of the range is just ahead of a clock near its end,
    not a whole episode behind it."""
    held = [{"start": 0, "end": 10}, {"start": 200, "end": 210}]
    assert drop(held=held, clock=395, keepBehind=20, keepAhead=60) == [200]
