// Unit test for the live-overlay transport invariants (run from pytest via
// test_transport_invariants_js.py, or directly with
// `node tests/gui/transport_invariants.test.js`).
//
// Locks the two defects observed while previewing SAM3 over a dataset: the
// transport button offered "Play" while the overlay's own stream was playing,
// and stored masks/stills were painted at the app playhead underneath tiles
// the stream was painting from its own clock.
const assert = require("assert");
const { transportViolations } = require("../../src/lerobot/gui/static/transport_invariants.js");

const healthy = {
  streaming: true, isPlaying: true, playBtnLabel: "⏸ Pause", liveActive: true,
  savedMasksDrawn: false, stillFetchInFlight: false, streamFrame: 120, playheadFrame: 120,
};
assert.deepStrictEqual(transportViolations(healthy), [], "a healthy stream reports nothing");

// The observed bug: togglePlay() delegated to the stream and returned before
// touching isPlaying, so the app stayed "paused" for the whole stream.
const paused = { ...healthy, isPlaying: false, playBtnLabel: "▶ Play" };
const pausedV = transportViolations(paused);
assert.ok(pausedV.some((m) => /reports paused/.test(m)), "must catch transport-says-paused");
assert.ok(pausedV.some((m) => /offers/.test(m)), "must catch the button offering Play");

// Two truths on one image.
assert.ok(transportViolations({ ...healthy, savedMasksDrawn: true })
  .some((m) => /two different truths/.test(m)), "stored masks under the live layer");
assert.ok(transportViolations({ ...healthy, stillFetchInFlight: true })
  .some((m) => /still frames/.test(m)), "stills fetched under the stream");

// Drift between the clocks; one frame of slack is normal (rounding on the video clock).
assert.deepStrictEqual(transportViolations({ ...healthy, streamFrame: 121 }), [], "1 frame of slack is fine");
assert.ok(transportViolations({ ...healthy, streamFrame: 140 })
  .some((m) => /not tracking/.test(m)), "a drifting playhead must be caught");

// Nothing is asserted about a stopped stream: the app owns the transport then.
assert.deepStrictEqual(
  transportViolations({ streaming: false, isPlaying: false, playBtnLabel: "▶ Play", liveActive: false,
    savedMasksDrawn: true, stillFetchInFlight: true, streamFrame: null, playheadFrame: 7 }),
  [], "with the stream off, stills and stored masks are exactly what should paint");

console.log("transport_invariants.test.js: ok");
