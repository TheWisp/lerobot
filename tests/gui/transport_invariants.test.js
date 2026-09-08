// Unit test for the transport rule and the live-overlay transport invariants
// (run from pytest via test_transport_invariants_js.py, or directly with
// `node tests/gui/transport_invariants.test.js`).
//
// Every combination of the transport's observable state is enumerated, so a
// routing that is right for the cases someone thought of and wrong for the
// rest cannot pass. The defects these lock, all seen while previewing SAM3
// over a dataset: Pause routed by the overlay badge instead of by the running
// engine (a still loop kept running under a button that said Pause); the
// button offering "Play" while the stream played; and stored masks or stills
// painted at the app playhead underneath tiles the stream painted from its
// own clock.
const assert = require("assert");
const { transportNext, transportViolations } = require("../../src/lerobot/gui/static/transport_invariants.js");

const ENGINES = [null, "still", "stream", "apply"];
const BOOLS = [false, true];

// ---- the rule: Pause stops what runs; Play starts what the state calls for ----
let cases = 0;
for (const isPlaying of BOOLS) for (const engine of ENGINES)
for (const applyArmed of BOOLS) for (const streamEligible of BOOLS) {
  const next = transportNext({ isPlaying, engine, applyArmed, streamEligible });
  const why = JSON.stringify({ isPlaying, engine, applyArmed, streamEligible });
  if (isPlaying) {
    // Whatever the badge or the Apply tick says NOW, Pause stops the engine that runs.
    assert.deepStrictEqual(next, { op: "stop", engine }, `pause must stop the running engine: ${why}`);
  } else if (applyArmed) {
    assert.deepStrictEqual(next, { op: "start", engine: "apply" }, `an armed Apply owns Play: ${why}`);
  } else if (streamEligible) {
    assert.deepStrictEqual(next, { op: "start", engine: "stream" }, `a live overlay plays the stream: ${why}`);
  } else {
    assert.deepStrictEqual(next, { op: "start", engine: "still" }, `otherwise Play plays stills: ${why}`);
  }
  cases += 1;
}
assert.strictEqual(cases, 32);

// The observed defect, spelled out: a still loop started while the worker was
// loading, the badge went live, and Pause was pressed.
assert.deepStrictEqual(
  transportNext({ isPlaying: true, engine: "still", applyArmed: false, streamEligible: true }),
  { op: "stop", engine: "still" },
  "Pause must stop the still loop even though the stream is now eligible");
// A state that omitted the engine (a caller from before the engine existed) still stops.
assert.deepStrictEqual(transportNext({ isPlaying: true }), { op: "stop", engine: null });

// ---- the invariants: one engine while playing, none while paused, honest button ----
const LABELS = ["▶ Play", "⏸ Pause"];
let consistent = 0, inconsistent = 0;
for (const isPlaying of BOOLS) for (const engine of ENGINES)
for (const streaming of BOOLS) for (const playBtnLabel of LABELS) {
  const ok = (isPlaying === (engine !== null))
          && (streaming === (engine === "stream"))
          && (isPlaying === /pause/i.test(playBtnLabel));
  const v = transportViolations({ isPlaying, engine, streaming, playBtnLabel });
  const why = JSON.stringify({ isPlaying, engine, streaming, playBtnLabel });
  if (ok) { assert.deepStrictEqual(v, [], `a consistent state reports nothing: ${why}`); consistent += 1; }
  else { assert.ok(v.length > 0, `an inconsistent state must be reported: ${why}`); inconsistent += 1; }
}
assert.strictEqual(consistent, 4, "exactly one consistent state per engine (incl. none)");
assert.strictEqual(inconsistent, 28);

// The messages name the defect they caught.
assert.ok(transportViolations({ isPlaying: true, engine: null, streaming: false, playBtnLabel: "⏸ Pause" })
  .some((m) => /no engine/.test(m)));
assert.ok(transportViolations({ isPlaying: false, engine: "still", streaming: false, playBtnLabel: "▶ Play" })
  .some((m) => /paused while the still engine runs/.test(m)));
assert.ok(transportViolations({ isPlaying: true, engine: "still", streaming: true, playBtnLabel: "⏸ Pause" })
  .some((m) => /stream is running while the engine is still/.test(m)));
assert.ok(transportViolations({ isPlaying: true, engine: "stream", streaming: true, playBtnLabel: "▶ Play" })
  .some((m) => /button offers/.test(m)));

// ---- the stream's own checks, unchanged from before the engine existed ----
const healthy = {
  streaming: true, isPlaying: true, playBtnLabel: "⏸ Pause", liveActive: true,
  savedMasksDrawn: false, stillFetchInFlight: false, streamFrame: 120, playheadFrame: 120,
};
assert.deepStrictEqual(transportViolations(healthy), [], "a healthy stream reports nothing");

// togglePlay() once delegated to the stream and returned before touching
// isPlaying, so the app stayed "paused" for the whole stream.
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

console.log("transport_invariants: ok");
