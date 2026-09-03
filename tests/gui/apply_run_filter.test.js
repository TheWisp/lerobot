// The write rule, applied where the coverage already is.
//
// A run may fill a (frame, label) only where that label is ABSENT. Detected and
// disabled are both left alone — disabled especially, because re-filling it
// would put back a detection the operator rejected, and the next run would do
// it again.
const assert = require("assert");
const { isAbsent, rowsToStage } = require("../../src/lerobot/gui/static/apply_run_filter.js");

// ── the three states, read out of the two bitsets ───────────────────────────
{
    const labels = ["ball", "cube"];
    // frame 0: ball detected. frame 1: cube disabled. frame 2: nothing.
    const enabled = [1, 0, 0];
    const disabled = [0, 2, 0];

    assert.strictEqual(isAbsent(enabled, disabled, labels, "ball", 0), false, "detected is not absent");
    assert.strictEqual(isAbsent(enabled, disabled, labels, "cube", 1), false, "disabled is not absent");
    assert.strictEqual(isAbsent(enabled, disabled, labels, "ball", 2), true);
    assert.strictEqual(isAbsent(enabled, disabled, labels, "cube", 0), true, "labels are independent");

    // A label the dataset has never declared is absent everywhere — that is how
    // a new object gets its first masks.
    assert.strictEqual(isAbsent(enabled, disabled, labels, "ring", 0), true);
}

// Bit 31 and beyond: a vocabulary grows and never shrinks, and JS bitwise ops
// are 32-bit signed, so `1 << 31` is negative and `1 << 40` wraps to 256.
{
    const labels = Array.from({ length: 45 }, (_, i) => `l${i}`);
    const enabled = [Math.pow(2, 40)];
    assert.strictEqual(isAbsent(enabled, [0], labels, "l40", 0), false, "bit 40 read as absent");
    assert.strictEqual(isAbsent(enabled, [0], labels, "l39", 0), true);
    assert.strictEqual(isAbsent([Math.pow(2, 31)], [0], labels, "l31", 0), false, "bit 31 read as absent");
}

// ── what gets staged ────────────────────────────────────────────────────────
{
    const coverage = { cam: { labels: ["ball", "cube"], enabled: [1, 0, 0], disabled: [0, 2, 0] } };
    const drained = [
        { episode: 0, frame: 0, camera: "cam", rle: { ball: "A", cube: "B" } },
        { episode: 0, frame: 1, camera: "cam", rle: { cube: "C" } },
        { episode: 0, frame: 2, camera: "cam", rle: { ball: "D" } },
    ];
    const { rows, filtered } = rowsToStage(drained, coverage);

    assert.deepStrictEqual(
        rows,
        [
            { camera: "cam", frame: 0, rle: { cube: "B" } },
            { camera: "cam", frame: 2, rle: { ball: "D" } },
        ],
        "the wrong pairs were kept",
    );
    assert.strictEqual(filtered, 2, "ball@0 (detected) and cube@1 (disabled) should both be dropped");
}

// A frame whose labels are ALL taken is not sent at all: an empty row would
// stage as "segmented, found nothing" and claim a frame the run did not fill.
{
    const coverage = { cam: { labels: ["ball"], enabled: [1], disabled: [0] } };
    const { rows, filtered } = rowsToStage(
        [{ episode: 0, frame: 0, camera: "cam", rle: { ball: "A" } }],
        coverage,
    );
    assert.deepStrictEqual(rows, [], "a fully-filtered frame was still staged");
    assert.strictEqual(filtered, 1);
}

// The complement, or "it filters" would be satisfied by filtering everything.
{
    const coverage = { cam: { labels: ["ball"], enabled: [0, 0], disabled: [0, 0] } };
    const { rows, filtered } = rowsToStage(
        [
            { episode: 0, frame: 0, camera: "cam", rle: { ball: "A" } },
            { episode: 0, frame: 1, camera: "cam", rle: { ball: "B" } },
        ],
        coverage,
    );
    assert.strictEqual(rows.length, 2, "an empty episode should stage every frame");
    assert.strictEqual(filtered, 0);
}

// A camera the client has no coverage for is counted, not guessed at: staging
// against unknown coverage is how a disabled mask gets refilled.
{
    const { rows, unknownCamera } = rowsToStage(
        [{ episode: 0, frame: 0, camera: "other", rle: { ball: "A" } }],
        { cam: { labels: ["ball"], enabled: [0], disabled: [0] } },
    );
    assert.deepStrictEqual(rows, []);
    assert.strictEqual(unknownCamera, 1);
}

// Nothing drained is not an error.
assert.deepStrictEqual(rowsToStage([], {}), { rows: [], filtered: 0, unknownCamera: 0 });
assert.deepStrictEqual(rowsToStage(undefined, undefined), { rows: [], filtered: 0, unknownCamera: 0 });

console.log("apply_run_filter.test.js: all assertions passed");

// ── hostile input: the filter must not be able to take the run down ─────────
//
// This function is called from inside the Apply run's drain. It used to sit
// outside that path's try, so anything it threw rejected all the way out of the
// run loop -- which left the run marked "in progress" for the rest of the
// session, after which the Play button did nothing at all, in silence, while
// scrubbing still worked. Reported as "the play button no longer starts it".
//
// The run now contains its own failures, but the cheaper guarantee is that
// there is nothing here to contain: a drain carrying an unexpected shape is a
// filtering decision, never an exception.
{
  const hostile = [
    ["null frames", null, {}],
    ["undefined frames", undefined, {}],
    ["null coverage", [{ episode: 0, frame: 1, camera: "cam", rle: "x" }], null],
    ["frame with no camera", [{ episode: 0, frame: 1, rle: "x" }], {}],
    ["frame with no rle", [{ episode: 0, frame: 1, camera: "cam" }], { cam: { labels: [], enabled: [], disabled: [] } }],
    ["coverage missing bitsets", [{ episode: 0, frame: 1, camera: "cam", rle: {} }], { cam: { labels: ["a"] } }],
    ["coverage with null labels", [{ episode: 0, frame: 1, camera: "cam", rle: {} }], { cam: { labels: null } }],
    ["a null entry in frames", [null], {}],
    ["frame index past the bitsets", [{ episode: 0, frame: 999, camera: "cam", rle: { a: "z" } }],
      { cam: { labels: ["a"], enabled: [0], disabled: [0] } }],
  ];
  for (const [what, frames, coverage] of hostile) {
    let out;
    assert.doesNotThrow(() => { out = rowsToStage(frames, coverage); }, `threw on ${what}`);
    assert.ok(out && Array.isArray(out.rows), `${what}: no rows array back`);
  }
}

console.log("apply_run_filter.test.js: hostile-input cases ok");

// ── the cost guard behind the flush cadence ─────────────────────────────────
//
// The run drains per frame, because in lock-step the drain is how it learns
// frame f came back. What keeps that from being a request per frame is this:
// a drain whose frames are all already covered must lower to ZERO rows, and
// the caller skips the POST when there are none. If this ever returned rows
// for covered ground, every re-run over an already-segmented episode would
// issue a request per frame to write nothing.
{
    const labels = ["ball", "cube"];
    const CAM = "observation.images.top";
    // Both labels detected on frames 0-2: bits 0 and 1 of each frame's word.
    const seen = [0b11, 0b11, 0b11];
    const covered = { [CAM]: { labels, enabled: seen, disabled: [0, 0, 0] } };
    const drained = [0, 1, 2].map((frame) => ({
        episode: 0, frame, camera: CAM, rle: { ball: "x", cube: "y" },
    }));
    const { rows } = rowsToStage(drained, covered);
    assert.strictEqual(rows.length, 0, "fully-covered frames must lower to no rows, so no request is sent");

    // The complement: with the same drain against EMPTY coverage the rows must
    // appear, or the assertion above passes for a filter that drops everything.
    const empty = { [CAM]: { labels, enabled: [0, 0, 0], disabled: [0, 0, 0] } };
    const { rows: fresh } = rowsToStage(drained, empty);
    assert.ok(fresh.length > 0, "absent frames must produce rows; otherwise nothing is ever written");
}
