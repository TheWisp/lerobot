// Every timeline row type, pinned to the pixels it draws.
//
// The defect this exists for is not visual: `renderTrackSvg` draws six kinds
// of row through one function, and a change aimed at one of them -- the lane
// geometry extraction, say -- silently moves another. Nothing in the suite
// would have said so, because the other row types are asserted only in shape
// ("three runs produce three rects"), never in position.
//
// A golden is the right tool here because the output is small, deterministic
// and meaningful to read: a diff names the row type and shows the moved
// coordinate. Regenerate with `node tests/gui/regen_track_render_golden.js`
// and read the diff -- an entry that changed without you meaning it to is the
// regression.
const assert = require("assert");
const fs = require("fs");
const { render, GOLDEN } = require("./track_render_golden.js");

const expected = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
const actual = render();

const missing = Object.keys(expected).filter((k) => !(k in actual));
const extra = Object.keys(actual).filter((k) => !(k in expected));
assert.deepStrictEqual(missing, [], `the golden covers row types the renderer no longer produces: ${missing}`);
assert.deepStrictEqual(extra, [], `new row types are unpinned; regenerate the golden: ${extra}`);

const moved = Object.keys(expected).filter((k) => expected[k] !== actual[k]);
assert.deepStrictEqual(
  moved,
  [],
  "these row types render differently than recorded:\n" +
    moved.map((k) => `  ${k}\n    was: ${expected[k].slice(0, 160)}\n    now: ${actual[k].slice(0, 160)}`).join("\n"),
);

// The golden must actually be load-bearing: an empty or trivial one would pass
// the comparison above while proving nothing.
assert.ok(Object.keys(expected).length >= 33, "the golden has lost coverage of the row types");
// The two things the extraction was justified by must be visible here, or the
// golden proves the move without proving what the move was for.
assert.ok(
  Object.values(expected).some((v) => v.includes("rgb(")),
  "no mask lane records the overlay palette; the golden is pinning the node fallback",
);
assert.ok(
  Object.keys(expected).some((k) => k.endsWith("bit40")),
  "no lane above bit 31 is covered, so the 32-bit shift fix is invisible here",
);
assert.ok(
  Object.values(expected).some((v) => v.includes("<rect")) &&
    Object.values(expected).some((v) => v.includes("<polyline")),
  "the golden no longer covers both the banded and the plotted row kinds",
);

console.log(`track_render_golden.test.js: ${Object.keys(expected).length} row renderings unchanged`);
