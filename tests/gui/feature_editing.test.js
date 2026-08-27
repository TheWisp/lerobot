// Unit tests for the pure render decisions in feature_editing.js.
//
// The module owns the Inspector panel and the per-feature timeline rows, and
// until now nothing loaded it — its render rules were reachable only by driving
// a browser, which is how the uniform-value case below shipped uncovered.
// Everything asserted here is a function of its arguments alone.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/feature_editing.js"),
  "utf8",
);

// The module registers a DOMContentLoaded listener at load time and publishes
// its API on window; nothing else runs without a dataset.
const noop = () => {};
const win = {};
const context = vm.createContext({
  console,
  window: win,
  document: { addEventListener: noop, getElementById: () => null, querySelector: () => null },
  localStorage: { getItem: () => null, setItem: noop },
});
vm.runInContext(source, context);

const F = win.FeatureEditing._internals;
const str = { dtype: "string", shape: [1] };
const bool = { dtype: "bool", shape: [1] };
const num = { dtype: "float32", shape: [1] };
const vec = { dtype: "float32", shape: [3] };

// ── Editability ─────────────────────────────────────────────────────────────
// `task` is read-only because it is recorded data the edit pipeline cannot
// express, not because of anything about its name. `subtask` stays editable.
assert.strictEqual(F.isRecordedFeature("task"), true);
assert.strictEqual(F.isRecordedFeature("action"), true);
assert.strictEqual(F.isRecordedFeature("observation.images.top"), true);
assert.strictEqual(F.isRecordedFeature("subtask"), false);
assert.strictEqual(F.isRecordedFeature("reward"), false);

assert.strictEqual(F.isEditable("task", str), false);
assert.strictEqual(F.isEditable("subtask", str), true);
assert.strictEqual(F.isEditable("reward", num), true);
assert.strictEqual(F.isEditable("timestamp", num), false, "DEFAULT_FEATURES are internal");
assert.strictEqual(F.isEditable("observation.images.top", { dtype: "video" }), false);

// ── Read-only value display ─────────────────────────────────────────────────
// The regression the adversarial review caught: a value that is the same on
// every frame must not carry "(frame N of M)". The hint says "this is one
// sample of many that may differ" — false for an episode-wide value, and it
// contradicts the card directly.
const uniform = F.readOnlyValueHtml(str, Array(385).fill("assemble cylinder into ring"), 0);
assert.ok(uniform.includes("assemble cylinder into ring"), "the value must be shown");
assert.ok(!uniform.includes("(frame"), `uniform value must not carry a frame hint: ${uniform}`);

// The other direction — suppression must not be so broad that a genuinely
// varying value loses the caveat that only one frame is displayed.
const varying = F.readOnlyValueHtml(str, ["grasp the ring", "insert the peg", "grasp the ring"], 0);
assert.ok(varying.includes("(frame 0 of 3)"), `varying value must keep the hint: ${varying}`);

// A single frame is not a range, so there is nothing to caveat either way.
assert.ok(!F.readOnlyValueHtml(str, ["only"], 7).includes("(frame"));

// Vectors are excluded from the uniform check by identity comparison, so they
// keep the hint. Pinned so the exclusion is deliberate rather than incidental.
const vectors = F.readOnlyValueHtml(vec, [[1, 2, 3], [1, 2, 3]], 0);
assert.ok(vectors.includes("(frame 0 of 2)"), "vectors keep the hint");
assert.ok(vectors.includes("readonly-vector"));

assert.ok(F.readOnlyValueHtml(str, null, 0).includes("no data in selection"));
assert.ok(F.readOnlyValueHtml(str, [], 0).includes("no data in selection"));
assert.ok(F.readOnlyValueHtml({ dtype: "video" }, null, 0).includes("rendered in viewer"));

// Strings are quoted so an empty value or stray whitespace is visible rather
// than looking like missing data.
assert.ok(F.readOnlyValueHtml(str, [""], 0).includes('""'));
assert.ok(F.readOnlyValueHtml(str, ["<script>"], 0).includes("&lt;script&gt;"), "escaped");

// ── Range summary ───────────────────────────────────────────────────────────
// Describes what an edit over the selection would overwrite. Only the "many
// values" answer carries information; the uniform answer exists so a caller
// does not read "range: X … X" as schema bounds.
assert.strictEqual(F.summarizeSlice(["a", "a", "a"]), 'uniform: "a" (3 frames)');
assert.strictEqual(F.summarizeSlice(["a", "b", "a"]), "2 unique values");
assert.strictEqual(F.summarizeSlice([true, true, false]), "2 true · 1 false");
assert.ok(F.summarizeSlice([1, 1, 1]).startsWith("uniform: "));
assert.ok(F.summarizeSlice([1, 5, 3]).startsWith("selection min … max"));
assert.strictEqual(F.summarizeSlice(["solo"]), 'value: "solo"');
assert.strictEqual(F.summarizeSlice([]), "&nbsp;");
assert.strictEqual(F.summarizeSlice(null), "&nbsp;");

// ── Timeline track ──────────────────────────────────────────────────────────
// A string constant across the episode collapses to one full-width rect. That
// is what a per-episode feature would look like on the timeline, and why it is
// routed to the Inspector instead of given a row.
const oneBand = F.renderTrackSvg("task", str, Array(120).fill("one instruction"), 120);
const widths = [...oneBand.matchAll(/<rect[^>]*width="([^"]+)"/g)].map(m => m[1]);
assert.strictEqual(widths.length, 1, `expected a single rect, got ${widths.length}`);
assert.strictEqual(widths[0], "100%");

// Run-length encoding: three runs, not one rect per frame.
const runs = F.renderTrackSvg("subtask", str, ["a", "a", "b", "b", "c", "c"], 6);
assert.strictEqual([...runs.matchAll(/<rect/g)].length, 3);

// Bools draw only the true spans, so an all-false row is empty rather than a
// misleading full-width band.
assert.strictEqual([...F.renderTrackSvg("s", bool, [false, false], 2).matchAll(/<rect/g)].length, 0);
assert.strictEqual([...F.renderTrackSvg("s", bool, [true, false], 2).matchAll(/<rect/g)].length, 1);

assert.strictEqual(F.renderTrackSvg("x", str, [], 0), "");
assert.strictEqual(F.renderTrackSvg("x", str, null, 10), "");

console.log("feature_editing.test.js: all assertions passed");

// ── Bit maths beyond 32 bits ────────────────────────────────────────────────
// JavaScript's &, | and ~ coerce to 32-bit integers, so `value & Math.pow(2, 40)`
// is 0 and a label at bit 31 or beyond would be silently invisible and
// untickable while the stored contract allows 63. These pin the arithmetic that
// replaced them.
assert.strictEqual(F.bitIsSet(Math.pow(2, 40), 40), true, "bitwise & gives 0 here");
assert.strictEqual(F.bitIsSet(Math.pow(2, 40), 39), false);
assert.strictEqual(F.bitIsSet(Math.pow(2, 31), 31), true, "bitwise & gives a negative here");

// Every bit JSON can carry faithfully must set and clear exactly.
for (let b = 0; b <= 52; b++) {
    const v = F.withBits(0, Math.pow(2, b), 0);
    assert.strictEqual(F.bitIsSet(v, b), true, `bit ${b} did not set`);
    assert.strictEqual(F.withBits(v, 0, Math.pow(2, b)), 0, `bit ${b} did not clear`);
}

// A high bit must not disturb a low one, which is the whole point of a bitset.
const mixedBits = F.withBits(Math.pow(2, 3), Math.pow(2, 40), 0);
assert.strictEqual(F.bitIsSet(mixedBits, 3), true);
assert.strictEqual(F.bitIsSet(mixedBits, 40), true);

assert.strictEqual(F.withBits(8, 0, Math.pow(2, 40)), 8, "clearing an unset bit changes nothing");
const onceSet = F.withBits(0, 5, 0);
assert.strictEqual(F.withBits(onceSet, 5, 0), onceSet, "setting what is set is idempotent");
