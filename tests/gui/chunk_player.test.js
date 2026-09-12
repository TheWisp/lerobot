// The chunk player's decisions as pure functions, driven directly (design: C1, R4).
//
// No server, no video, no link: what to fetch next, what to release, how much
// media is held ahead of the clock, and which chunk a frame is in. Every
// invariant the player asserts at runtime is enumerated here, so a violation
// is a failing test rather than a stall nobody can explain.
//
// Run from pytest via test_chunk_player_js.py, or directly with
// `node tests/gui/chunk_player.test.js`.
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");

function load() {
  global.window = { addEventListener() {} };
  global.performance = { now: () => 0 };
  global.requestAnimationFrame = () => 0;
  global.location = { origin: "http://test", pathname: "/" };
  new Function(fs.readFileSync(path.join(STATIC, "chunk_player.js"), "utf8"))();
  return global.window.ChunkPlayer;
}

const P = load();
const { chunkFor, fetchPlan, chunksToDrop, coverageFrom, RuleViolation } = P;
assert.ok(RuleViolation, "the player exports its violation type");

// ---- which chunk a frame is in ---------------------------------------------
assert.strictEqual(chunkFor(0, 20), 0);
assert.strictEqual(chunkFor(19, 20), 0);
assert.strictEqual(chunkFor(20, 20), 20);
assert.strictEqual(chunkFor(45, 20), 40);
assert.throws(() => chunkFor(5, 0), RuleViolation, "a zero-length chunk is a programming error");

// ---- fetch planning: one request per chunk, nothing per frame ---------------
const base = { held: [], inflight: [], clock: 0, rangeStart: 0, rangeEnd: 45, chunkFrames: 20, targetFrames: 40, maxInflight: 2 };

// Empty buffer: ask for the chunk under the clock and the next one, no more than maxInflight.
let plan = fetchPlan(base);
assert.deepStrictEqual(plan, [0, 20], "two chunks cover the 40-frame target");

// The clock inside a held chunk: only what is missing ahead is asked for.
plan = fetchPlan({ ...base, held: [{ start: 0, end: 20 }], clock: 7 });
assert.deepStrictEqual(plan, [20, 40], "held media is not fetched again");

// A chunk in flight counts as covered and as one of the requests standing.
plan = fetchPlan({ ...base, inflight: [{ start: 0, end: 20 }], clock: 3 });
assert.deepStrictEqual(plan, [20], "one slot left, one chunk planned");

// Reaching the end of the range wraps to its start, as playback does.
plan = fetchPlan({ ...base, held: [{ start: 40, end: 45 }], clock: 42 });
assert.deepStrictEqual(plan, [0, 20], "past the end, the walk continues from the range start");

// A trim range: the chunk holding the range start is asked for, even if it starts before it.
plan = fetchPlan({ ...base, clock: 25, rangeStart: 25, rangeEnd: 45 });
assert.deepStrictEqual(plan, [20, 40], "the chunk starts on the grid at or before the frame it serves");

// The clock outside the range: fetch for where playback will be, not where a stale clock is.
plan = fetchPlan({ ...base, clock: 3, rangeStart: 25, rangeEnd: 45 });
assert.deepStrictEqual(plan, [20, 40]);

// Every planned start is on the grid, none repeats, and the count respects the slots.
for (const s of [base, { ...base, clock: 44 }, { ...base, held: [{ start: 20, end: 40 }], clock: 21 }]) {
  const out = fetchPlan(s);
  assert.ok(out.every((st) => st % s.chunkFrames === 0), `on the grid: ${out}`);
  assert.strictEqual(new Set(out).size, out.length, "no chunk planned twice");
  assert.ok(out.length <= s.maxInflight - s.inflight.length, "no more than the free slots");
}
assert.throws(() => fetchPlan({ ...base, rangeEnd: 0 }), RuleViolation, "an empty range is a programming error");

// ---- coverage: seconds of ready media ahead of the clock, around the wrap ----
const held = [{ start: 0, end: 20, readyTo: 20 }, { start: 20, end: 40, readyTo: 30 }];
assert.strictEqual(coverageFrom({ held, clock: 5, rangeStart: 0, rangeEnd: 45 }), 25, "stops at the first frame not decoded");
assert.strictEqual(coverageFrom({ held: [{ start: 0, end: 20, readyTo: 20 }], clock: 30, rangeStart: 0, rangeEnd: 45 }), 0, "nothing held at the clock");
assert.strictEqual(
  coverageFrom({ held: [{ start: 40, end: 45, readyTo: 45 }, { start: 0, end: 20, readyTo: 20 }], clock: 41, rangeStart: 0, rangeEnd: 45 }),
  24, "counts across the wrap: 4 to the end, then 20 from the start",
);
assert.strictEqual(
  coverageFrom({ held: [{ start: 0, end: 20, readyTo: 20 }, { start: 20, end: 40, readyTo: 40 }, { start: 40, end: 45, readyTo: 45 }], clock: 10, rangeStart: 0, rangeEnd: 45 }),
  45, "a fully held range is counted once, not past the clock a second time",
);

// ---- release: never the chunk under the clock, keep a little behind and the reach ahead
const drop = chunksToDrop({ held: [{ start: 0, end: 20 }, { start: 20, end: 40 }, { start: 40, end: 45 }], clock: 42, rangeStart: 0, rangeEnd: 45, keepBehind: 1, keepAhead: 10 });
assert.ok(!drop.includes(40), "the chunk under the clock stays");
assert.ok(!drop.includes(0), "the chunk playback wraps into next is within reach ahead");
assert.deepStrictEqual(drop, [20], "the chunk 2 behind and 23 ahead across the wrap, outside both reaches, is released");
assert.throws(() => chunksToDrop({ held: [], clock: 0, rangeStart: 0, rangeEnd: 0, keepBehind: 1, keepAhead: 1 }), RuleViolation);

console.log("chunk_player: ok");

// A seek keeps the fetches the new position still wants: the chunk it lands in
// and those within reach ahead of it. Only what the plan would not ask for is
// aborted -- a save that lands on the frame already showing must not throw
// away the next chunk's transfer, which is what doubled every rebuild.
const { inflightToAbort } = P;
const flightBase = { inflight: [{ start: 0, end: 20 }, { start: 20, end: 40 }, { start: 40, end: 60 }], rangeStart: 0, rangeEnd: 100, keepAhead: 30 };
assert.deepStrictEqual(inflightToAbort({ ...flightBase, frame: 5 }), [40], "landing in the first chunk keeps it and the one in reach; the third is beyond reach");
assert.deepStrictEqual(inflightToAbort({ ...flightBase, frame: 25 }), [0], "the chunk behind the new position is not wanted");
assert.deepStrictEqual(inflightToAbort({ ...flightBase, frame: 90 }), [40], "far away: keep what the wrap reaches -- chunk 0 lies 10 frames ahead across it, chunk 20 exactly at the reach (kept, as a held chunk at the boundary is)");
assert.deepStrictEqual(inflightToAbort({ ...flightBase, frame: 90, keepAhead: 29 }), [20, 40], "one frame short of the reach, chunk 20 goes");
assert.deepStrictEqual(inflightToAbort({ ...flightBase, inflight: [], frame: 5 }), [], "nothing in flight, nothing to abort");
assert.deepStrictEqual(inflightToAbort({ ...flightBase, frame: -1 }), [0, 20, 40], "a frame outside the range (the whole-episode drop) aborts everything");
console.log("chunk_player inflightToAbort: ok");
