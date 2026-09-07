// Unit test for the timeline's lane primitives (run from pytest via
// test_timeline_lanes_js.py, or directly with `node tests/gui/timeline_lanes.test.js`).
//
// These were extracted from feature_editing.js, where nine sites recomputed
// the same lane band by hand and the hit test had to agree with the drawing by
// inspection. What is locked here is that agreement: `hit` is the inverse of
// `geometry`, for every lane of every plausible lane count.
const assert = require("assert");
const TL = require("../../src/lerobot/gui/static/timeline_lanes.js");

// ── Geometry ────────────────────────────────────────────────────────────────
// Lanes share the row's middle 80%, so the first starts at the 10% margin and
// the last ends at 90% — a lane drawn outside that is clipped by the row.
for (const n of [1, 2, 3, 5, 8, 40]) {
  const g = TL.geometry(n);
  assert.strictEqual(g.top(0), 10, `${n} lanes: first lane does not start at the margin`);
  assert.ok(Math.abs(g.top(n - 1) + g.pitch - 90) < 1e-9, `${n} lanes: last lane overruns the row`);
  assert.ok(g.height < g.pitch, `${n} lanes: bars touch, so two lanes read as one block`);
  for (let i = 1; i < n; i++) {
    assert.ok(g.top(i) >= g.top(i - 1) + g.height, `${n} lanes: lane ${i} overlaps the one above`);
  }
  // The middle of the BAR, not of the lane: a control pinned to `mid` must
  // land on the pixels the user aimed at.
  for (let i = 0; i < n; i++) {
    assert.ok(g.mid(i) > g.top(i) && g.mid(i) < g.top(i) + g.height, `${n} lanes: mid(${i}) off its bar`);
  }
}

// ── Hit-testing is the inverse of the geometry ──────────────────────────────
// The defect this guards is not visual: a hit test that disagrees with the
// drawing edits the lane below the bar you clicked.
{
  // `left`/`top` as well as `x`/`y`: a DOMRect carries both, and `hit` reads
  // the pair a plain object is easiest to forget.
  const ROW = { x: 100, y: 50, left: 100, top: 50, width: 400, height: 60 };
  const track = { getBoundingClientRect: () => ROW };
  const at = (yPct, xFrac, count, length) =>
    TL.hit(track, count, length, {
      clientX: ROW.x + ROW.width * xFrac,
      clientY: ROW.y + ROW.height * (yPct / 100),
    });

  for (const n of [1, 2, 3, 5, 8]) {
    const g = TL.geometry(n);
    for (let i = 0; i < n; i++) {
      assert.strictEqual(at(g.mid(i), 0.5, n, 10).lane, i, `${n} lanes: mid of lane ${i} hit another`);
      // Just inside each edge of the lane's own share of the row.
      assert.strictEqual(at(g.top(i) + 0.01, 0.5, n, 10).lane, i, `${n} lanes: top of lane ${i}`);
      assert.strictEqual(at(g.top(i) + g.pitch - 0.01, 0.5, n, 10).lane, i, `${n} lanes: foot of lane ${i}`);
    }
    // The row's margins belong to no lane — that is where a drag starts.
    assert.strictEqual(at(5, 0.5, n, 10), null, `${n} lanes: the top margin claimed a lane`);
    assert.strictEqual(at(95, 0.5, n, 10), null, `${n} lanes: the bottom margin claimed a lane`);
  }

  // Frames come off the x axis, clamped to the row: the pointer can sit one
  // pixel past the last frame's right edge and must not index past the series.
  assert.strictEqual(at(50, 0, 2, 10).frame, 0);
  assert.strictEqual(at(50, 0.55, 2, 10).frame, 5);
  assert.strictEqual(at(50, 1, 2, 10).frame, 9, "the row's right edge indexed past the last frame");

  // A row with no extent yet (not laid out, or an empty episode) has nothing
  // under the pointer — without this the frame is NaN and the edit is silent.
  assert.strictEqual(TL.hit({ getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }) }, 2, 10, { clientX: 0, clientY: 0 }), null);
  assert.strictEqual(at(50, 0.5, 2, 0), null, "a zero-length episode offered a frame");
}

// ── Runs ────────────────────────────────────────────────────────────────────
// The unit every lane edit acts on. A mis-split run does not draw wrong: it
// edits the wrong frames, in a direction read off the wrong state.
{
  const runs = (arr) => TL.runs((i) => arr[i], arr.length);
  assert.deepStrictEqual(runs(["a", "a", "b"]), [
    { from: 0, to: 2, state: "a" },
    { from: 2, to: 3, state: "b" },
  ]);
  // Adjacent runs of the same state must not split -- two segments where the
  // eye sees one bar would put two controls on it and stage two edits.
  assert.deepStrictEqual(runs([true, true, true]), [{ from: 0, to: 3, state: true }]);
  assert.deepStrictEqual(TL.runs(() => "x", 0), [], "an empty range is no runs, not one empty run");

  // Runs tile the range exactly, for any input: no gap, no overlap, no unsplit
  // neighbours, and the whole range covered.
  let s = 3;
  for (let seed = 0; seed < 30; seed++) {
    const arr = Array.from({ length: 41 }, () => (s = (s * 1103515245 + 12345) % 2147483648) % 3);
    const out = runs(arr);
    assert.strictEqual(out[0].from, 0);
    assert.strictEqual(out[out.length - 1].to, 41);
    for (let i = 1; i < out.length; i++) {
      assert.strictEqual(out[i].from, out[i - 1].to, `seed ${seed}: gap or overlap`);
      assert.notStrictEqual(out[i].state, out[i - 1].state, `seed ${seed}: unsplit run`);
    }
    for (let i = 0; i < 41; i++) {
      const r = out.find((o) => i >= o.from && i < o.to);
      assert.strictEqual(r.state, arr[i], `seed ${seed}: frame ${i} reports the wrong state`);
    }
  }
}

// ── The run under the pointer, clipped to the selection ─────────────────────
// The rule that makes the scope positional. Both lane rows share it, and it
// drifted between them while it was written twice.
{
  const all = TL.runs((i) => i >= 4, 7); // clear 0-4, set 4-7
  const sel = (from, to) => ({ frameFrom: from, frameTo: to });
  const plain = (v) => (v === null ? null : { from: v.from, to: v.to, state: v.state });

  // The reported case: `[....XXX]` selected whole. Pointing at the clear part
  // offers those four frames; pointing at the set part offers the other three.
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 1, sel(0, 7))), { from: 0, to: 4, state: false });
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 5, sel(0, 7))), { from: 4, to: 7, state: true });

  // Clipped to the selection, never to the run: a selection cutting a run in
  // half must edit only the half that was selected.
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 2, sel(2, 6))), { from: 2, to: 4, state: false });
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 5, sel(2, 6))), { from: 4, to: 6, state: true });

  // THE POINTER must be inside the selection, not merely the run. The run at
  // frame 5 overlaps a selection of 0..3, and acting on it is the reported
  // "my click outside the range edited the range".
  assert.strictEqual(TL.runUnderPointer(all, 5, sel(0, 3)), null);
  assert.strictEqual(TL.runUnderPointer(all, 3, sel(4, 7)), null);

  // A one-frame selection acts like any other. It was refused once, back when
  // the press was claimed at mousedown and one gesture could both create the
  // selection and edit it -- every seek would have become an edit. Deferring
  // the decision split those in two, so a click on a frame that is visibly
  // selected is deliberate, and refusing it made the lane disagree with the
  // Inspector's controls, which never imposed a minimum.
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 4, sel(4, 5))),
    { from: 4, to: 5, state: true });
  assert.deepStrictEqual(plain(TL.runUnderPointer(all, 1, sel(1, 2))),
    { from: 1, to: 2, state: false });

  // Still nothing without a selection at all.
  assert.strictEqual(TL.runUnderPointer(all, 4, null), null);
}

console.log("timeline_lanes.test.js: all assertions passed");
