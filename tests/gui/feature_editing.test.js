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

// A mask column is segmenter output: an RLE string whose meaning is positional
// against mask_labels, with the treatments every consumer reads living in its
// spec. Typing into it, or dropping it with the generic ✕, is never right --
// un-adopting belongs to the mask flow, which knows about recipes.
//
// This held by accident until the columns moved out of `observation.`, which
// isRecordedFeature already excluded. Keyed on the encoding, not the name,
// because the name has moved once already.
const maskCol = { dtype: "string", shape: [1], mask_encoding: "coco_rle", mask_labels: ["ball"] };
assert.strictEqual(F.isEditable("masks.top", maskCol), false, "a mask cell is not hand-editable");
assert.strictEqual(F.isDeletable("masks.top", maskCol), false, "a mask column is not generically deletable");
// The complement: an ordinary string column at the same key shape stays
// editable, so this cannot pass by locking every string column.
assert.strictEqual(F.isEditable("notes.top", str), true);
assert.strictEqual(F.isDeletable("notes.top", str), true);

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

// ── Mask segments ───────────────────────────────────────────────────────────
// The unit every mask edit acts on: a maximal run where one label is in one
// state. Getting this wrong is not a visual bug -- the click's direction is
// decided by the segment's state, so a mis-split segment toggles the wrong
// frames.
// Objects built inside the vm context have a different Object prototype, so
// deepStrictEqual would compare prototypes rather than content. Normalise.
const plain = (v) => JSON.parse(JSON.stringify(v));
const segs = (...a) => plain(win.FeatureEditing.maskSegments(...a));

// bit 0: detected 0-2, disabled 2-4, absent 4-6
{
  const enabled = [0b01, 0b01, 0, 0, 0, 0];
  const disabled = [0, 0, 0b01, 0b01, 0, 0];
  assert.deepStrictEqual(segs(enabled, disabled, 0, 6), [
    { from: 0, to: 2, state: "detected" },
    { from: 2, to: 4, state: "disabled" },
    { from: 4, to: 6, state: "absent" },
  ]);
}

// A pre-flag dataset has no disabled series at all: every carried frame reads
// as detected, and `undefined >> b` must not become a phantom disabled run.
assert.deepStrictEqual(segs([1, 1, 0], [], 0, 3), [
  { from: 0, to: 2, state: "detected" },
  { from: 2, to: 3, state: "absent" },
]);

// Labels are independent: bit 1's states say nothing about bit 0's.
{
  const enabled = [0b10, 0b11, 0b01];
  const disabled = [0b01, 0, 0];
  assert.deepStrictEqual(segs(enabled, disabled, 0, 3), [
    { from: 0, to: 1, state: "disabled" },
    { from: 1, to: 3, state: "detected" },
  ]);
  assert.deepStrictEqual(segs(enabled, disabled, 1, 3), [
    { from: 0, to: 2, state: "detected" },
    { from: 2, to: 3, state: "absent" },
  ]);
}

// Adjacent runs of the SAME state must not be split -- two segments where the
// eye sees one bar would put two x's on it and stage two edits.
assert.deepStrictEqual(segs([1, 1, 1], [0, 0, 0], 0, 3), [
  { from: 0, to: 3, state: "detected" },
]);

// A label absent everywhere is one absent run, not zero segments: hit-testing
// needs to know a click landed on an empty lane rather than off the row.
assert.deepStrictEqual(segs([0, 0], [0, 0], 0, 2), [{ from: 0, to: 2, state: "absent" }]);

// Segments tile the range exactly, with no gap and no overlap, for any input.
{
  const rnd = (n, seed) => {
    let s = seed;
    return Array.from({ length: n }, () => (s = (s * 1103515245 + 12345) % 2147483648) % 4);
  };
  for (let seed = 1; seed <= 20; seed++) {
    const enabled = rnd(37, seed);
    const disabled = rnd(37, seed * 7).map((v, i) => v & ~enabled[i]); // never both
    for (let b = 0; b < 2; b++) {
      const out = segs(enabled, disabled, b, 37);
      assert.strictEqual(out[0].from, 0, `seed ${seed} bit ${b}: does not start at 0`);
      assert.strictEqual(out[out.length - 1].to, 37, `seed ${seed} bit ${b}: does not end at len`);
      for (let i = 1; i < out.length; i++) {
        assert.strictEqual(out[i].from, out[i - 1].to, `seed ${seed} bit ${b}: gap or overlap`);
        assert.notStrictEqual(out[i].state, out[i - 1].state, `seed ${seed} bit ${b}: unsplit run`);
      }
    }
  }
}

// ── The camera a mask column describes ──────────────────────────────────────
// Wrong here and every staged edit names a camera that does not exist, which
// the server rejects — but only after the click looked like it worked.
assert.strictEqual(win.FeatureEditing.maskCameraOf("masks.top"), "observation.images.top");
assert.strictEqual(win.FeatureEditing.maskCameraOf("masks.left_wrist"), "observation.images.left_wrist");
assert.strictEqual(win.FeatureEditing.maskCameraOf("observation.state"), null);
assert.strictEqual(win.FeatureEditing.maskCameraOf("masksomething"), null);

// ── Staged mask edits merge into the lane ───────────────────────────────────
// Without this a click springs back: the lane draws from the stored column,
// which does not carry the edit until Save, so the operator clicks and sees
// nothing happen.
{
  const merge = win.FeatureEditing.applyPendingMaskEdits;
  const labels = ["ball", "tray"];
  const stage = (action, from, to, label = "ball") => [{
    edit_type: "mask_range",
    episode_index: 0,
    params: { camera: "observation.images.top", label, from_frame: from, to_frame: to, action },
  }];
  win.currentEpisode = 0;

  // disable: leaves enabled, enters disabled
  win.pendingEdits = stage("disable", 0, 2);
  assert.deepStrictEqual(plain(merge("masks.top", labels, [1, 1, 1], [0, 0, 0], 3)),
    [[0, 0, 1], [1, 1, 0]]);

  // delete: leaves both
  win.pendingEdits = stage("delete", 0, 2);
  assert.deepStrictEqual(plain(merge("masks.top", labels, [1, 1, 1], [0, 0, 0], 3)),
    [[0, 0, 1], [0, 0, 0]]);

  // enable on a muted run: the reverse of disable
  win.pendingEdits = stage("enable", 0, 2);
  assert.deepStrictEqual(plain(merge("masks.top", labels, [0, 0, 0], [1, 1, 0], 3)),
    [[1, 1, 0], [0, 0, 0]]);

  // An absent frame is skipped, exactly as the server skips it -- muting
  // nothing must not invent a mask.
  win.pendingEdits = stage("disable", 0, 3);
  assert.deepStrictEqual(plain(merge("masks.top", labels, [1, 0, 1], [0, 0, 0], 3)),
    [[0, 0, 0], [1, 0, 1]]);

  // Another label's edit does not touch this one's bit.
  win.pendingEdits = stage("disable", 0, 2, "tray");
  assert.deepStrictEqual(plain(merge("masks.top", labels, [0b11, 0b11, 0], [0, 0, 0], 3)),
    [[0b01, 0b01, 0], [0b10, 0b10, 0]]);

  // An edit for a different camera is not ours.
  win.pendingEdits = [{
    edit_type: "mask_range", episode_index: 0,
    params: { camera: "observation.images.wrist", label: "ball", from_frame: 0, to_frame: 3, action: "delete" },
  }];
  assert.deepStrictEqual(plain(merge("masks.top", labels, [1, 1, 1], [0, 0, 0], 3)),
    [[1, 1, 1], [0, 0, 0]]);

  // And neither is one from another episode.
  win.pendingEdits = stage("delete", 0, 3);
  win.pendingEdits[0].episode_index = 7;
  assert.deepStrictEqual(plain(merge("masks.top", labels, [1, 1, 1], [0, 0, 0], 3)),
    [[1, 1, 1], [0, 0, 0]]);

  win.pendingEdits = [];
}

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

// ── the dataset section ─────────────────────────────────────────────────────
//
// Rendered from `DatasetInfo`, whose fields are all optional in practice: a
// dataset can declare no robot, carry no cameras, or be reported with a count
// missing entirely. What it must never do is show a wrong number, or a blank
// where a number belongs.
{
  const facts = (html) => {
    const keys = [...html.matchAll(/class="ds-fact-key">([^<]*)</g)].map((m) => m[1]);
    const vals = [...html.matchAll(/class="ds-fact-val">([^<]*)</g)].map((m) => m[1]);
    return Object.fromEntries(keys.map((k, i) => [k, vals[i]]));
  };

  // A dataset with nothing open at all renders nothing, not a bare header.
  assert.strictEqual(F.renderDatasetSection("ds", null), "");
  assert.strictEqual(F.renderDatasetSection("ds", undefined), "");

  // The ordinary case.
  const ok = facts(F.renderDatasetSection("ds", {
    repo_id: "who/what", total_episodes: 24, total_frames: 5430, fps: 30,
    camera_keys: ["a", "b", "c", "d"], robot_type: "bi_so107_follower",
  }));
  assert.strictEqual(ok.repo, "who/what");
  assert.strictEqual(ok.episodes, "24");
  assert.strictEqual(ok.fps, "30");
  assert.strictEqual(ok.cameras, "4");
  assert.strictEqual(ok.robot, "bi_so107_follower");

  // Large counts are grouped: 556172 unseparated is a number you have to count
  // the digits of, and this panel exists to be read at a glance.
  const big = facts(F.renderDatasetSection("ds", {
    repo_id: "x/y", total_episodes: 1900, total_frames: 556172, fps: 50, camera_keys: ["c"],
  }));
  assert.strictEqual(big.frames, "556,172");

  // An empty dataset reports zero, not "?" — "?" means "not reported", and the
  // two are different facts.
  const empty = facts(F.renderDatasetSection("ds", {
    repo_id: "x/y", total_episodes: 0, total_frames: 0, fps: 30, camera_keys: [],
  }));
  assert.strictEqual(empty.episodes, "0", "0 episodes is a fact, not an unknown");
  assert.strictEqual(empty.frames, "0");
  assert.strictEqual(empty.cameras, "—", "no cameras reads as none, not as zero");

  // Fields genuinely absent read as unknown, and never as blank.
  const bare = facts(F.renderDatasetSection("ds", { id: "/some/path" }));
  assert.strictEqual(bare.repo, "/some/path", "falls back to the id when there is no repo_id");
  assert.strictEqual(bare.episodes, "?");
  assert.strictEqual(bare.fps, "?");
  assert.strictEqual(bare.robot, "—");
  assert.strictEqual(bare.frames, "0");
  for (const [k, v] of Object.entries(bare)) assert.ok(v !== "", `${k} rendered blank`);

  // A repo id is server data and goes through escaping like everything else.
  const nasty = F.renderDatasetSection("ds", { repo_id: '<img src=x onerror=1>', camera_keys: [] });
  assert.ok(!nasty.includes("<img"), "repo id was not escaped");
}

// ── Mask lanes: what `data-label` means ─────────────────────────────────────
// The rects and the delete button they offer both carry `data-label`, and they
// disagreed: the rect held the lane INDEX while the button held the NAME, as
// does every other data-label in the module. Nothing in the product read the
// rect's copy, so the two drifted with no symptom -- until anything did, and
// addressed lane 0 whenever the name happened not to parse as its index.
//
// Pinned as an AGREEMENT rather than against a literal, so a later rename of
// either attribute has to move both.
{
    const names = ["green ring", "yellow block"];
    const ft = { dtype: "string", mask_encoding: "coco_rle", mask_labels: names };
    //   frames 0-1: both labels enabled;  frames 2-3: label 1 muted
    const enabled = [3, 3, 1, 1];
    const muted = [0, 0, 2, 2];
    const svg = F.renderTrackSvg("masks.front", ft, enabled, enabled.length, muted);

    const attrs = [...svg.matchAll(/<rect class="mask-seg"[^>]*>/g)].map((m) => {
        const get = (k) => (m[0].match(new RegExp(`${k}="([^"]*)"`)) || [])[1];
        return { label: get("data-label"), lane: get("data-lane"), state: get("data-state") };
    });
    assert.ok(attrs.length >= 2, `expected mask segments, got ${attrs.length}`);

    // Every rect names a real label, and its lane index agrees with that name's
    // position -- the property the two attributes exist to express.
    for (const a of attrs) {
        assert.ok(names.includes(a.label), `data-label ${a.label!==undefined ? `"${a.label}"` : "(absent)"} is not a label name`);
        assert.strictEqual(
            String(names.indexOf(a.label)), a.lane,
            `data-lane ${a.lane} does not match "${a.label}" at index ${names.indexOf(a.label)}`,
        );
    }
    // And the muted lane really is drawn as disabled, or "the labels agree"
    // would hold over a renderer that drew only one state.
    assert.ok(attrs.some((a) => a.state === "detected"), "no detected segment drawn");
    assert.ok(
        attrs.some((a) => a.state === "disabled" && a.label === "yellow block"),
        `the muted label was not drawn disabled: ${JSON.stringify(attrs)}`,
    );

    // A label name is attacker-controllable text typed into the segmenter, and
    // it lands in an attribute: it must be escaped, not interpolated raw.
    const nasty = ['a" onload="x', "b"];
    const svg2 = F.renderTrackSvg(
        "masks.front", { ...ft, mask_labels: nasty }, [3, 3], 2, [0, 0],
    );
    assert.ok(!svg2.includes('onload="x'), "a quote in a label name escaped its attribute");
}

// ── pending mask edits past the 32-bit wall ─────────────────────────────────
//
// The per-frame coverage bitsets are plain numbers, and the vocabulary grows and
// never shrinks -- the design sizes the panel for 40 labels. JavaScript's
// bitwise operators are 32-bit signed, so `1 << 32` is 1: editing the 33rd
// label would have flipped the FIRST label's lane on the timeline. Read the same
// way `isAbsent` reads them in apply_run_filter.js, which has its own bit-31 and
// bit-40 tests.
{
    const labels = [];
    for (let i = 0; i < 40; i++) labels.push(`l${i}`);
    const BIT = 35;
    const frames = 3;
    // The label is present (enabled) on every frame, at a bit well past 31.
    const enabled = [Math.pow(2, BIT), Math.pow(2, BIT), Math.pow(2, BIT)];
    const disabled = [0, 0, 0];

    win.currentEpisode = 0;
    win.pendingEdits = [{
        edit_type: "mask_range",
        episode_index: 0,
        params: { camera: "observation.images.top", label: `l${BIT}`, action: "disable", from_frame: 0, to_frame: 3 },
    }];

    const [en, dis] = win.FeatureEditing.applyPendingMaskEdits(
        "masks.top", labels, enabled, disabled, frames);
    const isSet = (v, b) => Math.floor(Math.round(v) / Math.pow(2, b)) % 2 === 1;
    for (let i = 0; i < frames; i++) {
        assert.ok(!isSet(en[i], BIT), `frame ${i}: label stayed enabled after disable`);
        assert.ok(isSet(dis[i], BIT), `frame ${i}: label was not marked disabled`);
        // The bug's signature: bit 35 wrapping onto bit 3 (35 % 32).
        assert.ok(!isSet(en[i], BIT % 32) && !isSet(dis[i], BIT % 32),
            `frame ${i}: editing bit ${BIT} touched bit ${BIT % 32} — the 32-bit wrap`);
    }
    console.log("feature_editing.test.js: pending mask edits survive past bit 31");
}
