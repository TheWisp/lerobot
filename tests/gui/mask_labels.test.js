// How a stored-mask region's name is sized and placed (run from pytest via
// test_mask_labels_js.py, or directly with `node tests/gui/mask_labels.test.js`).
//
// Two things were reported together: "a weird large font ... that stacks on top
// of existing ones". They are separate faults with one shared cause -- the two
// layers that can draw a name had their own rules, and neither avoided the
// other's pixels.
//
// Size: the old rule floored the font at 11px in CANVAS space, which renders at
// 11 * scale on screen, so any tile shown larger than the stored mask grew the
// label. Sizing by a fraction of the SOURCE height instead makes it independent
// of the tile, and matches what the live overlay does in Python.
//
// Placement: regions near the top edge all clamp to the same y, so their pills
// landed on the same pixels and only the last was readable.
const assert = require("assert");

global.window = { addEventListener() {} };
global.document = { addEventListener() {} };
require("../../src/lerobot/gui/static/masks.js");
const { labelFontPx, freeSlot } = global.window.MaskOverlay;

// 1. The size depends on the frame, not on how big the tile happens to be.
{
  assert.strictEqual(labelFontPx(720), Math.round(720 * 0.032), "720p label should be ~23px");
  assert.strictEqual(labelFontPx(240), 14, "small frames get the legibility floor");
  // The property that matters: same frame -> same size, whatever the display.
  assert.strictEqual(labelFontPx(720), labelFontPx(720));
}

// 2. Monotonic in the frame height: a bigger frame never gets a smaller label.
{
  let prev = 0;
  for (const h of [120, 240, 480, 720, 1080, 2160]) {
    const px = labelFontPx(h);
    assert.ok(px >= prev, `label shrank from ${prev} to ${px} at h=${h}`);
    prev = px;
  }
}

// 3. The reported stacking: two names whose boxes land on the same pixels.
{
  const first = [10, 0, 100, 20];
  const moved = freeSlot([10, 0, 100, 20], [first], 720, 20);
  assert.ok(moved[1] >= first[3], `second label still overlaps the first: ${moved}`);
}

// 4. The complement, so "always moves down" cannot pass as a fix: a box that
//    collides with nothing must not be nudged at all.
{
  const box = [10, 300, 100, 320];
  assert.deepStrictEqual(freeSlot(box, [[200, 300, 300, 320]], 720, 20), box, "a clear box moved");
  assert.deepStrictEqual(freeSlot(box, [], 720, 20), box, "a box moved with nothing in the way");
}

// 5. Several stacked labels each end up somewhere of their own.
{
  const placed = [];
  for (let i = 0; i < 4; i++) {
    placed.push(freeSlot([10, 0, 100, 20], placed, 720, 20));
  }
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const [a, b] = [placed[i], placed[j]];
      const overlap = a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
      assert.ok(!overlap, `labels ${i} and ${j} overlap: ${a} vs ${b}`);
    }
  }
}

// 6. Rather off-screen than nowhere: a frame with no room left keeps the label
//    where it was instead of sliding it past the bottom edge.
{
  const box = [10, 700, 100, 719];
  assert.deepStrictEqual(freeSlot(box, [[10, 700, 100, 719]], 720, 20), box, "slid out of frame");
}

console.log("mask_labels.test.js: ok");
