// The depth tile's colour-range control, as markup and arithmetic.
const assert = require("assert");
const path = require("path");

const DepthChip = require(path.join(__dirname, "../../src/lerobot/gui/static/depth_chip.js"));

// Auto: the select says what it is about, the boxes show the episode's own
// range read-only, and there are no thumbs to drag.
{
  const html = DepthChip.render("observation.images.top_depth", { mode: "auto", lo: 0.2, hi: 1.0 }, { lo: 0.41, hi: 1.07 }, [0.4, 1.1]);
  assert.match(html, /colour/);
  assert.match(html, /<option value="auto" selected>auto range<\/option>/);
  assert.match(html, /<option value="fixed" >fixed range<\/option>/);
  assert.match(html, /data-depth-edge="lo"[^>]*value="0\.41"[^>]*readonly/);
  assert.match(html, /data-depth-edge="hi"[^>]*value="1\.07"[^>]*readonly/);
  assert.doesNotMatch(html, /type="range"/);
  assert.match(html, /depth-chip auto/);
}

// Fixed: editable boxes and two thumbs over the strip, with the slider reaching
// past the dataset's far end so both thumbs have room.
{
  const html = DepthChip.render("observation.images.top_depth", { mode: "fixed", lo: 0.4, hi: 0.55 }, { lo: 0.41, hi: 1.07 }, [0.4, 1.1]);
  assert.match(html, /<option value="fixed" selected>fixed range<\/option>/);
  assert.match(html, /data-depth-edge="lo"[^>]*value="0\.40"/);
  assert.doesNotMatch(html, /readonly/);
  assert.strictEqual((html.match(/type="range"/g) || []).length, 2);
  assert.match(html, /data-depth-knob="lo"[^>]*max="1\.7"/);
  assert.match(html, /data-depth-knob="hi"[^>]*max="1\.7"[^>]*value="0\.55"/);
  assert.match(html, /linear-gradient\(to right, #7a0403 0%/);
}

// The slider's reach: half again past the dataset's far end, rounded up to a
// decimetre, never below the values in hand, never below a metre.
assert.strictEqual(DepthChip.span([0.4, 1.1], 0.4, 0.55), 1.7);
assert.strictEqual(DepthChip.span([0.2, 0.3], 0.2, 0.3), 1.0);
assert.strictEqual(DepthChip.span(null, 0.5, 3.2), 3.2);

// The palette sits between lo and hi on the strip and is flat beyond them.
{
  const g = DepthChip.gradient(0.5, 1.0, 2.0);
  assert.match(g, /#7a0403 0%, #7a0403 25\.0%/);
  assert.match(g, /#30123b 50\.0%, #30123b 100%/);
}

// Edits cannot cross or go negative, and the untouched end stays put.
{
  const s = { mode: "fixed", lo: 0.4, hi: 0.55 };
  assert.deepStrictEqual(DepthChip.applyEdit(s, "lo", "0.7"), { mode: "fixed", lo: 0.54, hi: 0.55 });
  assert.deepStrictEqual(DepthChip.applyEdit(s, "hi", "0.3"), { mode: "fixed", lo: 0.4, hi: 0.41 });
  assert.deepStrictEqual(DepthChip.applyEdit(s, "lo", "-1"), { mode: "fixed", lo: 0, hi: 0.55 });
  assert.deepStrictEqual(DepthChip.applyEdit(s, "hi", "1.234"), { mode: "fixed", lo: 0.4, hi: 1.23 });
  assert.deepStrictEqual(DepthChip.applyEdit(s, "hi", "abc"), s);
  assert.deepStrictEqual(DepthChip.applyEdit(s, "mid", "0.5"), s);
}
console.log("depth_chip.test.js ok");
