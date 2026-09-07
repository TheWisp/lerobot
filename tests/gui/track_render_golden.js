// The inputs the golden covers, and the renderer under test. Shared by the
// generator (regen_track_render_golden.js) and the test, so the two cannot
// drift about what "every row type" means.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadRenderer() {
  // Straight from the module that owns it now -- no browser, no Inspector, no
  // vm. That this is possible at all is the point of the extraction.
  //
  // `window.MaskOverlay` is supplied because mask lanes take their colour from
  // the overlay's palette and fall back to FLAG_COLORS without it. Recording
  // the fallback would pin a palette the browser never renders -- and the two
  // deliberately diverge from the fourth entry, which is exactly the drift a
  // golden is here to catch.
  const dir = path.join(__dirname, "../../src/lerobot/gui/static");
  global.window = global.window || {};
  global.window.MaskOverlay = { PALETTE: MASK_PALETTE };
  return require(path.join(dir, "track_render.js")).renderTrackSvg;
}

// Copied from masks.js's PALETTE. A literal rather than a require: masks.js is
// a browser script with no module wrapper, and the golden's job is to notice
// when the rendered colour changes for ANY reason -- including that palette
// moving -- which a live import would hide.
const MASK_PALETTE = [
  [255, 99, 132], [54, 162, 235], [255, 206, 86], [75, 192, 192],
  [153, 102, 255], [255, 159, 64], [46, 204, 113], [231, 76, 60],
];

const N = 24;
const seq = (f) => Array.from({ length: N }, (_, i) => f(i));

// One entry per branch renderTrackSvg can take, plus the degenerate inputs.
// Deterministic by construction: no randomness, no clock, no DOM.
function cases() {
  const out = [];
  const str = { dtype: "string", shape: [1] };
  out.push(["string:runs", "subtask", str, seq((i) => `s${i % 3}`), N, undefined]);
  out.push(["string:uniform", "task", str, seq(() => "one instruction"), N, undefined]);

  const bool = { dtype: "bool", shape: [1] };
  out.push(["bool:alternating", "success", bool, seq((i) => i % 2 === 0), N, undefined]);
  out.push(["bool:all-false", "success", bool, seq(() => false), N, undefined]);

  for (const n of [2, 3, 6]) {
    const ft = { dtype: "int64", shape: [1], names: Array.from({ length: n }, (_, i) => `c${i}`) };
    out.push([`categorical:${n}`, "control_mode", ft, seq((i) => i % n), N, undefined]);
  }

  for (const shape of [[1], [3], [7]]) {
    const ft = { dtype: "float32", shape };
    const series = shape[0] === 1
      ? seq((i) => Math.sin(i))
      : seq((i) => Array.from({ length: shape[0] }, (_, k) => Math.sin(i + k)));
    out.push([`numeric:${shape.join("x")}`, "action", ft, series, N, undefined]);
  }

  // Lane indices past 31, where `>> bit & 1` wraps and `bitIsSet` does not.
  // Without these the golden cannot see the bit fix at all: reverting
  // `maskSegments` to the 32-bit shift changes nothing below bit 8.
  for (const hi of [31, 32, 40, 52]) {
    const n = hi + 1;
    const flags = Array.from({ length: n }, (_, i) => `f${i}`);
    const fseries = seq((i) => (i % 2 === 0 ? Math.pow(2, hi) : 0));
    out.push([`flags:bit${hi}`, "quality", { dtype: "int64", shape: [1], flags }, fseries, N, undefined]);

    const labels = Array.from({ length: n }, (_, i) => `obj${i}`);
    const mft = { dtype: "string", shape: [1], mask_encoding: "coco_rle", mask_labels: labels };
    const en = seq((i) => (i % 3 === 0 ? Math.pow(2, hi) : 0));
    const dis = seq((i) => (i % 3 === 1 ? Math.pow(2, hi) : 0));
    out.push([`masks:bit${hi}`, "masks.top", mft, en, N, dis]);
  }

  for (let n = 1; n <= 8; n++) {
    const ft = { dtype: "int64", shape: [1], flags: Array.from({ length: n }, (_, i) => `f${i}`) };
    out.push([`flags:${n}`, "quality", ft, seq((i) => i % 2 ** Math.min(n, 10)), N, undefined]);

    const labels = Array.from({ length: n }, (_, i) => `obj${i}`);
    const mft = { dtype: "string", shape: [1], mask_encoding: "coco_rle", mask_labels: labels };
    const en = [];
    const dis = [];
    for (let i = 0; i < N; i++) {
      let e = 0;
      let d = 0;
      for (let b = 0; b < n; b++) {
        const phase = (i + b) % 3;
        if (phase === 0) e |= 1 << b;
        else if (phase === 1) d |= 1 << b;
      }
      en.push(e);
      dis.push(d);
    }
    out.push([`masks:${n}`, "masks.top", mft, en, N, dis]);
  }

  out.push(["degenerate:empty", "x", str, [], 0, undefined]);
  out.push(["degenerate:null", "x", str, null, 10, undefined]);
  return out;
}

function render() {
  const renderTrackSvg = loadRenderer();
  const out = {};
  for (const [key, name, ft, series, length, muted] of cases()) {
    out[key] = renderTrackSvg(name, ft, series, length, muted);
  }
  return out;
}

module.exports = { render, GOLDEN: path.join(__dirname, "track_render_golden.json") };
