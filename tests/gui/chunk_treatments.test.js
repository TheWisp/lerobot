// The page's compositor module (mask_composite.js) on its own. The comparison
// against the library is test_mask_composite_equivalence.py; what is pinned
// here is what that comparison would not explain on its own: the pieces of
// cv2's 8-bit arithmetic the module mirrors, and the composite's ordering.
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");
global.window = { addEventListener() {} };
new Function(fs.readFileSync(path.join(STATIC, "mask_composite.js"), "utf8"))();
const MC = global.window.MaskComposite;

// numpy's rounding: ties to even, which is what makes a .5 blend agree with the library.
assert.strictEqual(MC.rint(4.5), 4); assert.strictEqual(MC.rint(5.5), 6); assert.strictEqual(MC.rint(4.4), 4); assert.strictEqual(MC.rint(4.6), 5);
// cv2's border: gfedcb|abcdefgh|gfedcba
assert.deepStrictEqual([-2, -1, 0, 7, 8, 9].map((i) => MC.reflect101(i, 8)), [2, 1, 0, 7, 6, 5]);
// cv2's 8-bit kernels: fixed tables for the small sizes, error-diffused 1/256ths otherwise, always summing to one
assert.deepStrictEqual(Array.from(MC.gaussianKernelFixed(3, 0)), [64, 128, 64]);
assert.deepStrictEqual(Array.from(MC.gaussianKernelFixed(9, 0)), [4, 13, 30, 51, 60, 51, 30, 13, 4]);
assert.deepStrictEqual(Array.from(MC.gaussianKernelFixed(11, 0)), [2, 7, 17, 31, 45, 52, 45, 31, 17, 7, 2], "feather 5: cv2's sigma for the size");
assert.deepStrictEqual(Array.from(MC.gaussianKernelFixed(13, 3)), [5, 8, 15, 21, 28, 33, 36, 33, 28, 21, 15, 8, 5]);
for (const [n, s] of [[49, 12], [41, 10.25], [21, 5]]) assert.strictEqual(MC.gaussianKernelFixed(n, s).reduce((a, b) => a + b, 0), 256);
// the blur of a flat frame is the frame, and rounds half up at the end
const flat = new Uint8Array(2 * 2 * 3).fill(77);
assert.deepStrictEqual(Array.from(MC.gaussianBlurU8(flat, 2, 2, 3, 49, 12)), Array.from(flat));
// the ellipse structuring element is symmetric and full at its middle row
const spans = MC.ellipseSpans(11); assert.deepStrictEqual(spans[5], [0, 11]); assert.deepStrictEqual(spans[0], spans[10]);
// the blend is float32, as cv2's: this triple rounds to 194 in float32 and 195 in double
assert.strictEqual(MC.blendU8(2, 210, Math.fround(236 / 255)), 194);
assert.strictEqual(MC.blendU8(100, 100, Math.fround(0.5)), 100);
assert.strictEqual(MC.blendU8(255, 0, 0), 255, "alpha 0 leaves the pixel");
// defaults under the recipe's own values, as resolve_params
assert.deepStrictEqual(MC.resolveParams("tint", { strength: 0.3 }), { color: [79, 195, 247], strength: 0.3 });
assert.deepStrictEqual(MC.resolveParams("solid", { color: null }), { color: [0, 200, 0] });

const W = 2, H = 2, N = W * H;
const frame = () => { const r = new Uint8ClampedArray(N * 4); for (let i = 0; i < N; i++) { r[i * 4] = 100; r[i * 4 + 1] = 10; r[i * 4 + 2] = 200; r[i * 4 + 3] = 255; } return r; };
const all = new Uint8Array([1, 1, 1, 1]);
// tint: float32 blend, ties to even (10*0.45 + 0*0.55 = 4.5 -> 4, not 5)
let rgba = frame();
MC.compositeFrame({ rgba, w: W, h: H, masks: { m: all }, treatments: { m: { key: "tint", params: { color: [0, 0, 0], strength: 0.55 } } }, textures: {}, feather: 0 });
assert.deepStrictEqual(Array.from(rgba.slice(0, 3)), [45, 4, 90], "4.5 rounds to even, as numpy does");
// solid replaces; random reads the region's texture pixel for pixel; an unknown key is an error
rgba = frame();
MC.compositeFrame({ rgba, w: W, h: H, masks: { m: all }, treatments: { m: { key: "solid", params: {} } }, textures: {}, feather: 0 });
assert.deepStrictEqual(Array.from(rgba.slice(0, 3)), [0, 200, 0]);
const tex = new Uint8Array(N * 3).map((_, i) => i * 7);
rgba = frame();
MC.compositeFrame({ rgba, w: W, h: H, masks: { m: all }, treatments: { m: { key: "random", params: {} } }, textures: { m: tex }, feather: 0 });
for (let i = 0; i < N; i++) assert.deepStrictEqual(Array.from(rgba.slice(i * 4, i * 4 + 3)), Array.from(tex.slice(i * 3, i * 3 + 3)));
assert.throws(() => MC.compositeFrame({ rgba: frame(), w: W, h: H, masks: { m: all }, treatments: { m: { key: "sparkle", params: {} } }, textures: {}, feather: 0 }));

// the composite: background first, objects on top, a `none` region keeps its pixels
rgba = frame();
MC.compositeFrame({ rgba, w: W, h: H, masks: { ball: new Uint8Array([1, 0, 0, 0]) },
  treatments: { ball: { key: "solid", params: { color: [0, 200, 0] } } }, background: { key: "solid", params: { color: [1, 2, 3] } }, textures: {}, feather: 0 });
assert.deepStrictEqual(Array.from(rgba.slice(0, 3)), [0, 200, 0], "the object wins where it is");
assert.deepStrictEqual(Array.from(rgba.slice(4, 7)), [1, 2, 3], "the background is everywhere the object is not");
const same = new Uint8ClampedArray(rgba);
MC.compositeFrame({ rgba: same, w: W, h: H, masks: { ball: new Uint8Array([1, 0, 0, 0]) }, treatments: { ball: { key: "none", params: {} } }, background: { key: "none", params: {} }, textures: {}, feather: 0 });
assert.deepStrictEqual(Array.from(same), Array.from(rgba), "nothing treated: the frame is untouched");

// one texture per key
const t1 = MC.noiseTexture(4, 3, "0:abc:ball"), t2 = MC.noiseTexture(4, 3, "0:abc:ball"), t3 = MC.noiseTexture(4, 3, "1:abc:ball");
assert.deepStrictEqual(Array.from(t1), Array.from(t2)); assert.notDeepStrictEqual(Array.from(t1), Array.from(t3));

console.log("chunk_treatments: ok");
