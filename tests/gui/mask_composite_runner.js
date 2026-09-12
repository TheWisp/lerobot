// Runs the page's compositor on a frame given as JSON on stdin, for the
// equivalence test against overlays/effects.py. Prints the composited RGB.
const fs = require("fs");
const path = require("path");
const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");
global.window = { addEventListener() {} };
new Function(fs.readFileSync(path.join(STATIC, "mask_composite.js"), "utf8"))();
const MC = global.window.MaskComposite;
const req = JSON.parse(fs.readFileSync(0, "utf8"));
const { w, h } = req;
const rgba = new Uint8ClampedArray(w * h * 4);
for (let i = 0; i < w * h; i++) { rgba[i * 4] = req.rgb[i * 3]; rgba[i * 4 + 1] = req.rgb[i * 3 + 1]; rgba[i * 4 + 2] = req.rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255; }
const masks = {};
for (const [name, m] of Object.entries(req.masks)) masks[name] = Uint8Array.from(m);
const textures = {};
for (const [name, t] of Object.entries(req.textures || {})) textures[name] = t ? Uint8Array.from(t) : null;
MC.compositeFrame({ rgba, w, h, masks, treatments: req.treatments, background: req.background, textures, feather: req.feather });
const out = new Array(w * h * 3);
for (let i = 0; i < w * h; i++) { out[i * 3] = rgba[i * 4]; out[i * 3 + 1] = rgba[i * 4 + 1]; out[i * 3 + 2] = rgba[i * 4 + 2]; }
process.stdout.write(JSON.stringify({ rgb: out }));
