/*
  The recipe's composite, in the page, as overlays/effects.py does it on the
  server (docs/dataset_playback.md, "Masks and treatments in the page").

  One definition in two languages: the library composites the JPEG path's
  pixels and the training input; this composites a decoded chunk frame. The
  same frame, masks, recipe and noise through both agree pixel for pixel --
  tests/gui/test_mask_composite_equivalence.py runs both -- which takes
  mirroring the library's arithmetic, not just its algorithm:

    build_and_sample_regions -- background first (alpha = 1 - feather(union of
      every mask)), then each object with its own feathered exclusive mask;
      overlaps go to the smallest mask; a region treated `none` keeps its pixels.
    feathered_alpha -- dilate with an ellipse of 2*feather+1, then the 8-bit
      Gaussian of the 0/255 mask over 255, computed on the mask's bounding box
      plus its support, as the library does.
    _treat -- tint blends toward a colour in float32 and rounds half to even;
      solid replaces; random reads a texture given per region; blur is the
      8-bit Gaussian of sigma max(1, strength), kernel int(4*sigma)|1.
    composite_regions -- cv2.blendLinear: (out*(1-a) + treated*a)/(1-a+a+1e-5)
      in float32, rounded half to even, region by region in order.
    cv2.GaussianBlur on 8-bit -- not a float convolution: the kernel is the
      bit-exact Gaussian quantized to 1/256 with error diffusion and the centre
      taking the residual, both passes exact in integers, one half-up rounding
      at the end; sigma 0 uses cv2's own sigma for the size, and the small odd
      sizes up to 9 use its fixed binomial-like tables.

  The one thing not mirrored is the noise itself: the library draws it from
  numpy's generator seeded by (episode, recipe); the page draws its own,
  seeded the same way, so it is fixed per episode and recipe but not the same
  pixels. Operators accepted per-run coherence; training reads the library's.
*/
(function () {
  'use strict';

  const f32 = Math.fround;
  const F1E5 = f32(1e-5);
  const DEFAULT_PARAMS = { tint: { color: [79, 195, 247], strength: 0.55 }, blur: { strength: 12 }, solid: { color: [0, 200, 0] } };

  /** numpy.rint / cvRound: nearest, ties to even. */
  function rint(x) {
    const fl = Math.floor(x);
    const frac = x - fl;
    if (frac === 0.5) return fl % 2 === 0 ? fl : fl + 1;
    return frac < 0.5 ? fl : fl + 1;
  }
  const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

  /** resolve_params: the treatment's defaults under the recipe's own values. */
  function resolveParams(key, params) {
    const out = {};
    const d = DEFAULT_PARAMS[key] || {};
    for (const k of Object.keys(d)) out[k] = Array.isArray(d[k]) ? d[k].slice() : d[k];
    for (const k of Object.keys(params || {})) if (params[k] != null) out[k] = params[k];
    return out;
  }

  /** getGaussianKernelBitExact: the kernel as doubles, sigma <= 0 meaning cv2's own. */
  function gaussianKernelBitExact(n, sigma) {
    if (!(sigma > 0)) {
      const small = {
        1: [1], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
        7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
        9: [4 / 256, 13 / 256, 30 / 256, 51 / 256, 60 / 256, 51 / 256, 30 / 256, 13 / 256, 4 / 256],
      };
      if (small[n]) return Float64Array.from(small[n]);
    }
    const sx = sigma > 0 ? sigma : n * 0.15 + 0.35;
    const scale2X = -0.125 / (sx * sx);
    const n2 = (n - 1) >> 1;
    const vals = new Float64Array(n2);
    let sum = 0;
    for (let i = 0, x = 1 - n; i < n2; i++, x += 2) { vals[i] = Math.exp((x * x) * scale2X); sum += vals[i]; }
    sum = 2 * sum + 1;
    const mul1 = 1 / sum;
    const res = new Float64Array(n);
    for (let i = 0; i < n2; i++) res[i] = res[n - 1 - i] = vals[i] * mul1;
    res[n2] = 1 * mul1;
    return res;
  }

  /** getGaussianKernelFixedPoint_ED: the kernel in 1/256ths, error-diffused, summing to exactly 256. */
  function gaussianKernelFixed(n, sigma) {
    const kb = gaussianKernelBitExact(n, sigma);
    const res = new Int32Array(n);
    const n2 = n >> 1;
    let err = 0, sum = 0;
    for (let i = 0; i < n2; i++) {
      const adj = kb[i] * 256 + err;
      const v0 = rint(adj);
      err = adj - v0;
      res[i] = res[n - 1 - i] = v0;
      sum += v0;
    }
    res[n2] = 256 - 2 * sum;
    return res;
  }

  /** BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba. */
  function reflect101(i, n) {
    if (n === 1) return 0;
    while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; }
    return i;
  }

  /** cv2.GaussianBlur on an 8-bit image with `ch` interleaved channels, bit for bit. */
  function gaussianBlurU8(src, w, h, ch, n, sigma) {
    const k = gaussianKernelFixed(n, sigma);
    const r = n >> 1;
    const xi = new Int32Array(w + 2 * r), yi = new Int32Array(h + 2 * r);
    for (let x = -r; x < w + r; x++) xi[x + r] = reflect101(x, w);
    for (let y = -r; y < h + r; y++) yi[y + r] = reflect101(y, h);
    const tmp = new Int32Array(w * h * ch);   // 8 fractional bits, at most 255*256: exact
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) {
        let s = 0;
        for (let t = 0; t < n; t++) s += k[t] * src[(row + xi[x + t]) * ch + c];
        tmp[(row + x) * ch + c] = s;
      }
    }
    const out = new Uint8Array(w * h * ch);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) {
      let s = 0;   // 16 fractional bits, at most 256*255*256: exact in a double
      for (let t = 0; t < n; t++) s += k[t] * tmp[(yi[y + t] * w + x) * ch + c];
      const v = Math.floor((s + 32768) / 65536);
      out[(y * w + x) * ch + c] = v > 255 ? 255 : v;
    }
    return out;
  }

  /** cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) as row spans [j1, j2). */
  function ellipseSpans(k) {
    const r = k >> 1, c = k >> 1, invR2 = r ? 1 / (r * r) : 0;
    const spans = [];
    for (let i = 0; i < k; i++) {
      const dy = i - r;
      let j1 = 0, j2 = 0;
      if (Math.abs(dy) <= r) {
        const dx = r ? rint(c * Math.sqrt((r * r - dy * dy) * invR2)) : 0;
        j1 = Math.max(c - dx, 0); j2 = Math.min(c + dx + 1, k);
      }
      spans.push([j1, j2]);
    }
    return spans;
  }

  /**
   * feathered_alpha: the soft alpha of a union of 0/1 masks, as float32 in
   * [0, 1]; `window` receives the bounding box the alpha is nonzero within.
   */
  function featherAlpha(masks, w, h, feather, window) {
    const n = w * h;
    const union = new Uint8Array(n);
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (const m of masks) {
      if (!m || m.length !== n) continue;
      for (let i = 0; i < n; i++) if (m[i]) {
        union[i] = 1;
        const x = i % w, y = (i - x) / w;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
    const alpha = new Float32Array(n);
    if (x1 < 0) { if (window) { window[0] = window[1] = window[2] = window[3] = 0; } return alpha; }
    if (!(feather > 0)) {
      for (let i = 0; i < n; i++) alpha[i] = union[i];
      if (window) { window[0] = x0; window[1] = y0; window[2] = x1 + 1; window[3] = y1 + 1; }
      return alpha;
    }
    const f = feather, k = 2 * f + 1;
    // The support is the box plus dilation and blur radius; compute on one
    // more radius of guard band and discard it, as the library does, so the
    // blur's border reflection only matters where the frame's own border is.
    const wx0 = Math.max(0, x0 - 2 * f), wy0 = Math.max(0, y0 - 2 * f);
    const wx1 = Math.min(w, x1 + 1 + 2 * f), wy1 = Math.min(h, y1 + 1 + 2 * f);
    const cx0 = Math.max(0, wx0 - f), cy0 = Math.max(0, wy0 - f);
    const cx1 = Math.min(w, wx1 + f), cy1 = Math.min(h, wy1 + f);
    const cw = cx1 - cx0, chh = cy1 - cy0;
    const spans = ellipseSpans(k);
    const dil = new Uint8Array(cw * chh);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!union[y * w + x]) continue;
      for (let dy = -f; dy <= f; dy++) {
        const yy = y + dy - cy0; if (yy < 0 || yy >= chh) continue;
        const [j1, j2] = spans[dy + f];
        for (let j = j1; j < j2; j++) { const xx = x + j - f - cx0; if (xx >= 0 && xx < cw) dil[yy * cw + xx] = 255; }
      }
    }
    const soft = gaussianBlurU8(dil, cw, chh, 1, k, 0);
    for (let y = wy0; y < wy1; y++) for (let x = wx0; x < wx1; x++) {
      const v = soft[(y - cy0) * cw + (x - cx0)];
      alpha[y * w + x] = f32(v / 255);
    }
    if (window) { window[0] = wx0; window[1] = wy0; window[2] = wx1; window[3] = wy1; }
    return alpha;
  }

  const treated = (t) => !!(t && t.key && t.key !== 'none' && t.key !== '');

  /** cv2.blendLinear for one channel of one pixel: float32 throughout, then cvRound. */
  function blendU8(o, t, a) {
    const w1 = f32(1 - a);
    const den = f32(f32(w1 + a) + F1E5);
    return clamp255(rint(f32(f32(f32(o * w1) + f32(t * a)) / den)));
  }

  /**
   * composite_regions over build_and_sample_regions, in place on `rgba`.
   *   masks: {name: Uint8Array(w*h) 0/1}; treatments: {name: {key, params}};
   *   background: {key, params}; textures: {name|'__bg__': Uint8Array(w*h*3)};
   *   feather: the radius (the library's 5 at the stored size); scale: the
   *   ratio of this frame's width to the stored width, scaling the blur's sigma.
   */
  function compositeFrame(o) {
    const { rgba, w, h } = o;
    const masks = o.masks || {}, treatments = o.treatments || {}, textures = o.textures || {};
    const background = o.background || { key: 'none', params: {} };
    const feather = o.feather == null ? 5 : o.feather;
    const scale = o.scale || 1;
    const n = w * h;
    const names = Object.keys(masks);
    if (!treated(background) && !names.some((nm) => treated(treatments[nm]))) return rgba;
    const rgb = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++) { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]; }
    const out = new Uint8Array(rgb);
    // Overlaps go to the smallest mask claiming them, never to whichever is listed later.
    const area = {};
    for (const nm of names) { let s = 0; const m = masks[nm]; for (let i = 0; i < n; i++) s += m[i]; area[nm] = s; }
    const order = names.slice().sort((a, b) => area[a] - area[b] || (a < b ? -1 : a > b ? 1 : 0));
    const claimed = new Uint8Array(n);
    const exclusive = {};
    for (const nm of order) {
      const m = masks[nm], e = new Uint8Array(n);
      for (let i = 0; i < n; i++) { e[i] = m[i] && !claimed[i] ? 1 : 0; if (m[i]) claimed[i] = 1; }
      exclusive[nm] = e;
    }
    const regions = [];
    if (treated(background)) {
      const a = featherAlpha(names.map((nm) => masks[nm]), w, h, feather, null);
      for (let i = 0; i < n; i++) a[i] = f32(1 - a[i]);
      regions.push({ alpha: a, box: [0, 0, w, h], treatment: background, texture: textures.__bg__ });
    }
    for (const nm of names) {
      if (!treated(treatments[nm])) continue;
      const box = [0, 0, 0, 0];
      regions.push({ alpha: featherAlpha([exclusive[nm]], w, h, feather, box), box, treatment: treatments[nm], texture: textures[nm] });
    }
    const blurred = new Map();
    for (const reg of regions) {
      const key = reg.treatment.key;
      const params = resolveParams(key, reg.treatment.params);
      let tex = null, oneMinus = 0, cs = null, solid = null;
      if (key === 'tint') {
        const c = params.color, s = +params.strength;
        oneMinus = f32(1 - s);
        cs = [f32(f32(c[0]) * f32(s)), f32(f32(c[1]) * f32(s)), f32(f32(c[2]) * f32(s))];
      } else if (key === 'solid') {
        solid = params.color;
      } else if (key === 'random') {
        tex = reg.texture && reg.texture.length === n * 3 ? reg.texture : null;
      } else if (key === 'blur') {
        const sigma = Math.max(1, +params.strength) * scale;
        const k = Math.trunc(sigma * 4) | 1;
        const bk = `${k}:${sigma}`;
        if (!blurred.has(bk)) blurred.set(bk, gaussianBlurU8(rgb, w, h, 3, k, sigma));
        tex = blurred.get(bk);
      } else {
        throw new Error(`unknown treatment ${key}`);
      }
      const a = reg.alpha, [bx0, by0, bx1, by1] = reg.box;
      for (let y = by0; y < by1; y++) for (let x = bx0; x < bx1; x++) {
        const i = y * w + x;
        const ai = a[i];
        if (ai <= 0) continue;   // the blend leaves an untouched pixel untouched
        for (let c = 0; c < 3; c++) {
          const j = i * 3 + c;
          let t;
          if (cs) t = clamp255(rint(f32(f32(f32(rgb[j]) * oneMinus) + cs[c])));
          else if (solid) t = solid[c];
          else t = tex ? tex[j] : 0;
          out[j] = blendU8(out[j], t, ai);
        }
      }
    }
    for (let i = 0; i < n; i++) { rgba[i * 4] = out[i * 3]; rgba[i * 4 + 1] = out[i * 3 + 1]; rgba[i * 4 + 2] = out[i * 3 + 2]; }
    return rgba;
  }

  /** One noise texture -- rgb per pixel -- for a key such as `${episode}:${fingerprint}:${region}`. */
  function noiseTexture(w, h, key) {
    let x = 2166136261;
    for (let i = 0; i < key.length; i++) { x ^= key.charCodeAt(i); x = Math.imul(x, 16777619) >>> 0; }
    x = x || 1;
    const out = new Uint8Array(w * h * 3);
    for (let i = 0; i < out.length; i++) { x = (Math.imul(x, 1103515245) + 12345) >>> 0; out[i] = (x >>> 8) & 255; }
    return out;
  }

  window.MaskComposite = {
    compositeFrame, featherAlpha, gaussianBlurU8, gaussianKernelBitExact, gaussianKernelFixed, ellipseSpans, blendU8,
    resolveParams, noiseTexture, rint, reflect101,
  };
})();
