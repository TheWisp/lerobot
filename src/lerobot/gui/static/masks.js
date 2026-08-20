// Segmentation-mask overlay: fetch an episode's masks once, draw them locally.
//
// The live overlay path costs a publish POST plus a pull GET per displayed
// frame. At the ~240 ms round trip an operator on the other side of the world
// actually has, that caps scrubbing near two frames per second however fast the
// segmenter runs. Stored masks are already computed, so an episode arrives in
// one gzipped response and every frame after that is local.
//
// This draws masks as DISPLAY CHROME only. It deliberately does not reproduce
// the treatments in effects.py — those are numpy/cv2 and are the single source
// of truth for committed pixels ("preview == commit"); a canvas reimplementation
// would be a second, subtly different one (Gaussian vs box blur, a noise texture
// no JS PRNG can reproduce, different rounding on the alpha blend). Judging a
// mask needs its boundary, not its treatment.

(function () {
    'use strict';

    // COCO's rleFrString, inverse of the Python encoder. Counts are delta-coded
    // against two positions back and written 5 bits per character, biased to
    // printable ASCII; bit 0x10 of a group is its sign bit.
    function decodeCounts(s) {
        const counts = [];
        let p = 0;
        while (p < s.length) {
            let x = 0, k = 0, more = true;
            while (more) {
                const c = s.charCodeAt(p) - 48;
                x |= (c & 0x1f) << (5 * k);
                more = (c & 0x20) !== 0;
                p++; k++;
                if (!more && (c & 0x10)) x |= -1 << (5 * k);
            }
            if (counts.length > 2) x += counts[counts.length - 2];
            counts.push(x);
        }
        return counts;
    }

    // Runs are COLUMN-major, starting with a run of zeros. Returns a row-major
    // Uint8Array so it can be indexed the way a canvas is.
    function decodeMask(counts, h, w) {
        const out = new Uint8Array(h * w);
        let pos = 0, value = 0;
        for (const run of decodeCounts(counts)) {
            if (value) {
                for (let i = pos; i < pos + run; i++) {
                    const col = (i / h) | 0, row = i % h;   // undo column-major
                    out[row * w + col] = 1;
                }
            }
            pos += run;
            value ^= 1;
        }
        if (pos !== h * w) throw new Error(`RLE covers ${pos} px, expected ${h * w}`);
        return out;
    }

    const PALETTE = [
        [79, 195, 247], [255, 138, 101], [174, 213, 129], [186, 104, 200],
        [255, 213, 79], [77, 208, 225], [240, 98, 146], [161, 136, 127],
    ];

    // One <canvas> worth of masks for a single frame. `frame` is the stored
    // [[labelId, rle], ...]; alpha is per-object so overlapping objects stay
    // distinguishable rather than compositing into a third colour.
    function drawFrame(canvas, frame, size, opts) {
        const [h, w] = size;
        const alpha = (opts && opts.alpha != null) ? opts.alpha : 0.45;
        const hidden = (opts && opts.hidden) || new Set();
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        if (!frame || !frame.length) return 0;

        const img = ctx.createImageData(w, h);
        const px = img.data;
        let drawn = 0;
        for (const [labelId, counts] of frame) {
            if (hidden.has(labelId)) continue;
            const [r, g, b] = PALETTE[labelId % PALETTE.length];
            const mask = decodeMask(counts, h, w);
            for (let i = 0; i < mask.length; i++) {
                if (!mask[i]) continue;
                const o = i * 4;
                // Later objects win the pixel outright; blending them would
                // invent a colour that belongs to no object.
                px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = Math.round(alpha * 255);
            }
            drawn++;
        }
        ctx.putImageData(img, 0, 0);
        return drawn;
    }

    // One request per (dataset, episode); scrubbing never touches the network.
    const cache = new Map();
    async function fetchEpisode(datasetId, episodeIdx) {
        const key = `${datasetId}::${episodeIdx}`;
        if (cache.has(key)) return cache.get(key);
        const p = fetch(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/masks`)
            .then((r) => (r.ok ? r.json() : null))
            .catch(() => null);
        cache.set(key, p);
        return p;
    }

    function invalidate(datasetId) {
        for (const key of [...cache.keys()]) {
            if (!datasetId || key.startsWith(`${datasetId}::`)) cache.delete(key);
        }
    }

    // ---- playhead integration ------------------------------------------
    // Driven from app.js's updateFrameUI, the same hook FeatureEditing and
    // Overlays use, so masks follow the playhead on both the still path and the
    // video clock without a second notion of "current frame".

    let enabled = true;
    const hiddenLabels = new Set();
    let loaded = null;   // { key, episode, cameras } for the episode in hand

    function _canvas(camKey) {
        return document.getElementById(`mask-${camKey.replace(/\./g, '-')}`);
    }

    // The mask column is named for the camera it describes: the feature key
    // swaps `images` for `masks`, so a tile can find its own without a map.
    function _maskKeyFor(camKey) {
        return camKey.replace('.images.', '.masks.');
    }

    function _clearAll() {
        document.querySelectorAll('canvas.mask-layer').forEach((c) => {
            c.style.display = 'none';
            const ctx = c.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, c.width, c.height);
        });
    }

    async function onPlayheadChanged() {
        const ds = window.currentDataset;
        const ep = window.currentEpisode;
        if (!enabled || !ds || ep === null || ep === undefined) { _clearAll(); return; }
        // Display arbitration: while the live overlay is tuning (worker active
        // or the stream playing), the tiles show CURRENT settings — drawing
        // stored masks at the same time stacks two different truths on one
        // image, and after a settings change they disagree. Saved masks render
        // only when the live layer is off; turning the overlay off is how you
        // review what is saved.
        const badge = document.getElementById('overlays-badge');
        const liveActive = (window.OverlayStream && window.OverlayStream.streaming)
            || (!!badge && (/\bok\b/.test(badge.className)
                || (/\bidle\b/.test(badge.className) && !/^busy/.test(badge.textContent || ''))
                || /\bloading\b/.test(badge.className)));
        if (liveActive) { _clearAll(); return; }

        const key = `${ds}::${ep}`;
        if (!loaded || loaded.key !== key) {
            loaded = { key, episode: ep, cameras: null };   // claim before awaiting
            const body = await fetchEpisode(ds, ep);
            if (!loaded || loaded.key !== key) return;      // playhead moved on while fetching
            loaded.cameras = (body && body.cameras) || {};
        }
        if (!loaded.cameras) return;                        // fetch still in flight

        const frameIdx = window.currentFrame || 0;
        for (const [maskKey, data] of Object.entries(loaded.cameras)) {
            const camKey = maskKey.replace('.masks.', '.images.');
            const canvas = _canvas(camKey);
            if (!canvas) continue;
            const frame = data.frames && data.frames[frameIdx];
            const drawn = drawFrame(canvas, frame, data.size, { hidden: hiddenLabels });
            canvas.style.display = drawn ? 'block' : 'none';
        }
    }

    function setEnabled(on) {
        enabled = !!on;
        if (!enabled) _clearAll();
        else onPlayheadChanged();
    }

    function setLabelHidden(labelId, hidden) {
        if (hidden) hiddenLabels.add(labelId);
        else hiddenLabels.delete(labelId);
        onPlayheadChanged();
    }

    /** Labels available for the episode in hand, for building a legend. */
    function currentLabels() {
        if (!loaded || !loaded.cameras) return [];
        const first = Object.values(loaded.cameras)[0];
        return (first && first.labels) || [];
    }

    window.MaskOverlay = {
        decodeCounts, decodeMask, drawFrame, fetchEpisode, invalidate, PALETTE,
        onPlayheadChanged, setEnabled, setLabelHidden, currentLabels,
        _maskKeyFor,
    };
})();
