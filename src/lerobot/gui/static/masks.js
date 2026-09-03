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
    // A stored entry is `[labelId, counts]`, or `[labelId, counts, enabled]`.
    // An absent third element means enabled -- the writer omits the default, so
    // every row predating the flag reads as enabled. Mirrors
    // `mask_codec._entries`; the two must agree or a mask the trainer ignores
    // still shows up here, which reads as the mute not having worked.
    function entryEnabled(entry) {
        return entry.length > 2 ? Boolean(entry[2]) : true;
    }

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
    /** How many display pixels one mask pixel occupies, >0 and <=1 in practice. */
    // Mirrors overlays/standalone.py's label_font_px: a constant fraction of the
    // frame, so the two layers that can draw a name agree on its size.
    function labelFontPx(frameH) {
        return Math.max(14, Math.floor(frameH * 0.032));   // truncates, as Python's int() does
    }

    // Slide a label box down until it clears the ones already placed. Regions
    // near the top edge all clamp to the same y, so without this the last name
    // drawn is the only one readable.
    function freeSlot(box, taken, h, step) {
        let [x0, y0, x1, y1] = box;
        for (let i = 0; i <= taken.length; i++) {
            const hit = taken.find((r) => x0 < r[2] && r[0] < x1 && y0 < r[3] && r[1] < y1);
            if (!hit) break;
            const shift = hit[3] - y0 + Math.max(1, Math.round(step / 4));
            y0 += shift; y1 += shift;
            if (y1 >= h) return box;   // out of frame: leave it where it started
        }
        return [x0, y0, x1, y1];
    }

    function _displayScale(canvas, w) {
        const shown = canvas.clientWidth || canvas.getBoundingClientRect().width || w;
        return Math.max(shown / w, 1 / 16);   // never divide the stroke to nothing
    }

    function drawFrame(canvas, frame, size, opts) {
        const [h, w] = size;
        const alpha = (opts && opts.alpha != null) ? opts.alpha : 0.45;
        const hidden = (opts && opts.hidden) || new Set();
        // Outline mode paints only boundary pixels, so it can sit on top of
        // composited pixels without double-painting them — which is what a
        // fill would do, and why the saved view used to show no mask at all.
        const outline = !!(opts && opts.outline);
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, w, h);
        if (!frame || !frame.length) return 0;

        const img = ctx.createImageData(w, h);
        const px = img.data;
        const named = [];        // where each outlined region can carry its name
        let drawn = 0;
        for (const entry of frame) {
            const [labelId, counts] = entry;
            // Disabled: excluded from the composite the trainer sees, so it is
            // not drawn here either.
            if (!entryEnabled(entry)) continue;
            if (hidden.has(labelId)) continue;
            const [r, g, b] = PALETTE[labelId % PALETTE.length];
            const mask = decodeMask(counts, h, w);
            if (outline) {
                // A boundary pixel is an on-pixel with an off 4-neighbour;
                // frame edges count as outside, so a region running off the
                // image still reads as bounded. The stroke is widened by the
                // canvas's display scale: masks are stored at source
                // resolution (1280 wide) and shown in a ~290 px tile, where a
                // one-pixel line lands on a fifth of a screen pixel and
                // disappears.
                const rad = Math.max(0, Math.min(6, Math.round(1 / _displayScale(canvas, w)) - 1));
                let minX = w, minY = h, maxX = -1;
                for (let y = 0; y < h; y++) {
                    const row = y * w;
                    for (let x = 0; x < w; x++) {
                        const i = row + x;
                        if (!mask[i]) continue;
                        if (x > 0 && mask[i - 1] && x < w - 1 && mask[i + 1]
                            && y > 0 && mask[i - w] && y < h - 1 && mask[i + w]) continue;
                        if (x < minX) minX = x;
                        if (y < minY) minY = y;
                        if (x > maxX) maxX = x;
                        for (let dy = -rad; dy <= rad; dy++) {
                            const yy = y + dy;
                            if (yy < 0 || yy >= h) continue;
                            for (let dx = -rad; dx <= rad; dx++) {
                                const xx = x + dx;
                                if (xx < 0 || xx >= w) continue;
                                const o = (yy * w + xx) * 4;
                                px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255;
                            }
                        }
                    }
                }
                if (maxX >= 0) named.push({ labelId, x: minX, y: minY, right: maxX });
            } else {
                for (let i = 0; i < mask.length; i++) {
                    if (!mask[i]) continue;
                    const o = i * 4;
                    // Later objects win the pixel outright; blending them would
                    // invent a colour that belongs to no object.
                    px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = Math.round(alpha * 255);
                }
            }
            drawn++;
        }
        ctx.putImageData(img, 0, 0);

        // Name each outlined region, the way the live overlay does — an
        // unlabelled outline tells you something was found but not what the
        // segmenter thought it was, which is the half that matters when two
        // objects touch. Sized against the display scale, since the canvas is
        // at mask resolution and shown a quarter that size.
        const names = (opts && opts.labels) || [];
        if (outline && names.length) {
            // Same size rule and same pill as the live overlay draws
            // (overlays/standalone.py: label_font_px), so a name does not
            // change size or shape when the segmenter is toggled over the same
            // frame. Sizing by a fraction of the SOURCE height rather than by
            // the display scale also keeps it stable while a tile resizes --
            // the previous rule had an 11px floor in canvas space, which
            // rendered at 11 x scale on screen and so grew on any tile shown
            // larger than the stored mask.
            const fontPx = labelFontPx(h);
            const pad = Math.max(2, Math.round(fontPx / 5));
            ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
            ctx.textBaseline = 'top';
            const placed = [];
            for (const { labelId, x, y, right } of named) {
                const name = names[labelId];
                if (!name) continue;
                const [r, g, b] = PALETTE[labelId % PALETTE.length];
                const tw = ctx.measureText(name).width;
                // Keep the name inside the frame: a region touching the right
                // edge would otherwise write its label off-canvas.
                const tx = Math.min(x, Math.max(0, Math.min(right, w - tw - 2 * pad)));
                const ty = Math.max(0, y - fontPx - 2 * pad);
                const box = freeSlot(
                    [tx, ty, tx + tw + 2 * pad, ty + fontPx + 2 * pad], placed, h, fontPx,
                );
                ctx.fillStyle = 'rgb(18, 22, 28)';
                ctx.fillRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillText(name, box[0] + pad, box[1] + pad);
                placed.push(box);
            }
        }
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

    // Saved recipe + per-camera fingerprints, one cheap /masks/status per
    // episode. The fingerprint rides composited frame/video URLs purely as a
    // cache-buster: an effects edit changes it, forcing the browser (and the
    // <video> element, whose src must change to reload) to refetch.
    const statusCache = new Map();
    function fetchStatus(datasetId, episodeIdx) {
        const key = `${datasetId}::${episodeIdx}`;
        if (statusCache.has(key)) return statusCache.get(key);
        const p = fetch(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/masks/status`)
            .then((r) => (r.ok ? r.json() : null))
            .catch(() => null);
        statusCache.set(key, p);
        return p;
    }
    let savedStatus = null;   // { key, data } for the episode in hand
    // Fingerprints from an effects write we made ourselves. They outrank the
    // status response (which may still be in flight, or cached from before the
    // write) until the episode changes.
    const fpOverride = new Map();

    // Bumped whenever the stored masks may have changed. The frame URL carries
    // it, because without a changing URL the browser answers a re-render from
    // its own cache: after a save the rows were new, the server composited
    // correctly, and the tile still showed the picture from before the edit.
    let _maskVersion = 0;

    function invalidate(datasetId) {
        _maskVersion++;
        for (const c of [cache, statusCache]) {
            for (const key of [...c.keys()]) {
                if (!datasetId || key.startsWith(`${datasetId}::`)) c.delete(key);
            }
        }
        savedStatus = null;
        fpOverride.clear();
        // Drop the episode in hand too, or a save of the episode being viewed
        // keeps answering from the pre-save decode: the tiles never switch to
        // the composited view and the saved-effects panel never appears until
        // the operator switches episodes and back.
        if (!datasetId || (loaded && loaded.key.startsWith(`${datasetId}::`))) loaded = null;
    }

    // ---- playhead integration ------------------------------------------
    // Driven from app.js's updateFrameUI, the same hook FeatureEditing and
    // Overlays use, so masks follow the playhead on both the still path and the
    // video clock without a second notion of "current frame".

    let enabled = true;
    // Outlines are the composited mode's only mask cue, so they default on;
    // the fill remains for the non-composited case, where nothing else marks
    // the region at all.
    let outlinesOnly = true;
    const hiddenLabels = new Set();
    let loaded = null;   // { key, episode, cameras } for the episode in hand

    function _canvas(camKey) {
        return document.getElementById(`mask-${camKey.replace(/\./g, '-')}`);
    }

    // The mask column is named for the camera it describes, under its own
    // top-level namespace rather than inside `observation.` -- see
    // MASK_NAMESPACE in datasets/mask_compositing.py for why. These two are
    // the only place the JS side constructs or reads that name; keep them in
    // step with the Python pair.
    const MASK_NS = 'masks.';
    const CAM_INFIX = '.images.';

    function _maskKeyFor(camKey) {
        const i = camKey.indexOf(CAM_INFIX);
        return i < 0 ? camKey : MASK_NS + camKey.slice(i + CAM_INFIX.length);
    }

    function _camKeyFor(maskKey, cameraKeys) {
        if (!maskKey.startsWith(MASK_NS)) return maskKey;
        const suffix = maskKey.slice(MASK_NS.length);
        for (const cam of cameraKeys || []) {
            if (cam.endsWith(CAM_INFIX + suffix)) return cam;
        }
        return 'observation' + CAM_INFIX + suffix;
    }

    function _clearAll() {
        document.querySelectorAll('canvas.mask-layer').forEach((c) => {
            c.style.display = 'none';
            const ctx = c.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, c.width, c.height);
        });
    }

    let _drewThisTick = false;

    async function onPlayheadChanged() {
        _drewThisTick = false;
        const ds = window.currentDataset;
        const ep = window.currentEpisode;
        if (!enabled || !ds || ep === null || ep === undefined) { _clearAll(); return; }
        // Display arbitration: while the live overlay is tuning (worker active
        // or the stream playing), the tiles show CURRENT settings — drawing
        // stored masks at the same time stacks two different truths on one
        // image, and after a settings change they disagree. Saved masks render
        // only when the live layer is off; turning the overlay off is how you
        // review what is saved.
        const liveActive = _liveLayerActive();
        if (liveActive) { _drewThisTick = false; _clearAll(); _setComposited(false); return; }

        const key = `${ds}::${ep}`;
        if (!savedStatus || savedStatus.key !== key) {
            fpOverride.clear();                            // they belonged to the old episode
            savedStatus = { key, data: null };              // claim before awaiting
            fetchStatus(ds, ep).then((body) => {
                if (savedStatus && savedStatus.key === key) savedStatus.data = body;
            });
        }
        if (!loaded || loaded.key !== key) {
            loaded = { key, episode: ep, cameras: null };   // claim before awaiting
            const body = await fetchEpisode(ds, ep);
            if (!loaded || loaded.key !== key) return;      // playhead moved on while fetching
            loaded.cameras = (body && body.cameras) || {};
        }
        if (!loaded.cameras) return;                        // fetch still in flight

        // Saved masks exist and the live layer is off: the tiles should show
        // the RECIPE's composite, served by the frame/video endpoints — chrome
        // fills on top of composited pixels would be double-painting, so the
        // canvases stay clear in this mode.
        const hasAny = Object.values(loaded.cameras).some(
            (d) => (d.frames || []).some((fr) => fr && fr.length));
        _setComposited(hasAny);
        // In composited mode the server has already painted the effect, so the
        // mask is drawn as an OUTLINE: the boundary is what tells you whether
        // the segmentation was right, and it is the one thing the composited
        // pixels cannot show you.
        const frameIdx = window.currentFrame || 0;
        for (const [maskKey, data] of Object.entries(loaded.cameras)) {
            const camKey = _camKeyFor(maskKey, window.datasets?.[window.currentDataset]?.camera_keys);
            const canvas = _canvas(camKey);
            if (!canvas) continue;
            const frame = data.frames && data.frames[frameIdx];
            const drawn = drawFrame(canvas, frame, data.size, {
                hidden: hiddenLabels, outline: hasAny && outlinesOnly, labels: data.labels || [],
            });
            canvas.style.display = drawn ? 'block' : 'none';
            if (drawn) _drewThisTick = true;
        }
        // Post-condition of the arbitration above: stored masks paint only
        // when the live layer does not own the tiles. Asserted here rather
        // than trusted, because the failure is silent -- two frames stacked
        // just look like a bad segmentation.
        if (_drewThisTick && _liveLayerActive()) {
            console.error('[masks] stored masks painted while the live layer owns the tiles');
            _clearAll();
            _drewThisTick = false;
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

    /** Does the live overlay own the tiles right now?
     *
     * Streaming is not the whole answer: the worker can be active with the
     * stream still starting, and in that window a still refresh both paints
     * the wrong frame and STOPS the stream (loadAllFrames treats any call as a
     * scrub). One definition, used by every decision that depends on it.
     */
    function _liveLayerActive() {
        if (window.OverlayStream && window.OverlayStream.streaming) return true;
        const badge = document.getElementById('overlays-badge');
        if (!badge) return false;
        const cls = badge.className || '';
        if (/\bloading\b/.test(cls)) return true;
        if (/\bok\b/.test(cls)) return true;
        return /\bidle\b/.test(cls) && !/^busy/.test(badge.textContent || '');
    }

    // ---- saved-view mode flag, read by app.js URL builders ----
    let compositedOn = false;
    function _setComposited(on) {
        on = !!on;
        if (on === compositedOn) return;
        compositedOn = on;
        // While the live preview owns the tiles, refreshing them here reloads
        // the stills underneath it and STOPS the stream (loadAllFrames treats
        // any call as a scrub) — which looked like the preview dying a second
        // after it started, since turning the overlay on is exactly what flips
        // this mode off. Checking only `streaming` left the window where the
        // worker is active and the stream is still starting.
        if (_liveLayerActive()) return;
        // The frame URLs change with the mode; refresh the tiles in place.
        if (typeof window.loadAllFrames === 'function' && window.currentDataset) {
            window.loadAllFrames(window.currentFrame || 0);
        }
        window.refreshVideoSources?.();
    }

    /** The requested camera's recipe fingerprint, '' until status arrives. */
    function compositedFingerprint(camKey) {
        const mk = _maskKeyFor(camKey);
        if (fpOverride.has(mk)) return fpOverride.get(mk);
        const cams = savedStatus && savedStatus.data && savedStatus.data.cameras;
        const e = cams && cams[mk];
        return (e && e.fingerprint) || '';
    }

    /** The saved recipe for the episode in hand, or null: what the panel's
     *  saved-effects controls initialize from. One recipe is returned (the
     *  first camera's); an effects apply then converges all cameras onto it. */
    function savedRecipe() {
        const d = savedStatus && savedStatus.data;
        if (!d || !d.adopted) return null;
        const keys = Object.keys(d.cameras || {});
        if (!keys.length) return null;
        const first = d.cameras[keys[0]];
        return {
            labels: first.labels || [],
            treatments: first.treatments || {},
            background: first.background || { key: 'none', params: {} },
            cameras: keys.map((k) => _camKeyFor(k, window.datasets?.[window.currentDataset]?.camera_keys)),
        };
    }

    /** The staged recipe changed: the server now composites playback from it,
     *  so drop the fingerprint overrides (they described the committed recipe)
     *  and re-pull. The cache-buster comes from the status endpoint, which
     *  reports the effective fingerprint. */
    function stagedTreatmentsChanged() {
        fpOverride.clear();
        savedStatus = null;
        statusCache.clear();
        if (typeof window.loadAllFrames === 'function' && window.currentDataset) {
            window.loadAllFrames(window.currentFrame || 0);
        }
        window.refreshVideoSources?.();
        onPlayheadChanged();
    }

    /** After a successful effects apply: adopt the new fingerprints (and the
     *  recipe the panel sent, so re-entering the panel shows what is saved)
     *  and refresh the tiles — the URL change re-pulls composited frames. */
    function applyEffectsResult(fingerprints, treatments, background) {
        const d = savedStatus && savedStatus.data;
        for (const [imgKey, fp] of Object.entries(fingerprints || {})) {
            const mk = _maskKeyFor(imgKey);
            fpOverride.set(mk, fp);        // wins until the episode changes
            const e = d && d.cameras && d.cameras[mk];
            if (!e) continue;
            e.fingerprint = fp;
            if (treatments) e.treatments = treatments;
            if (background) e.background = background;
        }
        // The status response is cached per episode; drop this episode's so a
        // later read reflects the write rather than the pre-write recipe.
        if (fingerprints && window.currentDataset != null && window.currentEpisode != null) {
            statusCache.delete(`${window.currentDataset}::${window.currentEpisode}`);
        }
        if (typeof window.loadAllFrames === 'function' && window.currentDataset) {
            window.loadAllFrames(window.currentFrame || 0);
        }
        window.refreshVideoSources?.();
        // The mask row names each lane's treatment, and it reads that from the
        // cached schema — which this apply just changed on disk.
        if (window.currentDataset) window.FeatureEditing?.refreshFromServer?.(window.currentDataset);
    }

    /** Draw stored masks as boundaries (true) or translucent fills (false). */
    function setOutlines(on) {
        outlinesOnly = !!on;
        onPlayheadChanged();
    }

    window.MaskOverlay = {
        // For the transport assertion: stored masks and the live layer must
        // never paint the same tiles at once.
        isDrawing: () => _drewThisTick,
        // Exposed so the arbitration rule can be tested directly: narrowing it
        // back to `streaming` is what let stills stop the stream.
        liveLayerActive: () => _liveLayerActive(),
        compositedActive: () => compositedOn,
        labelFontPx,
        freeSlot,
        setOutlines, outlinesActive: () => outlinesOnly,
        compositedFingerprint, savedRecipe, applyEffectsResult, stagedTreatmentsChanged,
        decodeCounts, decodeMask, drawFrame, entryEnabled, fetchEpisode, invalidate, PALETTE,
        maskVersion: () => _maskVersion,
        onPlayheadChanged, setEnabled, setLabelHidden, currentLabels,
        _maskKeyFor, _camKeyFor,
    };
})();
