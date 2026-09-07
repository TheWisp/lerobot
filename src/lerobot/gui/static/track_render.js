// How one feature becomes one row of the timeline.
//
// Six kinds of row are drawn through `renderTrackSvg`: a bitset's stack of flag
// lanes, a bool band, a mask column's three-state lanes, string stripes, a
// categorical band, and a numeric line. Splitting it out of feature_editing.js
// is not only for size -- it is the one part of that module that is a pure
// function of its arguments, so it is the part that can be pinned exactly.
// `tests/gui/track_render_golden.test.js` holds 28 renderings to the byte, and
// a diff there names the row type whose pixels moved.
//
// Depends on the lane geometry (so the bars agree with the hit test) and on the
// float bit helper (so a lane past bit 31 is not read off another lane).
// Loaded as a plain <script> (exposes window.TrackRender) and as a CommonJS
// module in the node tests.
(function (root, factory) {
    if (typeof module !== "undefined" && module.exports) {
        module.exports = factory(require("./timeline_lanes.js"), require("./bitset.js"));
    } else {
        root.TrackRender = factory(root.TimelineLanes, root.Bitset);
    }
})(typeof self !== "undefined" ? self : this, function (laneGeom, bitset) {
    "use strict";

    const bitIsSet = bitset.bitIsSet;

    const FLAG_COLORS = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6",
                         "#16a085", "#e15f9d", "#7f8c8d"];

    function renderTrackSvg(name, ft, series, length, mutedSeries) {
        if (!series || !series.length) return "";
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];

        // A bitset is not a magnitude: plotting the stored integer as a line
        // puts 3 above 2 and invites reading one flag as more than another.
        // Draw a lane per flag instead, filled where that flag is set --
        // which is also what makes the row usable for picking a range to edit.
        if (Array.isArray(ft.flags) && ft.flags.length && typeof series[0] === "number") {
            const count = ft.flags.length;
            const lanes = laneGeom.geometry(count);
            const segs = [];
            for (let bit = 0; bit < count; bit++) {
                const y = lanes.top(bit);
                const h = lanes.height;
                // A faint rail for every declared flag, drawn whether or not it
                // ever fires. Without it two filled bands are indistinguishable
                // from two-of-five, and a flag no frame carries would vanish
                // from the row entirely rather than reading as "none here".
                segs.push(
                    `<rect x="0%" y="${y.toFixed(2)}%" width="100%" height="${h.toFixed(2)}%" ` +
                    `fill="${flagColor(bit)}" opacity="0.13"/>`
                );
                for (let i = 0; i < series.length; i++) {
                    const v = typeof series[i] === "number" ? series[i] : 0;
                    if (!bitIsSet(v, bit)) continue;
                    const x = (i / length) * 100;
                    const w = (1 / length) * 100 + 0.05;  // overdraw to avoid seams
                    segs.push(
                        `<rect x="${x}%" y="${y.toFixed(2)}%" width="${w}%" ` +
                        `height="${h.toFixed(2)}%" fill="${flagColor(bit)}"/>`
                    );
                }
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${segs.join("")}</svg>`;
        }

        if (dtype === "bool" && (shape.length === 0 || (shape.length === 1 && shape[0] === 1))) {
            // band: green where true, light-grey where false.
            const segs = [];
            for (let i = 0; i < series.length; i++) {
                const x = (i / length) * 100;
                const w = (1 / length) * 100 + 0.05; // tiny overdraw to avoid gaps
                if (series[i] === true) {
                    segs.push(`<rect x="${x}%" y="20%" width="${w}%" height="60%" fill="#27ae60"/>`);
                }
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${segs.join("")}</svg>`;
        }

        // Stored masks: one thin lane per object, drawn in three states. The
        // value is the server's per-frame ENABLED bitset (bit i =
        // mask_labels[i]); the companion series carries the muted ones. Absent
        // is neither bit — see `_mask_disabled_bits` for why two series rather
        // than two bits per label.
        if (Array.isArray(ft.mask_labels) && ft.mask_labels.length && typeof series[0] === "number") {
            const names = ft.mask_labels;
            const n = names.length;
            const lanes = laneGeom.geometry(n);
            const rects = [];
            const laneNames = [];
            const muted = mutedSeries || [];
            for (let b = 0; b < n; b++) {
                const y = lanes.top(b);
                const color = maskLaneColor(b);
                laneNames.push(
                    `<div class="row-flag-name row-mask-name" style="top:${y}%; height:${lanes.height}%;">` +
                    `<i style="background:${color}"></i>${escapeHtml(names[b])}</div>`
                );
                // The faint rail is the lane even when the object is never
                // found — an object SAM never saw has to read as an empty
                // lane, not as a missing one.
                rects.push(
                    `<rect x="0%" y="${y}%" width="100%" height="${lanes.height}%" ` +
                    `fill="${color}" opacity="0.10"/>`
                );
                for (const seg of maskSegments(series, muted, b, series.length)) {
                    if (seg.state === "absent") continue;
                    const x = (seg.from / length) * 100;
                    const w = ((seg.to - seg.from) / length) * 100 + 0.05;
                    // FILLED means it reaches training; HOLLOW means stored but
                    // withheld. An outline, not a dimmer fill or a hatch: a lane
                    // is a few pixels tall, and at that size a texture or an
                    // opacity step is not a difference anyone can see -- which
                    // matters because the bar is also the control.
                    const detected = seg.state === "detected";
                    rects.push(
                        // `data-label` is the label NAME, matching the delete
                        // button this segment offers and every other data-label
                        // in this file. It carried the lane INDEX until the two
                        // were found to disagree, so anything reading one and
                        // writing the other silently addressed the wrong lane.
                        `<rect class="mask-seg" data-feature="${escapeHtml(name)}" ` +
                        `data-label="${escapeHtml(names[b])}" data-lane="${b}" ` +
                        `data-from="${seg.from}" data-to="${seg.to}" data-state="${seg.state}" ` +
                        `x="${x}%" y="${y}%" width="${w}%" height="${lanes.height}%" ` +
                        `fill="${detected ? color : "none"}" opacity="${detected ? 0.85 : 1}" ` +
                        `stroke="${detected ? "none" : color}" stroke-width="${detected ? 0 : 1.5}" ` +
                        `vector-effect="non-scaling-stroke"/>`
                    );
                }
            }
            return (
                `<svg class="mask-lanes" preserveAspectRatio="none" viewBox="0 0 100 100">` +
                `${rects.join("")}</svg>` +
                laneNames.join("")
            );
        }

        if (dtype === "string") {
            // Colored stripe — each unique string gets a color; render run-length segments.
            // The colored rectangles go in a stretched SVG (preserveAspectRatio="none")
            // so they fill the row exactly. Text flags go in HTML overlays — putting
            // them in the stretched SVG would non-uniformly scale the glyphs (the cause
            // of the "white stretched artifact" before the rewrite).
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085"];
            const colorMap = new Map();
            const rects = [];
            const labels = [];
            let i = 0;
            while (i < series.length) {
                const v = series[i];
                let j = i;
                while (j < series.length && series[j] === v) j++;
                if (!colorMap.has(v)) colorMap.set(v, colors[colorMap.size % colors.length]);
                const color = colorMap.get(v);
                const x = (i / length) * 100;
                const w = ((j - i) / length) * 100;
                rects.push(`<rect x="${x}%" y="10%" width="${w}%" height="80%" fill="${color}" opacity="0.7"/>`);
                if (j - i > 4) {
                    labels.push(
                        `<div class="row-string-label" ` +
                        `style="left:${x}%; width:${w}%;">` +
                        `${escapeHtml(String(v).slice(0, 24))}</div>`
                    );
                }
                i = j;
            }
            return (
                `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${rects.join("")}</svg>` +
                labels.join("")
            );
        }

        // Categorical (int + names): render as a colored band with the flag
        // for each segment, similar to strings but indexed via ft.names.
        const isScalar = (shape.length === 0 || (shape.length === 1 && shape[0] === 1));
        if (
            isScalar
            && dtype.startsWith("int")
            && Array.isArray(ft.names)
            && ft.names.length > 0
            && typeof series[0] === "number"
        ) {
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085"];
            const rects = [];
            const labels = [];
            let i = 0;
            while (i < series.length) {
                const v = series[i];
                let j = i;
                while (j < series.length && series[j] === v) j++;
                const idx = (typeof v === "number") ? Math.round(v) : -1;
                const label = (idx >= 0 && idx < ft.names.length) ? ft.names[idx] : `?(${v})`;
                const color = colors[((idx >= 0) ? idx : 0) % colors.length];
                const x = (i / length) * 100;
                const w = ((j - i) / length) * 100;
                rects.push(`<rect x="${x}%" y="10%" width="${w}%" height="80%" fill="${color}" opacity="0.7"/>`);
                if (j - i > 4) {
                    labels.push(
                        `<div class="row-string-label" style="left:${x}%; width:${w}%;">` +
                        `${escapeHtml(String(label).slice(0, 24))}</div>`
                    );
                }
                i = j;
            }
            return (
                `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${rects.join("")}</svg>` +
                labels.join("")
            );
        }

        // Numeric: scalar → line; vector → mini multi-line (up to MULTI_LINE_CAP);
        // very-large vectors fall back to L2-norm-per-frame.
        //
        // The cap was 8 originally — dropping a 14-DOF ALOHA action to a single
        // L2-norm line, surprising users (the row flag correctly says
        // float32[14] but the visualization shows one curve, looking like a
        // bug). 32 covers typical robot DOF (so-100 leader+follower=12, ALOHA
        // bimanual=14, humanoids ≤ 30) and keeps the SVG cheap.
        const MULTI_LINE_CAP = 32;
        const scalarSeries = (typeof series[0] === "number") ? series : null;
        if (scalarSeries) {
            return numericLineSvg(scalarSeries, length);
        }
        if (Array.isArray(series[0]) && series[0].length <= MULTI_LINE_CAP) {
            const dims = series[0].length;
            // 16-color palette; recycles on shape > 16. Palette tuned to be
            // distinguishable on a dark background and not collide with
            // common UI accent colors.
            const colors = [
                "#5b8def", "#d97757", "#4caf50", "#b58900",
                "#9b59b6", "#16a085", "#e74c3c", "#7f8c8d",
                "#3498db", "#e67e22", "#27ae60", "#f1c40f",
                "#8e44ad", "#1abc9c", "#c0392b", "#95a5a6",
            ];
            const lines = [];
            for (let d = 0; d < dims; d++) {
                const dim = series.map(row => row[d]);
                lines.push(numericLinePath(dim, length, colors[d % colors.length]));
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${lines.join("")}</svg>`;
        }
        if (Array.isArray(series[0])) {
            // Very large vector (> MULTI_LINE_CAP dims). A single L2-norm
            // line is dominated by whichever dims swing widest — on a 48-dim
            // bimanual state (16 pos + 16 vel + 16 torque) it looked like
            // "only the gripper is plotted". Draw one L2-norm line per
            // name-suffix group (.pos / .vel / .torque / ...) when the names
            // group cleanly, so each channel family stays visible.
            const names = ft.names || [];
            const groups = new Map();
            for (let d = 0; d < series[0].length; d++) {
                const n = names[d] || "";
                const dot = n.lastIndexOf(".");
                const suffix = dot >= 0 ? n.slice(dot + 1) : "";
                if (!groups.has(suffix)) groups.set(suffix, []);
                groups.get(suffix).push(d);
            }
            if (groups.size > 1 && groups.size <= 8) {
                const palette = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085", "#e74c3c", "#7f8c8d"];
                const lines = [];
                let gi = 0;
                for (const idxs of groups.values()) {
                    const norms = series.map(row => {
                        let s = 0;
                        for (const d of idxs) s += (typeof row[d] === "number" ? row[d] * row[d] : 0);
                        return Math.sqrt(s);
                    });
                    lines.push(numericLinePath(norms, length, palette[gi % palette.length]));
                    gi++;
                }
                return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${lines.join("")}</svg>`;
            }
            const norms = series.map(row => {
                let s = 0;
                for (const x of row) s += (typeof x === "number") ? x * x : 0;
                return Math.sqrt(s);
            });
            return numericLineSvg(norms, length);
        }
        return "";
    }

    function numericLinePath(values, length, color) {
        const finite = values.filter(v => typeof v === "number" && isFinite(v));
        if (!finite.length) return "";
        let lo = Math.min(...finite);
        let hi = Math.max(...finite);
        if (lo === hi) { lo -= 1; hi += 1; }
        const points = [];
        for (let i = 0; i < values.length; i++) {
            const v = (typeof values[i] === "number" && isFinite(values[i])) ? values[i] : (lo + hi) / 2;
            const x = (i / Math.max(1, length - 1)) * 100;
            const y = 100 - ((v - lo) / (hi - lo)) * 80 - 10; // 10% pad top/bottom
            points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
        }
        return `<polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    }

    function numericLineSvg(values, length) {
        return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${numericLinePath(values, length, "#5b8def")}</svg>`;
    }

    function maskSegments(enabled, disabled, bit, len) {
        // `bitIsSet`, not `>> bit & 1`: JavaScript's shift coerces to 32 bits
        // and takes the count mod 32, so lane 32 would read lane 0's bit --
        // while `applyPendingMaskEdits` deliberately merges edits past bit 31
        // in floats. The two disagreeing is a lane that draws from one label
        // and stages onto another. The stored contract allows 63 labels.
        return laneGeom.runs((i) => {
            if (bitIsSet(enabled[i] || 0, bit)) return "detected";
            if (bitIsSet(disabled[i] || 0, bit)) return "disabled";
            return "absent";
        }, len);
    }

    function flagSegments(series, bit, len) {
        return laneGeom.runs(
            (i) => bitIsSet(typeof series[i] === "number" ? series[i] : 0, bit),
            len,
        );
    }

    function maskLaneColor(bit) {
        // Reached for at call time rather than injected: masks.js may load
        // after this module, and a lane drawn before it arrives simply falls
        // back. Guarded for `window` so the module loads under node, where the
        // fallback is the only path anyway.
        const overlay = typeof window !== "undefined" ? window.MaskOverlay : null;
        const p = overlay && overlay.PALETTE;
        if (!p || !p.length) return flagColor(bit);
        const [r, g, b] = p[bit % p.length];
        return `rgb(${r}, ${g}, ${b})`;
    }

    function flagColor(bit) { return FLAG_COLORS[bit % FLAG_COLORS.length]; }


    function escapeHtml(s) {
        return String(s ?? "").replace(/[&<>"']/g, ch => (
            { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
        ));
    }
    return {
        renderTrackSvg,
        maskSegments,
        flagSegments,
        flagColor,
        maskLaneColor,
        escapeHtml,
        numericLinePath,
    };
});
