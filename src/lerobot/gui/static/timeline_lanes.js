// Lane geometry and the lane-click gesture for the timeline rows that stack
// bars: mask lanes and flag lanes.
//
// Extracted because the two rows had been kept in step by hand. A lane's band
// is written once here and read by the drawing, the legend, the pending
// overlay and the hit test — and the hit test agreeing with the drawing is not
// a nicety, it is the difference between editing the bar you clicked and the
// one below it. Nine sites in feature_editing.js recomputed `80 / count` and
// `10 + i * laneH` independently.
//
// Everything here is a function of its arguments or of the DOM node handed to
// it: no dataset, no schema, no edit vocabulary. What a run MEANS — a mask's
// three states, a flag's two — stays with the feature that owns it.
//
// Loaded as a plain <script> (exposes window.TimelineLanes) and as a CommonJS
// module in the node tests.
(function (root, factory) {
    if (typeof module !== "undefined" && module.exports) module.exports = factory();
    else root.TimelineLanes = factory();
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    // Lanes share the row's middle 80%, leaving a 10% margin top and bottom.
    // Each bar fills 80% of its lane, so two adjacent lanes read as two bars
    // rather than one block.
    const MARGIN_PCT = 10;
    const SPAN_PCT = 100 - 2 * MARGIN_PCT;
    const FILL = 0.8;

    /**
     * Where the lanes of a `count`-lane row sit, in row percentages.
     *
     * `top(i)` and `height` place a bar; `mid(i)` is the middle of the BAR,
     * which is where a control pinned to a lane belongs — the middle of the
     * lane is up to 20% of the pitch away from the pixels the user aimed at.
     *
     * Precondition: `count >= 1`.
     */
    function geometry(count) {
        const pitch = SPAN_PCT / count;
        return {
            count,
            pitch,
            height: pitch * FILL,
            top: (i) => MARGIN_PCT + i * pitch,
            mid: (i) => MARGIN_PCT + i * pitch + (pitch * FILL) / 2,
        };
    }

    /**
     * The lane and frame under a pointer, or null when the pointer is in the
     * row's padding, past the last lane, or the row has no extent yet.
     *
     * The inverse of `geometry`, and the reason both live here.
     */
    function hit(track, count, length, ev) {
        const rect = track.getBoundingClientRect();
        if (!rect.width || !rect.height || !length || !count) return null;
        const frame = Math.min(
            length - 1,
            Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * length)),
        );
        const yPct = ((ev.clientY - rect.top) / rect.height) * 100;
        const lane = Math.floor((yPct - MARGIN_PCT) / (SPAN_PCT / count));
        if (yPct < MARGIN_PCT || lane < 0 || lane >= count) return null;
        return { lane, frame };
    }

    /**
     * Maximal runs of one state over `[0, len)`, as `{from, to, state}`.
     *
     * The unit every lane edit acts on: within a run the state is constant by
     * construction, so a click never has to resolve a mixed range and the
     * direction of a toggle is decided by what was clicked. `stateAt(i)` may
     * return anything comparable with `===`.
     *
     * Postcondition: the runs tile `[0, len)` exactly — no gap, no overlap,
     * and no two adjacent runs of the same state.
     */
    function runs(stateAt, len) {
        const out = [];
        let i = 0;
        while (i < len) {
            const s = stateAt(i);
            // From i + 1: frame i's state is `s` by construction, and stateAt
            // is not free -- re-testing it costs one extra call per run, which
            // on a heavily segmented lane approaches one per two frames.
            let j = i + 1;
            while (j < len && stateAt(j) === s) j++;
            out.push({ from: i, to: j, state: s });
            i = j;
        }
        return out;
    }

    /**
     * The run under `frame`, clipped to `sel`, or null.
     *
     * Clipping is what makes the scope positional: the click acts on what you
     * selected AND what you pointed at, never on the whole run that happens to
     * extend past the selection's edge.
     *
     * Null when the pointer is outside the selection — not merely when the run
     * is. Those are different tests: a run spanning 0..40 still overlaps a
     * selection of 0..10 with the pointer at frame 30, which is how clicks far
     * outside the range were editing it.
     *
     * A one-frame selection is as good as any other. It was refused once,
     * because the press was claimed at mousedown: the row's own handler made a
     * one-frame selection and the same gesture could then commit a toggle on
     * it, so every seek would have become an edit. Deferring the decision
     * split those into two gestures -- the first click only selects, and a
     * second click on a frame that is visibly selected, with the band showing
     * what it will do, is a deliberate act. The Inspector's own controls never
     * imposed a minimum either.
     */
    function runUnderPointer(all, frame, sel) {
        if (!sel || frame < sel.frameFrom || frame >= sel.frameTo) return null;
        const run = all.find((s) => frame >= s.from && frame < s.to);
        if (!run) return null;
        const from = Math.max(run.from, sel.frameFrom);
        const to = Math.min(run.to, sel.frameTo);
        if (from >= to) return null;
        return { ...run, from, to };
    }

    // ── The press whose meaning is not yet known ────────────────────────
    //
    // A press inside an existing selection, on a bar, is genuinely ambiguous:
    // it may be the start of a new selection, or a click on the run under it.
    // Nothing about the mousedown says which, and the pointer has not moved
    // yet -- so the honest thing is to decide nothing.
    //
    // The earlier design decided immediately and undid the loser: the row
    // sought a frame and collapsed the selection to one on press, and a release
    // without travel then put the old selection back. That works, and it looks
    // wrong -- the range visibly disappears under the cursor and springs back
    // on release. It also means every click pays for a seek and two renders it
    // did not need.
    //
    // Here the decision waits for the event that makes it: the first movement
    // past DRAG_SLOP is a drag, and a release before that is a click. Exactly
    // one of the two callbacks runs, once, so nothing is ever done and undone.
    //
    // The listeners live on the document because the pointer routinely leaves
    // the row mid-drag, and they are removed by whichever outcome fires first.

    // How far the pointer may travel before a press stops being a click. Below
    // this, a release is a click even if the hand wobbled; above it, the
    // gesture is a drag even if the button comes up immediately after.
    const DRAG_SLOP = 4;

    let pending = null;

    function resolvePending(outcome, ev) {
        const p = pending;
        if (!p) return;
        pending = null;
        document.removeEventListener("mousemove", p.onMove, true);
        document.removeEventListener("mouseup", p.onUp, true);
        document.removeEventListener("mousedown", p.onOtherDown, true);
        window.removeEventListener("blur", p.onAbort);
        if (outcome === "drag") p.onDrag(ev);
        else if (outcome === "click") p.onClick(ev);
    }

    /**
     * Hold a press until it declares itself, then run exactly one of
     * `onDrag(moveEvent)` or `onClick(upEvent)`.
     *
     * Preconditions: called from a mousedown handler for the left button.
     * Postcondition: at most one callback runs, and it runs at the moment the
     * gesture becomes unambiguous -- never speculatively, so neither outcome
     * has to undo the other's work.
     *
     * A press abandoned without a release -- the pointer leaves the window,
     * the tab loses focus -- resolves to neither and is dropped.
     */
    function deferPress(ev, { onDrag, onClick }) {
        resolvePending(null);
        const x = ev.clientX;
        const y = ev.clientY;
        const p = {
            onDrag,
            onClick,
            onMove: (m) => {
                if (Math.abs(m.clientX - x) <= DRAG_SLOP && Math.abs(m.clientY - y) <= DRAG_SLOP) return;
                resolvePending("drag", m);
            },
            // Only the button that made the press may end it: a right-click
            // while the left is held is not this gesture's release.
            onUp: (u) => { if (u.button === 0) resolvePending("click", u); },
            // Another button going down abandons the press rather than leaving
            // it pending. Neither movement nor a release has happened, so the
            // gesture has still not declared itself -- but a context menu
            // between press and release makes the eventual release a different
            // act from the one that was started, and committing a toggle after
            // it would be a surprise.
            onOtherDown: (d) => { if (d.button !== 0) resolvePending(null); },
            onAbort: () => resolvePending(null),
        };
        pending = p;
        document.addEventListener("mousemove", p.onMove, true);
        document.addEventListener("mouseup", p.onUp, true);
        document.addEventListener("mousedown", p.onOtherDown, true);
        window.addEventListener("blur", p.onAbort);
    }

    /** Whether a press is currently being held undecided. For assertions. */
    function pressIsPending() {
        return pending !== null;
    }

    /**
     * Abandon a press without resolving it either way.
     *
     * For the callers that know the press has been overtaken -- Escape, an
     * episode or dataset change -- and would otherwise let it commit against a
     * world that no longer exists. Safe to call when nothing is pending.
     */
    function cancelPress() {
        resolvePending(null);
    }

    return { geometry, hit, runs, runUnderPointer, deferPress, pressIsPending, cancelPress, DRAG_SLOP };
});
