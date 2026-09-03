// Which of a run's masks are worth staging.
//
// The write rule is per (frame, label): a write may fill one only where that
// label is ABSENT there. A detected mask and a disabled mask are both left
// alone — the disabled one especially, since re-filling it would put back a
// detection the operator had rejected.
//
// This runs on the CLIENT, deliberately. The alternative — send every frame the
// run segmented and let the server drop what it must not touch — makes the
// request grow with the episode's existing coverage rather than with what the
// run produced, and moves the decision away from the side that already holds
// the coverage. The server enforces the rule again at save time, because rows
// can change between a run passing over a frame and the operator pressing Save.
(function (root) {
    "use strict";

    /** Is `label` absent at `frame`, per the two per-frame bitsets? */
    function isAbsent(enabled, disabled, labels, label, frame) {
        const i = labels.indexOf(label);
        if (i < 0) return true; // not in the vocabulary yet: a new label is absent everywhere
        const on = Number(enabled?.[frame] ?? 0);
        const off = Number(disabled?.[frame] ?? 0);
        // Bit i of either bitset means the frame carries the label already.
        return !(Math.floor(on / Math.pow(2, i)) % 2) && !(Math.floor(off / Math.pow(2, i)) % 2);
    }

    /**
     * Drop every (frame, label) the write rule would refuse.
     *
     * `frames` are what the server drained: {episode, frame, camera, rle}.
     * `coverage` is {camera: {labels, enabled, disabled}} as the client holds it.
     * Returns rows ready to stage, and counts of what was dropped and why, so a
     * run that stages nothing can say whether it found nothing or was filtered.
     */
    function rowsToStage(frames, coverage) {
        const rows = [];
        let filtered = 0;
        let unknownCamera = 0;
        for (const f of frames || []) {
            // A malformed entry is a filtering decision, not an exception. This
            // runs inside the run's drain, and anything thrown here used to
            // reject out of the run loop and leave it marked in progress -- after
            // which Play did nothing, silently.
            if (!f) {
                unknownCamera += 1;
                continue;
            }
            const cov = coverage?.[f.camera];
            if (!cov) {
                unknownCamera += 1;
                continue;
            }
            const keep = {};
            for (const [label, counts] of Object.entries(f.rle || {})) {
                if (isAbsent(cov.enabled, cov.disabled, cov.labels || [], label, f.frame)) {
                    keep[label] = counts;
                } else {
                    filtered += 1;
                }
            }
            // A frame whose labels were all taken is not sent at all; an empty
            // row would stage as "segmented, found nothing" and claim the frame.
            if (Object.keys(keep).length) {
                rows.push({ camera: f.camera, frame: f.frame, rle: keep });
            }
        }
        return { rows, filtered, unknownCamera };
    }

    const api = { isAbsent, rowsToStage };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    if (root) root.ApplyRunFilter = api;
})(typeof window !== "undefined" ? window : null);
