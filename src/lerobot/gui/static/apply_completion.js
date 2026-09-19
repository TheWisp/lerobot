// What is owed when an "Apply — write masks while playing" run stops.
//
// Extracted because the list is not obvious and one entry was missing for the
// life of the feature: a run may APPEND a label to the dataset's vocabulary,
// and the schema the client holds was read when the dataset was opened. Without
// a refresh the new label has no lane, no Inspector row and no entry in the
// fill-gaps dialog for the rest of the session — the masks are on disk while
// the UI shows the run as having done nothing, which reads as a failed run.
//
// A CANCELLED run can have appended too: the writer flushes what it computed
// before the stop. So every terminal status owes the same effects; only
// `running` owes nothing.
(function (root) {
    "use strict";

    const TERMINAL = ["complete", "failed", "cancelled"];

    /** Pre: `status` is a job status string or undefined. */
    function isTerminal(status) {
        return TERMINAL.indexOf(status) !== -1;
    }

    /**
     * Run the effects a stopped apply job owes, and name the ones that ran.
     *
     * Pre: `deps` supplies the collaborators; each is optional so a caller
     * missing one degrades rather than throwing mid-way through the others.
     * Post: returns the effect names in the order applied, or null when the
     * job has not stopped — so "nothing happened" is distinguishable from
     * "everything happened".
     */
    function applyTerminalEffects(job, deps) {
        if (!job || !isTerminal(job.status)) return null;
        const d = deps || {};
        const ran = [];
        const step = (name, fn) => {
            if (typeof fn !== "function") return;
            fn();
            ran.push(name);
        };
        step("uncheck", d.uncheck);
        step("invalidateMasks", d.invalidateMasks);
        // The one that was missing. After the masks, before the status line:
        // the status is what the user reads last.
        step("refreshSchema", d.refreshSchema);
        step("status", d.setStatus && (() => d.setStatus(`Apply ${job.status}`)));
        return ran;
    }

    const api = { isTerminal, applyTerminalEffects, TERMINAL };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    if (root) root.ApplyCompletion = api;
})(typeof window !== "undefined" ? window : null);
