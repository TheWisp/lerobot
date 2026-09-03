// A stopped "Apply — write masks while playing" run owes four effects, and one
// of them was missing for the life of the feature.
//
// A run can APPEND a label to the dataset's vocabulary. The client's copy of the
// schema is read when the dataset is opened and nothing else updates it, so
// without a refresh the new label had no timeline lane, no Inspector row and no
// entry in the fill-gaps dialog for the rest of the session. Measured before the
// fix: 20 s after a run that added "green cube", the server reported four labels
// and every client-side view still showed three.
//
// The effects are asserted as a SET rather than by re-listing the calls the
// caller makes, and `refreshSchema` is asserted separately by name, so deleting
// it cannot be absorbed by the others still passing.
const assert = require("assert");
const { isTerminal, applyTerminalEffects, TERMINAL } =
    require("../../src/lerobot/gui/static/apply_completion.js");

// ── which statuses stop a run ───────────────────────────────────────────────
for (const s of ["complete", "failed", "cancelled"]) {
    assert.strictEqual(isTerminal(s), true, `${s} should stop the run`);
}
// The case that must NOT fire, or every assertion below is vacuous: a poll sees
// far more running ticks than terminal ones, and refreshing the dataset schema
// on each would re-fetch it every second for the length of the job.
for (const s of ["running", "pending", "starting", undefined, ""]) {
    assert.strictEqual(isTerminal(s), false, `${s} is not terminal`);
}

const spy = () => {
    const calls = [];
    const rec = (n) => (...a) => calls.push([n, ...a]);
    return {
        calls,
        deps: {
            uncheck: rec("uncheck"),
            invalidateMasks: rec("invalidateMasks"),
            refreshSchema: rec("refreshSchema"),
            setStatus: rec("setStatus"),
        },
    };
};

// ── a stopped run refreshes the schema ─────────────────────────────────────
// Cancelled too: the writer flushes what it computed before the stop, so a
// cancelled run can have appended a label just as a complete one can.
for (const status of TERMINAL) {
    const { calls, deps } = spy();
    const ran = applyTerminalEffects({ status }, deps);
    const called = calls.map((c) => c[0]);
    assert.ok(
        called.includes("refreshSchema"),
        `${status}: the schema was not refreshed, so a label this run added stays invisible`,
    );
    assert.deepStrictEqual(
        new Set(called),
        new Set(["uncheck", "invalidateMasks", "refreshSchema", "setStatus"]),
        `${status}: wrong collaborators called — ${JSON.stringify(called)}`,
    );
    // The report must describe what really ran, or a caller cannot tell a
    // no-op poll tick from a completed one.
    assert.strictEqual(ran.length, called.length, `${status}: report and calls disagree`);
    assert.ok(ran.includes("refreshSchema"), `${status}: refresh missing from the report`);
    // The status line names the outcome, so "cancelled" cannot read as "complete".
    const status_call = calls.find((c) => c[0] === "setStatus");
    assert.ok(String(status_call[1]).includes(status), `status line lost the outcome: ${status_call[1]}`);
}

// ── a running job owes nothing ─────────────────────────────────────────────
{
    const { calls, deps } = spy();
    assert.strictEqual(applyTerminalEffects({ status: "running", frames_done: 7 }, deps), null);
    assert.deepStrictEqual(calls, [], "a running job triggered effects");
    assert.strictEqual(applyTerminalEffects(null, deps), null, "no job is not a stopped job");
    assert.deepStrictEqual(calls, [], "a missing job triggered effects");
}

// ── a caller missing one collaborator still runs the rest ──────────────────
// The poll has no error path: an exception here would abandon the remaining
// effects and leave the checkbox ticked with no run behind it.
{
    const calls = [];
    const ran = applyTerminalEffects(
        { status: "complete" },
        { refreshSchema: () => calls.push("refreshSchema") },
    );
    assert.deepStrictEqual(calls, ["refreshSchema"]);
    assert.deepStrictEqual(ran, ["refreshSchema"], "absent collaborators must not be reported as run");
    assert.strictEqual(ran.length, 1, "only the supplied collaborator ran");
}

console.log("apply_completion.test.js: all assertions passed");
