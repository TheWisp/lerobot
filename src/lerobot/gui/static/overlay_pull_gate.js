// The data-overlay PULL decision, isolated so it is unit-testable in node and shared
// with overlays.js. Root cause it fixes (measured by bisection): the page re-pulled +
// PNG-decoded each camera's overlay on EVERY frame change (~52/s at 2 cams) while the
// worker only produces ~8 new overlays/s — the wasted client decodes cost ~4 fps of
// worker throughput. Pull decisions are a pure function of the worker's overlay SEQ:
//   - an SSE event pulls iff its seq is NEW for that camera (inequality, not greater-
//     than: a worker respawn restarts seqs, and freshness must survive that);
//   - frame-change / status-poll ticks NEVER pull while SSE is connected, and act as a
//     rate-limited fallback (>= minTickMs per camera) when it is not.
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) module.exports = factory();
    else root.OverlayPullGate = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';
    function create(minTickMs) {
        var minMs = typeof minTickMs === 'number' ? minTickMs : 400;
        var lastSeq = {};   // cam -> last pulled overlay seq (SSE path)
        var lastTick = {};  // cam -> last fallback-pull wall time (ms)
        return {
            // SSE said camera `cam` has overlay seq `seq`. Pull iff it's new.
            onSse: function (cam, seq) {
                if (seq === undefined || seq === null) { lastSeq[cam] = null; return true; }
                if (lastSeq[cam] === seq) return false;
                lastSeq[cam] = seq;
                return true;
            },
            // A frame-change / status-poll tick. Only the SSE-down fallback pulls here.
            onTick: function (cam, sseConnected, nowMs) {
                if (sseConnected) return false;
                var now = typeof nowMs === 'number' ? nowMs : Date.now();
                if (lastTick[cam] !== undefined && now - lastTick[cam] < minMs) return false;
                lastTick[cam] = now;
                return true;
            },
            // Config/dataset change: forget history so the next event refreshes everything.
            reset: function () { lastSeq = {}; lastTick = {}; },
        };
    }

    // Completion-gated latest-wins image loading, per camera. Root cause it fixes
    // (reproduced under a 6.4 Mbit/s throttle): reassigning an <img> src ABORTS the
    // in-flight download, so when overlay pulls arrive faster than the link can carry
    // a composite PNG, NO load ever completes and the tile freezes on the last overlay
    // that finished — while the worker fps badge stays healthy. Remote browsers hit
    // this constantly in WYSIWYG composite mode (full-frame PNGs, 2 cams x ~10/s).
    // Contract: at most ONE in-flight load per camera; newer requests while busy
    // replace the pending slot (latest wins, intermediates dropped); completion
    // (load OR error) starts the pending one if any.
    function createLoader() {
        var inflight = {};  // cam -> true while a load is in flight
        var pending = {};   // cam -> newest url requested while busy
        return {
            // Want `url` shown for `cam`. Returns the url to assign now, or null to wait.
            request: function (cam, url) {
                if (inflight[cam]) { pending[cam] = url; return null; }
                inflight[cam] = true;
                return url;
            },
            // The current load finished (or failed). Returns the pending url to assign
            // next (staying in-flight), or null if idle.
            done: function (cam) {
                if (pending[cam] !== undefined) {
                    var url = pending[cam];
                    delete pending[cam];
                    return url;  // still in flight: the caller assigns it immediately
                }
                delete inflight[cam];
                return null;
            },
            reset: function () { inflight = {}; pending = {}; },
        };
    }
    return { create: create, createLoader: createLoader };
});
