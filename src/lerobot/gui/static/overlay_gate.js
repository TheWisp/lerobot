// The live-overlay DRAW decision, isolated so it is unit-testable in node and shared with
// overlays.js. The whole point of the started-race fix: this is a PURE function of the backend
// OverlayStateMachine's reported state — never a frontend flag that can desync during the
// worker's INACTIVE→LOADING→ACTIVE spawn. Loaded as a plain <script> (exposes window.OverlayGate)
// and as a CommonJS module in the node test.
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) module.exports = factory();
    else root.OverlayGate = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';
    // Draw iff a live model is selected, its objects (if any) are named, and the backend machine
    // is ACTIVE. Every other state (inactive / loading / stopping / error) → do not draw.
    function shouldDraw(mode, current, objectsReady, machineState) {
        return mode === 'live' && !!current && !!objectsReady && machineState === 'active';
    }
    return { shouldDraw: shouldDraw };
});
