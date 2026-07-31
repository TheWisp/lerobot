// When leaving a tab, should the backend's camera preview handles be released?
//
// Isolated so it is unit-testable in node and shared with app.js. The bug it
// encodes against: leaving the Robot tab called stopCameraPreview(), which
// clears the polling interval and hides the Stop Preview button but never tells
// the backend to let go. The GUI then held a V4L2 / librealsense handle per
// camera with nothing on screen to say so, and a run launched from another tab
// fought it for the same /dev/video* devices.
//
// Releasing is deliberately conditional on something actually being held, so an
// ordinary tab switch does not fire a pointless POST on every click.
//
// Loaded as a plain <script> (exposes window.CameraRelease) and as a CommonJS
// module in the node test.
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) module.exports = factory();
    else root.CameraRelease = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';
    // Release iff we are actually leaving the robot tab.
    //
    // Deliberately does NOT consult frontend state such as
    // `detectedCameras.length`. That was the first attempt and it failed an
    // end-to-end check: cameras opened by any path other than the frontend's
    // own detectCameras() (a page reload, a direct API call, a desynced array)
    // left the frontend believing nothing was held while the backend held five
    // devices, so the release never fired. The backend knows what it has and
    // /api/robot/stop-cameras is idempotent — let it decide.
    //
    // Keying on the *previous* tab rather than only the next one keeps an
    // unrelated switch (data -> run) from POSTing.
    function shouldReleaseCameras(previousTab, nextTab) {
        return previousTab === 'robot' && nextTab !== 'robot';
    }
    return { shouldReleaseCameras: shouldReleaseCameras };
});
