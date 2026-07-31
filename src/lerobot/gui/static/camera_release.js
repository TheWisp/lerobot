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
    // Release iff we are leaving the robot tab AND previews are actually open.
    // `heldCount` is the number of cameras the frontend believes are detected
    // and previewing (robot.js `detectedCameras.length`).
    function shouldReleaseCameras(tabName, heldCount) {
        return tabName !== 'robot' && Number(heldCount) > 0;
    }
    return { shouldReleaseCameras: shouldReleaseCameras };
});
