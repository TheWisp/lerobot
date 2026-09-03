// What must be true while the live overlay owns the video tiles.
//
// The tiles can be painted by two independent clocks: the app's playhead
// (stills fetched at currentFrame) and the live overlay's own MSE stream
// (server-composited frames arriving at whatever rate the model sustains).
// When both paint, you see two different frames stacked, and when the app
// believes it is paused while the stream runs, the transport button offers
// "Play" over moving video. Both were observed.
//
// A pure function so the rule is testable without a browser, and so the
// runtime check and the test cannot drift apart.

/**
 * @param {object} s
 *   streaming       - the live overlay's MSE stream is running
 *   isPlaying       - the app's transport state
 *   playBtnLabel    - what the transport button currently offers
 *   liveActive      - the live layer owns the tiles (worker active or streaming)
 *   savedMasksDrawn - the stored-mask canvases painted something this tick
 *   stillFetchInFlight - the app is fetching stills at its own playhead
 *   streamFrame     - frame index the stream is showing (null if unknown)
 *   playheadFrame   - the app's currentFrame
 * @returns {string[]} one message per violated invariant, empty when healthy.
 */
function transportViolations(s) {
  const v = [];
  const label = String(s.playBtnLabel || "");
  const offersPlay = /play/i.test(label) && !/pause/i.test(label);

  if (s.streaming && !s.isPlaying) {
    v.push("the live stream is running while the transport reports paused");
  }
  if (s.streaming && offersPlay) {
    v.push(`the live stream is running but the button offers "${label.trim()}"`);
  }
  if (s.liveActive && s.savedMasksDrawn) {
    v.push("stored masks are drawn while the live layer owns the tiles (two different truths on one image)");
  }
  if (s.streaming && s.stillFetchInFlight) {
    v.push("still frames are being fetched at the playhead while the stream paints the same tiles");
  }
  if (s.streaming && Number.isFinite(s.streamFrame) && Number.isFinite(s.playheadFrame)
      && Math.abs(s.streamFrame - s.playheadFrame) > 1) {
    v.push(`the playhead (${s.playheadFrame}) is not tracking the stream (${s.streamFrame})`);
  }
  return v;
}

if (typeof module !== "undefined" && module.exports) module.exports = { transportViolations };
if (typeof window !== "undefined") window.transportViolations = transportViolations;
