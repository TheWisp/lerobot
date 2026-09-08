// The transport: which engine advances the playhead, and what must hold while
// one does.
//
// Three engines can move the playhead: the still loop (a frame fetch per
// tick), the live overlay's composited stream (server-painted tiles arriving
// at whatever rate the model sustains), and an armed Apply run (lock-step with
// the worker). Exactly one runs while the transport plays. When two paint you
// see two different frames stacked; when the app believes it is paused while
// the stream runs, the button offers "Play" over moving video; and when Pause
// is routed to an engine that is not the one running, nothing stops. All three
// were observed.
//
// Pure functions so the rules are testable without a browser, and so the
// runtime checks and the tests cannot drift apart.

/**
 * What the transport button does next.
 *
 * @param {object} s
 *   isPlaying      - the app's transport state
 *   engine         - what is advancing the playhead: null | "still" | "stream" | "apply"
 *   applyArmed     - the Apply mode is armed (a run must drive playback itself)
 *   streamEligible - the live overlay can serve the composited stream
 * @returns {{op: "stop"|"start", engine: string|null}}
 *
 * Pause stops whatever is playing: the engine is read from the state, never
 * re-derived from the overlay badge, because the badge can change while a
 * still loop runs, and a Pause routed by it went to the stream module -- which
 * had nothing to stop and started a stream instead. Play picks the engine from
 * what is true now: an armed Apply run first (it must own the frame slot), then
 * the composited stream, then stills.
 */
function transportNext(s) {
  if (s.isPlaying) return { op: "stop", engine: s.engine === undefined ? null : s.engine };
  if (s.applyArmed) return { op: "start", engine: "apply" };
  if (s.streamEligible) return { op: "start", engine: "stream" };
  return { op: "start", engine: "still" };
}

/**
 * @param {object} s
 *   isPlaying       - the app's transport state
 *   engine          - null | "still" | "stream" | "apply" (omit to skip the engine checks)
 *   streaming       - the live overlay's MSE stream is running
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
  const offersPause = /pause/i.test(label);

  if (s.engine !== undefined) {
    const running = s.engine !== null && s.engine !== undefined;
    if (s.isPlaying && !running) v.push("the transport reports playing with no engine");
    if (!s.isPlaying && running) v.push(`the transport reports paused while the ${s.engine} engine runs`);
    if (s.streaming !== undefined) {
      if (s.streaming && s.engine !== "stream") {
        v.push(`the live stream is running while the engine is ${s.engine === null ? "none" : s.engine}`);
      }
      if (!s.streaming && s.engine === "stream") v.push("the engine is the stream but no stream is running");
    }
    if (label) {
      if (s.isPlaying && offersPlay) v.push(`the transport is playing but the button offers "${label.trim()}"`);
      if (!s.isPlaying && offersPause) v.push(`the transport is paused but the button offers "${label.trim()}"`);
    }
  }
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

if (typeof module !== "undefined" && module.exports) module.exports = { transportNext, transportViolations };
if (typeof window !== "undefined") {
  window.transportNext = transportNext;
  window.transportViolations = transportViolations;
}
