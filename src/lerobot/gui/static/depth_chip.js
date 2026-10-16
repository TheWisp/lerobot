// The depth tile's colour-range control: what the palette's two ends stand for.
//
// A depth tile is coloured by distance, near warm and far cool, and the only
// question the control answers is which two distances the ends of the palette
// are. "auto range" takes them from the episode's own 1st and 99th percentile;
// "fixed range" takes them from the user, on a two-thumb slider drawn over the
// palette itself with a number box at each end, so the legend and the control
// are one thing. Pure functions: the markup and the arithmetic live here so a
// node test can exercise them without a browser; app.js owns the storage and
// the requests.
(function () {
  'use strict';

  // The renderer's palette sampled from near (lo) to far (hi), see
  // lerobot/utils/depth_preview.py.
  const PALETTE = ['#7a0403', '#c52603', '#f05b12', '#fea732', '#e1dd37', '#a1fd3d', '#43f787', '#19d5cd', '#4099ff', '#4559cb', '#30123b'];
  const STEP = 0.01;

  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const fmt = (m) => (Number.isFinite(m) ? (+m).toFixed(2) : '');

  // The slider's reach: half again past the far end the dataset suggests,
  // rounded up to a decimetre, so both thumbs have room without wasting the
  // strip on distances nothing was recorded at; never less than the values
  // in hand, never under a metre.
  function span(hint, lo, hi) {
    const far = Math.max(hint && hint.length === 2 ? hint[1] * 1.5 : 0, lo || 0, hi || 0, 1.0);
    return Math.ceil(far * 10 - 1e-9) / 10;
  }

  // CSS for the strip: the palette stretched between lo and hi, flat beyond.
  function gradient(lo, hi, reach) {
    const at = (m) => Math.min(100, Math.max(0, (m / reach) * 100));
    const a = at(lo), b = at(hi);
    const stops = PALETTE.map((c, i) => `${c} ${(a + (b - a) * (i / (PALETTE.length - 1))).toFixed(1)}%`);
    return `linear-gradient(to right, ${PALETTE[0]} 0%, ${stops.join(', ')}, ${PALETTE[PALETTE.length - 1]} 100%)`;
  }

  // One edit applied to a state: thumbs and boxes may not cross, nothing goes
  // below zero. Returns the new state; the caller decides whether to keep it.
  function applyEdit(state, edge, value) {
    const v = Math.max(0, Math.round(parseFloat(value) / STEP) * STEP);
    if (!Number.isFinite(v)) return state;
    const s = { ...state };
    if (edge === 'lo') s.lo = Math.min(v, s.hi - STEP);
    else if (edge === 'hi') s.hi = Math.max(v, s.lo + STEP);
    else return state;
    s.lo = +s.lo.toFixed(2);
    s.hi = +s.hi.toFixed(2);
    return s;
  }

  // The control for one tile. `state` is {mode, lo, hi}; `auto` is the
  // episode's own range in metres when known, which Auto mode shows read-only.
  function render(cam, state, auto, hint) {
    const fixed = state.mode === 'fixed';
    const lo = fixed ? state.lo : auto ? auto.lo : state.lo;
    const hi = fixed ? state.hi : auto ? auto.hi : state.hi;
    const reach = span(hint, lo, hi);
    const opt = (m, label) => `<option value="${m}" ${state.mode === m ? 'selected' : ''}>${label}</option>`;
    const num = (edge, v) =>
      `<input type="number" class="depth-num" data-depth-edge="${edge}" step="${STEP}" min="0" value="${fmt(v)}" ` +
      `${fixed ? '' : 'readonly tabindex="-1"'} title="${edge === 'lo' ? 'near end' : 'far end'} of the palette, metres${fixed ? '' : ' (this episode, automatic)'}">`;
    const knob = (edge, v) =>
      `<input type="range" class="depth-knob" data-depth-knob="${edge}" min="0" max="${reach}" step="${STEP}" value="${fmt(v)}" ` +
      `aria-label="${edge === 'lo' ? 'near end' : 'far end'} of the palette, metres">`;
    return `
        <div class="camera-chip depth-chip ${fixed ? 'fixed' : 'auto'}" data-depth-cam="${esc(cam)}">
            <label class="depth-chip-mode" title="How distance becomes colour. Auto range: this episode's 1st to 99th percentile of depth. Fixed range: the two distances you set. The stored depth never changes.">colour
                <select class="depth-mode">${opt('auto', 'auto range')}${opt('fixed', 'fixed range')}</select>
            </label>
            <div class="depth-chip-range" title="${fixed ? 'Drag the thumbs or type the ends, in metres' : 'The range this episode is coloured with; pick fixed range to set your own'}">
                ${num('lo', lo)}
                <div class="depth-strip" style="background: ${gradient(lo, hi, reach)}">${fixed ? knob('lo', lo) + knob('hi', hi) : ''}</div>
                ${num('hi', hi)}<span class="depth-unit">m</span>
            </div>
        </div>`;
  }

  const api = { PALETTE, STEP, span, gradient, applyEdit, render };
  if (typeof window !== 'undefined') window.DepthChip = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
