// The treatment control: a segmented row of exclusive buttons, one per effect,
// whose `tint` button doubles as the colour swatch and opens a picker.
//
// It has two owners with different write semantics, which is why only the
// RENDERING lives here and each caller wires its own handlers:
//
//   * the Inspector's dataset tier, where a treatment is dataset metadata and a
//     change is STAGED and then saved;
//   * the Run tab's live panel, where a treatment is a rendering knob for the
//     preview and a change is pushed to the worker. Nothing is written there --
//     that panel has no save at all -- so the dataset-scope argument that took
//     this control out of the DATA panel never applied to it.
//
// Sharing the rendering is the point: the two used to be one control, and the
// only reason they diverged is that one copy was deleted.
(function (root, factory) {
    if (typeof module === "object" && module.exports) module.exports = factory();
    else root.TreatmentControl = factory();
})(typeof self !== "undefined" ? self : this, function () {
    const TINT_PRESETS = [
        [239, 68, 68], [34, 197, 94], [59, 130, 246], [234, 179, 8],
        [168, 85, 247], [20, 184, 166], [255, 255, 255], [15, 23, 42],
    ];
    const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
    const toHex = (c) => "#" + c.map((x) => Math.max(0, Math.min(255, x | 0)).toString(16).padStart(2, "0")).join("");
    const fromHex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

    // ∅ none · colour square = tint (click to pick) · dice = random · drop = blur.
    // The selected button is filled, so which one is active is unambiguous
    // whatever the glyphs read as.
    const TREAT_SVG = {
        none: '<svg viewBox="0 0 16 16" class="ti"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.4"/><line x1="4.2" y1="11.8" x2="11.8" y2="4.2" stroke="currentColor" stroke-width="1.4"/></svg>',
        random: '<svg viewBox="0 0 16 16" class="ti"><rect x="2.3" y="2.3" width="11.4" height="11.4" rx="2.6" fill="none" stroke="currentColor" stroke-width="1.3"/><g fill="currentColor"><circle cx="5.6" cy="5.6" r="1.05"/><circle cx="10.4" cy="5.6" r="1.05"/><circle cx="8" cy="8" r="1.05"/><circle cx="5.6" cy="10.4" r="1.05"/><circle cx="10.4" cy="10.4" r="1.05"/></g></svg>',
        blur: '<svg viewBox="0 0 16 16" class="ti"><path d="M8 1.6 C 8 1.6 3.4 7.6 3.4 10 a 4.6 4.6 0 1 0 9.2 0 C 12.6 7.6 8 1.6 8 1.6 Z" fill="currentColor"/></svg>',
    };

    function icon(key, tr) {
        if (key !== "tint") return TREAT_SVG[key] || esc((key || "?")[0]);
        const c = (tr && tr.params && tr.params.color) || TINT_PRESETS[2];
        return `<span class="ds-tint-chip" style="background:${rgbCss(c)}"></span>`;
    }

    /** The control's HTML. `attrs` identifies the region for delegated handlers. */
    function widget(tr, keys, attrs) {
        const cur = (tr && tr.key) || "none";
        const btns = (keys || []).map(
            (k) => `<button class="ds-treat-btn${k === cur ? " sel" : ""}" type="button" data-key="${esc(k)}"` +
                   ` title="${esc(k)}" aria-label="${esc(k)}">${icon(k, tr)}</button>`
        ).join("");
        return `<span class="ds-treat" ${attrs || ""}>${btns}</span>`;
    }

    /** The swatch popover, created once at body level so a re-render cannot kill it. */
    function makePopover() {
        let el = null;
        function ensure() {
            if (el) return el;
            el = document.createElement("div");
            el.className = "ds-tint-pop";
            el.style.display = "none";
            document.body.appendChild(el);
            document.addEventListener("click", (e) => {
                if (el.style.display === "none") return;
                const onTint = e.target.closest && e.target.closest('.ds-treat-btn[data-key="tint"]');
                if (!el.contains(e.target) && !onTint) el.style.display = "none";
            });
            return el;
        }
        const paint = (rgb) => el.querySelectorAll(".ds-tint-sw")
            .forEach((sw) => sw.classList.toggle("sel", sw.dataset.rgb === rgb.join(",")));
        return {
            close() { if (el) el.style.display = "none"; },
            /** `onPick(rgb)` fires per choice; the caller decides what that means. */
            open(anchor, current, onPick) {
                const box = ensure();
                const cur = current || TINT_PRESETS[2];
                box.innerHTML =
                    `<div class="ds-tint-sws">` +
                    TINT_PRESETS.map((c) => `<span class="ds-tint-sw${c.join(",") === cur.join(",") ? " sel" : ""}" data-rgb="${c.join(",")}" style="background:${rgbCss(c)}"></span>`).join("") +
                    `</div><label class="ds-tint-custom-row">Custom <input type="color" class="ds-tint-custom" value="${toHex(cur)}"></label>`;
                box.querySelectorAll(".ds-tint-sw").forEach((sw) => sw.addEventListener("click", () => {
                    const rgb = sw.dataset.rgb.split(",").map(Number);
                    onPick(rgb); paint(rgb);
                    box.querySelector(".ds-tint-custom").value = toHex(rgb);
                }));
                // `input` fires continuously while the native picker is dragged.
                box.querySelector(".ds-tint-custom").addEventListener("input", (e) => {
                    const rgb = fromHex(e.target.value); onPick(rgb); paint(rgb);
                });
                box.style.display = "block";
                const r = anchor.getBoundingClientRect();
                box.style.left = Math.max(6, Math.min(r.left, window.innerWidth - box.offsetWidth - 8)) + "px";
                box.style.top = (r.bottom + 4) + "px";
            },
        };
    }

    return { TINT_PRESETS, rgbCss, toHex, fromHex, icon, widget, makePopover };
});
