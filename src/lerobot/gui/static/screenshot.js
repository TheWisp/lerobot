// Hidden screenshot helper. Exposes:
//
//   * Keyboard shortcut Ctrl+Shift+S (or Cmd+Shift+S on macOS)
//   * Global function `takeScreenshot()` you can call from DevTools console
//
// Both capture the current GUI page via html2canvas (lazy-loaded from CDN)
// and trigger a browser download of a timestamped PNG.
//
// Same limitation as the bug-report screenshot: cross-origin iframes (the
// MeshCat URDF tile served from :7000 inside the GUI at :8000) cannot be
// read by html2canvas under same-origin policy — they render as blank
// rectangles. For capturing iframe content, use scripts/gui/screenshot_gui.py
// (server-side ffmpeg x11grab) or screenshot the MeshCat URL directly.
//
// All internal helpers are wrapped in an IIFE so they don't collide with
// bug_report.js, which has its own copies of the same html2canvas loader
// pattern under the same names.

(() => {
    const HTML2CANVAS_CDN = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";

    let _h2cPromise = null;
    function _loadHtml2Canvas() {
        if (_h2cPromise) return _h2cPromise;
        _h2cPromise = new Promise((resolve, reject) => {
            if (typeof window.html2canvas === "function") {
                resolve(window.html2canvas);
                return;
            }
            const s = document.createElement("script");
            s.src = HTML2CANVAS_CDN;
            s.async = true;
            s.onload = () => {
                if (typeof window.html2canvas === "function") resolve(window.html2canvas);
                else reject(new Error("html2canvas loaded but symbol missing"));
            };
            s.onerror = () => reject(new Error("failed to load html2canvas from CDN"));
            document.head.appendChild(s);
        });
        return _h2cPromise;
    }

    function _timestampedFilename() {
        const d = new Date();
        const pad = (n) => String(n).padStart(2, "0");
        return `lerobot-gui-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.png`;
    }

    window.takeScreenshot = async function takeScreenshot() {
        try {
            if (typeof showToast === "function") {
                showToast("Screenshot", "Capturing GUI page...", "info");
            }
            const html2canvas = await _loadHtml2Canvas();
            const canvas = await html2canvas(document.body, {
                backgroundColor: "#16213e",
                useCORS: true,
                logging: false,
                scale: 1,
            });
            const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
            if (!blob) throw new Error("canvas.toBlob returned null");
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = _timestampedFilename();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            if (typeof showToast === "function") {
                showToast("Screenshot", `Saved ${a.download}`, "success");
            } else {
                console.log(`Screenshot saved: ${a.download}`);
            }
        } catch (err) {
            console.error("takeScreenshot failed:", err);
            if (typeof showToast === "function") {
                showToast("Screenshot", `Failed: ${err.message}`, "error");
            }
        }
    };

    // Keyboard shortcut: Ctrl+Shift+S (Cmd+Shift+S on macOS). We use Shift to
    // avoid clashing with Ctrl+S (browser's save-page). Don't intercept when
    // typing in editable fields so user input is preserved.
    window.addEventListener("keydown", (ev) => {
        const isMac = navigator.platform.toUpperCase().includes("MAC");
        const modOk = isMac ? ev.metaKey : ev.ctrlKey;
        if (!modOk || !ev.shiftKey) return;
        if ((ev.key || "").toLowerCase() !== "s") return;
        const t = ev.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
            return;
        }
        ev.preventDefault();
        window.takeScreenshot();
    });
})();
