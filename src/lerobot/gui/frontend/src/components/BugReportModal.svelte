<script lang="ts">
  // Bug-report modal, ported from src/lerobot/gui/static/bug_report.js.
  //
  // The reactive win, even at this small scale:
  //   - the legacy version has 4 functions that imperatively flip
  //     overlay.style.display, button disabled state, status text
  //   - here, all of that is derived from 3 $state slots (form, capturing,
  //     submitting, status) — the template just reads them
  //
  // Deep-link: visiting /#bug-report still auto-opens the dialog. Esc closes.

  import { postJson } from "../lib/api";
  import { toast } from "../lib/toast";
  import type { BugReportPayload, BugReportResponse } from "../lib/types";

  const HTML2CANVAS_CDN =
    "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";

  type Phase = "idle" | "capturing" | "submitting" | "error";

  let open = $state(false);
  let title = $state("");
  let description = $state("");
  let attachScreenshot = $state(true);
  let phase = $state<Phase>("idle");
  let errorMsg = $state("");

  const canSubmit = $derived(title.trim().length > 0 && phase === "idle");

  let html2canvasLoading: Promise<typeof html2canvas> | null = null;
  type Html2CanvasFn = (el: HTMLElement, opts?: object) => Promise<HTMLCanvasElement>;
  let html2canvas: Html2CanvasFn | null = null;

  function loadHtml2Canvas(): Promise<Html2CanvasFn> {
    if (html2canvas) return Promise.resolve(html2canvas);
    if (html2canvasLoading) return html2canvasLoading;
    html2canvasLoading = new Promise((resolve, reject) => {
      const w = window as unknown as { html2canvas?: Html2CanvasFn };
      if (typeof w.html2canvas === "function") {
        html2canvas = w.html2canvas;
        resolve(html2canvas);
        return;
      }
      const s = document.createElement("script");
      s.src = HTML2CANVAS_CDN;
      s.async = true;
      s.onload = () => {
        const wAfter = window as unknown as { html2canvas?: Html2CanvasFn };
        if (typeof wAfter.html2canvas === "function") {
          html2canvas = wAfter.html2canvas;
          resolve(html2canvas);
        } else {
          reject(new Error("html2canvas loaded but symbol missing"));
        }
      };
      s.onerror = () => reject(new Error("failed to load html2canvas from CDN"));
      document.head.appendChild(s);
    });
    return html2canvasLoading;
  }

  function activeTabName(): string | null {
    const active = document.querySelector(".tab.active");
    if (!active) return null;
    return active.getAttribute("data-tab") || active.textContent?.trim() || null;
  }

  function openDialog() {
    title = "";
    description = "";
    attachScreenshot = true;
    phase = "idle";
    errorMsg = "";
    open = true;
    // Wait a tick for the input to mount then focus it.
    queueMicrotask(() => {
      document.getElementById("bug-report-title-input")?.focus();
    });
  }

  function closeDialog() {
    open = false;
  }

  async function captureScreenshot(): Promise<string | null> {
    // Hide the modal before the capture so it doesn't show up in the
    // saved screenshot. The Svelte reactivity will close+reopen the
    // overlay element across the html2canvas call.
    open = false;
    // Let the DOM re-render before html2canvas snapshots it.
    await new Promise((r) => requestAnimationFrame(r));
    try {
      const fn = await loadHtml2Canvas();
      const canvas = await fn(document.body, {
        backgroundColor: "#16213e",
        useCORS: true,
        logging: false,
        scale: 1,
      });
      return canvas.toDataURL("image/png");
    } finally {
      open = true;
    }
  }

  async function submit() {
    if (!canSubmit) return;
    errorMsg = "";

    let screenshotDataUrl: string | null = null;
    if (attachScreenshot) {
      phase = "capturing";
      try {
        screenshotDataUrl = await captureScreenshot();
      } catch (e) {
        // Soft-fail: keep the typed report; flag the screenshot loss.
        errorMsg = `Screenshot failed (${e instanceof Error ? e.message : e}). Submitting without it...`;
      }
    }

    phase = "submitting";
    const payload: BugReportPayload = {
      title: title.trim(),
      description,
      url: window.location.href,
      user_agent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        dpr: window.devicePixelRatio || 1,
      },
      active_tab: activeTabName(),
      screenshot_data_url: screenshotDataUrl,
      client_extra: {},
    };

    try {
      const res = await postJson<BugReportResponse>("/api/bug_reports", payload);
      closeDialog();
      const shotSuffix = res.screenshot_saved ? " (with screenshot)" : "";
      toast("Bug report saved", `${res.directory}${shotSuffix}`, "success", 8000);
    } catch (e) {
      phase = "error";
      errorMsg = `Failed to submit: ${e instanceof Error ? e.message : e}`;
    } finally {
      // Reset to idle so subsequent submissions are enabled. The error
      // message persists until the user types again.
      if (phase === "submitting") phase = "idle";
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Escape" && open) closeDialog();
  }

  // Public API for the rest of the app: legacy code calls openBugReportDialog().
  // Mounting (in islands/bug-report.ts) wires this into window.
  export function externalOpen() {
    openDialog();
  }
  export function externalClose() {
    closeDialog();
  }

  $effect(() => {
    document.addEventListener("keydown", onKeydown);
    return () => document.removeEventListener("keydown", onKeydown);
  });

  // Deep-link: /#bug-report opens the dialog after mount.
  $effect(() => {
    if (window.location.hash === "#bug-report") {
      openDialog();
    }
  });
</script>

{#if open}
  <div
    class="bug-report-overlay"
    role="dialog"
    aria-modal="true"
    aria-labelledby="bug-report-heading"
  >
    <div class="bug-report-panel">
      <h3 id="bug-report-heading">Report a bug</h3>
      <p class="bug-report-hint">
        Stored on the GUI server under
        <code>~/.cache/lerobot/bug_reports/</code>. Not sent anywhere else.
      </p>

      <label for="bug-report-title-input">Title</label>
      <input
        id="bug-report-title-input"
        type="text"
        placeholder="Short summary of the issue"
        bind:value={title}
        disabled={phase !== "idle"}
      />

      <label for="bug-report-desc-input">What happened? (optional)</label>
      <textarea
        id="bug-report-desc-input"
        rows="5"
        placeholder="Steps to reproduce, what you expected, what you saw instead."
        bind:value={description}
        disabled={phase !== "idle"}
      ></textarea>

      <label class="checkbox-row">
        <input type="checkbox" bind:checked={attachScreenshot} disabled={phase !== "idle"} />
        Attach screenshot of the current view
      </label>

      <div class="bug-report-status" class:visible={errorMsg !== "" || phase !== "idle"}>
        {#if phase === "capturing"}
          Capturing screenshot...
        {:else if phase === "submitting"}
          Saving report...
        {:else if errorMsg}
          {errorMsg}
        {/if}
      </div>

      <div class="bug-report-buttons">
        <button class="btn-ghost" onclick={closeDialog} disabled={phase !== "idle"}>Cancel</button>
        <button class="btn-primary" onclick={submit} disabled={!canSubmit}>Submit</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .bug-report-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 2000;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fade-in 0.12s ease-out;
  }

  .bug-report-panel {
    background: var(--bg-secondary, #1e1e1e);
    border: 1px solid var(--border, #333);
    border-radius: 8px;
    padding: 20px;
    min-width: 420px;
    max-width: 560px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    animation: pop-in 0.18s cubic-bezier(0.2, 0.9, 0.3, 1.1);
  }

  h3 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--text-primary, #ccc);
  }

  .bug-report-hint {
    margin: 0 0 14px;
    font-size: 11px;
    color: var(--text-secondary, #999);
  }

  .bug-report-hint code {
    color: var(--text-primary, #ccc);
  }

  label {
    display: block;
    font-size: 11px;
    color: var(--text-secondary, #999);
    margin-bottom: 4px;
  }

  input[type="text"],
  textarea {
    width: 100%;
    padding: 6px 8px;
    background: var(--bg-input, #2d2d30);
    color: var(--text-primary, #ccc);
    border: 1px solid var(--border, #333);
    border-radius: 4px;
    font-size: 12px;
    box-sizing: border-box;
    font-family: inherit;
    margin-bottom: 10px;
    transition: border-color 0.12s;
  }

  textarea {
    resize: vertical;
  }

  input[type="text"]:focus,
  textarea:focus {
    outline: none;
    border-color: var(--accent, #0e639c);
    box-shadow: 0 0 0 1px var(--accent, #0e639c);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 6px;
    user-select: none;
    margin-bottom: 12px;
  }

  .checkbox-row input {
    margin: 0;
  }

  .bug-report-status {
    font-size: 11px;
    color: #e5c07b;
    margin-bottom: 12px;
    min-height: 16px;
    opacity: 0;
    transition: opacity 0.15s;
  }

  .bug-report-status.visible {
    opacity: 1;
  }

  .bug-report-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  .btn-ghost,
  .btn-primary {
    padding: 5px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    border: 1px solid transparent;
    transition: background 0.12s, border-color 0.12s, opacity 0.12s;
  }

  .btn-ghost {
    background: transparent;
    color: var(--text-primary, #ccc);
    border-color: var(--border, #333);
  }

  .btn-ghost:not(:disabled):hover {
    border-color: var(--accent, #0e639c);
  }

  .btn-primary {
    background: var(--accent, #0e639c);
    color: #fff;
  }

  .btn-primary:not(:disabled):hover {
    background: #1177b3;
  }

  .btn-ghost:disabled,
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @keyframes fade-in {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes pop-in {
    from {
      opacity: 0;
      transform: translateY(8px) scale(0.98);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }
</style>
