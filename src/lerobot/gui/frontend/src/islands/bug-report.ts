// Island entry point for the bug-report modal.
//
// Mounts a single <BugReportModal/> into #bug-report-mount and exposes
// window.openBugReportDialog() so the legacy "Report bug" button in
// index.html (and any future caller) can trigger it without knowing about
// Svelte.
//
// The legacy implementation lived in static/bug_report.js and used four
// global functions + a chunk of inline HTML; this single island replaces all
// of that.

import { mount } from "svelte";
import BugReportModal from "../components/BugReportModal.svelte";

interface BugReportExports {
  externalOpen: () => void;
  externalClose: () => void;
}

declare global {
  interface Window {
    openBugReportDialog?: () => void;
    closeBugReportDialog?: () => void;
  }
}

function init() {
  const target = document.getElementById("bug-report-mount");
  if (!target) {
    // The legacy index.html may not have the mount point yet during a
    // staged rollout. Create one at the end of body so the modal still
    // works. Logged at warn level so the gap is visible during the
    // migration but doesn't break anything.
    const fallback = document.createElement("div");
    fallback.id = "bug-report-mount";
    document.body.appendChild(fallback);
    // eslint-disable-next-line no-console
    console.warn("[bug-report] mount point missing in DOM, created fallback");
    mountInto(fallback);
    return;
  }
  mountInto(target);
}

function mountInto(target: HTMLElement) {
  const api = mount(BugReportModal, { target }) as unknown as BugReportExports;
  window.openBugReportDialog = () => api.externalOpen();
  window.closeBugReportDialog = () => api.externalClose();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}
