# Implementation screenshots v2 — PR #15 (2026-05-25)

Re-shoot of the state-machine screenshots after addressing review feedback:

- **Cropped to top-right** (640×420) since that's where all HF/Transfers UI lives — full-page captures buried the relevant region.
- **New button language**: Cancel / Retry / Discard are now explicit text labels (Cancel + Discard tinted red as destructive). Hide on a completed card stays icon-only (`✕`) and is **non-red** — it's pure UI dismissal, no remote-state impact.
- **Toast no longer overlaps the popover** in D/E: completion toasts are suppressed when the popover is open (the card already shows the new status; duplicating with a toast on the same screen edge was visual noise).

| File | What's shown |
|---|---|
| `A_idle.png` | No transfers in registry. Tab bar shows only HF auth + Report bug. The Transfers pill is hidden when nothing's ever been queued in this session. |
| `B_modal_ready.png` | Hub Upload modal (kept full-window — modal is centered, not top-right). Repo ID prefilled, "New repo will be created" hint, local-state row. No mode dropdown — single unified pipeline. |
| `C_active_inflight.png` | One active upload: pulsing `● Transfers · 1` pill, popover open showing card with "Uploading files" milestone, current file, progress bar, and `Cancel` (red text) action. |
| `D_complete_with_hide.png` | Completed upload card with green left border, "Done · 47.7 GB", and `✕` Hide (neutral, non-red — UI-only dismissal, nothing to clean up server-side). |
| `E_failed_retry_discard.png` | Failed upload (auth error class — full remediation message) above a cancelled upload. Both rows have `Retry` (neutral) + `Discard` (red text). Discard closes the draft PR on HF; Retry resumes into the same PR. |

States A/C/D/E are synthesized via a `window.fetch` stub on `/api/datasets/hub/jobs` — no real Hub, no real worker, no real dataset. State B mounts a synthetic dataset into the in-memory `datasets` map and invokes the production `openHubModal('upload')`. All cards render through the production `_cardHtml` code path; nothing about the visual treatment is staged separately.
