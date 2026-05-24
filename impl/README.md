# Implementation screenshots — PR #15 (2026-05-24)

End-to-end working implementation of the design at `gui/docs/hub_transfers.md`.

| File | What's shown |
|---|---|
| `A_idle.png` | No transfers — tab bar shows only HF auth indicator + Report bug. The Transfers pill is hidden when nothing's in the registry. |
| `B_modal_ready.png` | Hub Upload modal — repo_id input, "New repo will be created" hint, "In sync — no differences" status. No mode dropdown (single unified pipeline). |
| `C_active_inflight.png` | Real upload in flight: tray pill "● Transfers · 1" (cyan + pulsing), popover open showing the upload card with stage "Uploading files". |
| `D_complete_with_hide.png` | (Synthesized state) Completed upload card with green border, "Done · 47.8 GB", Hide button (✕). PR link in the repo column. |
| `E_failed_retry_discard.png` | (Synthesized state) Failed upload with auth error class — full message: "Authentication failed. Your HF token may be expired or lacks write permission. Run `huggingface-cli login` and click Retry." Plus a cancelled card below showing 47/124 files (38%) with Retry + Discard buttons. |

States A/B/C are real (driven through actual GUI + real worker + real HF for C). States D/E are JS-synthesized snapshots: the polling loop's fetch is stubbed so the synthetic job entries render through the production `_cardHtml` path. Easier than provoking real failures on demand; verifies the UX shape that the production code emits.
