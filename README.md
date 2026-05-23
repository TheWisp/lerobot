# Hub Progress Feature — Screenshots (2026-05-23)

Real-world test of `feat(gui/hub): live progress + cancel for HF upload/download` (commit `0c4d3d36e`).

Drove the GUI via Playwright + Chromium against `lerobot/pusht`. Both directions completed end-to-end; the throwaway upload target `thewisp/_hub_progress_smoke_test` was deleted after verification.

| # | File | What it shows |
|---|------|---------------|
| 1 | `01_download_modal_opened.png` | Download modal initial state — local + remote info, no transfer yet |
| 2 | `02_download_in_progress.png` | **Download mid-flight: 3 / 8 files, 664.4 KB / 7.3 MB, 38%** — progress bar visible |
| 3 | `03_download_complete.png` | Modal auto-closed, "Download Complete" toast |
| 4 | `04_upload_modal_opened.png` | Upload modal initial state, target = `thewisp/_hub_progress_smoke_test` |
| 5 | `05_upload_in_progress.png` | **Upload mid-flight: 1 / 8 files, 2.4 KB / 7.3 MB, 13%** — `README.md` in flight |
| 6 | `06_upload_complete.png` | Both toasts visible — Download + Upload complete |

Remote verification (post-upload):
```
exists: thewisp/_hub_progress_smoke_test   private: True
remote has 8 files, total bytes: 7686801  ← matches progress-bar final
```

Throwaway repo deleted after verification.
