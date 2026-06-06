// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// Training panel — minimal frontend for the training API.
// Lives in the Model tab's sidebar (Training section) + main pane (detail).
//
// What this drives:
// - GET /api/training/hosts          → host info display
// - GET /api/training/runs           → run list + active polling
// - POST /api/training/runs          → start a run
// - GET /api/training/runs/{id}      → detail (progress, checkpoints, log)
// - POST /api/training/runs/{id}/stop → user-initiated stop

const TRAINING_POLL_MS = 3000;
let _trainingHosts = [];
let _trainingPollTimer = null;
let _trainingSelectedRunId = null;

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  trainingInit();
});

async function trainingInit() {
  await trainingLoadHosts();
  await trainingRefreshRuns();
  trainingSchedulePoll();
}

function trainingSchedulePoll() {
  if (_trainingPollTimer) clearInterval(_trainingPollTimer);
  _trainingPollTimer = setInterval(trainingRefreshRuns, TRAINING_POLL_MS);
}

// ── Data loaders ──────────────────────────────────────────────────────────────

async function trainingLoadHosts() {
  try {
    const resp = await fetch("/api/training/hosts");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    _trainingHosts = await resp.json();
  } catch (e) {
    _trainingHosts = [];
    console.error("training: failed to load hosts", e);
  }
  trainingRenderHostsInfo();
}

async function trainingRefreshRuns() {
  let runs = [];
  try {
    const resp = await fetch("/api/training/runs");
    if (resp.ok) runs = await resp.json();
  } catch (e) {
    console.error("training: failed to load runs", e);
  }
  trainingRenderRunsList(runs);
  // If a run is selected, refresh its detail too
  if (_trainingSelectedRunId) {
    await trainingRefreshDetail(_trainingSelectedRunId);
  }
}

// ── Render helpers ────────────────────────────────────────────────────────────

function trainingRenderHostsInfo() {
  const el = document.getElementById("training-hosts-info");
  if (!el) return;
  if (_trainingHosts.length === 0) {
    el.textContent = "No training hosts detected. Workstation mode needs a GPU.";
    const btn = document.getElementById("training-start-btn");
    if (btn) btn.disabled = true;
    return;
  }
  const h = _trainingHosts[0];
  const gpu = h.capabilities?.gpu_name || "GPU";
  const vram = h.capabilities?.vram_mb ? ` (${(h.capabilities.vram_mb / 1024).toFixed(1)} GB)` : "";
  el.textContent = `Host: ${h.display_name} — ${gpu}${vram}`;
}

function trainingRenderRunsList(runs) {
  const el = document.getElementById("training-runs-list");
  if (!el) return;
  if (runs.length === 0) {
    el.innerHTML =
      '<div style="font-size: 11px; color: var(--text-secondary, #666); padding: 8px;">No runs yet. Click + to start.</div>';
    return;
  }
  el.innerHTML = "";
  for (const r of runs) {
    const row = document.createElement("div");
    row.className = "training-run-row";
    row.style.cssText =
      "padding: 6px 8px; cursor: pointer; border-bottom: 1px solid var(--border, #eee); font-size: 12px;";
    if (r.run_id === _trainingSelectedRunId) {
      row.style.background = "var(--selected-bg, #f0f4ff)";
    }
    row.innerHTML = `
      <div style="display: flex; justify-content: space-between; gap: 8px;">
        <span style="font-weight: 500;">${escapeHtml(r.recipe_name)}</span>
        <span class="training-state-badge training-state-${r.state}">${r.state}</span>
      </div>
      <div style="font-size: 11px; color: var(--text-secondary, #666); margin-top: 2px;">
        ${escapeHtml(r.dataset_id)}
      </div>
    `;
    row.onclick = () => trainingSelectRun(r.run_id);
    el.appendChild(row);
  }
}

function trainingSelectRun(runId) {
  _trainingSelectedRunId = runId;
  document.getElementById("model-empty").style.display = "none";
  document.getElementById("model-detail").style.display = "none";
  document.getElementById("training-detail").style.display = "block";
  trainingRefreshDetail(runId);
  // Re-render sidebar to update selection highlight
  trainingRefreshRuns();
}

async function trainingRefreshDetail(runId) {
  const el = document.getElementById("training-detail");
  if (!el) return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const snap = await resp.json();
    el.innerHTML = trainingRenderDetailHtml(snap);
    const stopBtn = document.getElementById(`training-stop-${runId}`);
    if (stopBtn) stopBtn.onclick = () => trainingStopRun(runId);
  } catch (e) {
    el.innerHTML = `<div style="color: var(--error, #d00);">Failed to load run: ${escapeHtml(e.message)}</div>`;
  }
}

function trainingRenderDetailHtml(snap) {
  const r = snap.run;
  const progress = snap.progress;
  const checkpoints = snap.checkpoints || [];
  const isActive = !["completed", "failed", "aborted"].includes(r.state);

  let progressLine = "<em>No progress yet…</em>";
  if (progress) {
    const step = progress.step ?? 0;
    const total = progress.num_steps ?? "?";
    const loss = progress.loss != null ? progress.loss.toFixed(4) : "—";
    progressLine = `Step ${step}/${total} · loss ${loss}`;
  }

  const elapsed = r.started_at
    ? Math.round((Date.now() / 1000 - r.started_at))
    : 0;

  const stopBtnHtml = isActive
    ? `<button class="btn-small" id="training-stop-${r.run_id}" style="background: var(--btn-danger-bg, #d00); color: white;">Stop</button>`
    : "";

  const checkpointsHtml = checkpoints.length === 0
    ? "<em>None yet</em>"
    : `<ul style="margin: 4px 0; padding-left: 20px;">${checkpoints
        .map((c) => `<li>step ${c.step} — <code>${escapeHtml(c.path)}</code> <span style="color: #888;">(sha256 ${c.sha256.slice(0, 12)}…)</span></li>`)
        .join("")}</ul>`;

  return `
    <div style="margin-bottom: 12px;">
      <h2 style="margin: 0 0 4px;">${escapeHtml(r.recipe_name)}</h2>
      <div style="color: var(--text-secondary, #666); font-size: 12px;">
        ${escapeHtml(r.dataset_id)} · host ${escapeHtml(r.host_id)} · run ${escapeHtml(r.run_id)}
      </div>
    </div>
    <div style="margin-bottom: 12px;">
      <span class="training-state-badge training-state-${r.state}">${r.state}</span>
      ${stopBtnHtml}
      <span style="margin-left: 12px; color: var(--text-secondary, #666); font-size: 12px;">
        ${elapsed > 0 ? `${elapsed}s elapsed` : ""}
      </span>
    </div>
    <div style="margin-bottom: 16px;">
      <strong>Progress:</strong> ${progressLine}
    </div>
    <div style="margin-bottom: 16px;">
      <strong>Checkpoints (${checkpoints.length}):</strong> ${checkpointsHtml}
    </div>
    <div>
      <strong>Log tail:</strong>
      <pre style="background: #111; color: #ddd; padding: 8px; font-size: 11px; max-height: 240px; overflow: auto; white-space: pre-wrap;">${escapeHtml(snap.stderr_tail || "(no output yet)")}</pre>
    </div>
    ${r.error ? `<div style="color: var(--error, #d00); margin-top: 8px;"><strong>Error:</strong> ${escapeHtml(r.error)}</div>` : ""}
  `;
}

// ── Start form ────────────────────────────────────────────────────────────────

function trainingShowStartForm() {
  if (_trainingHosts.length === 0) {
    alert("No training hosts detected. Workstation mode needs a GPU on this machine.");
    return;
  }
  document.getElementById("model-empty").style.display = "none";
  document.getElementById("model-detail").style.display = "none";
  const el = document.getElementById("training-detail");
  el.style.display = "block";

  const hostOptions = _trainingHosts
    .map((h) => `<option value="${escapeHtml(h.id)}">${escapeHtml(h.display_name)}</option>`)
    .join("");

  el.innerHTML = `
    <h2 style="margin: 0 0 12px;">Start a training run</h2>
    <form id="training-start-form" onsubmit="trainingSubmitStart(event); return false;">
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 4px;">Host</label>
        <select name="host_id" required>${hostOptions}</select>
      </div>
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 4px;">Recipe name</label>
        <input type="text" name="recipe_name" value="fake-prototype" required style="width: 100%; max-width: 400px;" />
      </div>
      <div style="margin-bottom: 10px;">
        <label style="display: block; margin-bottom: 4px;">Dataset id</label>
        <input type="text" name="dataset_id" value="example/dataset" required style="width: 100%; max-width: 400px;" />
      </div>
      <div style="margin-bottom: 10px; display: flex; gap: 16px; flex-wrap: wrap;">
        <label style="display: block;">
          Num steps
          <input type="number" name="num_steps" value="100" min="1" max="100000" style="width: 100px;" />
        </label>
        <label style="display: block;">
          Save every
          <input type="number" name="save_every" value="20" min="1" max="100000" style="width: 100px;" />
        </label>
        <label style="display: block;">
          Step seconds
          <input type="number" name="step_seconds" value="0.1" step="0.01" min="0" max="10" style="width: 100px;" />
        </label>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 16px;">
        <button type="submit" class="btn-small" style="background: var(--btn-primary-bg, #006); color: white; padding: 6px 16px;">Start</button>
        <button type="button" class="btn-small" onclick="trainingHideStartForm()">Cancel</button>
      </div>
      <div id="training-start-error" style="color: var(--error, #d00); margin-top: 12px;"></div>
    </form>
  `;
}

function trainingHideStartForm() {
  document.getElementById("training-detail").style.display = "none";
  document.getElementById("model-empty").style.display = "";
}

async function trainingSubmitStart(ev) {
  ev?.preventDefault?.();
  const form = document.getElementById("training-start-form");
  if (!form) return;
  const fd = new FormData(form);
  // Idempotency key: random per submit; deflects double-clicks
  const idempotencyKey = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const body = {
    host_id: fd.get("host_id"),
    recipe_name: fd.get("recipe_name"),
    dataset_id: fd.get("dataset_id"),
    args: {
      num_steps: parseInt(fd.get("num_steps") || "100", 10),
      save_every: parseInt(fd.get("save_every") || "20", 10),
      step_seconds: parseFloat(fd.get("step_seconds") || "0.1"),
    },
    idempotency_key: idempotencyKey,
  };
  const errEl = document.getElementById("training-start-error");
  errEl.textContent = "";
  // Disable submit while in flight
  const submitBtn = form.querySelector("button[type=submit]");
  if (submitBtn) submitBtn.disabled = true;
  try {
    const resp = await fetch("/api/training/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const run = await resp.json();
    // Show the new run's detail
    _trainingSelectedRunId = run.run_id;
    await trainingRefreshRuns();
    await trainingRefreshDetail(run.run_id);
  } catch (e) {
    errEl.textContent = e.message || String(e);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

// ── Stop ──────────────────────────────────────────────────────────────────────

async function trainingStopRun(runId) {
  if (!confirm("Stop this training run?")) return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}/stop`, { method: "POST" });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    // Optimistic refresh
    await trainingRefreshDetail(runId);
    await trainingRefreshRuns();
  } catch (e) {
    alert(`Stop failed: ${e.message || e}`);
  }
}

// ── Utilities ────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Expose for the inline onclick handlers
window.trainingInit = trainingInit;
window.trainingShowStartForm = trainingShowStartForm;
window.trainingHideStartForm = trainingHideStartForm;
window.trainingSubmitStart = trainingSubmitStart;
window.trainingStopRun = trainingStopRun;
