// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// Training panel — frontend for the training API.
// Lives in the Model tab (sidebar + main pane).
//
// What this drives:
// - GET /api/training/hosts                      → host info display
// - GET /api/training/runs                       → run list + active polling
// - POST /api/training/runs                      → start a run
// - GET /api/training/runs/{id}                  → detail (progress, checkpoints, log)
// - POST /api/training/runs/{id}/stop            → user-initiated stop
// - GET /api/datasets/sources + .../datasets     → populate the dataset dropdown

const TRAINING_POLL_MS = 3000;
let _trainingHosts = [];
let _trainingDatasets = [];
let _trainingPollTimer = null;

// View mode: which thing the main pane is showing.
//   "empty"  — nothing selected (default)
//   "form"   — start-training form open
//   "detail" — run detail for _trainingSelectedRunId
//
// Tracking mode explicitly fixes the "polling overwrites the start form"
// bug — refresh logic only redraws the view that matches the current mode.
let _trainingMode = "empty";
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

async function trainingLoadDatasets() {
  // Fan out: list sources → list each source's datasets → flatten.
  _trainingDatasets = [];
  try {
    const sources = await fetch("/api/datasets/sources").then((r) => r.json());
    const lists = await Promise.all(
      sources.map(async (s) => {
        try {
          const enc = encodeURIComponent(s.path);
          const ds = await fetch(`/api/datasets/sources/${enc}/datasets`).then((r) => r.json());
          return ds.map((d) => ({
            name: d.name,
            episodes: d.total_episodes,
            frames: d.total_frames,
            source: s.path,
          }));
        } catch {
          return [];
        }
      })
    );
    _trainingDatasets = lists.flat().sort((a, b) => a.name.localeCompare(b.name));
  } catch (e) {
    console.error("training: failed to load datasets", e);
  }
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
  // Only refresh the detail when we're currently SHOWING a detail. Don't
  // overwrite the start form with a re-render every 3 seconds.
  if (_trainingMode === "detail" && _trainingSelectedRunId) {
    await trainingRefreshDetail(_trainingSelectedRunId);
  }
}

// ── Sidebar render ────────────────────────────────────────────────────────────

function trainingRenderHostsInfo() {
  const el = document.getElementById("training-hosts-info");
  const btn = document.getElementById("training-start-btn");
  if (!el) return;
  if (_trainingHosts.length === 0) {
    el.textContent = "No training hosts detected.";
    if (btn) {
      btn.disabled = true;
      btn.title = "No training hosts detected (workstation mode needs a GPU on this machine)";
    }
    return;
  }
  const h = _trainingHosts[0];
  const gpu = h.capabilities?.gpu_name || "GPU";
  const vram = h.capabilities?.vram_mb ? ` · ${(h.capabilities.vram_mb / 1024).toFixed(1)} GB` : "";
  el.textContent = `${h.display_name} — ${gpu}${vram}`;
  if (btn) {
    btn.disabled = false;
    btn.title = "Start a new training run";
  }
}

function trainingRenderRunsList(runs) {
  const el = document.getElementById("training-runs-list");
  if (!el) return;
  if (runs.length === 0) {
    el.innerHTML = '<div class="training-empty-hint">No runs yet. Click + to start.</div>';
    return;
  }
  el.innerHTML = "";
  for (const r of runs) {
    const row = document.createElement("div");
    row.className = "training-run-row";
    if (r.run_id === _trainingSelectedRunId && _trainingMode === "detail") {
      row.classList.add("selected");
    }
    row.innerHTML = `
      <div class="training-run-row-top">
        <span class="training-run-name">${escapeHtml(r.recipe_name)}</span>
        <span class="training-state-badge training-state-${r.state}">${r.state}</span>
      </div>
      <div class="training-run-row-sub">${escapeHtml(r.dataset_id)}</div>
    `;
    row.onclick = () => trainingSelectRun(r.run_id);
    el.appendChild(row);
  }
}

// ── Mode switches ─────────────────────────────────────────────────────────────

function trainingShowMain(mode) {
  // Hide existing model views; show the training-detail container for any
  // mode other than "empty".
  document.getElementById("model-empty").style.display = mode === "empty" ? "" : "none";
  document.getElementById("model-detail").style.display = "none";
  document.getElementById("training-detail").style.display = mode === "empty" ? "none" : "block";
  _trainingMode = mode;
}

function trainingSelectRun(runId) {
  _trainingSelectedRunId = runId;
  trainingShowMain("detail");
  trainingRefreshDetail(runId);
  trainingRefreshRuns(); // re-render sidebar selection highlight
}

// ── Detail pane ───────────────────────────────────────────────────────────────

async function trainingRefreshDetail(runId) {
  const el = document.getElementById("training-detail");
  if (!el || _trainingMode !== "detail") return;
  try {
    const resp = await fetch(`/api/training/runs/${runId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const snap = await resp.json();
    el.innerHTML = trainingRenderDetailHtml(snap);
    const stopBtn = document.getElementById(`training-stop-${runId}`);
    if (stopBtn) stopBtn.onclick = () => trainingStopRun(runId);
  } catch (e) {
    el.innerHTML = `<div class="training-error">Failed to load run: ${escapeHtml(e.message)}</div>`;
  }
}

function trainingRenderDetailHtml(snap) {
  const r = snap.run;
  const progress = snap.progress || {};
  const checkpoints = snap.checkpoints || [];
  const isActive = !["completed", "failed", "aborted"].includes(r.state);

  const step = progress.step ?? 0;
  const total = progress.num_steps ?? r.args?.num_steps ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0;
  const loss = progress.loss != null ? progress.loss.toFixed(4) : "—";
  const elapsedSec = r.started_at ? Math.round(Date.now() / 1000 - r.started_at) : 0;

  const stopBtn = isActive
    ? `<button class="btn-small danger" id="training-stop-${r.run_id}">Stop</button>`
    : "";

  return `
    <div class="training-detail-pane">
      <header class="training-detail-header">
        <div>
          <h2 class="training-detail-title">${escapeHtml(r.recipe_name)}</h2>
          <div class="training-detail-meta">
            <span>${escapeHtml(r.dataset_id)}</span>
            <span class="sep">·</span>
            <span>host ${escapeHtml(r.host_id)}</span>
            <span class="sep">·</span>
            <span class="training-mono">${escapeHtml(r.run_id)}</span>
          </div>
        </div>
        <div class="training-detail-actions">
          <span class="training-state-badge training-state-${r.state}">${r.state}</span>
          ${stopBtn}
        </div>
      </header>

      <section class="training-card">
        <div class="training-stats-row">
          <div class="training-stat"><div class="training-stat-label">Step</div><div class="training-stat-value">${step} / ${total}</div></div>
          <div class="training-stat"><div class="training-stat-label">Loss</div><div class="training-stat-value">${loss}</div></div>
          <div class="training-stat"><div class="training-stat-label">Elapsed</div><div class="training-stat-value">${elapsedSec}s</div></div>
          <div class="training-stat"><div class="training-stat-label">Checkpoints</div><div class="training-stat-value">${checkpoints.length}</div></div>
        </div>
        <div class="training-progress-bar"><div class="training-progress-fill" style="width: ${pct}%"></div></div>
      </section>

      <section class="training-card">
        <h3 class="training-card-heading">Checkpoints</h3>
        ${
          checkpoints.length === 0
            ? '<div class="training-empty-hint">None yet</div>'
            : `<ul class="training-checkpoint-list">${checkpoints
                .map(
                  (c) => `<li>
                <span class="training-mono">step ${c.step}</span>
                <span class="training-mono training-muted">${escapeHtml(c.path)}</span>
                <span class="training-mono training-muted">${c.sha256.slice(0, 12)}…</span>
              </li>`
                )
                .join("")}</ul>`
        }
      </section>

      <section class="training-card">
        <h3 class="training-card-heading">Log tail</h3>
        <pre class="training-log">${escapeHtml(snap.stderr_tail || "(no output yet)")}</pre>
      </section>

      ${r.error ? `<div class="training-error">Error: ${escapeHtml(r.error)}</div>` : ""}
    </div>
  `;
}

// ── Start form ────────────────────────────────────────────────────────────────

async function trainingShowStartForm() {
  // Button is disabled until hosts load, so this guard mostly catches
  // programmatic invocations. Keep it cheap and silent — no alert.
  if (_trainingHosts.length === 0) return;
  // Clear selected run so polling doesn't try to refresh detail over the form.
  _trainingSelectedRunId = null;
  trainingShowMain("form");
  // Load datasets lazily on first form-open
  if (_trainingDatasets.length === 0) {
    document.getElementById("training-detail").innerHTML =
      '<div class="training-detail-pane"><div class="training-empty-hint">Loading datasets…</div></div>';
    await trainingLoadDatasets();
  }
  if (_trainingMode === "form") trainingRenderStartForm(); // user might have nav'd away
  trainingRefreshRuns(); // unselect any previously-highlighted row
}

function trainingRenderStartForm() {
  const el = document.getElementById("training-detail");
  if (!el || _trainingMode !== "form") return;

  const hostOptions = _trainingHosts
    .map((h) => `<option value="${escapeHtml(h.id)}">${escapeHtml(h.display_name)}</option>`)
    .join("");

  const datasetOptions =
    _trainingDatasets.length === 0
      ? `<option value="" disabled>No datasets found — add a source in the Data tab first</option>`
      : _trainingDatasets
          .map(
            (d) =>
              `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)} (${d.episodes} ep · ${d.frames} fr)</option>`
          )
          .join("");

  el.innerHTML = `
    <div class="training-detail-pane">
      <header class="training-detail-header">
        <h2 class="training-detail-title">Start a training run</h2>
        <div class="training-detail-actions">
          <button type="button" class="btn-small secondary" onclick="trainingCancelForm()">Cancel</button>
        </div>
      </header>

      <form id="training-start-form" class="training-form" onsubmit="trainingSubmitStart(event); return false;">
        <label class="training-field">
          <span class="training-field-label">Host</span>
          <select name="host_id" required>${hostOptions}</select>
        </label>

        <label class="training-field">
          <span class="training-field-label">Dataset</span>
          <select name="dataset_id" required ${_trainingDatasets.length === 0 ? "disabled" : ""}>
            ${datasetOptions}
          </select>
          <span class="training-field-hint">Datasets are discovered from sources configured in the Data tab.</span>
        </label>

        <label class="training-field">
          <span class="training-field-label">Recipe name</span>
          <input type="text" name="recipe_name" value="prototype" required />
          <span class="training-field-hint">Free-form label used in the run list + Models tab.</span>
        </label>

        <div class="training-field-row">
          <label class="training-field">
            <span class="training-field-label">Num steps</span>
            <input type="number" name="num_steps" value="100" min="1" max="100000" required />
          </label>
          <label class="training-field">
            <span class="training-field-label">Save every (steps)</span>
            <input type="number" name="save_every" value="20" min="1" max="100000" required />
          </label>
          <label class="training-field">
            <span class="training-field-label">Step seconds (fake-train pacing)</span>
            <input type="number" name="step_seconds" value="0.1" step="0.01" min="0" max="10" required />
          </label>
        </div>

        <div class="training-form-actions">
          <button type="submit" class="btn-small">Start training</button>
          <button type="button" class="btn-small secondary" onclick="trainingCancelForm()">Cancel</button>
        </div>
        <div id="training-start-error" class="training-error" style="display:none;"></div>
      </form>
    </div>
  `;
}

function trainingCancelForm() {
  trainingShowMain("empty");
  trainingRefreshRuns();
}

async function trainingSubmitStart(ev) {
  ev?.preventDefault?.();
  const form = document.getElementById("training-start-form");
  if (!form) return;
  const fd = new FormData(form);
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
  errEl.style.display = "none";
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
    // Switch to detail view of the new run
    trainingSelectRun(run.run_id);
  } catch (e) {
    errEl.style.display = "block";
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
window.trainingCancelForm = trainingCancelForm;
window.trainingSubmitStart = trainingSubmitStart;
window.trainingStopRun = trainingStopRun;
