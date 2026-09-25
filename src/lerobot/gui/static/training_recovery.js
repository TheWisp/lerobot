/* Backend-owned recovery: this page displays state and sends explicit choices. */
function trainingRecoveryFieldsHtml() {
  return `<details class="training-section">
    <summary>Automatic recovery</summary>
    <label class="training-field"><span>
      <input type="checkbox" name="auto_recovery_enabled" /> Resume after an unexpected exit
    </span></label>
    <div class="training-field-row">
      <label class="training-field"><span>Maximum automatic retries</span>
        <input type="number" name="auto_recovery_retries" min="1" max="10" step="1" value="3" />
      </label>
      <label class="training-field"><span>Wait before retry (seconds)</span>
        <input type="number" name="auto_recovery_delay" min="0" max="3600" step="1" value="60" />
      </label>
    </div>
    <p class="training-field-hint">Off by default. Local training only. The GUI backend must remain running;
      the browser may be closed. Recovery waits for checkpoint and crash-diagnostic writes to settle.
      Stop cancels automatic recovery. Retries are counted across the whole training task.</p>
  </details>`;
}

function trainingRecoveryCardHtml(snap) {
  const state = snap.auto_recovery;
  const run = snap.run;
  const host = _trainingHosts.find(h => h.id === run.host_id);
  if (!state && host?.transport_kind !== "subprocess") return "";
  const active = !TERMINAL_STATES.has(run.state);
  const canToggle = active || state?.enabled;
  const rows = (state?.incidents || []).map((incident, index) => {
    const sources = [incident.log_path, incident.events_path, ...(incident.crash_materials || [])]
      .filter(Boolean).map(path => `<div class="training-mono">${escapeHtml(path)}</div>`).join("");
    const skipped = (incident.skipped_checkpoints || []).map(item =>
      `<div>${escapeHtml(item.path)}: ${escapeHtml(item.reason)}</div>`).join("");
    return `<tr><td>${index + 1}</td><td>
      <button type="button" class="btn-small secondary" data-recovery-run="${escapeHtml(incident.run_id)}">${escapeHtml(incident.run_id)}</button>
      <div>${escapeHtml(new Date(incident.detected_at * 1000).toLocaleString())}</div>
      <div>${escapeHtml(incident.reason)}</div>
      ${incident.last_reported_progress?.step != null ? `<div>Last reported progress: ${escapeHtml(String(incident.last_reported_progress.step))}${incident.last_reported_progress.total_steps != null ? " / " + escapeHtml(String(incident.last_reported_progress.total_steps)) : ""}</div>` : ""}
      <div>Exit code: ${incident.exit_code == null ? "unavailable" : escapeHtml(String(incident.exit_code))}</div>
      ${incident.checkpoint_step != null ? `<div>Resume checkpoint: ${escapeHtml(incident.checkpoint_run_id)} / step ${incident.checkpoint_step}</div>` : ""}
      <details><summary>Logs and crash materials</summary>${sources}
        <p class="training-field-hint">Crash-material paths are best-effort candidates by timestamp; availability depends on system diagnostics.</p>
        ${skipped ? `<p>Checkpoints skipped after writes settled:</p>${skipped}` : ""}
      </details></td></tr>`;
  }).join("");
  return `<section class="training-card" id="training-recovery-card">
    <h3>Automatic recovery: ${state?.enabled ? "enabled" : "off"}</h3>
    <p>${escapeHtml(state?.message || (state?.enabled ? "Monitoring training" : "Unexpected exits require manual Resume."))}
      ${state ? ` · ${state.attempts} / ${state.max_retries} retries · ${escapeHtml(state.status)}` : ""}</p>
    <div class="training-card-actions">
      ${canToggle ? `<button type="button" class="btn-small secondary" id="training-recovery-toggle">${state?.enabled ? "Disable automatic recovery" : "Enable automatic recovery"}</button>` : ""}
      ${state?.enabled && ["waiting", "resuming"].includes(state.status) ? '<button type="button" class="btn-small danger" id="training-recovery-stop">Stop recovery</button>' : ""}
      ${state && state.active_run_id !== run.run_id ? `<button type="button" class="btn-small" data-recovery-run="${escapeHtml(state.active_run_id)}">View latest attempt</button>` : ""}
    </div>
    ${rows ? `<table class="training-args-table"><tbody>${rows}</tbody></table>` : ""}
  </section>`;
}

function trainingBindRecovery(snap) {
  const card = document.getElementById("training-recovery-card");
  if (!card) return;
  card.querySelectorAll("[data-recovery-run]").forEach(button => {
    button.onclick = () => trainingSelectRun(button.dataset.recoveryRun);
  });
  const stop = card.querySelector("#training-recovery-stop");
  if (stop) stop.onclick = () => trainingStopRun(snap.run.run_id);
  const toggle = card.querySelector("#training-recovery-toggle");
  if (toggle) toggle.onclick = async () => {
    toggle.disabled = true;
    try {
      const state = snap.auto_recovery;
      const response = await fetch(`/api/training/runs/${snap.run.run_id}/auto-recovery`, {
        method: "PUT", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({enabled: !state?.enabled, max_retries: state?.max_retries ?? 3,
          delay_seconds: state?.delay_seconds ?? 60}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      await trainingRefreshDetail(snap.run.run_id);
    } catch (error) {
      showToast("Recovery settings failed", error.message, "error");
      toggle.disabled = false;
    }
  };
}
