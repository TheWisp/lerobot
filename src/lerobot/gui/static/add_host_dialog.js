// Copyright 2026 The HuggingFace Inc. team. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// "Add SSH host" dialog — POSTs a HostProfile to /api/training/hosts.
//
// State machine:
//   (closed)
//     ↓ "+ Host"
//   open, empty, save disabled
//     ↓ user fills Name + Host, clicks "Test"
//   probing (test button → "Testing…", disabled)
//     ↓ probe returns
//   probed (4-row checklist rendered, save enabled iff probe.ok)
//     ↓ user clicks Save
//   saving → success → close + refresh sidebar
//
// Saving is only allowed after a green probe — same posture as VS Code
// Remote-SSH's "verified" badge. Editing any field after a probe clears
// the result and re-disables Save (otherwise the user could pass a probe
// for host A and then save host B).

let _addHostLastProbeOk = false;

function openAddHostDialog() {
    const overlay = document.getElementById('add-host-overlay');
    if (!overlay) return;
    document.getElementById('add-host-name').value = '';
    document.getElementById('add-host-host').value = '';
    document.getElementById('add-host-display-name').value = '';
    document.getElementById('add-host-probe-results').innerHTML = '';
    document.getElementById('add-host-status').textContent = '';
    document.getElementById('add-host-test-btn').disabled = false;
    document.getElementById('add-host-test-btn').textContent = 'Test';
    document.getElementById('add-host-save-btn').disabled = true;
    _addHostLastProbeOk = false;
    overlay.style.display = 'flex';
    document.getElementById('add-host-name').focus();
}

function closeAddHostDialog() {
    const overlay = document.getElementById('add-host-overlay');
    if (overlay) overlay.style.display = 'none';
}

function _addHostInvalidateProbe() {
    // Editing a field after a green probe shouldn't leave Save clickable.
    _addHostLastProbeOk = false;
    document.getElementById('add-host-save-btn').disabled = true;
    document.getElementById('add-host-probe-results').innerHTML = '';
    document.getElementById('add-host-status').textContent = '';
}

function _renderProbeResults(result) {
    const el = document.getElementById('add-host-probe-results');
    const rowsHtml = result.checks.map((c) => {
        const colour = c.ok ? '#7ec699' : '#e06c75';
        const mark = c.ok ? '✓' : '✗';
        return `
          <div class="probe-check ${c.ok ? 'ok' : 'fail'}" style="display:flex; gap:8px; align-items:flex-start; padding:3px 0; font-size:11px; font-family: ui-monospace, monospace;">
            <span style="color:${colour}; min-width:16px;">${mark}</span>
            <span style="color:var(--text-primary,#ccc); min-width:64px;">${escapeHtmlSafe(c.name)}</span>
            <span style="color:var(--text-secondary,#999); flex:1;">${escapeHtmlSafe(c.detail)}</span>
          </div>`;
    }).join('');
    el.innerHTML = `
      <div style="border:1px solid var(--border,#333); border-radius:4px; padding:8px 10px; background:var(--bg-input,#2d2d30);">
        ${rowsHtml}
        <div style="margin-top:6px; font-size:10px; color:var(--text-secondary,#777);">probe completed in ${result.latency_ms} ms</div>
      </div>`;
}

function escapeHtmlSafe(s) {
    if (typeof window.escapeHtml === 'function') return window.escapeHtml(s);
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function addHostTest() {
    const host = document.getElementById('add-host-host').value.trim();
    const status = document.getElementById('add-host-status');
    const testBtn = document.getElementById('add-host-test-btn');
    const saveBtn = document.getElementById('add-host-save-btn');
    if (!host) {
        status.textContent = 'Enter the Host first (alias or user@host).';
        return;
    }
    const originalLabel = testBtn.textContent;
    testBtn.textContent = 'Testing…';
    testBtn.disabled = true;
    saveBtn.disabled = true;
    status.textContent = '';
    document.getElementById('add-host-probe-results').innerHTML = '';
    try {
        const resp = await fetch('/api/training/hosts/probe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host }),
        });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        const result = await resp.json();
        _renderProbeResults(result);
        _addHostLastProbeOk = !!result.ok;
        saveBtn.disabled = !result.ok;
        if (!result.ok && result.message) {
            status.textContent = result.message;
        }
    } catch (e) {
        status.textContent = `Probe failed: ${e.message}`;
        _addHostLastProbeOk = false;
        saveBtn.disabled = true;
    } finally {
        testBtn.textContent = originalLabel;
        testBtn.disabled = false;
    }
}

async function addHostSubmit() {
    const name = document.getElementById('add-host-name').value.trim();
    const host = document.getElementById('add-host-host').value.trim();
    const displayName = document.getElementById('add-host-display-name').value.trim();
    const status = document.getElementById('add-host-status');
    const saveBtn = document.getElementById('add-host-save-btn');

    if (!name) { status.textContent = 'Name is required.'; return; }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
        status.textContent = 'Name must contain only letters, digits, _ and -.'; return;
    }
    if (!host) { status.textContent = 'Host is required.'; return; }
    if (!_addHostLastProbeOk) {
        status.textContent = 'Run Test first; Save is enabled after a green probe.'; return;
    }

    saveBtn.disabled = true;
    status.textContent = 'Saving…';
    try {
        const resp = await fetch('/api/training/hosts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                host,
                display_name: displayName || null,
            }),
        });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        closeAddHostDialog();
        if (typeof window.trainingLoadHosts === 'function') {
            await window.trainingLoadHosts();
        }
        if (typeof window.showToast === 'function') {
            window.showToast('Host added', name, 'success', 4000);
        }
    } catch (e) {
        status.textContent = `Save failed: ${e.message}`;
        saveBtn.disabled = false;
    }
}

async function trainingDeleteHost(hostId, displayName) {
    const label = displayName || hostId;
    if (!confirm(`Remove host "${label}"? Saved profile will be deleted.`)) return;
    try {
        const resp = await fetch(`/api/training/hosts/${encodeURIComponent(hostId)}`, { method: 'DELETE' });
        if (!resp.ok && resp.status !== 204) {
            const err = await resp.text();
            throw new Error(`server returned ${resp.status}: ${err.slice(0, 200)}`);
        }
        if (typeof window.trainingLoadHosts === 'function') {
            await window.trainingLoadHosts();
        }
        if (typeof window.showToast === 'function') {
            window.showToast('Host removed', label, 'success', 3000);
        }
    } catch (e) {
        if (typeof window.showToast === 'function') {
            window.showToast('Remove failed', e.message, 'error', 5000);
        } else {
            alert(`Remove failed: ${e.message}`);
        }
    }
}

// Invalidate the probe whenever any field changes so Save stays in sync.
['add-host-name', 'add-host-host', 'add-host-display-name'].forEach((id) => {
    document.addEventListener('DOMContentLoaded', () => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', _addHostInvalidateProbe);
    });
});

// Esc closes the dialog when it's open.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const overlay = document.getElementById('add-host-overlay');
    if (overlay && overlay.style.display === 'flex') closeAddHostDialog();
});

window.trainingShowAddHostDialog = openAddHostDialog;
window.closeAddHostDialog = closeAddHostDialog;
window.addHostTest = addHostTest;
window.addHostSubmit = addHostSubmit;
window.trainingDeleteHost = trainingDeleteHost;
