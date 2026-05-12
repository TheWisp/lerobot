<script lang="ts">
  import TeleopForm from "./TeleopForm.svelte";
  import ReplayForm from "./ReplayForm.svelte";
  import { ensureRobotProfilesLoaded, ensureTeleopProfilesLoaded } from "../stores/profiles.svelte";
  import { ensureDatasetsLoaded } from "../stores/datasets.svelte";
  import { form, validateForLaunch } from "../stores/run.svelte";
  import type { RunWorkflow } from "../lib/types";

  // Reactive sidebar for the Run tab. Replaces (when ?reactive=1 is set)
  // the legacy renderRunForm() output. The Launch / Stop buttons still
  // belong to the legacy DOM and call window.launchRun() / window.stopRun()
  // — we expose form snapshots to those globals.

  const launchError = $derived(validateForLaunch());

  function selectWorkflow(w: RunWorkflow) {
    form.workflow = w;
  }

  // The Critical-TODO win at the data-loader level: kick off loads from the
  // sidebar mount, without any UI flag gating. The stores dedupe across
  // tabs, so if the Robot tab is also visible no extra fetch happens.
  $effect(() => {
    ensureRobotProfilesLoaded().catch(() => {});
    ensureTeleopProfilesLoaded().catch(() => {});
    ensureDatasetsLoaded().catch(() => {});
  });
</script>

<div class="run-sidebar-reactive">
  <div class="workflow-selector">
    <button
      class="workflow-btn"
      class:active={form.workflow === "teleop"}
      onclick={() => selectWorkflow("teleop")}
    >Teleop</button>
    <button
      class="workflow-btn"
      class:active={form.workflow === "replay"}
      onclick={() => selectWorkflow("replay")}
    >Replay</button>
    <button
      class="workflow-btn"
      class:active={form.workflow === "policy"}
      onclick={() => selectWorkflow("policy")}
    >Policy</button>
  </div>

  <div class="form-body">
    {#if form.workflow === "teleop"}
      <TeleopForm />
    {:else if form.workflow === "replay"}
      <ReplayForm />
    {:else if form.workflow === "policy"}
      <div class="policy-placeholder">
        <strong>Policy view — not yet ported to Svelte.</strong>
        <p>
          The Policy workflow has HVLA, RLT, intervention-recording, and
          checkpoint discovery — it's the most complex form in the GUI and
          the natural Phase 2 of the migration. For now, switch back to the
          legacy sidebar to use it:
        </p>
        <a class="legacy-link" href="?reactive=0">Switch to legacy sidebar</a>
      </div>
    {/if}
  </div>

  <div class="validation-banner" class:ok={!launchError}>
    {#if launchError}
      <span class="validation-icon">!</span> {launchError}
    {:else}
      <span class="validation-icon ok">OK</span> Ready to launch
    {/if}
  </div>
</div>

<style>
  .run-sidebar-reactive {
    display: flex;
    flex-direction: column;
    padding: 12px;
    gap: 12px;
    color: var(--text-primary, #ccc);
    font-family:
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      Roboto,
      sans-serif;
  }

  .workflow-selector {
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border, #2a4a6f);
    padding-bottom: 8px;
  }

  .workflow-btn {
    flex: 1;
    padding: 6px 12px;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-secondary, #888);
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    font-weight: 500;
    transition:
      color 0.12s,
      border-color 0.12s,
      background 0.12s;
  }

  .workflow-btn:hover {
    color: var(--text-primary, #ccc);
    background: rgba(255, 255, 255, 0.04);
  }

  .workflow-btn.active {
    color: var(--accent, #4fc3f7);
    border-color: var(--accent, #4fc3f7);
    background: rgba(79, 195, 247, 0.08);
  }

  .form-body {
    min-height: 200px;
  }

  .policy-placeholder {
    padding: 16px;
    background: var(--bg-secondary, #1e1e1e);
    border: 1px dashed var(--border, #2a4a6f);
    border-radius: 6px;
    color: var(--text-secondary, #aaa);
    font-size: 12px;
    line-height: 1.5;
  }

  .policy-placeholder strong {
    color: var(--text-primary, #ccc);
    display: block;
    margin-bottom: 8px;
  }

  .policy-placeholder p {
    margin: 0 0 12px;
  }

  .legacy-link {
    display: inline-block;
    color: var(--accent, #4fc3f7);
    text-decoration: none;
    font-size: 12px;
    border: 1px solid var(--accent, #4fc3f7);
    padding: 4px 10px;
    border-radius: 4px;
    transition: background 0.12s;
  }

  .legacy-link:hover {
    background: rgba(79, 195, 247, 0.1);
  }

  .validation-banner {
    padding: 8px 10px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(243, 156, 18, 0.12);
    color: #f39c12;
    border: 1px solid rgba(243, 156, 18, 0.3);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .validation-banner.ok {
    background: rgba(39, 174, 96, 0.12);
    color: #27ae60;
    border-color: rgba(39, 174, 96, 0.3);
  }

  .validation-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: currentColor;
    color: var(--bg-primary, #1a1a2e);
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
  }
</style>
