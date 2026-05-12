<script lang="ts">
  import { robotProfiles, teleopProfiles } from "../stores/profiles.svelte";
  import { datasets } from "../stores/datasets.svelte";
  import {
    form,
    teleopShowRecordFields,
    teleopShowNewDatasetName,
  } from "../stores/run.svelte";

  // The classic record-fields show/hide is now a $derived expression. The
  // legacy code calls _toggleHvlaRecordFields() from 4 places (init, dataset
  // change, profile refresh, oninput); here, the template just reads
  // showRecordFields and Svelte handles the rest.
  const showRecord = $derived(teleopShowRecordFields());
  const showNewDataset = $derived(teleopShowNewDatasetName());

  // Same idea for the task selector — a $derived list rather than imperatively
  // rebuilt with innerHTML. The legacy app calls _getDatasetTasks() then
  // populates the <select> from JS; here the {#each} block does it.
  const taskOptions = $derived.by(() => {
    const dsId = form.teleop.recordDataset;
    if (!dsId || dsId === "__new__") {
      // Default suggestions when there's no dataset to crib from.
      return ["Pick up the cube", "Assemble the parts", "Place into bin"];
    }
    // In a fully-ported world we'd query the dataset's tasks here. The
    // prototype keeps the legacy default while the rest of the migration
    // lands — Phase 2 in the playbook covers the task fetch.
    return ["Pick up the cube"];
  });
</script>

<div class="form-grid">
  <label for="teleop-robot-sel">Robot <span class="req">*</span></label>
  <select id="teleop-robot-sel" bind:value={form.teleop.robot}>
    <option value="" disabled>Select a robot profile</option>
    {#each robotProfiles.value as p (p.name)}
      <option value={p.name}>{p.name}</option>
    {/each}
  </select>

  <label for="teleop-teleop-sel">Teleop <span class="req">*</span></label>
  <select id="teleop-teleop-sel" bind:value={form.teleop.teleop}>
    <option value="" disabled>Select a teleop profile</option>
    {#each teleopProfiles.value as p (p.name)}
      <option value={p.name}>{p.name}</option>
    {/each}
  </select>

  <label for="teleop-fps">FPS</label>
  <input
    id="teleop-fps"
    type="number"
    bind:value={form.teleop.fps}
    min="1"
    max="200"
  />
</div>

<div class="form-section">
  <div class="form-section-title">Record dataset</div>
  <div class="form-grid">
    <label for="teleop-record-dataset">Dataset</label>
    <select id="teleop-record-dataset" bind:value={form.teleop.recordDataset}>
      <option value="">None (don't record)</option>
      <option value="__new__">+ New dataset...</option>
      {#each datasets.value as d (d.id)}
        <option value={d.id}>{d.repo_id || d.id}</option>
      {/each}
    </select>
  </div>

  {#if showNewDataset}
    <div class="form-grid">
      <label for="teleop-new-dataset-name">Name <span class="req">*</span></label>
      <input
        id="teleop-new-dataset-name"
        type="text"
        bind:value={form.teleop.newDatasetName}
        placeholder="my_new_dataset"
      />
    </div>
  {/if}

  {#if showRecord}
    <div class="form-grid reveal">
      <label for="teleop-task">Task <span class="req">*</span></label>
      <select id="teleop-task" bind:value={form.teleop.task}>
        <option value="">— pick a task —</option>
        {#each taskOptions as t (t)}
          <option value={t}>{t}</option>
        {/each}
      </select>

      <label for="teleop-num-episodes">Episodes</label>
      <input
        id="teleop-num-episodes"
        type="number"
        bind:value={form.teleop.numEpisodes}
        min="1"
      />

      <label for="teleop-episode-time">Episode duration (s)</label>
      <input
        id="teleop-episode-time"
        type="number"
        bind:value={form.teleop.episodeTime}
        min="1"
      />

      <label for="teleop-reset-time">Reset duration (s)</label>
      <input
        id="teleop-reset-time"
        type="number"
        bind:value={form.teleop.resetTime}
        min="0"
      />
    </div>
  {/if}
</div>

<style>
  .form-grid {
    display: grid;
    grid-template-columns: minmax(120px, max-content) 1fr;
    gap: 8px 12px;
    align-items: center;
    margin-bottom: 12px;
  }

  .form-section {
    border-top: 1px solid var(--border, #2a4a6f);
    padding-top: 12px;
    margin-top: 8px;
  }

  .form-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary, #888);
    margin-bottom: 8px;
  }

  label {
    font-size: 12px;
    color: var(--text-secondary, #aaa);
  }

  .req {
    color: #e06c75;
    margin-left: 2px;
  }

  select,
  input {
    width: 100%;
    padding: 5px 8px;
    background: var(--bg-input, #2d2d30);
    color: var(--text-primary, #ccc);
    border: 1px solid var(--border, #333);
    border-radius: 4px;
    font-size: 12px;
    font-family: inherit;
    box-sizing: border-box;
    transition: border-color 0.12s;
  }

  select:focus,
  input:focus {
    outline: none;
    border-color: var(--accent, #4fc3f7);
    box-shadow: 0 0 0 1px var(--accent, #4fc3f7);
  }

  .reveal {
    animation: reveal-fade 0.18s ease-out;
  }

  @keyframes reveal-fade {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
