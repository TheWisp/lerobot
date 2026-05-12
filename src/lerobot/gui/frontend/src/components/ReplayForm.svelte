<script lang="ts">
  import { robotProfiles } from "../stores/profiles.svelte";
  import { datasets } from "../stores/datasets.svelte";
  import { form } from "../stores/run.svelte";

  // Replay is the simplest workflow — pick a robot + an episode and play it
  // back. The legacy version still requires renderRunForm() to build it, plus
  // _onReplayEpisodeChange() handler. Here it's two binds and a {#each}.
  //
  // The episode list is keyed by "datasetId:episodeIdx" because the API takes
  // both. In the real port we'd fetch episodes per opened dataset; for the
  // prototype we show a sample list of the dataset IDs and the user picks an
  // episode by typing the index.

  // For the prototype we just list opened datasets. A real port would expand
  // each dataset into its episodes (50 entries per dataset is normal, so a
  // tree or filterable list is better UX than a flat <select>).
  const episodes = $derived(
    datasets.value.flatMap((d) =>
      // Without per-dataset episode counts loaded yet, expose just episode 0
      // as a placeholder. Real port will fetch /api/datasets/{id}/episodes.
      [{ key: `${d.id}:0`, label: `${d.repo_id || d.id} · ep 0` }],
    ),
  );
</script>

<div class="form-grid">
  <label for="replay-robot-sel">Robot <span class="req">*</span></label>
  <select id="replay-robot-sel" bind:value={form.replay.robot}>
    <option value="" disabled>Select a robot profile</option>
    {#each robotProfiles.value as p (p.name)}
      <option value={p.name}>{p.name}</option>
    {/each}
  </select>

  <label for="replay-episode-sel">Episode <span class="req">*</span></label>
  <select id="replay-episode-sel" bind:value={form.replay.episode}>
    <option value="" disabled>Select an episode</option>
    {#each episodes as ep (ep.key)}
      <option value={ep.key}>{ep.label}</option>
    {/each}
  </select>
</div>

<div class="hint">
  Replay loads the recorded actions from the dataset and plays them back on
  the robot. Use the dataset tab to scrub episodes first if you're not sure
  which to pick.
</div>

<style>
  .form-grid {
    display: grid;
    grid-template-columns: minmax(120px, max-content) 1fr;
    gap: 8px 12px;
    align-items: center;
    margin-bottom: 12px;
  }

  label {
    font-size: 12px;
    color: var(--text-secondary, #aaa);
  }

  .req {
    color: #e06c75;
    margin-left: 2px;
  }

  select {
    width: 100%;
    padding: 5px 8px;
    background: var(--bg-input, #2d2d30);
    color: var(--text-primary, #ccc);
    border: 1px solid var(--border, #333);
    border-radius: 4px;
    font-size: 12px;
    box-sizing: border-box;
  }

  select:focus {
    outline: none;
    border-color: var(--accent, #4fc3f7);
    box-shadow: 0 0 0 1px var(--accent, #4fc3f7);
  }

  .hint {
    font-size: 11px;
    color: var(--text-secondary, #888);
    line-height: 1.4;
    margin-top: 8px;
  }
</style>
