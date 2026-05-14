// Preact + signals port of components/ReplayForm.svelte.

import { computed } from "@preact/signals";
import type { JSX } from "preact";
import { robotProfiles } from "../stores-preact/profiles";
import { datasets } from "../stores-preact/datasets";
import { form, updateReplay } from "../stores-preact/run";

export function ReplayForm(): JSX.Element {
  // Placeholder episode list: 1 entry per opened dataset until per-dataset
  // episodes are fetched in Phase 2. Same shape as the Svelte version so
  // the A/B is fair.
  const episodes = computed(() =>
    datasets.value.value.flatMap((d) => [
      { key: `${d.id}:0`, label: `${d.repo_id || d.id} · ep 0` },
    ]),
  );

  const r = form.value.replay;

  return (
    <>
      <div class="form-grid">
        <label for="replay-robot-sel">
          Robot <span class="req">*</span>
        </label>
        <select
          id="replay-robot-sel"
          value={r.robot}
          onChange={(e) =>
            updateReplay({ robot: (e.target as HTMLSelectElement).value })
          }
        >
          <option value="" disabled>
            Select a robot profile
          </option>
          {robotProfiles.value.value.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>

        <label for="replay-episode-sel">
          Episode <span class="req">*</span>
        </label>
        <select
          id="replay-episode-sel"
          value={r.episode}
          onChange={(e) =>
            updateReplay({ episode: (e.target as HTMLSelectElement).value })
          }
        >
          <option value="" disabled>
            Select an episode
          </option>
          {episodes.value.map((ep) => (
            <option key={ep.key} value={ep.key}>
              {ep.label}
            </option>
          ))}
        </select>
      </div>

      <div class="hint">
        Replay loads the recorded actions from the dataset and plays them back
        on the robot. Use the dataset tab to scrub episodes first if you're not
        sure which to pick.
      </div>
    </>
  );
}
