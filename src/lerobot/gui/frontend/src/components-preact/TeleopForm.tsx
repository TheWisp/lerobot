// Preact + signals port of components/TeleopForm.svelte.
//
// The reactive win is identical: showRecord / showNewDataset are
// computed() values, and the template (JSX) reads them directly. No
// imperative _toggleHvla* equivalents anywhere.

import { computed } from "@preact/signals";
import type { JSX } from "preact";
import { robotProfiles, teleopProfiles } from "../stores-preact/profiles";
import { datasets } from "../stores-preact/datasets";
import {
  form,
  teleopShowRecordFields,
  teleopShowNewDatasetName,
  updateTeleop,
} from "../stores-preact/run";

export function TeleopForm(): JSX.Element {
  // computed values local to this component, derived from form state.
  // Equivalent to Svelte's $derived block, including the placeholder
  // task-options fallback for the prototype.
  const taskOptions = computed(() => {
    const dsId = form.value.teleop.recordDataset;
    if (!dsId || dsId === "__new__") {
      return ["Pick up the cube", "Assemble the parts", "Place into bin"];
    }
    return ["Pick up the cube"];
  });

  // Read once for terseness. Components re-run on signal change so we
  // pick up updates correctly.
  const t = form.value.teleop;

  return (
    <>
      <div class="form-grid">
        <label for="teleop-robot-sel">
          Robot <span class="req">*</span>
        </label>
        <select
          id="teleop-robot-sel"
          value={t.robot}
          onChange={(e) => updateTeleop({ robot: (e.target as HTMLSelectElement).value })}
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

        <label for="teleop-teleop-sel">
          Teleop <span class="req">*</span>
        </label>
        <select
          id="teleop-teleop-sel"
          value={t.teleop}
          onChange={(e) => updateTeleop({ teleop: (e.target as HTMLSelectElement).value })}
        >
          <option value="" disabled>
            Select a teleop profile
          </option>
          {teleopProfiles.value.value.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>

        <label for="teleop-fps">FPS</label>
        <input
          id="teleop-fps"
          type="number"
          value={t.fps}
          min="1"
          max="200"
          onInput={(e) =>
            updateTeleop({ fps: Number((e.target as HTMLInputElement).value) || 0 })
          }
        />
      </div>

      <div class="form-section">
        <div class="form-section-title">Record dataset</div>
        <div class="form-grid">
          <label for="teleop-record-dataset">Dataset</label>
          <select
            id="teleop-record-dataset"
            value={t.recordDataset}
            onChange={(e) =>
              updateTeleop({ recordDataset: (e.target as HTMLSelectElement).value })
            }
          >
            <option value="">None (don't record)</option>
            <option value="__new__">+ New dataset...</option>
            {datasets.value.value.map((d) => (
              <option key={d.id} value={d.id}>
                {d.repo_id || d.id}
              </option>
            ))}
          </select>
        </div>

        {teleopShowNewDatasetName.value && (
          <div class="form-grid">
            <label for="teleop-new-dataset-name">
              Name <span class="req">*</span>
            </label>
            <input
              id="teleop-new-dataset-name"
              type="text"
              value={t.newDatasetName}
              placeholder="my_new_dataset"
              onInput={(e) =>
                updateTeleop({
                  newDatasetName: (e.target as HTMLInputElement).value,
                })
              }
            />
          </div>
        )}

        {teleopShowRecordFields.value && (
          <div class="form-grid reveal">
            <label for="teleop-task">
              Task <span class="req">*</span>
            </label>
            <select
              id="teleop-task"
              value={t.task}
              onChange={(e) => updateTeleop({ task: (e.target as HTMLSelectElement).value })}
            >
              <option value="">— pick a task —</option>
              {taskOptions.value.map((task) => (
                <option key={task} value={task}>
                  {task}
                </option>
              ))}
            </select>

            <label for="teleop-num-episodes">Episodes</label>
            <input
              id="teleop-num-episodes"
              type="number"
              min="1"
              value={t.numEpisodes}
              onInput={(e) =>
                updateTeleop({
                  numEpisodes: Number((e.target as HTMLInputElement).value) || 0,
                })
              }
            />

            <label for="teleop-episode-time">Episode duration (s)</label>
            <input
              id="teleop-episode-time"
              type="number"
              min="1"
              value={t.episodeTime}
              onInput={(e) =>
                updateTeleop({
                  episodeTime: Number((e.target as HTMLInputElement).value) || 0,
                })
              }
            />

            <label for="teleop-reset-time">Reset duration (s)</label>
            <input
              id="teleop-reset-time"
              type="number"
              min="0"
              value={t.resetTime}
              onInput={(e) =>
                updateTeleop({
                  resetTime: Number((e.target as HTMLInputElement).value) || 0,
                })
              }
            />
          </div>
        )}
      </div>
    </>
  );
}
