// Cached list of *opened* datasets. The Run sidebar uses this to populate
// the dataset selector in the teleop and policy forms (for record-dataset
// targets and inference dataset references).
//
// Same ensureLoaded pattern as profiles.svelte.ts — single in-flight load
// dedupes races, no UI-state coupling. The legacy app.js has a window-global
// `datasets` array that's mutated imperatively across many call sites; here we
// own a $state-backed copy so reactive consumers don't need to know about
// that legacy global.

import { getJson } from "../lib/api";
import type { DatasetSummary } from "../lib/types";

interface OpenedDataset {
  id: string;
  repo_id: string;
}

interface OpenedDatasetsResponse {
  datasets: OpenedDataset[];
}

const s: { value: DatasetSummary[]; loaded: boolean; error: string | null } =
  $state({ value: [], loaded: false, error: null });
let inflight: Promise<DatasetSummary[]> | null = null;

export const datasets = s;

export async function ensureDatasetsLoaded(): Promise<DatasetSummary[]> {
  if (s.loaded) return s.value;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const data = await getJson<OpenedDatasetsResponse>("/api/datasets/opened");
      s.value = data.datasets;
      s.loaded = true;
      s.error = null;
      return data.datasets;
    } catch (e) {
      s.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

export function reloadDatasets(): Promise<DatasetSummary[]> {
  s.loaded = false;
  return ensureDatasetsLoaded();
}
