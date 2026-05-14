// Preact + signals version of stores/datasets.svelte.ts.

import { signal } from "@preact/signals";
import { getJson } from "../lib/api";
import type { DatasetSummary } from "../lib/types";

interface DatasetsState {
  value: DatasetSummary[];
  loaded: boolean;
  error: string | null;
}

export const datasets = signal<DatasetsState>({
  value: [],
  loaded: false,
  error: null,
});

let inflight: Promise<DatasetSummary[]> | null = null;

export async function ensureDatasetsLoaded(): Promise<DatasetSummary[]> {
  if (datasets.value.loaded) return datasets.value.value;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      // GET /api/datasets returns a flat array of opened datasets — the
      // root endpoint is the list, not /api/datasets/opened (404).
      const data = await getJson<DatasetSummary[]>("/api/datasets");
      datasets.value = { value: data, loaded: true, error: null };
      return data;
    } catch (e) {
      datasets.value = {
        ...datasets.value,
        error: e instanceof Error ? e.message : String(e),
      };
      throw e;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

export function reloadDatasets(): Promise<DatasetSummary[]> {
  datasets.value = { ...datasets.value, loaded: false };
  return ensureDatasetsLoaded();
}
