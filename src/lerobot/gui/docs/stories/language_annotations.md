# Language Annotations (v3.1) — GUI parity

**Story — not implemented.** The dataset layer landed with the 2026-07 upstream
sync; the GUI has no surface for it. See [stories/README.md](README.md).

Upstream's account of the format and its tooling:
["Extensive Language Support in LeRobot Dataset"](https://huggingface.co/spaces/lerobot/robots-that-talk),
[`language_and_recipes.mdx`](../../../../../docs/source/language_and_recipes.mdx),
[`annotation_pipeline.mdx`](../../../../../docs/source/annotation_pipeline.mdx),
[`tools.mdx`](../../../../../docs/source/tools.mdx). The reference GUI is
upstream's [dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset),
whose Annotations panel is the bar to clear.

---

## Why it matters

An episode used to carry one sentence: `task`. The v3.1 schema replaces that with
typed language rows — the plan being followed, the subtask active right now, a
memory of what already happened, corrections shouted mid-episode, and grounded
question/answer pairs tied to a specific camera. That is the training data for
long-horizon and talking policies, and it is why a robot can be told "no, the
other one" halfway through a run.

We can already read, write, render, and generate all of it. We cannot see or
edit any of it.

## Desired outcome

An operator working in the Data tab can:

- **See** every language annotation a dataset carries, positioned in time —
  including the ones grounded in a specific camera view, drawn on that view.
- **Correct** any of them by hand, and author new ones, without leaving the GUI
  or learning the schema. Hand-authored annotations are indistinguishable from
  pipeline-generated ones to any downstream reader.
- **Generate** annotations for a dataset by running the VLM pipeline from a form,
  watch it progress, and land in the editor on the result to fix what it got
  wrong.

That last loop — generate, inspect, correct — is the point. Annotation quality
is what the operator is actually iterating on; everything else is plumbing to
make the iteration fast.

## Where we are

**The dataset layer is done.** Merged from upstream (`7ab4936b1` #3467,
`cec8ee0be` #3471, `279c6c7af` #3896): the column schema and its invariants
([datasets/language.py](../../../datasets/language.py)), row-to-message rendering
([language_render.py](../../../datasets/language_render.py)), the recipe layer
([configs/recipe.py](../../../configs/recipe.py),
[render_messages_processor.py](../../../processor/render_messages_processor.py)),
the VLM pipeline ([annotations/steerable_pipeline/](../../../annotations/steerable_pipeline/)),
and the `lerobot-annotate` CLI.

**The GUI layer is empty.** `grep -rn "language_persistent\|language_events"
src/lerobot/gui/` matches one line, and it is a TODO.

## The gap

1. **The editor writes an incompatible format.** Ours is the fork's older model —
   an int64 `subtask_index` resolved against a fork-only `meta/subtasks.parquet`
   ([feature_value_edits.py](../../../datasets/feature_value_edits.py),
   [gui/api/datasets.py](../../api/datasets.py),
   [feature_editing.js](../../static/feature_editing.js)). The upstream merge
   deleted that layer with no conflict; it was restored as-is to avoid regressing
   the editor. So two subtask models coexist in the tree, datasets from the two
   paths are not interchangeable, and every sync re-litigates it. **This is the
   one gap that costs us something today** — the others are absent features, this
   one is active debt.

2. **Only subtasks are visible at all.** Plans, memory, interjections, speech,
   VQA and traces are invisible even on a dataset that carries them.

3. **Nothing is drawn on the video.** VQA answers carry pixel bounding boxes and
   keypoints naming a specific camera. Upstream renders these over the player and
   lets you drag out new ones; we render nothing. Authoring a spatial annotation
   by drawing on a camera tile is the only interaction here with no existing
   analogue in the GUI.

4. **The pipeline is CLI-only.** `lerobot-annotate` wants a vLLM server, a model
   id, per-module flags, and a Hub push. Everything else in this GUI launches
   from a form.

Gap 1 gates the rest — until the GUI and the pipeline agree on where annotations
live, nothing built on top of either is worth much. Gaps 2–4 are independent of
each other.

## Decisions still open

- **Where annotations live in the UI.** Extending the Inspector keeps one editing
  surface and inherits the pending-edits pipeline; a dedicated panel matches
  upstream and has room for per-atom-type controls. Spatial authoring may
  outgrow the Inspector.
- **Whether existing datasets migrate automatically.** Rewriting a dataset on
  open is not something this GUI does elsewhere; an offer-to-migrate banner
  matches the schema-add path in [add_feature.md](../add_feature.md).
- **What happens to `subtask_index` readers.** HVLA S2 and its analysis scripts
  consume subtasks today. Either they port, or a compatibility shim survives —
  and a shim needs an expiry date decided up front.
- **How much of the VQA answer space is editable.** Boxes and keypoints are
  drawable; count, attribute and spatial answers are text and may need no
  bespoke widget at all.

## Name collision to resolve first

`TODO.md` already promises an **Annotations panel** for the MCP `tag_episode`
sidecar — per-episode key/value comments, unrelated to language columns. Two
features cannot both be "Annotations" in the Data tab. The MCP one is closer to
"Notes"; the language columns are what the format itself calls annotations.

## Out of scope

- Upstream's other visualizer panels (Statistics, Action Insights, Filtering).
  They overlap existing `TODO.md` items and are tracked there.
- Runtime tool execution (`say` actually speaking). Upstream marks the executor
  side unimplemented too; the dataset carries only the tool catalog.
- Recipe authoring in the GUI. Recipes are YAML consumed at training time, with
  no evidence yet that they need a visual editor.
