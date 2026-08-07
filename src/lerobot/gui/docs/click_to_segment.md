# Click to segment

While a segmenter runs, click — or drag a box around — an object on a camera tile. It is
segmented, tracked, and added to the objects list as `object_N`. Both tabs.

## Why it exists

Naming is the part that fails. One frame, SO-107 rig, `sam3_track` at 672 px, same object:

| Prompt                                                                     | Detected      |
| -------------------------------------------------------------------------- | ------------- |
| **`robotic arm`**                                                          | **63,658 px** |
| `robot arm`                                                                | nothing       |
| `gripper`, `mechanical arm`, `white robot arm`, `robot`, `claw`, `machine` | nothing       |

One phrasing in eight, and not the natural one. It fails silently: your word is in the
panel, the frame rate is healthy, the tile has no outline. It gets worse as the rig fills
with things English has no word for — a printed fixture, one connector among five.

A click sidesteps vocabulary. Here it segments the tissue pack, which no prompt ever found.

## How it works

A clicked object is a concept like any other — same tracker session, chrome, compositing.
Only the seed differs: a point instead of a text detection.

| Gesture / control                 | Effect                                            |
| --------------------------------- | ------------------------------------------------- |
| Click a tile                      | Segment that object, add it as `object_N`         |
| Drag a box                        | Segment what is inside — always a **new** object  |
| Click inside one you already have | Refines it                                        |
| Rename the row                    | Relabels the tracked mask; never becomes a prompt |
| Row `×`                           | Removes it (the last row clears in place)         |
| Treatment buttons                 | Tint / Random / Blur, same as any region          |

The name is a **key, not a query** — call it `blob`. Two consequences: renaming relabels
the mask rather than re-detecting it, and a lost clicked object stays lost until clicked
again. Handing its label to the text detector made `stick` jump onto some other stick.

## Activation: there isn't any

No toggle. Picking a segmenter starts the worker and arms the tiles; the crosshair shows
when they are live. Nothing else on a tile consumes a click, so there is nothing to
disambiguate.

With nothing designated, the background region is the whole frame — so its treatment stays
off until an object exists. Otherwise picking a model would bury the scene in static, and
`Process dataset` would write that.

## Traps found

**Prompt coordinates normalise against the native 1008 px**, while `_pv()` feeds the
configured preset — a point lands `1008/672 = 1.5x` out (measured 1.47–1.52x). Nasty
because the misplaced mask looks plausible: one click returned 47% of the frame.

**The control channel is state, not events.** It is re-read every frame and never consumed,
so a click replayed ~13x/second. Ops carry a monotonic `click_seq` and apply once.

**The server assigns that id, not the browser.** It was `Date.now()`-derived per client, and
the worker ignores any id below the highest it has applied — so with two clients (two tabs
landing on the same millisecond, or two machines with skewed clocks) one client's gestures
were dropped in silence. Measured: a session where six objects were seeded and no removal
ever arrived, so rows left the panel while their detections stayed in the scene. A dropped
op that is not simply a replay is now logged as a warning, since a lost gesture and a
gesture that did nothing are otherwise identical.

**One latched slot loses racing writes**, so the latest op rides along on every control
write — server-side only, and dropped on teardown. A client-held copy would outlive the
worker: a respawn resets `click_seq` to 0, so a stale op would re-seed from a click made
against a different dataset.

**Chrome colour must not depend on the other objects.** It was the concept's index in the
list of objects VISIBLE this frame, so removing a row — or an object dropping out of view for
a frame — recoloured everything after it. Colour is assigned per name on first sight and
released only when the object is explicitly deleted: never reassigning at all sounds safer
but exhausts the 8-entry palette in one session of clicking and deleting, after which new
objects fall back to a hash and stop being reliably distinguishable.

**Names must never be reused.** Numbering by row count reused one after a delete and the
worker replaced the live object holding it — a row labelled "tissue pack" showing a yellow
cube. Names come from a monotonic per-camera counter.

## Cost

**~24 ms base, plus 3.9 ms per additional object** at 672 px. Clicked and typed objects
share one `MAX_OBJECTS` budget, because they share one tracker session.

## Boxes

The gesture resolves on **release**: under 5 display px of travel is a click, more is a
box. Nothing is ambiguous at press time, so no mode is needed. A sliver under 3 frame px
is discarded; a release off-frame clamps.

**A box always creates a new object, never refines.** A point inside a mask is evidence of
sameness; a box enclosing one is not — surrounding something is how you select the thing
next to it. Boxes also replaced negative points: constraining up front beats subtracting a
distractor, without the anchor bookkeeping.

A box can be read by either model, selectable because they fail differently. The box is the
only prompt in both — typed object names belong to their own rows and play no part, or a
gesture would mean different things depending on rows it has nothing to do with. Measured
with a ring touching a dowel:

| `box read by` | What it does                                    | Result on that scene |
| ------------- | ----------------------------------------------- | -------------------- |
| **Tracker**   | Cuts out whatever the box encloses              | Merged both (0.445)  |
| **Detector**  | Shown the box as an example, searches the image | Took the dowel (0.0) |

Neither is good there, which is why neither is a default worth hiding. An earlier version
fed the typed concepts to the Detector alongside the box, which did score 0.949 — but that
number came from the word `green ring` being in another row, not from the box, so it
measured the wrong thing and made the gesture depend on unrelated state.

Handlers are delegated per grid, not per tile: the data tab's grid outlives its tiles
(`innerHTML` is replaced each episode), so per-tile listeners would bind to dead elements
and per-grid ones would pile up.

## Cross-camera: what was tried, and why it failed

A click marks a place in one image, so it does not carry to other cameras. SAM3 accepts
`text_embeds` — the (1, 32, 256) block a word produces — so the obvious repair is to fit
those 8192 numbers until the detector reproduces the clicked mask, then reuse them.

**It half-works, and the failing half is the one that matters.** One point click on the
green ring, top camera, frame 400, 60 steps, model frozen:

|                                       | result            |
| ------------------------------------- | ----------------- |
| fit time                              | 1.9 s             |
| reproduces the clicked mask           | IoU 0.96          |
| finds it on front / left wrist        | IoU 0.988 / 0.983 |
| finds it 240 frames earlier           | IoU 0.939         |
| **instances returned**                | **2–5, not 1**    |
| **frames where the object is absent** | **still fires**   |

Every IoU is "best of the several masks returned", which is not "it found the object". The
vector includes the target and over-includes others. Penalising non-target queries removes
the absent-frame firing but not the extra instances; pushing harder collapses it to
nothing.

Two dead ends worth not repeating:

- **Seeding the fit from a box in clutter.** The box merged ring and dowel (IoU 0.445), and
  the detector cannot express two objects as one instance, so the fit plateaued at dice
  0.42. A point click gave 0.93 and converged to 0.028. Verify the seed is one object first.
- **Reading the vector back as text.** The nearest word is whatever seeded the fit
  (`thing`, +0.96); the true description sits at +0.10. It is a key, not a name.

**Why, per the literature.** Visual-prompt objectives are contrastive, and
[T-Rex2](https://arxiv.org/abs/2403.14610) is explicit that abundant negatives are
essential. One positive with no negative set is the textbook recipe for over-inclusion.

What the field does instead: [PerSAM](https://arxiv.org/abs/2305.03048) is training-free —
it builds positive _and negative_ location priors from one image + mask and tunes **2**
parameters, not 8192. T-Rex2 and DINOv **train** the model to accept exemplars; SAM3's
exemplar route is in that family. A PerSAM-shaped attempt here failed instructively:
pooling SAM3's FPN features over the mask gave peak cosine 0.95–0.96 on _every_ frame,
including those without the ring — no rejection signal. A faithful attempt needs SAM's
image-encoder features and PerSAM's target-guided attention.

### What did work, parked

Searching real **words** beats inventing a vector. Score each candidate as overlap minus
0.15 per instance beyond the first — includes, does not over-include. Same click, 40 words:

| word         | score | IoU vs the click | instances |
| ------------ | ----- | ---------------- | --------- |
| round object | 0.961 | 0.961            | 1         |
| teal ring    | 0.960 | 0.960            | 1         |
| green ring   | 0.931 | 0.931            | 1         |
| plastic part | 0.802 | 0.952            | 2         |
| object       | 0.646 | 0.946            | 3         |
| hole         | −3.28 | 0.174            | 24        |

**0.6 s for 40 words, and exactly one instance** — the exclusivity every fitted vector
lacked, because a real word denotes a real category. The instance penalty demotes
permissive words automatically; nobody would have thought to ban `toy` (4) or `hole` (24).
A real word also dissolves cross-camera, persistence and naming at once.

**The catch: that vocabulary was hand-written and contained the answer.** Leakage. The
premise of clicking is that you have no word, so candidates must be generated:

    click -> mask -> crop -> VLM proposes 3-6 nouns -> this search verifies -> best word

The proposer only has to get the right word into the set; the search decides which
reproduces the click, so a mediocre one is fine. Candidates: Florence-2 base (0.23B, native
region→category task), RAM++ (returns a tag list), or a small VLM. It degrades honestly —
nothing scoring well is evidence the object is unnameable, and the plain click still works.

Parked, not built: the same VLM step is wanted for auto-labelling, so it should ride that
work. **The cross-camera problem stays open** — clicks are per-camera, as the row tooltip says.

## Limitations

- **Negative prompts.** A box is the way to exclude context.
- **Persistence.** Clicks are lost on any discontinuity — a scrub, an episode change, and
  the wrap at the end of playback, as well as a worker respawn. A clicked object's seed was
  a point on one frame, and the only re-seed path is the text detector, which by
  construction cannot find it; so a discontinuity forgets it rather than searching for a
  label that matches nothing. Note what counts as one: continuity is a short forward step
  within the same episode, not exactly `last + 1`. Requiring adjacency made a single dropped
  frame a new stream, and playback drops frames whenever inference is the slower side — so
  clicked objects used to die at the first skip, mid-episode. A dataset is
  deterministic so a stored mask at (episode, frame) would reproduce; against a live camera
  a stored coordinate is a lie. Appearance-based retrieval is ruled out above, so this
  probably waits on finding a real word, which survives a restart by itself.
- **Panel/worker reconciliation.** The panel adds a row optimistically; when the worker
  folds a click into an existing object, the row has no mask behind it. The fix is for the
  worker to report its object list back, which no channel does.
- **Two fast gestures can produce one object.** The worker picks up at most one gesture per
  frame it processes, so a second click ~200 ms after the first can replace it or merge
  into it — while the panel has already added both rows. Deliberate clicking is reliable.
  The fix is a queue rather than the current single slot, which reconciliation needs too.
