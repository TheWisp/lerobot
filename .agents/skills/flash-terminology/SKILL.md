---
name: flash-terminology
description: Consult the repository glossary before using, explaining, or introducing FLASH tracking terminology in conversation, documentation, code comments, or reviews. Keep definitions and code citations consistent with the implementation.
---

# FLASH terminology

Before discussing FLASH tracking terminology, read the relevant entries in
[`src/lerobot/policies/flash/docs/glossary.md`](../../../src/lerobot/policies/flash/docs/glossary.md).
That Markdown file is the single source for definitions. Resolve its path from
the repository root; do not rely on a browser tab or a locally served copy.

When adding or correcting a definition, check it against the current
[tracking design](https://github.com/TheWisp/lerobot/blob/29a575fe8a8621ebf38990a8af578d3ed02f5ede/src/lerobot/policies/flash/docs/tracking_pipeline.md),
[maintained frame architecture](https://github.com/TheWisp/lerobot/blob/29a575fe8a8621ebf38990a8af578d3ed02f5ede/src/lerobot/gui/static/flash_architecture.html),
and the cited implementation. Respect documents' status: `architecture.md` is
explicitly superseded. If the intended design and running code differ, describe
both and label the difference; do not redefine a term to make a bug look intended.

- Use the glossary's exact names and meanings. Follow its code citations when
  explaining implementation behavior; a definition alone is not evidence that
  an estimate is correct or that a failure has been fixed.
- Link the first use of a technical term to its glossary heading. Also explain
  it briefly inline: remote readers may be unable to open local links. Do not
  give a localhost URL as the only explanation.
- Specify whether a point is 2D or 3D, its coordinate system, and its units.
  Distinguish point track IDs, object identity, object coordinates, and accepted
  images. Separate position differences in distance units from rotation
  differences in degrees; do not describe a pose as merely “too far away.”
- Prefer existing terms and plain descriptions. If a new technical term is
  necessary, first add a precise definition and a verified code citation to the
  glossary. If code and glossary disagree, identify the discrepancy and correct
  the definition before using it as fact. Label proposed behavior as proposed.
- Make each definition understandable without the conversation. Do not include
  “earlier wording,” fix-status notes, or replay results in the glossary. Keep
  diagnoses and dated evidence in the regression report; the glossary defines
  the terms used to explain them.

Keep definitions in the glossary, not in this skill or a second HTML version.
This skill applies to FLASH terminology; it does not require adding unrelated
general technical vocabulary to the tracking glossary.
