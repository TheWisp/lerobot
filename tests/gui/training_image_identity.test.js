// Unit test for how a run's training image is described (run from pytest via
// test_training_image_identity.py, or directly with
// `node tests/gui/training_image_identity.test.js`).
//
// Locks the fix for: every run showed the bare tag `lerobot-training:dev-local`,
// which is rebuilt whenever the trainer changes, so runs spanning several
// rebuilds all claimed the same image.
// Pinned before the module loads: formatBuiltAt renders in local time, so
// without this every date assertion has to be a shape check.
process.env.TZ = "UTC";
const assert = require("assert");
const { text, formatBuiltAt } = require("../../src/lerobot/gui/static/training_image_identity.js");

const CREATED = "2026-08-10T14:22:31.123456789Z";
const REVISION = "9469cdf39abcdef0123456789abcdef012345678";

// ── The reason this exists: same tag, different builds, now distinguishable ──
const runA = text({ __image__: "lerobot-training:dev-local", __image_created__: CREATED, __image_revision__: REVISION });
const runB = text({
  __image__: "lerobot-training:dev-local",
  __image_created__: "2026-08-06T09:05:00Z",
  __image_revision__: "300ec75c4aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
});
assert.notStrictEqual(runA, runB, "two builds of the same tag must not render identically");
assert.ok(runA.includes("lerobot-training:dev-local"), "the tag is still shown");
assert.ok(runA.includes("9469cdf"), "revision is abbreviated to 7 chars");
assert.ok(!runA.includes(REVISION), "the full 40-char sha is not shown");

// ── Backward compatibility: runs that predate the recording ─────────────────
// Must render exactly as before: bare tag, no separator, no "built undefined".
assert.strictEqual(
  text({ __image__: "lerobot-training:dev-local" }),
  "lerobot-training:dev-local",
  "a run with no identity keys renders as the bare tag"
);
assert.strictEqual(text({}), "(default)", "no image key at all keeps the old default marker");

// "(default)" names nothing, and means something different once the pin moves.
assert.strictEqual(
  text({ __image_resolved__: "ghcr.io/thewisp/lerobot-training:proto-e6bf147" }),
  "ghcr.io/thewisp/lerobot-training:proto-e6bf147",
  "resolved tag replaces (default)"
);
assert.strictEqual(
  text({ __image__: "asked-for:tag", __image_resolved__: "actually-ran:tag" }),
  "actually-ran:tag",
  "what ran beats what was asked for"
);
assert.strictEqual(text(), "(default)", "missing args object must not throw");
assert.strictEqual(text(null), "(default)", "null args must not throw");

// Partial records: a date but no label, normal for any pre-label image.
assert.strictEqual(
  text({ __image__: "img", __image_created__: CREATED }),
  "img · built 2026-08-10 14:22",
  "date without revision"
);
assert.strictEqual(
  text({ __image__: "img", __image_revision__: REVISION }),
  "img · 9469cdf",
  "revision without date"
);

// ── Formatting ──────────────────────────────────────────────────────────────
// Minute precision: day precision would collapse same-day rebuilds back into
// looking identical, which is the original bug.
assert.strictEqual(formatBuiltAt("2026-08-10T09:05:00Z"), "2026-08-10 09:05");
assert.strictEqual(formatBuiltAt("2026-08-10T21:47:00Z"), "2026-08-10 21:47");

// Verbatim `docker image inspect {{.Created}}` output from real images here.
// An invented ISO string would not have caught the offset form, which is what
// a local build reports.
// The offset must actually be applied, not ignored: 22:58+01:00 is 21:58 UTC.
assert.strictEqual(formatBuiltAt("2026-06-08T22:58:17.472388969+01:00"), "2026-06-08 21:58");
assert.strictEqual(formatBuiltAt("2026-03-23T21:33:59.562202219Z"), "2026-03-23 21:33");

assert.strictEqual(formatBuiltAt(""), "", "empty in, empty out");
assert.strictEqual(formatBuiltAt(null), "", "null in, empty out");
assert.strictEqual(formatBuiltAt(undefined), "", "undefined in, empty out");
// If docker's format changes, the raw string beats an empty cell.
assert.strictEqual(formatBuiltAt("not-a-date"), "not-a-date", "unparsable passes through");

// ── The full record must not be mutated ─────────────────────────────────────
const args = { __image__: "img", __image_created__: CREATED, __image_revision__: REVISION };
assert.deepStrictEqual(
  args,
  { __image__: "img", __image_created__: CREATED, __image_revision__: REVISION },
  "must not mutate the run's args"
);

// All three keys together — what a run recorded after this change looks like.
assert.strictEqual(
  text({
    __image__: "lerobot-training:dev-local",
    __image_resolved__: "lerobot-training:dev-local",
    __image_created__: CREATED,
    __image_revision__: REVISION,
  }),
  "lerobot-training:dev-local · built 2026-08-10 14:22 · 9469cdf"
);

console.log("training_image_identity: all assertions passed");
