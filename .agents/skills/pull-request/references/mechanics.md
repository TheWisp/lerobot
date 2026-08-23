# PR mechanics — the parts that silently fail

Each item here has cost a real round trip. Check them before pushing a body.

## Links must be absolute

GitHub does **not** resolve relative markdown links in PR or issue bodies.
`[foo](src/lerobot/x.py)` renders as a dead link. Use the full
`https://github.com/<owner>/<repo>/blob/<sha>/<path>` URL.

Pre-check before every PR create/edit:

```bash
grep -nE "\]\([^h#)]" <body-file>   # any hit is a relative link
```

`^h` allows `https://`, `#` allows in-page anchors. Anything else is a bug.

## Images must be commit-pinned

GitHub proxies images through camo and **caches by source URL**. A branch-based
raw URL (`.../raw/main/shot.png`) will keep serving the first version it ever
saw, so a re-captured screenshot silently shows the old image.

Always pin to a commit SHA:

```
https://raw.githubusercontent.com/<owner>/<repo>/<full-sha>/<path>/shot.png
```

After re-capturing an image, you must bump the SHA in the body or the update is
invisible. This is the single most common way PR evidence goes stale and wrong.

## LFS-tracked images need a different host

`raw.githubusercontent.com` returns the **pointer text**, not the image, for
files tracked by Git LFS. Use:

```
https://media.githubusercontent.com/media/<owner>/<repo>/<full-sha>/<path>
```

If an embedded image renders as a wall of `version https://git-lfs...`, this is
why.

## A closing keyword closes the issue, negation included

GitHub closes an issue when `close/closes/closed/fix/fixes/fixed/resolve/
resolves/resolved` is followed by an issue reference — anywhere in a PR body, or
in a commit message merged to the default branch. It is pattern matching, with
no grammar and **no repository or organisation setting to disable it**.

It does not understand negation. Issue #98 was closed one second after PR #108
merged, by a sentence written to prevent exactly that:

> "It does not close #98. The option naming ... are untouched"

The issue then read as fixed while the defect it described was untouched. The
same sentence was in a commit body, so it would have fired twice.

Commit messages are covered by the `no-accidental-issue-close` commit-msg hook.
**PR and issue bodies are not** — they never pass through git. Check the body
file before every post or edit, alongside the relative-link grep above:

```bash
python scripts/lint/no_accidental_issue_close.py body.md
grep -nE "\]\([^h#)]" body.md
```

Mid-sentence, drop the keyword: `Refs #98`, `see #98`, `the problem #98
describes` all read identically to a human and are inert to GitHub. When you do
mean to close, put it on its own line as a trailer — `Closes #98`.

## Rebasing a branch that sits on another branch

A branch's base is not always `main`, and "rebase onto main" applied to a
stacked branch **flattens the stack**: the base's commits are absorbed into the
child, the layers stop being separately reviewable, and the base's work would
get reviewed twice. Nothing errors -- the rebase succeeds and looks clean,
which is why this has to be checked rather than assumed.

**Establish the topology first, and say it back before touching anything:**

```bash
gh pr list --head <branch> --json baseRefName          # the declared base, if a PR exists
git merge-base --is-ancestor origin/<candidate> <branch> && echo "stacked on <candidate>"
```

A bare "rebase" is not authorisation to change what a branch is stacked on. If
flattening genuinely looks better, ask.

**The two-step, when a stack exists.** Keep the child's pre-rebase tip: its
boundary with the old base is the only exact record of which commits are the
child's own.

```bash
CHILD_TIP=$(git rev-parse <child>)      # BEFORE anything
OLD_BASE=$(git rev-parse origin/<base>)

git checkout <base> && git rebase origin/main            # 1. base onto main
NEW_BASE=$(git rev-parse HEAD)

git rebase --onto "$NEW_BASE" "$OLD_BASE" "$CHILD_TIP"   # 2. only the child's own commits
```

Rebasing the child directly onto `$NEW_BASE` would replay the base's commits a
second time; `--onto` with the old base as the boundary is what excludes them.

**Verify before pushing** -- a rebase renames every commit, so SHA comparisons
prove nothing and only content does:

```bash
git merge-base --is-ancestor "$NEW_BASE" HEAD          # the stack is a stack again
git merge-base --is-ancestor origin/main "$NEW_BASE"   # the base actually moved
while read -r s; do git log --format=%s "$NEW_BASE"..HEAD | grep -Fqx "$s" || echo "LOST: $s"; done \
  < <(git log --format=%s "$OLD_BASE".."$CHILD_TIP")
```

A commit already cherry-picked to main drops itself here ("patch contents
already upstream") -- that is the mechanism working, not work being lost.
Confirm the subject is in `origin/main` before believing it.

Push both refs with `--force-with-lease=<ref>:<old-sha>`, base first: a child
pushed onto a base that has not moved leaves the remote stack inconsistent.

**Auto-resolving repetitive conflicts.** Version and fingerprint bookkeeping
conflicts on nearly every commit of a long branch. A resolver script is fine,
but it must **exit non-zero when it cannot resolve**, and the driver must check
that status before `git add` -- staging an unresolved file put conflict markers
into most of a 114-commit branch, which only surfaces later, when a bisect
lands on one.

## `gh pr edit` can fail silently

With Projects-classic enabled on the repo, `gh pr edit --body-file ...` may
report success while changing nothing. Use the API directly and verify:

```bash
gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -F body=@body.md
gh pr view <n> --json body -q .body | head -20
```

Always read the body back after editing. Do not assume the write landed.

## Capturing evidence

- **GUI stills** — use `scripts/gui/screenshot_gui.py` (`GuiScreenshotSession`).
  For pages with cross-origin iframes (MeshCat), that path fails; use
  `ffmpeg x11grab` instead of CDP `captureScreenshot`.
- **GUI video** — Playwright `record_video_dir` **with** the OOPIF-disable
  flags, otherwise the recording stutters.
- **Never point evidence capture at real datasets.** Synthesize throwaway
  datasets in a temp dir. Say so in the PR — it tells the reviewer the evidence
  is reproducible and that nothing of the user's was touched.
- Commit the rendered artefact (PNG, transcript), not the one-off script that
  produced it.
