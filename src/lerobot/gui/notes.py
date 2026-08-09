"""Free-text notes on datasets, training runs, and checkpoints.

The GUI can find every dataset and checkpoint on a machine but cannot record
what a human knows about them, so that knowledge ends up in directory names
("...(Backup before adding RL)") or in a handoff document nobody updates.

A note is a plain file next to the artifact it describes:

- ``<dataset>/NOTES.md`` — the whole file is the note.
- ``<run>/NOTES.md`` — the run's note, then one ``## <checkpoint>`` section per
  checkpoint. One file per run, not per checkpoint: checkpoints get pruned and
  rotated, and a note should outlive the thing it describes.

Markdown, not JSON/YAML, because the file has to stay useful outside the GUI —
read over SSH on the robot host, grepped, diffed, rsync'd to a colleague. The
failure modes decide it too: a malformed JSON file loses *every* note in it,
while a mistyped heading here merges one section into its neighbour, visibly,
losing nothing.

Correctness rests on two invariants, both enforced in code and covered by
property tests over adversarial bodies:

**Round-trip.** ``write(p, text)`` then ``read(p)`` returns ``text``. ``write``
checks this before returning and raises if it does not hold, so a body that
would parse back differently fails loudly instead of being silently truncated.

**Non-interference.** Writing one entity's note never changes what any other
entity's ``read`` returns. Writes replace the target section's line range in
place; every line outside it is preserved byte-for-byte, and the file is never
reserialized from a parsed model.

What makes both achievable is that **parsing depends only on the bytes of the
file**. A section heading is recognised by its shape — a lone checkpoint-style
token — not by looking up which checkpoints currently exist on disk. An earlier
version passed the on-disk checkpoint names in, and that ambient input was the
source of three separate data-loss bugs: an empty placeholder ``checkpoints/``
dir shadowing the real one produced an empty key set, and a pruned checkpoint
dropped out of the set so its section merged into the run's note and was
destroyed by the next save.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

NOTES_FILENAME = "NOTES.md"

#: Key of the container's own note (the dataset's, or the run's) within a file.
SELF_KEY = ""

#: A section heading: a lone checkpoint-style token on a markdown heading line.
#: Matches every convention real trainers write — a bare step (``003000``) and
#: ``checkpoint-<N>`` — and nothing else, so a prose heading a human types
#: inside a note (``## Background``) stays prose. Deliberately syntactic: the
#: filesystem is not consulted, so the same bytes always parse the same way.
_HEADING = re.compile(r"^#{1,6}[ \t]*(\d+|checkpoint-\d+)[ \t]*$")


def is_section_heading(line: str) -> str | None:
    """Return the section key this line starts, or None if it is ordinary text."""
    m = _HEADING.match(line)
    return m.group(1) if m else None


# ============================================================================
# Locating the file and the section within it
# ============================================================================


def _is_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _is_run_dir(path: Path) -> bool:
    """True iff ``path`` is a training run directory (either checkpoint layout).

    ``lerobot.gui.api.models`` owns the authoritative layout detection; reuse it
    rather than re-deriving what counts as a step directory. Imported lazily so
    this module stays importable without the models router.

    ``<run>/output/checkpoints/...`` is the GUI orchestrator's layout — that
    extra ``output/`` level exists only because the run dir is bind-mounted into
    the training container. It satisfies the checkpoint-layout test on its own,
    so without excluding it a checkpoint's note would land in ``output/NOTES.md``
    instead of with its run.
    """
    from lerobot.gui.api.models import _dir_has_step_subdirs  # noqa: PLC0415

    if path.name == "output":
        return False
    return _dir_has_step_subdirs(path / "checkpoints") or _dir_has_step_subdirs(
        path / "output" / "checkpoints"
    )


def locate(path: str | Path) -> tuple[Path, str]:
    """Resolve an artifact path to ``(notes_file, section_key)``.

    A dataset or run owns its file under :data:`SELF_KEY`; a checkpoint is a
    section in its run's file, keyed by the checkpoint directory name.

    Precondition: ``path`` is absolute. It need not exist — a pruned checkpoint
    keeps its note in the run's file.
    Postcondition: the returned file sits in ``path`` or one of its ancestors.
    """
    p = Path(path).expanduser()
    assert p.is_absolute(), f"notes are keyed by absolute path, got {p}"

    if _is_dataset_root(p) or _is_run_dir(p):
        return p / NOTES_FILENAME, SELF_KEY

    for ancestor in p.parents:
        if _is_run_dir(ancestor) or (ancestor / "run.json").is_file():
            # A checkpoint under this run. The section is named by the
            # checkpoint's own directory ("003000"), not its path within the
            # run, so the note survives a layout change around it.
            return ancestor / NOTES_FILENAME, p.name
        if _is_dataset_root(ancestor):
            return ancestor / NOTES_FILENAME, p.name

    # Standalone artifact (a flat converted checkpoint, a dataset whose meta/
    # could not be found): it owns its own file.
    return p / NOTES_FILENAME, SELF_KEY


# ============================================================================
# Parsing
# ============================================================================


def _split(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(key, start_line, end_line)`` spans, end-exclusive.

    A section's span covers the lines *after* its heading. Pure function of the
    text — no filesystem, no caller-supplied key set.

    Postcondition: the first span is always ``SELF_KEY`` (possibly empty), and
    the spans tile the whole text in order.
    """
    lines = text.split("\n")
    spans: list[tuple[str, int, int]] = []
    current_key = SELF_KEY
    current_start = 0

    for i, line in enumerate(lines):
        key = is_section_heading(line)
        if key is None:
            continue
        spans.append((current_key, current_start, i))
        current_key = key
        current_start = i + 1

    spans.append((current_key, current_start, len(lines)))
    return spans


def _read_file(notes_file: Path) -> str:
    try:
        # Explicit UTF-8, not the locale default: notes are written in whatever
        # language the operator thinks in, and a process started with LANG=C
        # would otherwise fail to read back a note it wrote itself.
        return notes_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        logger.warning("Could not read %s", notes_file, exc_info=True)
        return ""


# ============================================================================
# Public API
# ============================================================================


def read_all(notes_file: Path) -> dict[str, str]:
    """Read every note in one file: ``{section_key: text}``.

    Empty sections are dropped, so the result contains only notes that exist.
    A section whose checkpoint has since been pruned is still returned — a note
    outlives the thing it describes.
    """
    text = _read_file(notes_file)
    if not text.strip():
        return {}

    lines = text.split("\n")
    out: dict[str, str] = {}
    for key, start, end in _split(text):
        body = "\n".join(lines[start:end]).strip()
        if body:
            out[key] = body
    return out


def read(path: str | Path) -> str:
    """Read one artifact's note, or ``""`` if it has none."""
    notes_file, key = locate(path)
    return read_all(notes_file).get(key, "")


class NoteNotRepresentableError(ValueError):
    """The note's text would not read back unchanged, so it was not stored.

    Raised when a body line would parse as a section heading — the only text a
    note cannot hold, because storing it would silently split the note in two.
    """


def write(path: str | Path, note: str) -> str:
    """Set one artifact's note. An empty note removes it.

    Only the target section's lines are rewritten; every other byte of the file
    is preserved, including content this module does not understand.

    Precondition: ``path`` is absolute.
    Postcondition (checked, not assumed): ``read(path) == note.strip()``, and
    every other entity's note is unchanged. The file is deleted once it holds
    nothing.

    Raises:
        NoteNotRepresentableError: a line in ``note`` would parse as a section
            heading. Rejecting is the honest option — the alternative is
            storing text that reads back as something else.
    """
    notes_file, key = locate(path)
    note = note.strip()

    for line in note.split("\n"):
        clashing = is_section_heading(line)
        if clashing is not None:
            raise NoteNotRepresentableError(
                f"the line {line.strip()!r} would be read back as the start of "
                f"checkpoint {clashing}'s note; rephrase it"
            )

    text = _read_file(notes_file)
    # Split the same way _split() does, so its line indices address this list.
    lines = text.split("\n")
    spans = _split(text)

    target = next((s for s in spans if s[0] == key), None)

    if target is not None:
        _, start, end = target
        heading_line = start - 1  # the "## <key>" line itself, if any
        if note:
            # Keep the blank lines that separated this section from the next
            # one, so editing a note never runs two sections together.
            trailing = 0
            while trailing < (end - start) and not lines[end - 1 - trailing].strip():
                trailing += 1
            replacement = note.split("\n") + [""] * trailing
        elif key != SELF_KEY and heading_line >= 0:
            # Dropping a checkpoint note takes its heading with it, so an empty
            # section never lingers.
            start, replacement = heading_line, []
        else:
            replacement = []
        lines[start:end] = replacement
    elif note:
        # No section for this key yet: append one, separated by exactly one
        # blank line from whatever the file already ends with.
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            lines.append("")
        lines.extend([f"## {key}", *note.split("\n")] if key != SELF_KEY else note.split("\n"))

    _persist(notes_file, "\n".join(lines).strip())

    # Verify the postcondition rather than trusting the splice. This is the
    # check that turns a parser/writer disagreement into a loud failure instead
    # of a note that looks saved in the UI and is truncated on disk.
    stored = read(path)
    if stored != note:
        raise NoteNotRepresentableError(
            f"note for {path} did not round-trip: stored {stored!r}, expected {note!r}"
        )
    return note


def _persist(notes_file: Path, text: str) -> None:
    """Write ``text`` atomically, or delete the file when nothing is left."""
    if not text:
        if notes_file.is_file():
            # safe-destruct: our own NOTES.md, and only once every note in it is gone
            notes_file.unlink()
        return

    if not notes_file.parent.is_dir():
        # Refuse rather than mkdir(parents=True): a note is written *about* an
        # artifact, so a missing directory means the caller has the wrong path,
        # and materialising it would leave an empty tree behind on delete.
        raise FileNotFoundError(f"no such artifact directory: {notes_file.parent}")

    try:
        tmp = notes_file.with_suffix(".md.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, notes_file)
    except OSError as e:
        raise OSError(f"Could not save notes to {notes_file}: {e}") from e
