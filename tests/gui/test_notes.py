"""Notes store + API tests.

Every dataset/run here is synthesised in ``tmp_path`` — writing a note creates a
file inside the directory it describes, so pointing these at a real dataset
would scribble on it.
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui import notes


def make_dataset(root: Path, episodes: int = 3, frames: int = 100) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": frames, "fps": 30, "robot_type": "so101"})
    )
    return root


def make_run(run_dir: Path, steps=(1000, 2000), nested_output: bool = False) -> Path:
    ckpts = run_dir / ("output/checkpoints" if nested_output else "checkpoints")
    for step in steps:
        pretrained = ckpts / f"{step:06d}" / "pretrained_model"
        pretrained.mkdir(parents=True)
        (pretrained / "config.json").write_text(json.dumps({"type": "act"}))
        (pretrained / "train_config.json").write_text(
            json.dumps({"dataset": {"repo_id": "acme/demo"}, "policy": {"type": "act"}})
        )
    return run_dir


def notes_file(container: Path) -> Path:
    return container / notes.NOTES_FILENAME


# ============================================================================
# Locating
# ============================================================================


def test_dataset_owns_its_file(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    assert notes.locate(ds) == (notes_file(ds), notes.SELF_KEY)


def test_run_owns_its_file(tmp_path):
    run = make_run(tmp_path / "run")
    assert notes.locate(run) == (notes_file(run), notes.SELF_KEY)


def test_checkpoint_is_a_section_of_its_run(tmp_path):
    run = make_run(tmp_path / "run")
    assert notes.locate(run / "checkpoints" / "001000") == (notes_file(run), "001000")


def test_checkpoint_is_a_section_in_the_gui_layout(tmp_path):
    """The GUI orchestrator writes ``<run>/output/checkpoints/<step>``."""
    run = make_run(tmp_path / "run", nested_output=True)
    assert notes.locate(run / "output" / "checkpoints" / "002000") == (notes_file(run), "002000")


def test_standalone_dir_owns_its_file(tmp_path):
    lone = tmp_path / "converted_ckpt"
    lone.mkdir()
    assert notes.locate(lone) == (notes_file(lone), notes.SELF_KEY)


def test_relative_path_is_rejected():
    with pytest.raises(AssertionError):
        notes.locate(Path("relative/dataset"))


# ============================================================================
# Read / write
# ============================================================================


def test_dataset_note_is_the_whole_file(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    notes.write(ds, "left arm barely moves")

    assert notes_file(ds).read_text() == "left arm barely moves\n"
    assert notes.read(ds) == "left arm barely moves"


def test_missing_note_reads_empty(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    assert notes.read(ds) == ""


def test_run_and_checkpoint_notes_share_one_file(tmp_path):
    run = make_run(tmp_path / "run")
    notes.write(run, "normfloor + boundary fix")
    notes.write(run / "checkpoints" / "002000", "30k stalls")

    assert notes_file(run).read_text() == "normfloor + boundary fix\n\n## 002000\n30k stalls\n"
    assert notes.read(run) == "normfloor + boundary fix"
    assert notes.read(run / "checkpoints" / "002000") == "30k stalls"
    assert notes.read(run / "checkpoints" / "001000") == ""


def test_editing_one_section_leaves_the_others_alone(tmp_path):
    run = make_run(tmp_path / "run")
    notes.write(run, "run note")
    notes.write(run / "checkpoints" / "001000", "first")
    notes.write(run / "checkpoints" / "002000", "second")

    notes.write(run / "checkpoints" / "001000", "first, revised")

    assert notes.read(run) == "run note"
    assert notes.read(run / "checkpoints" / "001000") == "first, revised"
    assert notes.read(run / "checkpoints" / "002000") == "second"


def test_multiline_notes_round_trip(tmp_path):
    run = make_run(tmp_path / "run")
    text = "Val loss bottoms near 3k.\n\nCompare 2k/3k/4k against 10k — do not assume 10k is best."
    notes.write(run, text)
    assert notes.read(run) == text


def test_clearing_a_dataset_note_deletes_the_file(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    notes.write(ds, "x")
    notes.write(ds, "")

    assert notes.read(ds) == ""
    assert not notes_file(ds).exists()


def test_clearing_a_checkpoint_note_removes_its_heading(tmp_path):
    run = make_run(tmp_path / "run")
    notes.write(run, "run note")
    notes.write(run / "checkpoints" / "002000", "temporary")

    notes.write(run / "checkpoints" / "002000", "")

    assert notes_file(run).read_text() == "run note\n"
    assert notes.read(run) == "run note"


def test_clearing_every_note_deletes_the_file(tmp_path):
    run = make_run(tmp_path / "run")
    notes.write(run, "run note")
    notes.write(run / "checkpoints" / "002000", "ckpt note")

    notes.write(run / "checkpoints" / "002000", "")
    notes.write(run, "")

    assert not notes_file(run).exists()


def test_note_outlives_a_deleted_checkpoint(tmp_path):
    """Notes are loosely tied: pruning a checkpoint must not lose its note."""
    import shutil

    run = make_run(tmp_path / "run")
    notes.write(run / "checkpoints" / "002000", "this one stalled")
    shutil.rmtree(run / "checkpoints" / "002000")

    assert notes.read(run / "checkpoints" / "002000") == "this one stalled"


# ============================================================================
# Non-ASCII — operators write notes in the language they think in
# ============================================================================


def test_chinese_note_round_trips(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    text = "左臂基本不动，覆盖不足。核桃/圆柱相对位置固定。"
    notes.write(ds, text)

    assert notes.read(ds) == text
    assert notes_file(ds).read_bytes().decode("utf-8").strip() == text


def test_files_are_utf8_whatever_the_locale_is(tmp_path, monkeypatch):
    """A process started with LANG=C must still read back what it wrote."""
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    run = make_run(tmp_path / "run")
    notes.write(run, "训练时没有 normalization floor")
    notes.write(run / "checkpoints" / "002000", "30k 会卡住 — 需要 E-stop")

    assert notes.read(run) == "训练时没有 normalization floor"
    assert notes.read(run / "checkpoints" / "002000") == "30k 会卡住 — 需要 E-stop"


def test_emoji_and_mixed_scripts_survive(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    text = "⚠️ walnut 数据集 — σ≈18,083, do not train 🚫"
    notes.write(ds, text)
    assert notes.read(ds) == text


def test_api_serves_non_ascii(client):
    ds = str(client.ds)  # type: ignore[attr-defined]
    text = "左臂基本不动"
    assert client.put("/api/notes", params={"path": ds}, json={"note": text}).json()["note"] == text
    assert client.get("/api/notes", params={"path": ds}).json()["note"] == text
    assert client.get("/api/notes/bulk", params={"paths": [ds]}).json()[ds] == text


# ============================================================================
# Hand-edited files — the reason the format is markdown
# ============================================================================


def test_a_prose_heading_stays_part_of_the_note(tmp_path):
    """``## Background`` is not a checkpoint, so it must not split the note."""
    run = make_run(tmp_path / "run")
    text = "Summary line.\n\n## Background\nWhy this run exists."
    notes.write(run, text)

    assert notes.read(run) == text


def test_a_hand_written_file_is_read_as_is(tmp_path):
    run = make_run(tmp_path / "run")
    notes_file(run).write_text("typed over ssh\n\n## 002000\nstalls\n")
    assert notes.read(run) == "typed over ssh"
    assert notes.read(run / "checkpoints" / "002000") == "stalls"


def test_a_mistyped_checkpoint_heading_orphans_but_loses_nothing(tmp_path):
    """A typo'd checkpoint heading becomes a section nothing claims.

    The text stays in the file, is not attributed to the run, and — the part
    that matters — is not destroyed when the run's own note is rewritten.
    """
    run = make_run(tmp_path / "run")
    notes_file(run).write_text("run note\n\n## 00200\nmeant for 002000\n")

    assert notes.read(run / "checkpoints" / "002000") == ""
    assert notes.read(run) == "run note"
    assert notes.read_all(notes_file(run))["00200"] == "meant for 002000"

    notes.write(run, "run note, revised")
    assert "meant for 002000" in notes_file(run).read_text(encoding="utf-8")


def test_an_unknown_heading_belongs_to_the_note_containing_it(tmp_path):
    """What the editor shows is what gets stored — no hidden text to clobber."""
    run = make_run(tmp_path / "run")
    notes_file(run).write_text("run note\n\n## 002000\nckpt note\n\n## Scratch\nkeep me\n")

    shown = notes.read(run / "checkpoints" / "002000")
    assert shown == "ckpt note\n\n## Scratch\nkeep me"

    notes.write(run / "checkpoints" / "002000", shown)
    assert notes.read(run / "checkpoints" / "002000") == shown
    assert notes.read(run) == "run note"


def test_lines_outside_the_edited_section_are_untouched(tmp_path):
    run = make_run(tmp_path / "run")
    notes_file(run).write_text("run note\n\n## 001000\nfirst\n\n## 002000\nsecond\n")

    notes.write(run / "checkpoints" / "001000", "rewritten")

    assert notes_file(run).read_text() == "run note\n\n## 001000\nrewritten\n\n## 002000\nsecond\n"


# ============================================================================
# API
# ============================================================================


@pytest.fixture
def client(tmp_path):
    from lerobot.gui.api import notes as notes_api

    app = FastAPI()
    app.include_router(notes_api.router)
    ds = make_dataset(tmp_path / "ds")
    run = make_run(tmp_path / "run")
    with TestClient(app) as c:
        c.ds, c.run = ds, run  # type: ignore[attr-defined]
        yield c


def test_api_get_put_roundtrip(client):
    ds = str(client.ds)  # type: ignore[attr-defined]
    assert client.get("/api/notes", params={"path": ds}).json()["note"] == ""

    res = client.put("/api/notes", params={"path": ds}, json={"note": "  left arm static  "})
    assert res.status_code == 200
    assert res.json()["note"] == "left arm static"
    assert client.get("/api/notes", params={"path": ds}).json()["note"] == "left arm static"


def test_api_checkpoint_note_needs_no_key_list_from_the_caller(client):
    """The server derives the run's checkpoint names itself."""
    ckpt = str(client.run / "checkpoints" / "002000")  # type: ignore[attr-defined]
    client.put("/api/notes", params={"path": ckpt}, json={"note": "stalls"})

    assert client.get("/api/notes", params={"path": ckpt}).json()["note"] == "stalls"
    assert client.get("/api/notes", params={"path": str(client.run)}).json()["note"] == ""  # type: ignore[attr-defined]


def test_api_bulk_returns_an_entry_for_each_path(client):
    ds, run = str(client.ds), str(client.run)  # type: ignore[attr-defined]
    client.put("/api/notes", params={"path": ds}, json={"note": "dataset note"})

    got = client.get("/api/notes/bulk", params={"paths": [ds, run]}).json()
    assert got == {ds: "dataset note", run: ""}


def test_api_bulk_of_a_runs_checkpoints(client):
    run = client.run  # type: ignore[attr-defined]
    client.put("/api/notes", params={"path": str(run / "checkpoints" / "001000")}, json={"note": "a"})

    got = client.get(
        "/api/notes/bulk", params={"paths": [str(run / "checkpoints" / c) for c in ("001000", "002000")]}
    ).json()
    assert got == {str(run / "checkpoints" / "001000"): "a", str(run / "checkpoints" / "002000"): ""}


def test_api_rejects_a_relative_path(client):
    assert client.get("/api/notes", params={"path": "relative/thing"}).status_code == 400
    assert client.put("/api/notes", params={"path": "relative/thing"}, json={"note": "x"}).status_code == 400


# ============================================================================
# Invariants — the properties the CRUD surface is built to hold
# ============================================================================

#: Bodies chosen to attack the parser: prose that looks structural, headings
#: that are not checkpoints, whitespace edges, non-ASCII, and the empty note.
ADVERSARIAL_BODIES = [
    "plain",
    "two\nlines",
    "blank\n\nline between",
    "## Background\nprose heading, not a checkpoint",
    "#### Deeper heading\nstill prose",
    "trailing whitespace   ",
    "   leading whitespace",
    "left arm static #walnut",
    "左臂基本不动 ⚠️",
    "a line that mentions checkpoint-3000 inline",
    "## not-a-checkpoint",
    "---\nhorizontal rule",
]


@pytest.mark.parametrize("body", ADVERSARIAL_BODIES)
def test_roundtrip_for_every_entity_kind(tmp_path, body):
    """write(x, t) then read(x) == t.strip(), for datasets, runs, checkpoints."""
    ds = make_dataset(tmp_path / "ds")
    run = make_run(tmp_path / "run")
    for target in (ds, run, run / "checkpoints" / "001000"):
        assert notes.write(target, body) == body.strip()
        assert notes.read(target) == body.strip()


@pytest.mark.parametrize("body", ADVERSARIAL_BODIES)
def test_writing_one_note_never_disturbs_another(tmp_path, body):
    """Non-interference: a write is invisible to every other entity's read."""
    run = make_run(tmp_path / "run", steps=(1000, 2000, 3000))
    others = {
        run: "the run",
        run / "checkpoints" / "001000": "first",
        run / "checkpoints" / "003000": "third",
    }
    for target, text in others.items():
        notes.write(target, text)

    notes.write(run / "checkpoints" / "002000", body)

    for target, text in others.items():
        assert notes.read(target) == text, f"{target} changed"


def test_a_body_that_cannot_round_trip_is_rejected(tmp_path):
    """The one text a note cannot hold fails loudly instead of truncating."""
    run = make_run(tmp_path / "run")
    notes.write(run, "keep me")

    with pytest.raises(notes.NoteNotRepresentableError):
        notes.write(run, "summary\n\n## 001000\nthis would split the note")

    # The parser is line-based on purpose, so a fenced block does not shelter a
    # heading. Rejecting is the honest outcome; silently storing it would mean
    # reading back two notes where one was written.
    with pytest.raises(notes.NoteNotRepresentableError):
        notes.write(run, "```\n## 003000\n```")

    assert notes.read(run) == "keep me"


def test_rejected_body_is_a_400_not_a_500(client):
    res = client.put(
        "/api/notes",
        params={"path": str(client.run / "checkpoints" / "001000")},  # type: ignore[attr-defined]
        json={"note": "text\n## 002000\nsplit"},
    )
    assert res.status_code == 400
    assert "002000" in res.json()["detail"]


def test_pruned_checkpoint_note_is_readable_and_survives_a_run_write(tmp_path):
    """The bug the ambient key set caused: a pruned section merging and dying."""
    import shutil

    run = make_run(tmp_path / "run")
    notes.write(run, "the run")
    notes.write(run / "checkpoints" / "002000", "this one stalled")
    shutil.rmtree(run / "checkpoints" / "002000")

    assert notes.read(run / "checkpoints" / "002000") == "this one stalled"
    assert notes.read(run) == "the run"

    notes.write(run, "the run, revised")
    assert notes.read(run / "checkpoints" / "002000") == "this one stalled"


def test_an_empty_placeholder_checkpoints_dir_does_not_shadow_the_real_one(tmp_path):
    """GUI layout with a stray empty <run>/checkpoints/ still resolves sections."""
    run = make_run(tmp_path / "run", nested_output=True)
    (run / "checkpoints").mkdir()

    ckpt = run / "output" / "checkpoints" / "002000"
    notes.write(run, "the run")
    notes.write(ckpt, "30k stalls")

    assert notes.read(run) == "the run"
    assert notes.read(ckpt) == "30k stalls"


# ============================================================================
# Regressions — one test per defect found in review
# ============================================================================


def test_write_refuses_to_create_an_artifact_directory(tmp_path):
    """A PUT to a bogus path must not materialise a directory tree."""
    ghost = tmp_path / "not" / "a" / "real" / "dataset"

    with pytest.raises(FileNotFoundError):
        notes.write(ghost, "note about nothing")

    assert not (tmp_path / "not").exists()


def test_api_missing_artifact_directory_is_a_404(client, tmp_path):
    res = client.put("/api/notes", params={"path": str(tmp_path / "ghost" / "ds")}, json={"note": "x"})
    assert res.status_code == 404


def test_bulk_read_is_unaffected_by_a_stray_checkpoints_dir(client):
    """Reading notes must not depend on which checkpoints dir wins a probe."""
    run = client.run  # type: ignore[attr-defined]
    (run / "output").mkdir(exist_ok=True)
    (run / "output" / "checkpoints").mkdir(exist_ok=True)

    ckpt = str(run / "checkpoints" / "002000")
    client.put("/api/notes", params={"path": ckpt}, json={"note": "still mine"})

    got = client.get("/api/notes/bulk", params={"paths": [ckpt, str(run)]}).json()
    assert got[ckpt] == "still mine"
    assert got[str(run)] == ""
